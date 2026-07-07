#!/usr/bin/env python
"""Full Qwen demo behind books/callflow.html — the exact code that produced every capture.

Follows the torch -> vLLM -> Inductor -> Triton/C++ call flow on ONE op
(`fused_add_rms_norm`) across two targets.

    .venv/bin/python books/captures/callflow/capture_callflow.py cuda          # full vLLM engine (Triton)
    .venv/bin/python books/captures/callflow/capture_callflow.py cpu-cpp       # torch.compile front-end -> C++/OpenMP
    .venv/bin/python books/captures/callflow/capture_callflow.py cpu-vllm-op   # vLLM RMSNorm CustomOp under stock inductor

Environment used for the shipped captures:
  GPU    : NVIDIA RTX 4090 (24 GB), driver 610.43.02, CUDA 13.3
  CPU    : Intel i9-13900F (32 threads)
  Python : 3.12.13   torch 2.11.0+cu130   triton 3.6.0
  vLLM   : 0.1.dev18451+ge35e4cf43  (CUDA build)
  Model  : Qwen/Qwen2.5-0.5B-Instruct  (24 layers, hidden 896, bf16)

NOTE: the full vLLM *engine* only runs the `cuda` mode here. A CPU engine run is not possible in
this env — this is a CUDA build (CpuPlatform is gated on the version string, platforms/__init__.py:162)
and no `torch==2.11.0+cpu` wheel exists for 2.11 on any index. The two cpu-* modes therefore drive the
*same Inductor front-end* directly, which is exactly the branch the book explains.
"""
import os
import sys
import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
H = 896  # Qwen2.5-0.5B hidden size


# --------------------------------------------------------------------------- #
# CUDA — full vLLM engine: Dynamo -> VllmBackend split -> Inductor -> Triton   #
# --------------------------------------------------------------------------- #
def run_cuda(out_dir):
    # These env vars MUST be set before importing torch/vllm. A fresh cache root forces a real
    # recompile so the depyf dump (and the generated Triton kernels) are produced from scratch.
    os.environ.setdefault("VLLM_CACHE_ROOT", os.path.join(out_dir, "vllm_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(out_dir, "inductor"))
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    depyf_dir = os.path.join(out_dir, "depyf")

    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationConfig, CompilationMode

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.35,
        max_model_len=2048,
        max_num_seqs=8,
        enforce_eager=False,  # enable torch.compile + cudagraph
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,   # piecewise: dynamo -> split -> inductor -> cudagraph
            debug_dump_path=depyf_dir,           # depyf dumps FX graphs + generated kernels here
        ),
    )
    print(f"[cuda] engine + compile init: {time.time() - t0:.1f}s", flush=True)

    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=16, temperature=0.0),
    )
    print("[cuda] generation:", repr(out[0].outputs[0].text), flush=True)
    print(f"[cuda] artifacts under {depyf_dir}/rank_0_dp_0/ :", flush=True)
    print("         __compiled_fn_1.before_split.0.py     # Dynamo/AOT FX graph (pre-split)", flush=True)
    print("         __compiled_fn_1.after_split.0.py       # submod_0..submod_48 (compute/attn)", flush=True)
    print("         __compiled_fn_1.kernel_*.py            # generated Triton kernels", flush=True)
    print(f"       and {os.environ['VLLM_CACHE_ROOT']}/torch_compile_cache/<h>/rank_0_0/backbone/", flush=True)
    print("         artifact_compile_range_1_8192_subgraph_{0,1,24}   # 3 deduped inductor artifacts", flush=True)


# --------------------------------------------------------------------------- #
# CPU — plain-torch RMSNorm -> Inductor CPU backend -> C++/OpenMP kernel       #
# --------------------------------------------------------------------------- #
def run_cpu_cpp():
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ["TORCH_LOGS"] = "output_code"  # prints the generated C++ wrapper + kernel
    import torch

    torch.manual_seed(0)

    def rmsnorm(x, w, eps=1e-6):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * w).to(x.dtype)

    w = torch.randn(H, dtype=torch.bfloat16)
    x = torch.randn(8, H, dtype=torch.bfloat16)
    y0 = rmsnorm(x, w)
    cm = torch.compile(rmsnorm, backend="inductor", fullgraph=True)  # Dynamo -> FX -> Inductor -> C++
    with torch.no_grad():
        y1 = cm(x, w)
    # Inductor emits cpp_fused__to_copy_add_mean_mul_pow_rsqrt_0 via async_compile.cpp_pybinding,
    # vectorized with at::vec::Vectorized<at::BFloat16> + vec_reduce_all.
    print("[cpu-cpp] max|delta| vs eager:", (y1.float() - y0.float()).abs().max().item(), flush=True)
    print("[cpu-cpp] see the cpp_fused_* kernel in the TORCH_LOGS=output_code dump above.", flush=True)


# --------------------------------------------------------------------------- #
# CPU — vLLM's REAL RMSNorm CustomOp under stock inductor (no VllmBackend)      #
#       shows the vllm_ir op staying opaque + the Linear becoming extern mm     #
# --------------------------------------------------------------------------- #
def run_cpu_vllm_op():
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ["TORCH_LOGS"] = "output_code"
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):  # CustomOp needs an ambient vLLM config
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = RMSNorm(H, eps=1e-6)
                self.proj = torch.nn.Linear(H, H, bias=False)

            def forward(self, x, residual):
                normed, residual = self.norm.forward_native(x, residual)
                return self.proj(normed), residual

        m = Block().eval()

    x = torch.randn(8, H)
    residual = torch.randn(8, H)
    with torch.no_grad():
        y0, _ = m(x, residual)
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    with torch.no_grad():
        y1, _ = cm(x, residual)
    # Without VllmBackend's VllmIRLoweringPass, torch.ops.vllm_ir.fused_add_rms_norm stays an opaque
    # extern call; only the Linear lowers -> extern_kernels.mm (ATen/MKL GEMM). See the dump above.
    print("[cpu-vllm-op] max|delta| vs eager:", (y1 - y0).abs().max().item(), flush=True)
    print("[cpu-vllm-op] note: fused_add_rms_norm survives as an extern op; mm -> extern_kernels.mm", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "cuda":
        out = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT", "./callflow_out")
        os.makedirs(out, exist_ok=True)
        run_cuda(out)
    elif mode == "cpu-cpp":
        run_cpu_cpp()
    elif mode == "cpu-vllm-op":
        run_cpu_vllm_op()
    else:
        print(__doc__)
        print("Pick a mode: cuda | cpu-cpp | cpu-vllm-op")
