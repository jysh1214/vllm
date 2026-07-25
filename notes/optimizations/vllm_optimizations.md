# vLLM LLM / Transformer Optimizations — v0.25.1

A catalog of optimization techniques shipped in the vLLM codebase, grouped by category.
Most knobs are toggleable via env vars in `vllm/envs.py` or fields on the `vllm/config/`
dataclasses (`CompilationConfig`, `SchedulerConfig`, `ParallelConfig`, `CacheConfig`,
`AttentionConfig`, `KernelConfig`, `SpeculativeConfig`, …).

## Version stamp

| | |
| --- | --- |
| Version | **v0.25.1** |
| Tag commit | `752a3a504485790a2e8491cacbb35c137339ad34` (2026-07-12) |
| Previous revision documented | `740f379fa` (2026-07-06) |
| Delta absorbed | 61 commits, 265 files, +13710 / −1295 |

Every path, env var and config field quoted below was checked against that commit. Re-measured
2026-07-25 against a fresh v0.25.1 checkout: 296 repo paths, 42 bare-filename shorthands,
102 `VLLM_*` env vars, 125 config fields, 7 CLI flags, 336 symbols — all resolve.
To re-verify against a checkout:

```bash
git -C <vllm> checkout v0.25.1
# paths
grep -o '`[^`]*`' notes/optimizations/vllm_optimizations.md | tr -d '`' \
  | grep '/' | sort -u | while read p; do test -e "$p" || echo "MISSING $p"; done
# env vars
grep -o '`VLLM_[A-Z0-9_]*`' notes/optimizations/vllm_optimizations.md | tr -d '`' \
  | sort -u | while read v; do grep -q "\"$v\"" vllm/envs.py || echo "MISSING $v"; done
```

Two references are expected to report MISSING because the note cites them precisely as things that
no longer exist: `csrc/mamba/` and `VLLM_ATTENTION_BACKEND`.

## What changed for v0.25.1

### Structural corrections (these were already wrong before this release)

- **CUDA kernels moved under `csrc/libtorch_stable/`** (torch stable-ABI migration, `22a58640b`,
  2026-05-29). Every `csrc/*.cu` path in the previous revision was stale:
  `custom_all_reduce.cu`, `fused_qknorm_rope_kernel.cu`, `nvfp4_kv_cache_kernels.cu`,
  `persistent_topk.cuh`, `quantization/fused_kernels/*`, and all of `csrc/mamba/`.
- **`VLLM_ATTENTION_BACKEND` no longer exists** (zero occurrences in `vllm/`). Attention backend
  selection is now `AttentionConfig.backend` (`vllm/config/attention.py`) over the
  `AttentionBackendEnum` registry.
- **PagedAttention v1/v2 CUDA kernels are gone.** `csrc/attention/` retains only dtype headers.
  The paged *concept* survives as the block table + KV-cache layout; the only surviving paged
  attention *kernel* is the ROCm ll4mi one (`csrc/rocm/attention.cu`). See the note on
  [PagedAttention.md](PagedAttention.md) below.

### Removed upstream (verified absent under any name)

| Entry | Evidence |
| --- | --- |
| Tree Attention | no backend, kernel or scheduler support |
| Vertical Slash Index | no MInference-style sparse index anywhere |
| Multi-step scheduling | V1 has no multi-step scheduler; `--num-scheduler-steps` gone |
| P2P NCCL connector | `grep -r 'p2p_nccl\|P2pNccl'` empty — superseded by `vllm/v1/kv_offload/tiering/p2p` |
| SmoothQuant | not a registered quantization method |
| GGUF | migrated out-of-tree to `vllm-gguf-plugin`; `docs/features/quantization/gguf.md`, `tests/plugins_tests/gguf/` and a `setup.py` `extra-quant` extra remain |
| Copy-on-write block sharing | V1 blocks are immutable; sharing is ref-counting in `BlockPool` |
| pplx all-to-all | dropped from the all2all backend list |
| compressed-tensors sparsity | schemes + transforms remain, sparsity support removed |

### New or materially changed **in the v0.25.1 window** (`740f379fa..v0.25.1`)

- **Helion kernel backend grows a 7th op** — `silu_and_mul_per_block_quant` with checked-in
  B200/H100 tuned configs (#43994).
- **Transformers modeling backend made as fast as native vLLM** — FX-trace-driven fusers rewrite
  arbitrary HF models into `QKVParallelLinear` / `MergedColumnParallelLinear+act-and-mul` /
  `FusedMoE` / `RMSNorm` (#47187), with CUDA-graph + embed-scaling follow-up (#48010).
- **Humming extended from MoE-only to all dense + MoE kernel oracles** — `"humming"` added to the
  `LinearBackend` literal, registered for int8/fp8/fp8-block/wfp8a16/mxfp8/nvfp4/mxfp4 (#41652).
- **Shared `token_to_req_indices` cache** on `CommonAttentionMetadata` — built once per step,
  reused across layers/builders; 5–6× kernel speedup on DSv4 sparse MLA (#47474).
- **Blocking CUDA events** replace spin-waits at output-copy / draft-token / prepare-inputs sync
  points so a TP rank yields the CUDA driver lock instead of busy-polling (#47081); XPU
  counterpart in (#47868).
- **MiniMax-M3 cross-layer all-reduce + norm fusion** generalized — `FusedMoE(reduce_results=False)`
  replaces poking `skip_final_all_reduce`, and now covers MoE layers too (#47631).
- **MiniMax-M3 sparse indexer** gains an SM100 MSA `sparse_topk_select` path alongside the Triton
  top-k, plus a unified prefill+decode score buffer (#47502).
- **CPU ShortConv** — `ShortConv.forward_native` went from a no-op stub to a real CPU
  prefill+decode causal-conv1d path (#35059).
- **Fused Kimi image preprocessing** — numba single-pass pad+normalize+patchify (#47416).
- **TorchCodec video decoding backend** — one batched GIL-releasing `get_frames_at`, NHWC layout
  (#46609); PyNvVideoCodec pinned to 2.0.4 (#48056).
- **B12x MoE backend** extended to non-gated (RELU2) MoEs; now incompatible with expert
  parallelism (#43328). **FlashInfer A2A backends** allowed for TRT-LLM FP8 modular MoE (#46661).
- **Sequence-parallel MoE now additionally requires `data_parallel_size > 1`** — disabled for
  TP-only + EP deployments (#47902, labeled a temp fix).
- **Mixed-dtype AR+RMSNorm quant fusions guarded** — patterns no longer fire when norm input dtype
  ≠ norm weight dtype (#48330, the tag commit); **MNNVL one-shot selection delegated** to
  FlashInfer AUTO (#47589).
- **P/D async KV load lookahead** widened from EAGLE-only to any `num_lookahead_tokens > 0`, i.e.
  MTP (#46694), plus a hybrid/GDN correctness follow-up (#47466); **DP-safe drafter dummy run**
  fixes an MTP collective hang (#40589).
- **DiffusionGemma sampler** tiled over requests to bound peak transient memory (#45672).
- **Per-request timing metrics** on Chat/Completions responses (`--enable-per-request-metrics`,
  #46768).

Zero files were deleted in this window, and `vllm/envs.py` and `csrc/` were unchanged — so no row
below is marked removed on the strength of anything in the v0.25.1 delta.

## Status table

`Status` column: ✓ means a deep-dive note exists in this directory (linked from the
`Optimization` cell). Blank means no note yet.

| Category | Optimization | Status |
| --- | --- | --- |
| Attention | [PagedAttention (block table; CUDA v1/v2 kernels removed)](PagedAttention.md) | ✓ |
| Attention | [FlashAttention v2 / v3 / v4](FlashAttention.md) | ✓ |
| Attention | FlashAttention DiffKV | |
| Attention | Triton DiffKV | |
| Attention | FlashInfer | |
| Attention | Triton Attention (unified) | |
| Attention | Flex Attention | |
| Attention | HPC Attention | |
| Attention | TurboQuant Attention | |
| Attention | MLA (Multi-head Latent Attention) | |
| Attention | MLA Sparse (DSA) + FP8 MQA-logits indexer | |
| Attention | MLA prefill backend selector | |
| Attention | Sliding Window Attention | |
| Attention | RSWA (reference sliding-window attention) | |
| Attention | Chunked Local Attention | |
| Attention | Static Sink Attention | |
| Attention | Cross-Attention | |
| Attention | Encoder-only attention | |
| Attention | Prefill prefix-LM attention | |
| Attention | MM Encoder Attention | |
| Attention | Cascade attention | |
| Attention | ROCm AITER Attention / Unified | |
| Attention | ROCm standard attention | |
| Attention | CPU Attention | |
| Attention | Merge Attn States | |
| Attention | Fused QK-Norm + RoPE | |
| Attention | Sparse-indexer top-k kernels | |
| Attention | Shared `token_to_req_indices` cache | |
| Attention | DCP all-to-all attention combine | |
| Attention | Pluggable attention backend registry | |
| KV Cache | Paged KV cache | |
| KV Cache | Hash-based prefix caching | |
| KV Cache | Partial-block prefix caching | |
| KV Cache | Decoupled prefix-cache hash block size | |
| KV Cache | Prefix-cache retention interval (SWA / Mamba) | |
| KV Cache | Chunked prefill | |
| KV Cache | KV cache quantization (FP8 / INT8 / INT4 / NVFP4 / TurboQuant) | |
| KV Cache | NVFP4 KV-cache kernels | |
| KV Cache | Fused concat + RoPE + MLA KV-cache insert | |
| KV Cache | Ref-counted block sharing | |
| KV Cache | CPU KV-cache offload | |
| KV Cache | Multi-tier KV offload (fs / obj / p2p) | |
| KV Cache | KV-cache coordinator | |
| KV Cache | Single-type KV managers | |
| KV Cache | Sliding-window / chunked-local block reclamation | |
| KV Cache | Attention-sink KV cache spec | |
| KV Cache | Cross-layer KV sharing fast prefill | |
| KV Cache | Mamba / SSM state cache modes | |
| KV Cache | Encoder cache manager | |
| KV Cache | KV events | |
| KV Cache | Hybrid KV cache manager (HMA) | |
| KV Cache | KV cache layout selection (NHD/HND, ROCm shuffle) | |
| KV Cache | KV cache residency metrics | |
| Quantization | FP8 (E4M3 / E5M2) | |
| Quantization | W8A8 (INT8 / FP8) | |
| Quantization | AWQ + AWQ-Marlin | |
| Quantization | GPTQ + GPTQ-Marlin | |
| Quantization | Marlin / Machete | |
| Quantization | AllSpark W8A16 | |
| Quantization | CUTLASS W4A8 (grouped) | |
| Quantization | Compressed-tensors (schemes + transforms) | |
| Quantization | QuTLASS / hadacore Hadamard transforms | |
| Quantization | MXFP4 | |
| Quantization | MXFP8 (native + emulation) | |
| Quantization | NVFP4 | |
| Quantization | NVIDIA ModelOpt | |
| Quantization | Bitsandbytes | |
| Quantization | TorchAO | |
| Quantization | Quark | |
| Quantization | Intel Neural Compressor (INC) | |
| Quantization | Humming (mixed-precision linear + MoE) | |
| Quantization | TurboQuant (KV compression) | |
| Quantization | Experts INT8 | |
| Quantization | MoE WNA16 | |
| Quantization | Online (dynamic) quantization | |
| Quantization | Input FP8 quant | |
| Quantization | Fused activation + quant | |
| Quantization | Fused layernorm + quant | |
| Quantization | FBGEMM FP8 (deprecated) | |
| Kernel Dispatch | Linear/MoE kernel oracles (`--linear-backend` / `--moe-backend`) | |
| Kernel Dispatch | vLLM IR op dispatch priority | |
| Kernel Dispatch | Helion autotuned kernels | |
| Kernel Dispatch | OINK fused custom ops | |
| Kernel Dispatch | Custom-op enable/disable list | |
| Kernel Dispatch | Startup kernel warmup / autotune | |
| Parallelism | Tensor Parallel (TP) | |
| Parallelism | Pipeline Parallel (PP) | |
| Parallelism | Async PP token broadcast | |
| Parallelism | Expert Parallel (EP) | |
| Parallelism | Data Parallel (DP) + coordinator | |
| Parallelism | Context Parallel (PCP / DCP) | |
| Parallelism | CP KV-cache interleaving | |
| Parallelism | Sequence Parallel (IR-level) | |
| Parallelism | Sequence-parallel MoE | |
| Parallelism | Async TP (GEMM ↔ collective fusion) | |
| Parallelism | DCP all-to-all (`--dcp-comm-backend a2a`) | |
| Parallelism | Dual Batch Overlap (DBO) | |
| Parallelism | Elastic Expert Parallel | |
| Parallelism | NUMA binding | |
| MoE | Fused MoE kernel | |
| MoE | Grouped GEMM / Batched MoE | |
| MoE | Modular Kernel | |
| MoE | MoE kernel oracles | |
| MoE | Topk-Softmax / Grouped Topk routers | |
| MoE | Zero-expert routing | |
| MoE | Permute / Unpermute ops | |
| MoE | `moe_align_block_size` / padding skip | |
| MoE | MoE FP8 / INT8 / WNA16 | |
| MoE | Marlin MoE | |
| MoE | FlashInfer Cutlass MoE | |
| MoE | FlashInfer CuteDSL MoE | |
| MoE | FlashInfer B12x MoE | |
| MoE | TensorRT-LLM MoE | |
| MoE | GPT-OSS Triton-kernels MoE | |
| MoE | HPC-Ops MoE | |
| MoE | DeepGEMM | |
| MoE | DeepEP HT/LL/v2 all-to-all | |
| MoE | Mori all-to-all (ROCm) | |
| MoE | NIXL-EP all-to-all | |
| MoE | FlashInfer NVLink all-to-all | |
| MoE | Shared experts (side stream) | |
| MoE | AITER shared-experts fusion | |
| MoE | Deferred MoE all-reduce (cross-layer AR+norm fusion) | |
| MoE | EPLB (expert-parallel load balancing) | |
| MoE | Expert placement strategy | |
| MoE | MoE routing simulator | |
| MoE | Routed-experts capture / replay | |
| Speculative Decoding | [Draft-target spec decoding](SpeculativeDecoding.md) | ✓ |
| Speculative Decoding | [EAGLE / EAGLE-3](SpeculativeDecoding.md#how-to-draft-cheaply) | ✓ |
| Speculative Decoding | [Medusa](SpeculativeDecoding.md#how-to-draft-cheaply) | ✓ |
| Speculative Decoding | [N-gram proposer (CPU + GPU)](SpeculativeDecoding.md#how-to-draft-cheaply) | ✓ |
| Speculative Decoding | [Lookahead KV slot reservation](SpeculativeDecoding.md#how-to-draft-cheaply) | ✓ |
| Speculative Decoding | Suffix tree decoding | |
| Speculative Decoding | DFlash parallel drafting | |
| Speculative Decoding | DSpark semi-autoregressive drafting | |
| Speculative Decoding | MTP (multi-token prediction) | |
| Speculative Decoding | MLP speculator | |
| Speculative Decoding | Custom proposer plug-in | |
| Speculative Decoding | Extract-hidden-states proposer | |
| Speculative Decoding | Dynamic draft length per batch size | |
| Speculative Decoding | Heterogeneous draft/target vocab (TLI) | |
| Speculative Decoding | Vocab-parallel local argmax reduction | |
| Speculative Decoding | Padded drafter batch | |
| Speculative Decoding | Parallel drafting | |
| Speculative Decoding | [Rejection sampler (standard / block / synthetic)](SpeculativeDecoding.md#stochastic-correctness) | ✓ |
| Speculative Decoding | Probabilistic draft sampling | |
| Speculative Decoding | Drafter dummy-run / CUDA graph capture | |
| Speculative Decoding | Acceptance metrics | |
| Compilation | torch.compile / Inductor backend | |
| Compilation | CUDA Graphs (full + piecewise) | |
| Compilation | Breakable CUDA graphs | |
| Compilation | Inductor graph partition | |
| Compilation | Stitching-graph codegen | |
| Compilation | Dynamic-shapes control | |
| Compilation | Compile ranges / shape specialization | |
| Compilation | LoRA-specialized CUDA graphs | |
| Compilation | AOT compile + mega AOT artifact | |
| Compilation | Custom Inductor passes | |
| Compilation | Compile cache | |
| Compilation | Inductor max-autotune | |
| Compilation | Inductor coordinate-descent tuning | |
| Compilation | Fusion: RMSNorm + Quant | |
| Compilation | Fusion: All-Reduce + RMSNorm | |
| Compilation | Fusion: Attention + Quant | |
| Compilation | Fusion: QK-Norm + RoPE | |
| Compilation | Fusion: Activation + Quant | |
| Compilation | Fusion: MLA Attention + Quant | |
| Compilation | [Fusion: RoPE + KV-cache update (ROCm)](RoPEKVCacheFusion.md) | ✓ |
| Compilation | Fusion: MLA RoPE + KV-cache cat | |
| Compilation | Fusion: Collective + Compute (async TP) | |
| Compilation | Fusion: ROCm AITER | |
| Compilation | Sequence-parallelism IR pass | |
| Compilation | Lowering pass + inplace functionalization | |
| Compilation | Clone elimination | |
| Compilation | Utility passes (noop elim, split coalesce, scatter-split, post-cleanup) | |
| Compilation | Fix-functionalization pass | |
| Compilation | CUDAGraph GC | |
| Compilation | Fast MoE cold start | |
| Scheduling | Continuous batching | |
| Scheduling | Chunked prefill | |
| Scheduling | Prefix-cache-aware scheduling | |
| Scheduling | Priority scheduling (FCFS / priority) | |
| Scheduling | Async scheduler | |
| Scheduling | Preemption and recompute | |
| Scheduling | KV cache admission watermark | |
| Scheduling | Full-ISL admission reservation | |
| Scheduling | Concurrent partial-prefill caps / long-prompt jumping | |
| Scheduling | DP-aligned prefill throttling | |
| Scheduling | Spec-decode lookahead slot budgeting | |
| Scheduling | Mamba block-aligned prefill split | |
| Scheduling | Chunked multimodal input control | |
| Scheduling | Batched structured-output grammar bitmask | |
| Scheduling | Ubatching / micro-batching | |
| Scheduling | Disaggregated prefill / decode | |
| Scheduling | Encoder budget for multimodal | |
| Scheduling | Output batching (`stream_interval`) | |
| Scheduling | Pluggable scheduler class | |
| Communication | Custom all-reduce | |
| Communication | Quick all-reduce (ROCm, quantized) | |
| Communication | NVIDIA Symmetric Memory all-reduce | |
| Communication | NCCL symmetric-memory allocator | |
| Communication | FlashInfer all-reduce | |
| Communication | AITER custom all-reduce (ROCm) | |
| Communication | PyNCCL + tuning | |
| Communication | Shared-memory broadcast / object storage | |
| Communication | Ray compiled-graph communicator | |
| Communication | All-to-all backends (DeepEP / Mori / NIXL-EP / FlashInfer NVLink) | |
| Communication | KV-transfer connectors (umbrella) | |
| Communication | Weight-transfer engines (NCCL / IPC / sparse-NCCL) | |
| Communication | EC (encoder cache) transfer | |
| Sampling | FlashInfer sampling | |
| Sampling | Triton top-k / top-p | |
| Sampling | Fused per-platform top-k / top-p sampler | |
| Sampling | ROCm AITER fused sampler | |
| Sampling | XPU fused sampler kernel | |
| Sampling | Exponential-race (Gumbel-max) sampling | |
| Sampling | Sort-free top-k-only path | |
| Sampling | All-greedy / all-random fast paths | |
| Sampling | V2-runner Triton sampler stack | |
| Sampling | Logits processors | |
| Sampling | Frequency / presence / repetition penalties | |
| Sampling | Bad-words masking | |
| Sampling | Thinking-budget logits forcing | |
| Sampling | Structured output: xGrammar | |
| Sampling | Structured output: Outlines | |
| Sampling | Structured output: Guidance | |
| Sampling | Structured output: lm-format-enforcer | |
| Sampling | Triton grammar-bitmask kernel | |
| Sampling | Async grammar compilation / bitmask fill | |
| Sampling | Logprobs computation | |
| Sampling | Chunked prompt-logprobs | |
| Memory & Runtime | Block pool | |
| Memory & Runtime | Custom cumem allocator / sleep mode | |
| Memory & Runtime | XPU sleep-mode allocator | |
| Memory & Runtime | Pinned host memory + `CpuGpuBuffer` | |
| Memory & Runtime | Prefix-cache eviction (LRU free-block queue) | |
| Memory & Runtime | Weight offloading (UVA + prefetch backends) | |
| Memory & Runtime | Workspace manager | |
| Memory & Runtime | Memory profiler (CUDA-graph estimation) | |
| Memory & Runtime | Explicit KV-cache byte budget | |
| Memory & Runtime | Blocking CUDA events (no spin-wait) | |
| Memory & Runtime | Auxiliary-stream parallel execution | |
| Memory & Runtime | GC debugger / tuning | |
| Model-specific Kernels | Fused RMSNorm + residual | |
| Model-specific Kernels | Fused activations (act-and-mul family) | |
| Model-specific Kernels | Fused QK-norm + RoPE + KV-insert (DSv4 / MiniMax-M3) | |
| Model-specific Kernels | DeepSeek fused A-GEMM | |
| Model-specific Kernels | FP32 router GEMM | |
| Model-specific Kernels | MiniMax fused all-reduce + RMSNorm QK | |
| Model-specific Kernels | MHC (hyper-connection) kernels | |
| Model-specific Kernels | MLA latent attention | |
| Model-specific Kernels | Mamba / Mamba2 SSM kernels | |
| Model-specific Kernels | Mamba SSU backend + stochastic rounding | |
| Model-specific Kernels | Linear attention (Lightning / MiniMax / Bailing) | |
| Model-specific Kernels | GDN linear attention | |
| Model-specific Kernels | GDN chunked-prefill CuteDSL kernels (SM100) | |
| Model-specific Kernels | Short-conv (Mamba local context) | |
| Model-specific Kernels | Gated RMSNorm for Mamba2 | |
| Model-specific Kernels | FLA packed recurrent decode | |
| Model-specific Kernels | KDA (Kimi Delta Attention) | |
| Model-specific Kernels | Batch-invariant mode (deterministic) | |
| Model-specific Kernels | `vllm/models/` vendor fast paths (DSv4 / DSv3.2 / MiniMax-M3) | |
| Model-specific Kernels | Transformers-backend graph fusers | |
| Model-specific Kernels | RoPE variants (16 implementations) | |
| Model-specific Kernels | RoPE instance cache | |
| LoRA | [Punica kernels](LoRA.md#bgmv--batched-grouped-matrix-vector) | ✓ |
| LoRA | [Multi-LoRA batching](LoRA.md#the-multi-tenant-serving-problem) | ✓ |
| LoRA | [LoRA dual-stream execution](LoRA.md#other-lora-optimizations-in-vllm) | ✓ |
| LoRA | [PDL projection optimization](LoRA.md#other-lora-optimizations-in-vllm) | ✓ |
| LoRA | [LoRA model manager](LoRA.md#other-lora-optimizations-in-vllm) | ✓ |
| LoRA | [LoRA weight resolver](LoRA.md#other-lora-optimizations-in-vllm) | ✓ |
| LoRA | Fused MoE LoRA | |
| LoRA | FP8 quantized LoRA kernels | |
| LoRA | Fully sharded LoRA | |
| LoRA | LoRA CUDA-graph specialization | |
| LoRA | LoRA kernel metadata (no-LoRA early exit) | |
| LoRA | Tuned multi-LoRA kernel configs | |
| LoRA | Dummy-LoRA warmup | |
| LoRA | Non-CUDA punica backends (CPU / XPU) | |
| Multi-Modal | Encoder output cache | |
| Multi-Modal | Multimodal processor cache | |
| Multi-Modal | Shared-memory object-store cache | |
| Multi-Modal | Zero-copy multimodal tensor IPC | |
| Multi-Modal | Frontend GPU multimodal memory pool | |
| Multi-Modal | Multimodal prefix caching | |
| Multi-Modal | Media cache + media hasher | |
| Multi-Modal | Parallel media fetching | |
| Multi-Modal | Video decoding backends (opencv / pyav / TorchCodec / PyNvVideoCodec) | |
| Multi-Modal | Fused Kimi image preprocessing | |
| Multi-Modal | Efficient Video Sampling (EVS) token pruning | |
| Multi-Modal | Chunked multimodal encoder scheduling | |
| Multi-Modal | Encoder CUDA graphs + encoder compile | |
| Multi-Modal | Data-parallel multimodal encoder | |
| Multi-Modal | Encoder-only execution | |
| Multi-Modal | Skip multimodal profiling | |
| Multi-Modal | Multimodal registry | |
| Distributed / Disaggregated | Disaggregated prefill / decode (KV transfer) | |
| Distributed / Disaggregated | NIXL connector (pull/push, heterogeneous TP) | |
| Distributed / Disaggregated | LMCache / Mooncake / MoRIIO / HF3FS / FlexKV connectors | |
| Distributed / Disaggregated | OffloadingConnector + MultiConnector | |
| Distributed / Disaggregated | P2P KV tiering (ZMQ + NIXL) | |
| Distributed / Disaggregated | SupportsHMA (connectors under hybrid KV) | |
| Distributed / Disaggregated | Mamba/SSM conv-state transfer | |
| Distributed / Disaggregated | KV load failure policy | |
| Distributed / Disaggregated | KV connector stats / metrics | |
| Distributed / Disaggregated | KV events lifecycle tracking | |
| Distributed / Disaggregated | Stateless group coordinator | |
| Distributed / Disaggregated | Elastic EP (dynamic expert scaling) | |
| ROCm (AITER) | AITER paged attention | |
| ROCm (AITER) | AITER MHA / Unified attention | |
| ROCm (AITER) | AITER MLA + sparse MLA | |
| ROCm (AITER) | AITER Triton RoPE | |
| ROCm (AITER) | AITER RMSNorm | |
| ROCm (AITER) | AITER linear / Triton GEMM / hipBLASLt | |
| ROCm (AITER) | AITER MoE + dispatch policy | |
| ROCm (AITER) | AITER fused topk / routing | |
| ROCm (AITER) | AITER FP4 ASM GEMM | |
| ROCm (AITER) | AITER FP8 / FP4 BMM | |
| ROCm (AITER) | AITER shared-experts fusion | |
| ROCm (AITER) | AITER GDN Triton kernels | |
| ROCm (AITER) | AITER sparse attention indexer | |
| ROCm (AITER) | AITER MHC kernels | |
| ROCm (AITER) | Quick-reduce quantization for all-reduce | |
| ROCm (AITER) | Skinny GEMM | |
| ROCm (AITER) | RDNA3 W4A16 GPTQ GEMM | |
| ROCm (AITER) | FP8 padding / MoE padding | |
| ROCm (AITER) | KV cache shuffle layout | |
| CPU / Other HW | CPU attention (per-ISA) + MLA decode | |
| CPU / Other HW | CPU micro-GEMM tiles (AMX / NEON / RVV / VSX / VXE) | |
| CPU / Other HW | CPU SGL kernel suite (GEMM + MoE, fp8/int8/int4) | |
| CPU / Other HW | CPU oneDNN / ACL GEMM | |
| CPU / Other HW | ZenDNN (zentorch) weight prepacking | |
| CPU / Other HW | CPU WNA16 + dynamic 4-bit INT MoE | |
| CPU / Other HW | CPU LUT-based bf16 activation | |
| CPU / Other HW | CPU shared-memory collectives | |
| CPU / Other HW | CPU Mamba / GDN / ShortConv kernels | |
| CPU / Other HW | CPU speculative-decoding kernels | |
| CPU / Other HW | CPU OpenMP thread pinning | |
| CPU / Other HW | MWAITX spin-loop extension | |
| CPU / Other HW | XPU graph capture + XPU kernels | |
| Frontend & Serving | Rust frontend (native API server) | |
| Frontend & Serving | API-server scale-out (`--api-server-count`) | |
| Frontend & Serving | External / hybrid DP load balancing | |
| Frontend & Serving | Incremental detokenization fast path | |
| Frontend & Serving | fastokens BPE tokenizer backend | |
| Frontend & Serving | msgpack zero-copy tensor (de)serialization | |
| Frontend & Serving | Pooling / late-interaction runners | |
| Frontend & Serving | Per-request timing metrics | |
| Model Loading | Safetensors load strategies + prefetch | |
| Model Loading | fastsafetensors pipelined loading | |
| Model Loading | `instanttensor` load format | |
| Model Loading | Parallel loading workers | |
| Model Loading | EP weight filtering | |
| Model Loading | Startup kernel warmup / autotune | |
| Misc | Host-device transfer overlap | |
| Misc | Attention metadata caching across KV groups | |
| Misc | Triton custom kernels (broadly) | |
| Misc | Profiling scopes (NVTX, layerwise) | |
| Misc | MFU debug metrics | |
| Misc | GPU↔CPU sync detector | |
| Misc | Post-warmup JIT compile monitor | |
| Misc | Dry-run / dummy forward for memory profiling | |

---

## Detailed catalog

Paths are repo-relative and verified at v0.25.1. Within a bullet, a bare filename is relative to
the directory named earlier in that same bullet.

## 1. Attention Algorithms & Kernels

Backend selection is `AttentionConfig.backend` (`vllm/config/attention.py`) resolved through the
`AttentionBackendEnum` registry (`vllm/v1/attention/backends/registry.py`), which also accepts
third-party backends via `register_backend()`. The old `VLLM_ATTENTION_BACKEND` env var is gone.

- **PagedAttention** — block-table KV addressing. The CUDA `paged_attention_v1` / `paged_attention_v2` kernels no
  longer exist; `csrc/attention/` holds only dtype headers. The surviving paged-attention kernel is
  ROCm's ll4mi (`csrc/rocm/attention.cu`), reached through
  `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`. **[PagedAttention.md](PagedAttention.md)
  describes the removed v1/v2 kernels and is stale for CUDA at this version.**
- **FlashAttention v2 / v3 / v4** — IO-aware tiled attention (`vllm/v1/attention/backends/flash_attn.py`);
  version picked by `get_flash_attn_version()` in `vllm/v1/attention/backends/fa_utils.py`.
- **FlashAttention DiffKV** — heterogeneous K/V head dims, with its own
  `triton_reshape_and_cache_flash_diffkv` writer (`vllm/v1/attention/backends/flash_attn_diffkv.py`).
- **Triton DiffKV** — Triton counterpart of the above
  (`vllm/v1/attention/backends/triton_attn_diffkv.py`).
- **FlashInfer** — decode/prefill incl. TRT-LLM-gen kernels
  (`vllm/v1/attention/backends/flashinfer.py`); `AttentionConfig.use_trtllm_attention` toggles the
  TRT-LLM path.
- **Triton Attention** — unified prefill+decode backend
  (`vllm/v1/attention/backends/triton_attn.py`); kernels in
  `vllm/v1/attention/ops/triton_unified_attention.py`. v0.25.1 passes literal `None` for dead-branch
  args so Triton skips materialising them, and fixes an int32 page-offset overflow in
  `vllm/v1/attention/ops/triton_decode_attention.py`.
- **Flex Attention** — `torch.compile` programmable `mask_mod`
  (`vllm/v1/attention/backends/flex_attention.py`); tile sizes via `AttentionConfig.flex_attn_block_m`
  and siblings.
- **HPC Attention** — Tencent hpc-ops prefill+decode backend for Hopper
  (`vllm/v1/attention/backends/hpc_attn.py`).
- **TurboQuant Attention** — Hadamard-rotated low-bit KV store + decode
  (`vllm/v1/attention/backends/turboquant_attn.py`).
- **MLA (Multi-head Latent Attention)** — compressed-KV latent attention with Triton / CUTLASS /
  FlashMLA / FlashAttn / FlashInfer / AITER / Tokenspeed variants (`vllm/v1/attention/backends/mla/`).
  CUTLASS SM100 kernel at `csrc/libtorch_stable/attention/mla/`.
- **MLA Sparse (DSA)** — top-k sparse-indexed MLA: `flashmla_sparse`, `flashinfer_mla_sparse`
  (+SM120), `flashattn_mla_sparse`, `sparse_swa`, `rocm_aiter_mla_sparse`, `xpu_mla_sparse`.
  The indexer (`vllm/v1/attention/backends/mla/indexer.py`,
  `vllm/model_executor/layers/sparse_attn_indexer.py`) uses FP8 MQA logits
  (DeepGEMM `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` via `vllm/utils/deep_gemm.py`; `vllm/v1/attention/ops/triton_fp8_mqa_logits.py` is a temporary ROCm gfx942 fallback), capped by
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`; KV dtype via `AttentionConfig.indexer_kv_dtype`.
- **MLA prefill backend selector** — prefill phase gets its own pluggable backend
  (`vllm/v1/attention/backends/mla/prefill/`, `AttentionConfig.mla_prefill_backend`).
- **Sparse-indexer top-k kernels** — `csrc/libtorch_stable/persistent_topk.cuh` (persistent-CTA,
  k=512/1024/2048), `csrc/libtorch_stable/cooperative_topk.cu`, and `top_k_per_row_*` in
  `csrc/libtorch_stable/sampler.cu`. These serve the sparse-attention indexer, **not** the sampler.
- **Sliding Window Attention** — driven by `SlidingWindowSpec` (`vllm/v1/kv_cache_interface.py`);
  windowing is applied inside each backend.
- **RSWA** — reference sliding-window attention reporting `RSWASpec` so the KV manager evicts gap
  blocks, bounding KV at O(prefix + window)
  (`vllm/model_executor/layers/attention/rswa_attention.py`).
- **Chunked Local Attention** — Llama4-style
  (`vllm/model_executor/layers/attention/chunked_local_attention.py`).
- **Static Sink Attention** — persistent leading sink tokens beside a sliding window
  (`vllm/model_executor/layers/attention/static_sink_attention.py`).
- **Cross-Attention** — encoder-decoder (`vllm/model_executor/layers/attention/cross_attention.py`).
  v0.25.1 realigns mixed encoder/decoder KV views on ROCm (`vllm/v1/worker/gpu/attn_utils.py`).
- **Encoder-only attention** — bidirectional, no persistent KV
  (`vllm/model_executor/layers/attention/encoder_only_attention.py`).
- **Prefill prefix-LM attention** — bidirectional over the prompt, causal after
  (`vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py`).
- **MM Encoder Attention** — `vllm/model_executor/layers/attention/mm_encoder_attention.py`; ViT
  wrappers (FA / Triton / SDPA / FlashInfer) in `vllm/v1/attention/ops/vit_attn_wrappers.py`;
  selected by `MultiModalConfig.mm_encoder_attn_backend`.
- **Cascade attention** — one shared-prefix pass + per-request suffix pass merged via
  `merge_attn_states`; backends opt in with `use_cascade_attention()`, disabled by
  `ModelConfig.disable_cascade_attn`.
- **ROCm AITER Attention / Unified** — `vllm/v1/attention/backends/rocm_aiter_fa.py` and
  `rocm_aiter_unified_attn.py` (`VLLM_ROCM_USE_AITER_MHA`,
  `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`).
- **ROCm standard attention** — non-AITER backend (`vllm/v1/attention/backends/rocm_attn.py`) using
  the Triton `chunked_prefill_paged_decode` path, which calls `ops.paged_attention_rocm` for decode.
- **CPU Attention** — `vllm/v1/attention/backends/cpu_attn.py` over per-ISA kernels in
  `csrc/cpu/cpu_attn.cpp` (+ AMX / NEON / RVV / VSX / VXE headers), plus `csrc/cpu/mla_decode.cpp`.
- **Merge Attn States** — LSE-weighted merge of partial outputs (chunked prefill / cascade / CP);
  `csrc/libtorch_stable/attention/merge_attn_states.cu`.
- **Fused QK-Norm + RoPE** — `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu`, with a Helion
  variant at `vllm/kernels/helion/ops/fused_qk_norm_rope.py`.
- **Shared `token_to_req_indices` cache** *(new in v0.25.1)* — `CommonAttentionMetadata`
  (`vllm/v1/attention/backend.py`) builds the per-token request-index map once and shares it across
  layers and builders; 5–6× kernel win on DSv4 sparse MLA.
- **DCP all-to-all attention combine** — exchanges partial outputs + LSE in one packed all-to-all
  (`vllm/v1/attention/ops/dcp_alltoall.py`); see §5.
- **Query quantization / split pinning** — `AttentionConfig.use_prefill_query_quantization`,
  `disable_flashinfer_q_quantization`, `flash_attn_max_num_splits_for_cuda_graph`,
  `tq_max_kv_splits_for_cuda_graph`.

*Removed at v0.25.1:* Tree Attention, Vertical Slash Index. (`AttentionBackendEnum.NO_ATTENTION`
points at a module that does not exist — do not treat it as a backend.)

## 2. KV Cache Management

- **Paged KV cache** — logical→physical block mapping via `BlockPool` + `KVCacheManager`
  (`vllm/v1/core/`), block size `CacheConfig.block_size`.
- **Hash-based prefix caching** — rolling content hashes (`vllm/v1/core/kv_cache_utils.py`);
  algorithm selectable (sha256, sha256_cbor, xxhash, xxhash_cbor).
- **Partial-block prefix caching** — `BlockPool.cache_partial_block` registers a sub-block prefix
  key on an existing block: hits inside a block with no allocation or copy.
- **Decoupled hash block size** — `CacheConfig.hash_block_size` lets hashing be finer-grained than
  the allocation block size.
- **Prefix-cache retention interval** — `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` sparsifies checkpoints
  for sliding-window and Mamba groups.
- **Chunked prefill** — interleaves prefill/decode (`SchedulerConfig.enable_chunked_prefill`).
- **KV cache quantization** — FP8 per-tensor, INT8/FP8 per-token-head, INT4 per-token-head, NVFP4,
  `fp8_inc`, `fp8_ds_mla`, plus the TurboQuant presets — all via `CacheConfig.cache_dtype`
  (`vllm/model_executor/layers/quantization/kv_cache.py`). *There is no MXFP4 KV dtype.*
- **NVFP4 KV-cache kernels** — `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`.
- **Fused concat + RoPE + MLA KV-cache insert** — `concat_and_cache_mla_rope_fused` in
  `csrc/libtorch_stable/cache_kernels_fused.cu`.
- **Ref-counted block sharing** — V1 has no copy-on-write; blocks are immutable and shared by
  ref-count (`BlockPool.touch` / `free_blocks`).
- **CPU KV-cache offload** — `vllm/v1/kv_offload/cpu/` with LRU/ARC policies and Triton swap kernels.
- **Multi-tier KV offload** — secondary tiers `fs` / `obj` / `p2p` registered in
  `vllm/v1/kv_offload/tiering/` (CPU is the *primary* tier, not a tiering backend);
  `csrc/fs_io.cpp` provides a GIL-releasing batch lookup for the filesystem tier.
- **KV-cache coordinator** — `NoPrefixCache` / `Unitary` / `Hybrid` coordinators
  (`vllm/v1/core/kv_cache_coordinator.py`).
- **Single-type KV managers** — `FullAttention`, `RSWA`, `SlidingWindow`, `ChunkedLocalAttention`,
  `Mamba`, `CrossAttention`, `SinkFullAttention`
  (`vllm/v1/core/single_type_kv_cache_manager.py`); the windowed managers reclaim out-of-window
  blocks.
- **Attention-sink KV spec** — `SinkFullAttentionSpec` reserves `sink_len // block_size` blocks up front
  (`vllm/v1/kv_cache_interface.py`).
- **Cross-layer KV sharing fast prefill** — `CacheConfig.kv_sharing_fast_prefill` skips redundant
  prefill compute on layers reusing an earlier layer's KV.
- **Mamba / SSM state cache modes** — `CacheConfig.mamba_cache_mode` (`all` / `align` / `none`)
  trades memory for prefix-cache hit rate; dtype via `CacheConfig.mamba_ssm_cache_dtype`.
- **Encoder cache manager** — `vllm/v1/core/encoder_cache_manager.py`.
- **KV events** — `BlockStored` / `BlockRemoved` / `AllBlocksCleared` published over ZMQ
  (`vllm/distributed/kv_events.py`, `KVEventsConfig.enable_kv_cache_events`).
- **Hybrid KV cache manager (HMA)** — the manager for mixed-spec models (full + SWA / local /
  Mamba). It is *not* CPU/GPU tiering. Disable with
  `SchedulerConfig.disable_hybrid_kv_cache_manager`; connectors opt in via `SupportsHMA`.
- **KV cache layout selection** — `VLLM_KV_CACHE_LAYOUT` (NHD vs HND; XPU forces NHD),
  `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` for the AITER shuffled layout.
- **KV cache residency metrics** — sampled per-block lifetime / idle / reuse-gap tracking
  (`vllm/v1/core/kv_cache_metrics.py`, `ObservabilityConfig.kv_cache_metrics`, sample rate
  `ObservabilityConfig.kv_cache_metrics_sample`, default 1%).

## 3. Quantization

Checkpoint formats live in `vllm/model_executor/layers/quantization/`; the GEMM backends that
execute them live in `vllm/model_executor/kernels/linear/` (see §4).

- **FP8 (E4M3 / E5M2)** — per-tensor / per-channel / per-128-block scales
  (`vllm/model_executor/layers/quantization/fp8.py`).
- **W8A8** — INT8/FP8 per-tensor / per-token / per-channel / per-group; CUDA at
  `csrc/libtorch_stable/quantization/w8a8` (incl. `per_token_group_quant_8bit.h`, CUTLASS blockwise
  128×128 SM90/100/120, and INT8 asymmetric AZP).
- **AWQ + AWQ-Marlin** — `auto_awq.py`; CUDA at `csrc/libtorch_stable/quantization/awq`.
- **GPTQ + GPTQ-Marlin** — `auto_gptq.py`; CUDA at `csrc/libtorch_stable/quantization/gptq`.
- **Marlin / Machete** — mixed-precision GEMMs in
  `vllm/model_executor/kernels/linear/mixed_precision/`. `VLLM_MARLIN_USE_ATOMIC_ADD` picks the
  reduction; `VLLM_MARLIN_INPUT_DTYPE` lets Marlin take int8/fp8 activations
  (`csrc/libtorch_stable/quantization/marlin/marlin_int4_fp8_preprocess.cu`).
- **AllSpark W8A16** — Ampere-optimized GPTQ GEMM + repack
  (`csrc/libtorch_stable/quantization/gptq_allspark`).
- **CUTLASS W4A8** — mm and grouped-mm for compressed-tensors W4A8 dense + MoE
  (`csrc/libtorch_stable/quantization/cutlass_w4a8`).
- **Compressed-tensors** — W4A4 / W4A8 / W8A8 / WNa16 schemes for linear and MoE
  (`vllm/model_executor/layers/quantization/compressed_tensors`), plus online
  **QuTLASS / hadacore Hadamard transforms**
  (`vllm/model_executor/layers/quantization/compressed_tensors/transform`). *Sparsity support was
  removed upstream.*
- **MXFP4** — OCP E2M1 + E8M0 block scales (`vllm/model_executor/kernels/linear/mxfp4`).
- **MXFP8** — native plus a dequant-at-load emulation
  (`vllm/model_executor/kernels/linear/mxfp8`, `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`).
- **NVFP4** — FP4 + FP8 block scales via CUTLASS / FlashInfer / fbgemm / Marlin / Humming /
  emulation (`vllm/model_executor/kernels/linear/nvfp4`); blockwise MoE kernels in
  `csrc/libtorch_stable/quantization/fp4/`.
- **NVIDIA ModelOpt** — `modelopt` (FP8), `modelopt_fp4`, `modelopt_mxfp8` and `modelopt_mixed`
  per-layer mixed precision (`vllm/model_executor/layers/quantization/modelopt.py`).
- **Bitsandbytes** — NF4 / INT8, on-the-fly or pre-quantized.
- **TorchAO** — int4 / int8 / fp8 AQT configs.
- **Quark** — AMD checkpoints: W8A8 FP8/INT8, NVFP4, OCP-MX, W4A8 MXFP4+FP8.
- **Intel Neural Compressor (INC)** — AutoRound INT2/3/4/8 weight-only checkpoints (`auto_round:auto_gptq` / `auto_round:auto_awq` packing; gptq / awq / marlin backends)
  (`vllm/model_executor/layers/quantization/inc`).
- **Humming (mixed-precision linear + MoE)** *(extended in v0.25.1)* — now backs dense linear
  (fp8 / int8 / fp8-block / wfp8a16 / mxfp8 / nvfp4 / mxfp4) as well as the MoE oracles.
  `vllm/model_executor/layers/quantization/humming.py`,
  `vllm/model_executor/kernels/linear/scaled_mm/humming.py`; knobs
  `VLLM_HUMMING_MOE_GEMM_TYPE` (`indexed` | `grouped` | `auto`),
  `VLLM_HUMMING_INPUT_QUANT_CONFIG`, `VLLM_HUMMING_ONLINE_QUANT_CONFIG`,
  `VLLM_HUMMING_USE_F16_ACCUM`.
- **TurboQuant** — 3/4-bit Lloyd-Max + norm-corrected KV compression presets
  (`vllm/model_executor/layers/quantization/turboquant`).
- **Experts INT8** — load-time per-channel INT8 of expert weights.
- **MoE WNA16** — int4/int8 weight-only MoE (`moe_wna16.py`,
  `csrc/libtorch_stable/moe/moe_wna16.cu`), with Marlin and RDNA3 variants.
- **Online (dynamic) quantization** — quantize an unquantized checkpoint at load time: fp8
  (per-tensor / block / channel), int8 weight-only, mxfp8
  (`vllm/model_executor/layers/quantization/online`).
- **Input FP8 quant** — `QuantFP8` static/dynamic per-tensor, per-token, per-group activation quant
  (`vllm/model_executor/layers/quantization/input_quant_fp8.py`).
- **Fused activation + quant** — `csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu`,
  plus the Helion variant (§4).
- **Fused layernorm + quant** — `csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu`.
- **FBGEMM FP8** — still registered but listed in `DEPRECATED_QUANTIZATION_METHODS`; requires
  `ModelConfig.allow_deprecated_quantization`.

*Removed at v0.25.1:* SmoothQuant; GGUF (moved to the out-of-tree `vllm-gguf-plugin`).

## 4. Kernel Selection & Dispatch

The layer that decides *which* implementation of an op actually runs — new enough that the previous
revision had no section for it.

- **Linear / MoE kernel oracles** — per-dtype registries under
  `vllm/model_executor/kernels/linear/` (`scaled_mm`, `mixed_precision`, `mxfp4`, `mxfp8`, `nvfp4`)
  and `vllm/model_executor/layers/fused_moe/oracle/`, picking the best backend for the current
  hardware and parallel config. Overridable with `KernelConfig.linear_backend` and
  `KernelConfig.moe_backend` (`vllm/config/kernel.py`).
- **vLLM IR op dispatch priority** — `vllm/ir/` defines high-level ops whose implementations are
  ranked by `KernelConfig.ir_op_priority` / `IrOpPriorityConfig`, then lowered by the IR lowering
  pass (§8).
- **Helion autotuned kernels** — `vllm/kernels/helion/ops/` (7 modules) served from checked-in
  per-GPU tuned configs in `vllm/kernels/helion/configs/` rather than autotuning at runtime;
  registered through `vllm/kernels/helion/register.py`. `silu_and_mul_per_block_quant` was added in
  v0.25.1; the others cover `rms_norm_*_quant`, `per_token_group_fp8_quant`,
  `dynamic_per_token_scaled_fp8_quant`, `fused_qk_norm_rope`, `silu_mul_fp8`.
- **OINK fused custom ops** — dispatches vLLM IR ops to externally-registered `oink::` ops on
  SM100+ (`vllm/kernels/oink_ops.py`, `VLLM_USE_OINK_OPS`).
- **Custom-op enable/disable list** — `CompilationConfig.custom_ops` (`all` / `none` / per-op `+op` / `-op`)
  chooses hand-written `CustomOp` kernels vs Inductor-compiled native impls
  (`vllm/model_executor/custom_op.py`).
- **Startup kernel warmup / autotune** — `vllm/model_executor/warmup/` pre-runs DeepGEMM JIT
  (`VLLM_DEEP_GEMM_WARMUP` skip/full/relax), the FlashInfer autotuner
  (`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`), CuteDSL and sparse-MLA warmups. At v0.25.1 the FlashInfer
  persistent cache is disabled whenever `world_size > 1` so all ranks tune in lockstep, and
  DeepGEMM warmup is skipped unless the FP8 linear kernel really is the DeepGEMM one.
- **Disabled-kernel escape hatch** — `VLLM_DISABLED_KERNELS`.

## 5. Parallelism

- **Tensor Parallel (TP)** — Column/Row/QKV/MergedColumn `ParallelLinear`
  (`vllm/model_executor/layers/linear.py`), `ParallelConfig.tensor_parallel_size`.
- **Pipeline Parallel (PP)** — layer sharding with p2p hidden-state transfer; split overridable via
  `VLLM_PP_LAYER_PARTITION`.
- **Async PP token broadcast** — the sampled-token broadcast runs on a side stream with events so
  the default stream isn't gated by the peer call (`vllm/v1/worker/gpu/pp_utils.py`).
- **Expert Parallel (EP)** — `ParallelConfig.enable_expert_parallel`, `FusedMoEParallelConfig.use_ep`.
- **Data Parallel (DP)** — replicated engine ranks with per-step token-count sync and cudagraph/DP
  padding (`vllm/v1/worker/gpu/dp_utils.py`, `VLLM_DP_SIZE`). The `DPCoordinator`
  (`vllm/v1/engine/coordinator.py`, `VLLM_DP_MASTER_PORT`) publishes per-rank queue depths and
  synchronizes request waves; `data_parallel_external_lb` / `data_parallel_hybrid_lb` hand
  balancing to the front end.
- **Context Parallel** — separate prefill (PCP) and decode (DCP) sizes
  (`ParallelConfig.decode_context_parallel_size`, `vllm/v1/worker/gpu/cp_utils.py`).
- **CP KV-cache interleaving** — `ParallelConfig.cp_kv_cache_interleave_size`, with a cudagraph-safe
  Triton kernel for per-rank local sequence lengths.
- **Sequence Parallel (IR-level)** — `vllm/compilation/passes/fusion/sequence_parallelism.py`,
  `PassConfig.enable_sp`.
- **Sequence-parallel MoE** — not a flag but a derived property
  (`ParallelConfig.use_sequence_parallel_moe`): true when `all2all_backend` is one of
  allgather_reducescatter / deepep_high_throughput / deepep_low_latency / mori_* / nixl_ep.
  **Changed in v0.25.1:** it now additionally requires `data_parallel_size > 1`, disabling it for
  TP-only + EP deployments (labeled a temp fix upstream).
- **Async TP** — `AsyncTPPass` fuses GEMM with reduce-scatter and all-gather with GEMM
  (`vllm/compilation/passes/fusion/collective_fusion.py`, `PassConfig.fuse_gemm_comms`).
- **DCP all-to-all attention** — Ulysses-style A2A exchanging partial attention out + LSE in one
  all-to-all instead of allgather + reduce-scatter
  (`vllm/v1/attention/ops/dcp_alltoall.py`, `ParallelConfig.dcp_comm_backend`).
- **Dual Batch Overlap (DBO)** — splits a step into two ubatches so one's EP/TP communication
  overlaps the other's compute (`ParallelConfig.enable_dbo`, gating in
  `vllm/v1/worker/gpu_ubatch_wrapper.py`, contexts in `vllm/v1/worker/ubatching.py`);
  `VLLM_DBO_COMM_SMS` partitions SMs between comm and compute.
- **Elastic Expert Parallel** — scale EP/DP world size at runtime via standby stateless groups and
  staged expert-weight transfer (`vllm/distributed/elastic_ep`,
  `ParallelConfig.enable_elastic_ep`, `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`); requires `enable_eplb`
  and PP == 1.
- **NUMA binding** — `ParallelConfig.numa_bind` pins each worker's CPU/memory to its GPU's NUMA
  node (`vllm/utils/numa_utils.py`).

## 6. MoE Optimizations

- **Fused MoE kernel** — Triton `fused_moe_kernel` / `fused_experts` over a sorted token-expert map
  (`vllm/model_executor/layers/fused_moe/fused_moe.py`).
- **Grouped GEMM / Batched MoE** — `vllm/model_executor/layers/fused_moe/experts/`
  (`fused_batched_moe`, `batched_deep_gemm_moe`, `cutlass_moe`), backed by CUTLASS grouped GEMM in
  `csrc/libtorch_stable/quantization/w8a8/cutlass/moe/`.
- **Modular Kernel** — pluggable `FusedMoEPrepareAndFinalize`
  (`vllm/model_executor/layers/fused_moe/prepare_finalize/`) + `FusedMoEExperts`
  (`vllm/model_executor/layers/fused_moe/experts/`) composed by the MoE runner
  (`vllm/model_executor/layers/fused_moe/modular_kernel.py`).
- **MoE kernel oracles** — per-quant-dtype backend + weight-schema selection
  (`vllm/model_executor/layers/fused_moe/oracle/`), see §4.
- **Routers** — fused topk / topk-bias / grouped-topk / custom
  (`vllm/model_executor/layers/fused_moe/router/`) over CUDA `topk_softmax`,
  `topk_softplus_sqrt` and `grouped_topk` kernels (`csrc/libtorch_stable/moe/`);
  `VLLM_USE_FUSED_MOE_GROUPED_TOPK`.
- **Zero-expert routing** — tokens routed to zero/identity experts skip expert GEMMs
  (`vllm/model_executor/layers/fused_moe/router/zero_expert_router.py`).
- **Permute / Unpermute** — `vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py` over
  `csrc/libtorch_stable/moe/permute_unpermute_kernels/`.
- **`moe_align_block_size` / padding skip** — sorts tokens by expert and pads to the GEMM tile;
  `VLLM_MOE_SKIP_PADDING` forces cudagraph/DP padding tokens' expert ids to `-1` so dispatch and the experts drop them.
- **MoE FP8 / INT8 / WNA16** — `vllm/model_executor/layers/fused_moe/experts/` plus
  `csrc/libtorch_stable/moe/moe_wna16.cu`.
- **Marlin MoE** — `experts/marlin_moe.py` over `csrc/libtorch_stable/moe/marlin_moe_wna16/`.
- **FlashInfer Cutlass MoE** — `experts/flashinfer_cutlass_moe.py`.
- **FlashInfer CuteDSL MoE** — contiguous and batched Blackwell NVFP4 experts
  (`experts/flashinfer_cutedsl_moe.py`).
- **FlashInfer B12x MoE** — `experts/flashinfer_b12x_moe.py`. **Changed in v0.25.1:** supports
  non-gated RELU2 MoEs (Nemotron-style) via a lazily-built `B12xMoEWrapper` with cached MMA-layout
  scale factors, and is now **incompatible with expert parallelism**.
- **TensorRT-LLM MoE** — BF16 / FP8 / NVFP4 / MXFP4 / MXINT4 experts (`experts/trtllm_fp8_moe.py`).
  **Changed in v0.25.1:** accepts FlashInfer NVLink one-/two-sided A2A backends.
- **GPT-OSS Triton-kernels MoE** — MXFP4 via OpenAI `triton_kernels`
  (`experts/gpt_oss_triton_kernels_moe.py`).
- **HPC-Ops MoE** — `vllm/model_executor/layers/fused_moe/hpc_moe.py`.
- **DeepGEMM** — FP8 block-scaled grouped GEMM (`experts/deep_gemm_moe.py`,
  `experts/triton_deep_gemm_moe.py`, `experts/batched_deep_gemm_moe.py`); `VLLM_USE_DEEP_GEMM`,
  `VLLM_MOE_USE_DEEP_GEMM`, `VLLM_USE_DEEP_GEMM_E8M0`, `VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES`.
- **All-to-all expert backends** — `vllm/model_executor/layers/fused_moe/prepare_finalize/`:
  DeepEP HT/LL/v2 (`VLLM_DEEPEP_V2_PREFER_OVERLAP`, `VLLM_DEEPEP_BUFFER_SIZE_MB`,
  `VLLM_DEEPEPLL_NVFP4_DISPATCH` for 4-bit dispatch payloads), Mori HT/LL,
  NIXL-EP, FlashInfer NVLink one-/two-sided, and allgather-reducescatter. Selected by
  `ParallelConfig.all2all_backend`. *pplx and naive were removed.*
- **Shared experts** — run concurrently with routed experts on a side CUDA stream
  (`runner/shared_experts.py`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM`).
- **AITER shared-experts fusion** — folds the shared expert into the routed AITER top-k path
  (`router/aiter_shared_routed_fused_moe_router.py`,
  `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`).
- **Deferred MoE all-reduce** *(generalized in v0.25.1)* — `FusedMoE(reduce_results=False)` skips
  the final MoE all-reduce so the next layer's RMSNorm fuses it
  (`vllm/model_executor/layers/fused_moe/layer.py`,
  `vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py`). Now applies to dense layers as
  well as MoE; disabled for MTP blocks and PP > 1.
- **EPLB** — runtime expert rearrangement from a load-history window plus redundant experts
  (`vllm/distributed/eplb/`, `ParallelConfig.enable_eplb`, `EPLBConfig.window_size` /
  `step_interval` / `num_redundant_experts` / `use_async` / `communicator`), with per-rank maps in
  `vllm/model_executor/layers/fused_moe/expert_map_manager.py`. Rebalance transports:
  `torch_nccl`, `torch_gloo`, `nixl`, `pynccl`; async rebalance overlaps model execution.
- **Expert placement strategy** — `ParallelConfig.expert_placement_strategy`
  (`linear` vs `round_robin`).
- **MoE routing simulator** — synthetic routing distributions for EP load-balance benchmarking
  (`router/routing_simulator_router.py`, `VLLM_MOE_ROUTING_SIMULATION_STRATEGY`).
- **Routed-experts capture / replay** — records per-token expert indices
  (`routed_experts_capturer.py`).

## 7. Speculative Decoding

- **Draft-target spec decoding** — `vllm/v1/spec_decode/draft_model.py`, `SpeculativeConfig.method`.
- **EAGLE / EAGLE-3** — `vllm/v1/spec_decode/eagle.py`; eagle3 optionally consumes auxiliary target
  hidden states.
- **Medusa** — extra heads on the target's last hidden state (`vllm/v1/spec_decode/medusa.py`).
- **MLP speculator** — chained MLP heads (`vllm/model_executor/models/mlp_speculator.py`).
- **N-gram proposer** — CPU `NgramProposer` plus a vectorized `torch.compile`d
  `NgramProposerGPU` (`vllm/v1/spec_decode/ngram_proposer.py`, `ngram_proposer_gpu.py`).
- **Suffix tree decoding** — per-prompt suffix trees (`vllm/v1/spec_decode/suffix_decoding.py`,
  `SpeculativeConfig.suffix_decoding_max_tree_depth`).
- **DFlash parallel drafting** — all N drafts in one pass with its own prepare-inputs Triton kernel
  and CUDA-graph manager (`vllm/v1/spec_decode/dflash.py`).
- **DSpark semi-autoregressive drafting** — drafts a whole block in one parallel pass then adds
  intra-block dependency via a Markov head
  (`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`).
- **MTP** — model-native multi-token-prediction heads (`vllm/v1/spec_decode/llm_base_proposer.py`;
  the V2 runner's opt-in subclass is `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`). Legacy
  per-model `*_mtp` method names are deprecated aliases.
- **Custom proposer plug-in** — `method='custom_class'`
  (`vllm/v1/spec_decode/custom_class_proposer.py`).
- **Extract-hidden-states proposer** — 1-token proposer exporting target hidden states for external
  drafters (`vllm/v1/spec_decode/extract_hidden_states.py`).
- **Dynamic draft length** — `SpeculativeConfig.num_speculative_tokens_per_batch_size` shrinks the
  draft as batch size grows (`vllm/v1/spec_decode/dynamic/`).
- **Heterogeneous draft/target vocab (TLI)** — token-level intersection constrains draft logits to
  shared tokens (`vllm/v1/spec_decode/vocab_mapping.py`,
  `SpeculativeConfig.use_heterogeneous_vocab`).
- **Vocab-parallel local argmax reduction** — reduces per-rank argmax instead of all-gathering
  logits: O(2·tp_size) instead of O(vocab) per token
  (`SpeculativeConfig.use_local_argmax_reduction`).
- **Padded drafter batch** — uniform draft lengths keep EAGLE drafting CUDA-graph friendly
  (`SpeculativeConfig.disable_padded_drafter_batch` opts out).
- **Parallel drafting** — all speculative tokens in one pass; EAGLE and draft_model only
  (`SpeculativeConfig.parallel_drafting`).
- **Rejection sampler** — Triton greedy + random kernels with recovered-token sampling
  (`vllm/v1/sample/rejection_sampler.py`). `SpeculativeConfig.rejection_sample_method` selects
  `standard`, `block` (joint block verification of the whole draft block, Sun et al.) or `synthetic` (decaying
  calibrated accept rates); `SpeculativeConfig.draft_sample_method` picks greedy vs probabilistic
  draft sampling.
- **Lookahead KV slot reservation** — the scheduler reserves `num_lookahead_tokens` KV slots per
  step (+1 for DFlash's in-fill layout). **Changed in v0.25.1:** async P/D KV loads are clipped
  whenever `num_lookahead_tokens > 0` (was EAGLE-only), so MTP is covered, via
  the scheduler's `limit_lookahead_tokens` gate (`KVCacheManager.get_block_ids_for_computed_tokens()` clips the connector's block list in the same fix).
- **Drafter dummy-run / CUDA graph capture** — `vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py`.
  **Fixed in v0.25.1:** with DP > 1 the runner issues `drafter.dummy_run(num_tokens=1)` on both
  paths when ranks disagree on `input_fits_in_drafter`, avoiding a collective hang.
- **Draft `hf_overrides` composition** — callable target overrides now reach the draft
  `ModelConfig` through a picklable partial (`SpeculativeConfig.compose_draft_hf_overrides()` — a
  staticmethod, not a field).
- **Acceptance metrics** — per-step `SpecDecodingStats` → Prometheus acceptance-rate and
  per-position accepted-token counters (`vllm/v1/spec_decode/metrics.py`).
- **CPU spec-decode kernels** — `csrc/cpu/spec_decode_utils.cpp`.

## 8. Compilation & Graph Optimizations

- **torch.compile / Inductor backend** — `VllmBackend` splits, compiles and stitches the FX graph
  (`vllm/compilation/backends.py`, `CompilationConfig.backend`).
- **CUDA Graphs** — `FULL` / `PIECEWISE` / `FULL_AND_PIECEWISE`
  (`vllm/compilation/cuda_graph.py`, `CompilationConfig.cudagraph_mode`).
- **Breakable CUDA graphs** — runtime stream-capture breaks instead of FX splitting
  (`vllm/compilation/breakable_cudagraph.py`, `VLLM_USE_BREAKABLE_CUDAGRAPH`).
- **Inductor graph partition** — let Inductor partition for cudagraphs with vLLM partition rules
  (`vllm/compilation/partition_rules.py`, `CompilationConfig.use_inductor_graph_partition`).
- **Stitching-graph codegen** — generates a plain Python execution function for the split graph,
  removing `nn.Module.__call__` dispatch overhead (`vllm/compilation/codegen.py`).
- **Dynamic-shapes control** — `BACKED` / `UNBACKED` / `BACKED_SIZE_OBLIVIOUS`
  (`CompilationConfig.dynamic_shapes_config`).
- **Compile ranges / shape specialization** — `CompilationConfig.compile_ranges_endpoints` and
  `compile_sizes`.
- **LoRA-specialized CUDA graphs** — `CompilationConfig.cudagraph_specialize_lora` (on by default).
- **AOT compile + mega artifact** — `VLLM_USE_AOT_COMPILE`, `VLLM_USE_MEGA_AOT_ARTIFACT`
  (`vllm/compilation/decorators.py`), plus a guard-free bytecode-hook wrapper.
- **Custom Inductor passes** — `PostGradPassManager` (`vllm/compilation/passes/`);
  user passes via `CompilationConfig.inductor_passes`.
- **Compile cache** — `vllm/compilation/caching.py`, `CompilationConfig.cache_dir`,
  `VLLM_DISABLE_COMPILE_CACHE`, `VLLM_COMPILE_CACHE_SAVE_FORMAT` (`binary` is multiprocess-safe).
- **Inductor max-autotune / coordinate-descent tuning** — both on by default; toggle with
  `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` and
  `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING`.
- **Fusion passes** (`vllm/compilation/passes/fusion/`):
      - RMSNorm + Quant — `rms_quant_fusion.py`, `PassConfig.fuse_norm_quant`
      - All-Reduce + RMSNorm (+ static quant) — `allreduce_rms_fusion.py`,
    `PassConfig.fuse_allreduce_rms`, size threshold
    `PassConfig.fi_allreduce_fusion_max_size_mb`. **Changed in v0.25.1:** static-quant patterns are
    guarded by a norm-input/weight dtype match, and MNNVL one-shot selection is delegated to
    FlashInfer AUTO.
      - Attention + Quant — `attn_quant_fusion.py`, `PassConfig.fuse_attn_quant`
      - MLA Attention + Quant — `mla_attn_quant_fusion.py`
      - QK-Norm + RoPE — `qk_norm_rope_fusion.py`, `PassConfig.enable_qk_norm_rope_fusion`
      - Activation + Quant — `act_quant_fusion.py`, `PassConfig.fuse_act_quant`
      - [RoPE + KV-cache update](RoPEKVCacheFusion.md) — `rope_kvcache_fusion.py`,
    `PassConfig.fuse_rope_kvcache`. **ROCm-only:** force-disabled on non-ROCm platforms
    (`vllm/config/compilation.py`); token cap `rope_kvcache_fusion_max_token_num`
      - MLA RoPE + KV-cache cat — `mla_rope_kvcache_cat_fusion.py`
      - Collective + Compute (async TP) — `collective_fusion.py`, `PassConfig.fuse_gemm_comms`
      - ROCm AITER — `rocm_aiter_fusion.py`: RMSNorm+quant, SiLU-mul+FP8 group quant,
    add-RMSNorm+router pad (`PassConfig.fuse_act_padding`), MLA dual-RMSNorm
    (`PassConfig.fuse_mla_dual_rms_norm`)
- **Sequence-parallelism IR pass** — `sequence_parallelism.py`, `PassConfig.enable_sp`,
  `PassConfig.sp_min_token_num`.
- **Lowering + functionalization** — `passes/ir/lowering_pass.py` lowers vLLM IR ops after fusion;
  `passes/ir/inplace_functionalization.py` rewrites maybe-inplace overloads pre-grad so activations
  can be donated; `passes/ir/clone_elimination.py` removes the resulting redundant clones.
- **Utility passes** — `vllm/compilation/passes/utility/`: `NoOpElimination`
  (`PassConfig.eliminate_noops`),
  `SplitCoalescing`, `ScatterSplitReplacement`, `PostCleanup` (topo-sort + DCE), and the mandatory
  `FixFunctionalization` de-functionalization pass.
- **CUDAGraph GC** — vLLM freezes the GC heap around capture by default;
  `VLLM_ENABLE_CUDAGRAPH_GC=1` re-enables it (`vllm/v1/worker/gpu_model_runner.py`).
- **Fast MoE cold start** — `CompilationConfig.fast_moe_cold_start`.
- **XPU graph capture** — the Intel GPU cudagraph equivalent (`vllm/v1/worker/xpu_model_runner.py`).

## 9. Scheduling & Batching

- **Continuous batching** — `Scheduler.schedule()` rebuilds the running batch every step under
  token/seq budgets (`vllm/v1/core/sched/scheduler.py`, `SchedulerConfig.max_num_seqs`).
- **Chunked prefill** — on by default in V1 (`SchedulerConfig.enable_chunked_prefill`).
- **Prefix-cache-aware scheduling** — only the uncached suffix is scheduled
  (`vllm/v1/core/kv_cache_manager.py`).
- **Priority scheduling** — `SchedulingPolicy` FCFS (deque) or PRIORITY (heap)
  (`vllm/v1/core/sched/request_queue.py`, `SchedulerConfig.policy`).
- **Async scheduler** — overlaps the next `schedule()` with the current forward by writing output
  placeholders (`vllm/v1/core/sched/async_scheduler.py`, `SchedulerConfig.async_scheduling`).
  **Fixed in v0.25.1:** `num_output_placeholders` underflow with spec decode.
- **Preemption and recompute** — running requests are preempted back to the waiting queue with KV
  blocks and encoder cache freed when the KV cache is exhausted.
- **KV cache admission watermark** — `SchedulerConfig.watermark` keeps a fraction of blocks free to
  avoid eviction-driven repeated preemption.
- **Full-ISL admission reservation** — `SchedulerConfig.scheduler_reserve_full_isl` checks the whole
  input length fits, not just the first chunk.
- **Concurrent partial-prefill caps** — `SchedulerConfig.max_num_partial_prefills` and
  `long_prefill_token_threshold` are declared, but only the latter works — a non-default `max_num_partial_prefills` raises `NotImplementedError` ("Concurrent Partial Prefill is not supported") at config time, and the threshold only caps a long prompt's tokens per step so shorter ones still fit.
- **DP-aligned prefill throttling** — `SchedulerConfig.prefill_schedule_interval` admits new
  prefills only every N steps, aligned across DP ranks.
- **Spec-decode lookahead slot budgeting** — see §7.
- **Mamba block-aligned prefill split** — chunk boundaries snap to SSM state-block alignment.
- **Chunked multimodal input control** — `SchedulerConfig.disable_chunked_mm_input` keeps an
  image/video whole within one step.
- **Separate scheduled-token cap** — `SchedulerConfig.max_num_scheduled_tokens`, distinct from
  `max_num_batched_tokens` because the model may append draft tokens.
- **Batched structured-output grammar bitmask** — one batched `get_grammar_bitmask` per step.
- **Ubatching / micro-batching** — `vllm/v1/worker/ubatching.py`, see DBO in §5.
- **Disaggregated prefill/decode** — requests wait in `_update_waiting_for_remote_kv` until async
  remote KV loads land (`VllmConfig.kv_transfer_config`).
- **Encoder budget for multimodal** — `compute_mm_encoder_budget` +
  `SchedulerConfig.encoder_cache_size`.
- **Output batching** — `OutputProcessor` emits every `SchedulerConfig.stream_interval` tokens.
- **Pluggable scheduler class** — `SchedulerConfig.scheduler_cls` over `SchedulerInterface`.

*Removed:* multi-step scheduling (`--num-scheduler-steps`).

## 10. Communication

- **Custom all-reduce** — one/two-shot IPC all-reduce for small TP messages
  (`csrc/libtorch_stable/custom_all_reduce.cu`,
  `vllm/distributed/device_communicators/custom_all_reduce.py`,
  `ParallelConfig.disable_custom_all_reduce`); used only below a per-arch max size (8 MiB default),
  world sizes 2/4/6/8.
- **Quick all-reduce (ROCm)** — quantized all-reduce with FP / INT8 / INT6 / INT4 / INT3 regimes
  (INT3 restricted to TP2); kernels in `csrc/quickreduce/` and `csrc/custom_quickreduce.cu`;
  `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB`.
- **NVIDIA Symmetric Memory all-reduce** — torch symmetric-memory multimem
  (`device_communicators/symm_mem.py`, `VLLM_ALLREDUCE_USE_SYMM_MEM`).
- **NCCL symmetric-memory allocator** — allocate tensors in NCCL symmetric memory for multimem
  collectives (`device_communicators/pynccl_allocator.py`, `VLLM_USE_NCCL_SYMM_MEM`).
- **FlashInfer all-reduce** — trtllm / MNNVL with shared workspaces
  (`device_communicators/flashinfer_all_reduce.py`, `VLLM_ALLREDUCE_USE_FLASHINFER`,
  `VLLM_FLASHINFER_ALLREDUCE_BACKEND`, `VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB`).
- **AITER custom all-reduce (ROCm)** — AITER's `CustomAllreduce` wired directly into
  `CudaCommunicator` (`device_communicators/aiter_custom_all_reduce.py`,
  `VLLM_ROCM_USE_AITER_CUSTOM_AR`); prerequisite for the ROCm fused AR+RMSNorm patterns.
- **PyNCCL** — ctypes NCCL wrapper for hot collectives with tuning tables in `all_reduce_utils.py`
  (`VLLM_DISABLE_PYNCCL`). `distributed_timeout_seconds` now propagates to NCCL device groups.
- **Shared-memory broadcast / object storage** — lock-free shm ring buffer
  (`device_communicators/shm_broadcast.py`) plus a shared-memory object store
  (`shm_object_storage.py`, `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`); polling can use the MWAITX
  spin-loop extension (`csrc/spinloop.cpp`, `VLLM_USE_SPINLOOP_EXT`).
- **Ray compiled-graph communicator** — `RayPPCommunicator` channels for PP transport under Ray.
- **All-to-all backends** — see §6.
- **KV-transfer connectors** — registered in `KVConnectorFactory`
  (`vllm/distributed/kv_transfer/kv_connector/v1/`): NIXL (pull/push + heterogeneous TP mapping),
  LMCache (+MP), Mooncake (+Store), MoRIIO, HF3FS, FlexKV, Offloading, SimpleCPUOffload, Multi,
  DecodeBench. Knobs: `KVTransferConfig.kv_connector`, `VLLM_NIXL_SIDE_CHANNEL_HOST`/`_PORT`,
  `VLLM_MOONCAKE_BOOTSTRAP_PORT`. Per-connector stats and Prometheus metrics in
  `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py`.
- **Weight-transfer engines** — push updated weights into a live engine for RL / online sync
  (`vllm/distributed/weight_transfer/`, `WeightTransferConfig.backend` = `nccl` | `ipc` |
  `sparse_nccl`).
- **EC (encoder cache) transfer** — connector-style plumbing for **encoder-cache** transfer between
  engines (`vllm/distributed/ec_transfer/`, `ECTransferConfig.ec_connector`). *Not* "expert
  collection", and not MoE-related.
- **CPU shared-memory collectives** — SHM allreduce / allgather / tensor-list send-recv
  (`csrc/cpu/shm.cpp`).

*Removed:* P2P NCCL connector (superseded by `vllm/v1/kv_offload/tiering/p2p`).

## 11. Sampling & Output

- **FlashInfer sampling** — `TopKTopPSampler.forward_cuda` dispatches to `flashinfer.sampling`
  (`vllm/v1/sample/ops/topk_topp_sampler.py`, `VLLM_USE_FLASHINFER_SAMPLER`).
- **Triton top-k / top-p** — sort-free fused mask kernel
  (`vllm/v1/sample/ops/topk_topp_triton.py`); used on CPU and for GPU batches ≥ 8 rows.
- **Fused per-platform sampler** — `TopKTopPSampler` binds `self.forward` at construction to
  `forward_cuda` / `forward_hip` / `forward_xpu` / `forward_cpu` / `forward_native`.
- **ROCm AITER fused sampler** — `forward_hip` routes to `aiter.ops.sampling`.
- **XPU fused sampler kernel** — `torch.ops.vllm.xpu_topk_topp_sampler` does mask+sample+logits in
  one kernel with manual RNG offset advance (`vllm/_xpu_ops.py`,
  `VLLM_XPU_USE_SAMPLER_KERNEL`).
- **Exponential-race (Gumbel-max) sampling** — `random_sample` draws exponential noise and argmaxes
  `probs/q`, avoiding `torch.multinomial`'s CPU-GPU sync; optional FP64 noise via
  `ModelConfig.use_fp64_gumbel`.
- **Sort-free top-k-only path** — `apply_top_k_only` masks via `torch.topk` + gather instead of
  sorting the vocab.
- **All-greedy / all-random fast paths** — `Sampler.sample` short-circuits to argmax when all
  requests are greedy, skips greedy when all random, applies temperature in place
  (`vllm/v1/sample/sampler.py`).
- **V2-runner Triton sampler stack** — all-Triton temperature/Gumbel and
  penalties/min-p/logit-bias/bad-words kernels over persistent GPU state
  (`vllm/v1/worker/gpu/sample/`).
- **Logits processors** — batch-persistent framework with builtin min-p, logit-bias, min-tokens plus
  plugin/FQCN-loaded custom processors (`vllm/v1/sample/logits_processor/`).
- **Frequency / presence / repetition penalties** — CUDA `apply_repetition_penalties_`
  (`csrc/libtorch_stable/sampler.cu`) via `vllm/v1/sample/ops/penalties.py`; Triton version in the
  V2 runner.
- **Bad-words masking** — suffix-matched, draft-aware (`vllm/v1/sample/ops/bad_words.py`).
- **Thinking-budget logits forcing** — forces end-of-think tokens directly in logits, including
  under spec decode (`vllm/v1/sample/thinking_budget_state.py`).
- **Structured output** — xGrammar with a compiled-grammar cache (`VLLM_XGRAMMAR_CACHE_MB`,
  default 512 MB), Outlines with an on-disk compiled-index cache (`VLLM_V1_USE_OUTLINES_CACHE`),
  llguidance, and lm-format-enforcer — all under `vllm/v1/structured_output/`,
  selected by `StructuredOutputsConfig.backend`.
- **Triton grammar-bitmask kernel** — applies the token bitmask to logits on GPU, avoiding a host
  round-trip (`vllm/v1/worker/gpu/structured_outputs.py`).
- **Async grammar compilation / bitmask fill** — `StructuredOutputManager` uses thread pools so FSM
  work overlaps the forward pass (`vllm/v1/structured_output/__init__.py`).
- **Logprobs computation** — a `torch.compile`d batched rank count plus one raw-logprobs pass
  reused for the whole step (`vllm/v1/sample/ops/logprobs.py`, `ModelConfig.logprobs_mode`).
  *There is no logprobs cache.*
- **Chunked prompt-logprobs** — prompt-position-chunked (1024 tokens at a time) Triton computation to bound peak memory on long
  prompts (`vllm/v1/worker/gpu/sample/prompt_logprob.py`).

## 12. Memory & Runtime

- **Block pool** — ref-counted `KVCacheBlock`s with a null block, an intrusive free list and a
  hash→block map (`vllm/v1/core/block_pool.py`).
- **Custom cumem allocator / sleep mode** — CUDA VMM (`cuMemCreate`/`cuMemMap`) pluggable torch
  allocator backing tagged pools (`csrc/cumem_allocator.cpp`,
  `vllm/device_allocator/sleep_mode_backend.py`, `ModelConfig.enable_sleep_mode`,
  `ModelConfig.sleep_mode_backend`). Level 1 offloads weights to host RAM, level 2 discards them.
- **XPU sleep-mode allocator** — `vllm/device_allocator/xpumem.py`.
- **Pinned host memory** — global `PIN_MEMORY` flag (`vllm/utils/torch_utils.py`,
  `VLLM_WSL2_ENABLE_PIN_MEMORY`) plus `CpuGpuBuffer` (`vllm/v1/utils.py`), the runtime's standard
  paired pinned-CPU/GPU staging tensor. `csrc/libtorch_stable/cuda_view.cu` exposes a zero-copy
  CUDA view of a pinned CPU tensor.
- **Prefix-cache eviction (LRU)** — `FreeKVCacheBlockQueue`, an O(1) intrusive doubly-linked free
  list ordered LRU-first with longer hash chains evicted first (`vllm/v1/core/kv_cache_utils.py`).
- **Weight offloading** — `vllm/model_executor/offloader/` with two backends selected by
  `OffloadConfig.offload_backend` (`auto` | `uva` | `prefetch`):
      - **UVA** (`offloader/uva.py`) keeps weights in pinned CPU memory and exposes zero-copy GPU
    views, virtually enlarging GPU memory over PCIe (`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA`).
      - **Prefetch** (`offloader/prefetch.py`, `offloader/prefetch_ops.py`) does group-based offload
    with async H2D prefetch on a copy stream, using static buffers + events so it stays
    CUDA-graph capturable (`PrefetchOffloadConfig.offload_group_size`).
- **Workspace manager** — one growable scratch buffer per ubatch slot, then locked so size is fixed
  at runtime (`vllm/v1/worker/workspace.py`, `VLLM_DEBUG_WORKSPACE`).
- **Memory profiler** — `determine_available_memory()` profiles weights/activations and subtracts a
  CUDA-graph estimate (`vllm/v1/worker/gpu_worker.py`,
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, `CacheConfig.gpu_memory_utilization`).
- **Explicit KV-cache byte budget** — `CacheConfig.kv_cache_memory_bytes` pins the KV cache size
  directly, bypassing utilization-based profiling.
- **Blocking CUDA events** *(new in v0.25.1)* — `torch.cuda.Event(blocking=True)` at output-copy,
  draft-token and prepare-inputs sync points (`vllm/v1/worker/gpu/async_utils.py`,
  `vllm/v1/worker/gpu/spec_decode/utils.py`, `vllm/v1/worker/gpu_model_runner.py`) so a waiting TP
  rank sleeps instead of spinning on the CUDA driver lock. XPU drops the `blocking` kwarg in
  `vllm/v1/worker/xpu_model_runner.py`.
- **Auxiliary-stream parallel execution** — `maybe_execute_in_parallel` runs two independent
  callables on the main and an auxiliary stream joined by events
  (`vllm/utils/multi_stream_utils.py`, `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`); used by DSv4
  attention and LoRA layers.
- **GC debugger** — collect-time and top-collected-type instrumentation for engine-loop GC pauses
  (`vllm/utils/gc_utils.py`, `VLLM_GC_DEBUG`).

## 13. Model-Architecture-Specific Kernels

- **Fused RMSNorm + residual** — `csrc/libtorch_stable/layernorm_kernels.cu` (quantized variants in
  `layernorm_quant_kernels.cu`).
- **Fused activations** — `silu_and_mul` (+clamp), `mul_and_silu`, `gelu_and_mul`,
  `gelu_tanh_and_mul` and siblings (`csrc/libtorch_stable/activation_kernels.cu`); CPU LUT-based
  bf16 variant in `csrc/cpu/activation_lut_bf16.cpp`.
- **Fused QK-norm + RoPE + KV-insert** — per-family single kernels:
  `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` (bf16 / fp8 / rope-quant
  variants) and `csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`.
- **DeepSeek fused A-GEMM** — `csrc/libtorch_stable/dsv3_fused_a_gemm.cu`.
- **FP32 router GEMM** — `csrc/libtorch_stable/fp32_router_gemm.cu` plus the DSv3 variants in
  `csrc/libtorch_stable/moe/dsv3_router_gemm_*.cu`.
- **MiniMax fused all-reduce + RMSNorm QK** — the `minimax_allreduce_rms_qk` custom op
  (`csrc/libtorch_stable/minimax_reduce_rms_kernel.cu`), a Lamport one-shot fused AR+norm for TP.
- **MHC kernels** — multi-head **hyper-connection** pre/post blocks used by DSpark-DSv4, with
  torch / triton / tilelang / AITER backends (`vllm/model_executor/kernels/mhc/`).
- **MLA latent attention** — see §1.
- **Mamba / Mamba2 SSM kernels** — `selective_scan_fwd` (`csrc/libtorch_stable/mamba/`) plus Triton
  SSD chunked-scan ops (`vllm/model_executor/layers/mamba/ops/`), with per-GPU tuned configs for
  `selective_state_update`.
- **Mamba SSU backend + stochastic rounding** — `MambaConfig.backend` dispatches triton vs
  flashinfer (`vllm/model_executor/layers/mamba/ops/ssu_dispatch.py`);
  `MambaConfig.enable_stochastic_rounding` (+ `stochastic_rounding_philox_rounds`) makes fp16 SSM
  state caches viable.
- **Gated RMSNorm for Mamba2** — Triton gated LayerNorm/RMSNorm fused with the SSM output gate
  (`vllm/model_executor/layers/mamba/ops/layernorm_gated.py`).
- **Linear attention** — MiniMax lightning attention and Bailing linear attention
  (`vllm/model_executor/layers/mamba/linear/`).
- **GDN linear attention** — gated DeltaNet layers (Qwen / Olmo / Kimi) over the FLA Triton
  chunk/fused-recurrent ops (`vllm/model_executor/layers/mamba/gdn/`), with fused post-conv prefill
  prep and fused sigmoid gating in `vllm/model_executor/layers/fla/ops/`.
- **GDN chunked-prefill CuteDSL kernels (SM100)** —
  `vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/`.
- **Short-conv** — `vllm/model_executor/layers/mamba/short_conv.py`. **New in v0.25.1:**
  `forward_native` is a real CPU prefill+decode causal-conv1d path (was a no-op), over
  `vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py`.
- **FLA packed recurrent decode** — `fused_recurrent_gated_delta_rule_packed_decode`
  (`vllm/model_executor/layers/fla/ops/fused_recurrent.py`).
- **KDA — Kimi Delta Attention** — `chunk_kda_with_fused_gate`, `fused_kda_gate`,
  `fused_recurrent_kda` (`vllm/model_executor/layers/fla/ops/kda.py`).
- **Batch-invariant mode** — deterministic mm/bmm/softmax/mean overrides
  (`vllm/model_executor/layers/batch_invariant.py`, `VLLM_BATCH_INVARIANT`): SM80 uses Triton
  kernels, SM90/SM100 disable cuBLAS split-k.
- **`vllm/models/` vendor fast paths** — heavily-optimized per-model implementations outside the
  flat `vllm/model_executor/models/` layout: `vllm/models/deepseek_v4/` (fused compress-quant cache, fused indexer q,
  fused inverse-RoPE FP8 quant, CuteDSL sparse-attention compress, per-vendor `nvidia`/`amd`/`xpu`
  subtrees), `vllm/models/deepseek_v32/`, `vllm/models/minimax_m3/`.
- **Transformers-backend graph fusers** *(new in v0.25.1)* — FX-trace-driven fusion of arbitrary HF
  models into vLLM primitives (`vllm/model_executor/models/transformers/fusers/`: `qkv.py`,
  `glu.py`, `moe.py`, `rms_norm.py`), driven by
  `vllm/model_executor/models/transformers/fuser.py` +
  `vllm/model_executor/models/transformers/fx_utils.py`. Models without a HF
  `tp_plan`/`_pp_plan` now warn and infer the split instead of raising.
- **RoPE variants** — one module per scheme in `vllm/model_executor/layers/rotary_embedding/`:
  linear scaling, NTK, dynamic-NTK scaling, dynamic-NTK alpha, YaRN, Llama3, Llama4-vision,
  DeepSeek scaling, TeleChat3 scaling, Phi3-long, MRoPE, MRoPE interleaved, dual-chunk, FoPE,
  XDRoPE, Ernie4.5-VL, Gemma4 (proportional: `inv_freq` exponents use `head_dim` rather than
  `rotary_dim`). Instances are deduplicated across layers by a `_ROPE_DICT` cache
  (`rotary_embedding/__init__.py`); FlashInfer offers a fused
  `apply_rope_with_cos_sin_cache_inplace` fast path.

## 14. LoRA / Multi-LoRA

- **Punica kernels** — batched LoRA shrink/expand; on CUDA these are Triton kernels
  (`vllm/lora/ops/triton_ops/lora_shrink_op.py`, `lora_expand_op.py`), not CUDA BGMV.
- **Multi-LoRA batching** — the scheduler caps concurrently scheduled distinct adapters at
  `LoRAConfig.max_loras`.
- **LoRA dual-stream execution** — shrink/expand on an auxiliary stream overlapped with the base
  GEMM (`VLLM_LORA_ENABLE_DUAL_STREAM`); guarded to CUDA-alike platforms
  (`current_platform.is_cuda_alike()`, so ROCm is allowed) and incompatible with
  `fully_sharded_loras`.
- **PDL projection optimization** — Programmatic Dependent Launch (GDC) in the Triton LoRA kernels
  (`vllm/lora/ops/triton_ops/utils.py`, opt-out `VLLM_LORA_DISABLE_PDL`). Dense paths enable GDC
  only when dual-stream is on; FP8 dense paths never do.
- **LoRA model manager** — adapter slots, activation, pinning and dummy-LoRA warmup via an
  `AdapterLRUCache` (`vllm/lora/model_manager.py`, `LoRAConfig.max_cpu_loras`).
- **LoRA weight resolver** — pluggable `LoRAResolver` registry for request-time adapter fetching
  (`vllm/lora/resolver.py`, `VLLM_LORA_RESOLVER_CACHE_DIR`).
- **Fused MoE LoRA** — Triton fused-MoE LoRA kernels plus `FusedMoEWithLoRA`
  (`vllm/lora/ops/triton_ops/fused_moe_lora_op.py`, `vllm/lora/layers/fused_moe.py`).
- **FP8 quantized LoRA kernels** — `lora_shrink_fp8_op.py`, `lora_expand_fp8_op.py`,
  `fused_moe_lora_fp8_op.py`.
- **Fully sharded LoRA** — shards both A and B across TP ranks (`LoRAConfig.fully_sharded_loras`).
- **LoRA CUDA-graph specialization** — separate graphs per power-of-two active-adapter count so
  kernel grids shrink when few adapters are live (`vllm/v1/worker/gpu/lora_utils.py`,
  `LoRAConfig.specialize_active_lora`).
- **LoRA kernel metadata** — token→LoRA mapping built once per step, passing `no_lora_flag` /
  `num_active_loras` as CPU tensors so compiled graphs can early-exit
  (`vllm/lora/ops/triton_ops/lora_kernel_metadata.py`).
- **Tuned kernel configs** — per-shape configs loaded from JSON with nearest-key fallback
  (`get_lora_op_configs`, `vllm/lora/ops/triton_ops/README_TUNING.md`).
- **Dummy-LoRA warmup** — installs low-rank (min(`max_lora_rank`, 8)) dummy adapters for profiling and graph capture
  (`vllm/v1/worker/lora_model_runner_mixin.py`).
- **Non-CUDA punica backends** — `vllm/lora/punica_wrapper/` dispatches GPU / CPU / XPU.
- **Mixed 2D/3D MoE LoRA format** — `LoRAConfig.enable_mixed_moe_lora_format`.
- **Target-module restriction** — `LoRAConfig.target_modules`.

## 15. Multi-Modal

- **Encoder output cache** — `EncoderCacheManager` caches encoder embeddings keyed by `mm_hash`,
  ref-counted and shared across requests (`vllm/v1/core/encoder_cache_manager.py`).
- **Multimodal processor cache** — hash-keyed cache of processed inputs (P0 keeps keys, P1 the
  tensors) (`vllm/multimodal/cache.py`, `MultiModalConfig.mm_processor_cache_gb`).
- **Shared-memory object-store cache** — `mm_processor_cache_type="shm"` stores processed tensors in
  a ring shared-memory object store (`vllm/distributed/device_communicators/shm_object_storage.py`).
- **Zero-copy multimodal tensor IPC** — `MultiModalConfig.mm_tensor_ipc="torch_shm"` uses
  `torch.multiprocessing` shared memory instead of msgspec RPC
  (transport: `vllm/v1/engine/tensor_ipc.py`).
- **Frontend GPU multimodal memory pool** — byte-counting semaphore carving GPU memory out of the
  KV cache so API-server-side GPU media decode cannot OOM the engine
  (`vllm/multimodal/gpu_ipc_memory.py`, `MultiModalConfig.mm_ipc_gpu_memory_gb`).
- **Multimodal prefix caching** — block hashes include `(mm_hash, start_offset)` extra keys.
- **Media cache + hasher** — opt-in on-disk cache of downloaded media bounded by size and TTL
  (`vllm/multimodal/media/connector.py`, `VLLM_MEDIA_CACHE`, `VLLM_MEDIA_CACHE_MAX_SIZE_MB`,
  `VLLM_MEDIA_CACHE_TTL_HOURS`); content hashes (blake3 by default) from
  `vllm/multimodal/hasher.py` (`VLLM_MM_HASHER_ALGORITHM`).
- **Parallel media fetching** — shared thread pool, 8 workers by default
  (`VLLM_MEDIA_LOADING_THREAD_COUNT`).
- **Video decoding backends** — opencv / pyav / **TorchCodec** / **PyNvVideoCodec**
  (`vllm/multimodal/video.py`, `VLLM_VIDEO_LOADER_BACKEND`). *New in v0.25.1:*
  `TorchCodecVideoBackendMixin` decodes frames in one batched GIL-releasing `get_frames_at` call
  with NHWC layout (no transpose), tunable `seek_mode` and ffmpeg thread count.
- **Fused Kimi image preprocessing** *(new in v0.25.1)* — numba parallel single-pass
  pad+normalize+patchify for K2.5/K2.6 vision chunks
  (`vllm/transformers_utils/processors/kimi_k25_vision_fused.py`).
- **Efficient Video Sampling (EVS)** — prunes a configurable fraction of video tokens per item with
  retention mask + mrope position recomputation (`vllm/multimodal/evs.py`,
  `MultiModalConfig.video_pruning_rate`).
- **Chunked multimodal encoder scheduling** — a multimodal item may be split across scheduler steps
  under chunked prefill; `SchedulerConfig.disable_chunked_mm_input` forbids it. (The Kimi-specific
  `VisionChunk` types in `vllm/multimodal/inputs.py` are a separate, model-gated mechanism.)
- **Encoder CUDA graphs + encoder compile** — `EncoderCudaGraphManager` captures per-token-budget
  graphs (`vllm/v1/worker/encoder_cudagraph.py`); `CompilationConfig.compile_mm_encoder` and
  `cudagraph_mm_encoder` control compile/capture budgets.
- **Data-parallel multimodal encoder** — `MultiModalConfig.mm_encoder_tp_mode="data"` shards encoder
  work by data instead of weights (`run_dp_sharded_vision_model`).
- **Encoder-only execution** — `MultiModalConfig.mm_encoder_only` skips the language component in
  the worker (intended for a disaggregated encoder process).
- **Skip multimodal profiling** — `MultiModalConfig.skip_mm_profiling` cuts startup time, shifting
  peak-memory estimation to the operator.
- **Multimodal registry** — `vllm/multimodal/registry.py`.

## 16. Distributed / Disaggregated Serving

- **Disaggregated prefill / decode** — prefill and decode instances exchange KV blocks through the
  connector layer (`vllm/distributed/kv_transfer/`, `KVTransferConfig.kv_role`).
- **Connectors** — see §10. NIXL adds heterogeneous TP mapping between prefill and decode
  (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`); `MultiConnector` composes
  several; `OffloadingConnector` drives the
  `vllm/v1/kv_offload/` backends.
- **P2P KV tiering** — ZMQ control plane + NIXL data plane
  (`vllm/v1/kv_offload/tiering/p2p`), successor to the removed P2P NCCL connector.
- **SupportsHMA** — marks a connector safe under hybrid (multi-group) KV layouts; required for P/D
  on hybrid attention models.
- **Mamba/SSM conv-state transfer** — splits SSM/conv state into transferable blocks so hybrid
  models can be disaggregated, not just attention KV
  (`vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py`).
- **KV load failure policy** — `KVTransferConfig.kv_load_failure_policy` (`recompute` | `fail`).
- **KV connector stats / metrics** —
  `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py`.
- **KV events lifecycle tracking** — `vllm/distributed/kv_events.py`,
  `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`.
- **Stateless group coordinator** — builds CPU/device/TCPStore groups independent of torch's WORLD
  group, so new groups form without tearing down existing ones
  (`vllm/distributed/stateless_coordinator.py`).
- **Elastic EP** — see §5.

## 17. ROCm-Specific (AITER)

Master switch `VLLM_ROCM_USE_AITER`; op surface enumerated in `vllm/_aiter_ops.py` and
`vllm/kernels/aiter_ops.py`.

- **AITER paged attention** — `paged_attention_common` via
  `vllm/v1/attention/backends/rocm_aiter_fa.py` (`VLLM_ROCM_USE_AITER_MHA` selects the backend; `VLLM_ROCM_USE_AITER_PAGED_ATTN` is declared but unread).
- **AITER MHA / Unified attention** — `VLLM_ROCM_USE_AITER_MHA`,
  `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`.
- **AITER MLA + sparse MLA** — `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`,
  `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`,
  `vllm/v1/attention/backends/mla/aiter_triton_mla.py` (`VLLM_ROCM_USE_AITER_MLA`). v0.25.1 fixes a
  memory-access fault for DPA + FP8 KV.
- **AITER Triton RoPE** — swapped into `RotaryEmbedding.forward`
  (`VLLM_ROCM_USE_AITER_TRITON_ROPE`).
- **AITER RMSNorm** — registered as IR op impls; ROCm prioritises `"aiter"` for `rms_norm` under
  cudagraphs (`VLLM_ROCM_USE_AITER_RMSNORM`).
- **AITER linear / Triton GEMM / hipBLASLt** —
  `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` (`VLLM_ROCM_USE_AITER_LINEAR`,
  `VLLM_ROCM_USE_AITER_TRITON_GEMM` for the unquantized GEMM in `vllm/model_executor/layers/utils.py`, `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM` for the autotuned
  `hipb_mm` path).
- **AITER MoE + dispatch policy** —
  `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` plus MXFP4-W4A8 and MXFP8
  variants (`VLLM_ROCM_USE_AITER_MOE`, `VLLM_ROCM_AITER_MOE_DISPATCH_POLICY`).
- **AITER fused topk / routing** — `topk_softmax` / `topk_sigmoid` / grouped and biased-grouped
  topk in `vllm/_aiter_ops.py`.
- **AITER FP4 ASM GEMM** — selected by the Quark OCP-MX scheme
  (`VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`).
- **AITER FP8 / FP4 BMM** — MLA up-projection BMMs (`VLLM_ROCM_USE_AITER_FP8BMM`,
  `VLLM_ROCM_USE_AITER_FP4BMM`).
- **AITER shared-experts fusion** — see §6.
- **AITER GDN Triton kernels** — fused reshape + `causal_conv1d` update and chunked GDN for
  Qwen3-Next (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`).
- **AITER sparse attention indexer** — `vllm::rocm_aiter_sparse_attn_indexer`, listed in `CompilationConfig._attention_ops` so it becomes a default splitting op — not excluded from
  splitting/functionalization in the compile config.
- **AITER MHC kernels** — `mhc_pre_aiter` / `mhc_post_aiter`
  (`vllm/model_executor/kernels/mhc/aiter.py`).
- **Quick-reduce quantization for all-reduce** — see §10.
- **Skinny GEMM** — LLMM1 / wvSplitK / wvSplitKrc / wvSplitKQ for small-M linear layers
  (`csrc/rocm/skinny_gemms.cu`, `VLLM_ROCM_USE_SKINNY_GEMM`).
- **RDNA3 W4A16 GPTQ GEMM** — scalar, WMMA and MoE variants adapted from exllamav2 for gfx1100
  (`csrc/rocm/q_gemm_rdna3.cu`, `csrc/rocm/q_gemm_rdna3_wmma.cu`, `csrc/rocm/moe_q_gemm_rdna3.cu`).
- **FP8 padding / MoE padding** — pads FP8 weight and expert buffers by 256 B to space them apart in memory, not to dodge LDS bank
  conflicts (`VLLM_ROCM_FP8_PADDING`, `VLLM_ROCM_MOE_PADDING`).
- **KV cache shuffle layout** — `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`.

## 18. CPU & Alternative Hardware

- **CPU attention** — `csrc/cpu/cpu_attn.cpp` with per-ISA kernel headers (AMX, NEON, RVV, VSX,
  VXE) and split-KV decode; MLA decode in `csrc/cpu/mla_decode.cpp`.
- **CPU micro-GEMM tiles** — a tile abstraction with AMX / NEON / RVV / generic backends
  (`csrc/cpu/micro_gemm/`).
- **CPU SGL kernel suite** — `csrc/cpu/sgl-kernels/`: `gemm_fp8.cpp`, `gemm_int4.cpp`,
  `moe_fp8.cpp`, `moe_int4.cpp`, `conv.cpp`, `fla.cpp` (`VLLM_CPU_SGL_KERNEL`).
- **CPU oneDNN / ACL GEMM** — `csrc/cpu/dnnl_kernels.cpp`.
- **ZenDNN (zentorch) weight prepacking** — load-time prepack for AMD CPUs
  (`vllm/model_executor/kernels/linear/mixed_precision/zentorch.py`).
- **CPU WNA16 + dynamic 4-bit INT MoE** — `csrc/cpu/cpu_wna16.cpp`,
  `csrc/moe/dynamic_4bit_int_moe_cpu.cpp`.
- **CPU MoE** — `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py` and
  `experts/cpu_moe.py` (bf16 / FP8-W8A16 / INT4), with weight prepacking. *Note:
  `VLLM_CPU_MOE_PREPACK` appears only inside `compile_factors()` — it is **not** a runtime env var
  at v0.25.1.*
- **CPU LUT-based bf16 activation** — `csrc/cpu/activation_lut_bf16.cpp`.
- **CPU shared-memory collectives** — `csrc/cpu/shm.cpp`.
- **CPU Mamba / GDN / ShortConv** — `vllm/model_executor/layers/mamba/ops/cpu/`
  (`causal_conv1d`, gated delta-net); wired into `ShortConv.forward_native` in v0.25.1.
- **CPU speculative-decoding kernels** — `csrc/cpu/spec_decode_utils.cpp`.
- **CPU sizing / pinning** — `VLLM_CPU_KVCACHE_SPACE`, OpenMP thread pinning and core reservation.
- **MWAITX spin-loop extension** — `csrc/spinloop.cpp`, `VLLM_USE_SPINLOOP_EXT`.
- **XPU** — `vllm/v1/worker/xpu_model_runner.py`, `vllm/_xpu_ops.py`, XPU graph capture, XPU sleep
  allocator, XPU punica wrapper, XPU MLA sparse backend, XPU linear kernels.

## 19. Frontend & Serving Path

- **Rust frontend** — a native API server, tokenizer and engine-core client replacing the Python
  hot path (`rust/src/`: `server`, `llm`, `engine-core-client`, `tokenizer`, `parser`, `metrics`).
  v0.25.1 avoids extra copies for multimodal tensors, caches metric handles for scheduler/request
  stats, and stamps `arrival_time` at frontend entry.
- **API-server scale-out** — `--api-server-count` runs multiple front-end processes
  (`vllm/entrypoints/cli/serve.py`); mutually exclusive with the Rust frontend and multi-port external LB.
- **External / hybrid DP load balancing** — `ParallelConfig.data_parallel_external_lb` and
  `data_parallel_hybrid_lb`.
- **Incremental detokenization** — `FastIncrementalDetokenizer` with a slow fallback
  (`vllm/v1/engine/detokenizer.py`), on the per-token CPU hot path of every request.
- **fastokens BPE tokenizer backend** — faster tokenizer path selectable at the tokenizer layer.
- **msgpack zero-copy tensor (de)serialization** — above a size threshold tensors bypass copying in
  the engine RPC path.
- **Pooling / late-interaction runners** — `vllm/v1/worker/gpu/pool/` (`pooling_runner.py`,
  `late_interaction_runner.py`) with `vllm/model_executor/layers/pooler/` for embedding and
  reranker serving.
- **Per-request timing metrics** *(new in v0.25.1)* — opt-in `metrics` field on Chat/Completions
  responses (`--enable-per-request-metrics`); conflicts with `--disable-log-stats`.

## 20. Model Loading & Startup

- **Safetensors load strategies + prefetch** — `LoadConfig.safetensors_load_strategy`,
  `safetensors_prefetch_num_threads`, `safetensors_prefetch_block_size`.
- **fastsafetensors pipelined loading** — `VLLM_FASTSAFETENSORS_QUEUE_SIZE`.
- **`instanttensor` load format** — `LoadConfig.load_format`.
- **Parallel loading workers** — `ParallelConfig.max_parallel_loading_workers`.
- **EP weight filtering** — `ParallelConfig.enable_ep_weight_filter` skips loading experts a rank
  does not own.
- **Startup kernel warmup / autotune** — see §4.

## 21. Miscellaneous / Observability

- **Host-device transfer overlap** — sampled/pooled outputs copied D2H `non_blocking` on a
  dedicated copy stream, joined with blocking events (`vllm/v1/worker/gpu/async_utils.py`).
- **Attention metadata caching across KV groups** — within a step, metadata builds are cached per
  `(KVCacheSpec, builder)` across hybrid KV groups and only the block table is updated
  (`vllm/v1/worker/gpu_model_runner.py`). Distinct from the `token_to_req_indices` cache in §1.
- **V2 GPU model runner** — the newer runner tree (`vllm/v1/worker/gpu/`) gated by
  `VLLM_USE_V2_MODEL_RUNNER`, hosting the Triton sampler stack, pooling runners, spec-decode
  speculators and LoRA graph specialization referenced above.
- **Triton custom kernels** — throughout attention, cache, quant, FLA/Mamba and MoE; wrapper layer
  in `vllm/triton_utils`. `VLLM_TRITON_FORCE_FIRST_CONFIG`, `VLLM_TRITON_ATTN_USE_TD`.
- **Profiling scopes** — layerwise NVTX tracing hooks (`vllm/utils/nvtx_pytorch_hooks.py`,
  `ObservabilityConfig.enable_layerwise_nvtx_tracing`) plus opt-in engine-loop scopes
  (`VLLM_NVTX_SCOPES_FOR_PROFILING`); layerwise profiler in `vllm/profiler`.
- **MFU debug metrics** — `ObservabilityConfig.enable_mfu_metrics`, `VLLM_DEBUG_MFU_METRICS`.
- **GPU↔CPU sync detector** — flags unintended device-host syncs in decorated hot paths after
  warmup (`vllm/utils/gpu_sync_debug.py`, `VLLM_GPU_SYNC_CHECK=warn|error`).
- **Post-warmup JIT compile monitor** — reports Triton/CuteDSL JIT compiles that happen after
  warmup, i.e. steady-state stalls (`vllm/utils/jit_monitor.py`); also forces the numba workqueue
  threading layer so a forked EngineCore doesn't abort under GNU OpenMP.
- **Dry-run / dummy forward** — `GPUModelRunner.profile_run()` / `_dummy_run()` run synthetic
  max-length batches (including the MM encoder) to size activation peak and KV budget.
- **Batch-size logging** — `VLLM_LOG_BATCHSIZE_INTERVAL`.
