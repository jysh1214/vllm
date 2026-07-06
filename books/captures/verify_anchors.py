#!/usr/bin/env python
"""Verify notes' file:line anchors against current HEAD; report actual lines."""
import re
import sys
from pathlib import Path

ROOT = Path("/home/alex/vllm")

# (file, cited_line_or_None, regex, label)
CHECKS = [
    # scheduler.py
    ("vllm/v1/core/sched/scheduler.py", 349, r"no .?prefill phase|no .?decoding phase|There's no", "unified-model comment"),
    ("vllm/v1/core/sched/scheduler.py", 348, r"def schedule\(", "schedule()"),
    ("vllm/v1/core/sched/scheduler.py", 961, r"def _preempt_request", "_preempt_request"),
    ("vllm/v1/core/sched/scheduler.py", 983, r"def _update_after_schedule", "_update_after_schedule"),
    ("vllm/v1/core/sched/scheduler.py", 1299, r"def update_from_output", "update_from_output"),
    ("vllm/v1/core/sched/scheduler.py", 157, r"self\.requests[:\s]", "self.requests dict"),
    # core.py
    ("vllm/v1/engine/core.py", 402, r"def step\(", "EngineCore.step"),
    ("vllm/v1/engine/core.py", 443, r"def step_with_batch_queue", "step_with_batch_queue"),
    # request / output
    ("vllm/v1/request.py", 299, r"class RequestStatus", "RequestStatus"),
    ("vllm/v1/core/sched/output.py", 178, r"class SchedulerOutput", "SchedulerOutput"),
    ("vllm/v1/core/sched/request_queue.py", None, r"class RequestQueue|SchedulingPolicy", "request_queue"),
    # kv_cache_manager.py
    ("vllm/v1/core/kv_cache_manager.py", 176, r"def get_computed_blocks", "get_computed_blocks"),
    ("vllm/v1/core/kv_cache_manager.py", 218, r"def can_fit_full_sequence", "can_fit_full_sequence"),
    ("vllm/v1/core/kv_cache_manager.py", 257, r"def allocate_slots", "allocate_slots"),
    ("vllm/v1/core/kv_cache_manager.py", 429, r"def free\(", "free"),
    # block_pool.py
    ("vllm/v1/core/block_pool.py", 34, r"class BlockHashToBlockMap", "BlockHashToBlockMap"),
    ("vllm/v1/core/block_pool.py", 130, r"class BlockPool", "BlockPool"),
    ("vllm/v1/core/block_pool.py", 176, r"null_block", "null block"),
    ("vllm/v1/core/block_pool.py", 211, r"def cache_full_blocks", "cache_full_blocks"),
    ("vllm/v1/core/block_pool.py", 322, r"def get_new_blocks", "get_new_blocks"),
    ("vllm/v1/core/block_pool.py", 391, r"def touch", "touch"),
    ("vllm/v1/core/block_pool.py", 408, r"def free_blocks", "free_blocks"),
    # kv_cache_utils.py
    ("vllm/v1/core/kv_cache_utils.py", 87, r"NONE_HASH", "NONE_HASH"),
    ("vllm/v1/core/kv_cache_utils.py", 110, r"class KVCacheBlock", "KVCacheBlock"),
    ("vllm/v1/core/kv_cache_utils.py", 158, r"class FreeKVCacheBlockQueue", "FreeKVCacheBlockQueue"),
    ("vllm/v1/core/kv_cache_utils.py", 497, r"def generate_block_hash_extra_keys", "extra_keys"),
    ("vllm/v1/core/kv_cache_utils.py", 535, r"def hash_block_tokens", "hash_block_tokens"),
    ("vllm/v1/core/kv_cache_coordinator.py", 28, r"class KVCacheCoordinator", "KVCacheCoordinator"),
    ("vllm/v1/kv_cache_interface.py", 80, r"class KVCacheSpec|class AttentionSpec|class FullAttentionSpec", "KVCacheSpec"),
    # distributed
    ("vllm/distributed/parallel_state.py", 290, r"class GroupCoordinator", "GroupCoordinator"),
    ("vllm/distributed/parallel_state.py", 1486, r"def initialize_model_parallel", "initialize_model_parallel"),
    ("vllm/distributed/parallel_state.py", 1221, r"def get_tp_group", "get_tp_group"),
    ("vllm/distributed/device_communicators/base_device_communicator.py", 118, r"class DeviceCommunicatorBase", "DeviceCommunicatorBase"),
    ("vllm/distributed/device_communicators/base_device_communicator.py", 344, r"def dispatch", "dispatch"),
    ("vllm/distributed/device_communicators/base_device_communicator.py", 363, r"def combine", "combine"),
    ("vllm/distributed/device_communicators/all2all.py", 41, r"class AgRsAll2AllManager", "AgRs"),
    ("vllm/distributed/device_communicators/all2all.py", 197, r"class DeepEPHTAll2AllManager", "DeepEP-HT"),
    ("vllm/distributed/device_communicators/all2all.py", 261, r"class DeepEPLLAll2AllManager", "DeepEP-LL"),
    ("vllm/model_executor/layers/fused_moe/layer.py", 71, r"def determine_expert_map", "determine_expert_map"),
    ("vllm/model_executor/layers/fused_moe/layer.py", 219, r"class FusedMoE", "FusedMoE"),
    ("vllm/model_executor/layers/linear.py", 410, r"class ColumnParallelLinear", "ColumnParallelLinear"),
    ("vllm/model_executor/layers/linear.py", 609, r"class MergedColumnParallelLinear", "MergedColumnParallelLinear"),
    ("vllm/model_executor/layers/linear.py", 977, r"class QKVParallelLinear", "QKVParallelLinear"),
    ("vllm/model_executor/layers/linear.py", 1389, r"class RowParallelLinear", "RowParallelLinear"),
    ("vllm/model_executor/layers/linear.py", 214, r"def process_weights_after_loading", "process_weights_after_loading"),
    ("vllm/model_executor/layers/linear.py", 220, r"def apply", "UnquantizedLinearMethod.apply"),
    ("vllm/model_executor/layers/vocab_parallel_embedding.py", 192, r"class VocabParallelEmbedding", "VocabParallelEmbedding"),
    ("vllm/compilation/passes/fusion/sequence_parallelism.py", 44, r"def get_sequence_parallelism_threshold|_sequence_parallelism_threshold", "SP threshold"),
    ("vllm/compilation/passes/fusion/sequence_parallelism.py", 133, r"class SequenceParallelismPass|reduce_scatter", "SP pass"),
    ("vllm/config/compilation.py", 129, r"enable_sp", "enable_sp"),
    ("vllm/config/parallel.py", 293, r"def world_size|world_size", "world_size"),
    ("vllm/config/parallel.py", 611, r"use_sequence_parallel_moe", "use_sequence_parallel_moe"),
    # cpu gemm chain
    ("vllm/model_executor/layers/utils.py", 213, r"def check_cpu_sgl_kernel", "check_cpu_sgl_kernel"),
    ("vllm/model_executor/layers/utils.py", 222, r"def dispatch_cpu_unquantized_gemm", "dispatch_cpu_unquantized_gemm"),
    ("vllm/model_executor/layers/utils.py", 293, r"def cpu_unquantized_gemm", "cpu_unquantized_gemm"),
    ("vllm/model_executor/layers/utils.py", 302, r"def dispatch_unquantized_gemm", "dispatch_unquantized_gemm"),
    ("vllm/_custom_ops.py", 3220, r"def create_onednn_mm", "create_onednn_mm"),
    ("vllm/_custom_ops.py", 3234, r"def onednn_mm", "onednn_mm"),
    ("csrc/cpu/torch_bindings.cpp", 201, r"TORCH_LIBRARY_EXPAND", "TORCH_LIBRARY_EXPAND"),
    ("csrc/cpu/torch_bindings.cpp", 284, r'impl\("onednn_mm', "onednn_mm impl"),
    ("csrc/cpu/torch_bindings.cpp", 346, r'impl\("weight_packed_linear', "weight_packed_linear impl"),
    ("csrc/cpu/dnnl_kernels.cpp", 497, r"create_onednn_mm_handler", "create handler"),
    ("csrc/cpu/dnnl_kernels.cpp", 519, r"void onednn_mm|onednn_mm\(", "onednn_mm fn"),
    ("csrc/cpu/cpu_types.hpp", 16, r"__riscv_v", "riscv_v gate"),
    ("vllm/compilation/backends.py", 797, r"class VllmBackend", "VllmBackend"),
    ("vllm/compilation/decorators.py", 115, r"def support_torch_compile", "support_torch_compile"),
    ("vllm/compilation/wrapper.py", 96, r"init_backend", "init_backend"),
    ("vllm/compilation/wrapper.py", 164, r"torch\.compile", "torch.compile call"),
    ("vllm/platforms/cpu.py", 170, r"DYNAMO_TRACE_ONCE", "cpu forces trace_once"),
    ("cmake/cpu_extension.cmake", 294, r"oneDNN\.git", "oneDNN fetch"),
    ("cmake/cpu_extension.cmake", 280, r"FETCHCONTENT_SOURCE_DIR_ONEDNN", "oneDNN source dir hook"),
    # existence-only checks
    ("csrc/cpu/micro_gemm/cpu_micro_gemm_vec.hpp", None, r"MicroGemm", "MicroGemm vec"),
    ("csrc/cpu/sgl-kernels/gemm.cpp", None, r"weight_packed_linear", "SGL gemm"),
    ("vllm/v1/spec_decode/eagle.py", None, r"class Eagle|Eagle", "eagle"),
    ("vllm/v1/spec_decode/medusa.py", None, r"Medusa", "medusa"),
    ("vllm/v1/spec_decode/ngram_proposer.py", None, r"[Nn]gram", "ngram"),
    ("vllm/v1/spec_decode/suffix_decoding.py", None, r"[Ss]uffix", "suffix"),
    ("vllm/v1/sample/rejection_sampler.py", None, r"class RejectionSampler", "rejection sampler"),
    ("vllm/lora/punica_wrapper/punica_base.py", None, r"class PunicaWrapper", "punica"),
    ("vllm/lora/model_manager.py", None, r"class LoRAModelManager", "lora manager"),
    ("vllm/lora/resolver.py", None, r"class LoRAResolver|Resolver", "lora resolver"),
    ("vllm/compilation/passes/fusion/rope_kvcache_fusion.py", None, r"RopeKVCacheFusionPass|fused_rope", "rope-kv fusion pass"),
    ("vllm/compilation/passes/fusion/collective_fusion.py", 409, r"class AsyncTPPass|GEMMReduceScatter|fuse_gemm_comms", "async TP"),
    ("vllm/v1/attention/backends/flash_attn.py", None, r"class FlashAttention", "flash_attn backend"),
    ("vllm/v1/attention/backends/flashinfer.py", None, r"[Ff]lash[Ii]nfer", "flashinfer backend"),
    ("vllm/v1/attention/backends/triton_attn.py", None, r"[Tt]riton", "triton backend"),
    ("csrc/attention/paged_attention_v1.cu", None, r"paged_attention", "paged attn v1 kernel"),
    ("csrc/attention/paged_attention_v2.cu", None, r"paged_attention", "paged attn v2 kernel"),
    ("csrc/custom_all_reduce.cu", None, r"all_reduce|allreduce", "custom all-reduce"),
    ("vllm/distributed/device_communicators/pynccl.py", None, r"class PyNccl|Nccl", "pynccl"),
    ("vllm/distributed/device_communicators/shm_broadcast.py", None, r"class MessageQueue", "shm MessageQueue"),
    ("vllm/config/compilation.py", None, r"rope_kvcache_fusion_max_token_num", "rope fusion threshold"),
    ("vllm/config/compilation.py", None, r"fuse_rope_kvcache", "fuse_rope_kvcache flag"),
    ("vllm/v1/attention/backend.py", None, r"fused_rope_kvcache_supported", "fused rope support hook"),
    ("vllm/engine/protocol.py", None, r"class EngineClient", "EngineClient protocol"),
    ("vllm/v1/worker/gpu_worker.py", None, r"class Worker", "gpu worker"),
    ("vllm/v1/attention/selector.py", None, r"backend", "attention selector"),
]

def main():
    drift = ok = missing = 0
    for f, cited, pat, label in CHECKS:
        p = ROOT / f
        if not p.exists():
            print(f"MISSING-FILE  {f}  ({label})")
            missing += 1
            continue
        lines = p.read_text(errors="replace").splitlines()
        rx = re.compile(pat)
        hits = [i + 1 for i, ln in enumerate(lines) if rx.search(ln)]
        if not hits:
            print(f"NO-MATCH      {f}:{cited}  ({label})  /{pat}/")
            missing += 1
        elif cited is None:
            print(f"EXISTS        {f}  ({label})  first hit @ {hits[0]}")
            ok += 1
        else:
            # nearest hit to the cited line
            near = min(hits, key=lambda h: abs(h - cited))
            if abs(near - cited) <= 3:
                print(f"OK            {f}:{cited}  ({label})  actual {near}")
                ok += 1
            else:
                print(f"DRIFT         {f}:{cited} -> {near}  ({label})  all={hits[:6]}")
                drift += 1
    print(f"\nSummary: {ok} ok, {drift} drifted, {missing} missing")

if __name__ == "__main__":
    main()
