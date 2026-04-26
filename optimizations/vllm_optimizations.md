# vLLM LLM / Transformer Optimizations

A catalog of optimization techniques shipped in the vLLM codebase, grouped by category. Most knobs are toggleable via env vars in `vllm/envs.py` or fields on the `vllm/config/` dataclasses (`CompilationConfig`, `SchedulerConfig`, `ParallelConfig`, `CacheConfig`, `SpeculativeConfig`, etc.).

## Status table

`Status` column: ✓ means a deep-dive note exists in this directory (linked from the `Optimization` cell). Blank means no note yet.

| Category | Optimization | Status |
|---|---|---|
| Attention | [PagedAttention v1/v2](PagedAttention.md) | ✓ |
| Attention | [FlashAttention v2 / v3](FlashAttention.md) | ✓ |
| Attention | FlashAttention DiffKV | |
| Attention | FlashInfer | |
| Attention | Triton Attention | |
| Attention | Flex Attention | |
| Attention | Tree Attention | |
| Attention | TurboQuant Attention | |
| Attention | MLA (Multi-head Latent Attention) | |
| Attention | MLA Sparse | |
| Attention | Sliding Window Attention | |
| Attention | Chunked Local Attention | |
| Attention | Static Sink Attention | |
| Attention | Cross-Attention | |
| Attention | MM Encoder Attention | |
| Attention | ROCm AITER Attention / Unified | |
| Attention | ROCm standard attention | |
| Attention | CPU Attention | |
| Attention | Vertical Slash Index | |
| Attention | Merge Attn States | |
| Attention | Fused QK-Norm + RoPE | |
| KV Cache | Paged KV cache | |
| KV Cache | Hash-based prefix caching | |
| KV Cache | Chunked prefill | |
| KV Cache | KV cache quantization (FP8 / INT8 / NVFP4 / MXFP4) | |
| KV Cache | NVFP4 KV-cache kernels | |
| KV Cache | Copy-on-write block sharing | |
| KV Cache | CPU KV-cache offload | |
| KV Cache | KV-cache coordinator | |
| KV Cache | Single-type KV manager | |
| KV Cache | Encoder cache manager | |
| KV Cache | KV events | |
| KV Cache | Hybrid Memory Allocator (HMA) | |
| KV Cache | KV cache layout selection (NHD/HND, ROCm shuffle) | |
| Quantization | FP8 (E4M3 / E5M2) | |
| Quantization | FBGEMM FP8 | |
| Quantization | W8A8 | |
| Quantization | AWQ + AWQ-Marlin | |
| Quantization | GPTQ + GPTQ-Marlin | |
| Quantization | GGUF | |
| Quantization | Marlin / Machete kernels | |
| Quantization | Compressed-tensors (sparsity + structured) | |
| Quantization | MXFP4 | |
| Quantization | NVFP4 | |
| Quantization | Bitsandbytes | |
| Quantization | TorchAO | |
| Quantization | Quark | |
| Quantization | Humming (MoE PTQ) | |
| Quantization | SmoothQuant | |
| Quantization | Experts INT8 | |
| Quantization | MoE WNA16 | |
| Quantization | Online (dynamic) quantization | |
| Quantization | Input FP8 quant | |
| Quantization | Fused activation + quant | |
| Quantization | Fused layernorm + quant | |
| Parallelism | Tensor Parallel (TP) | |
| Parallelism | Pipeline Parallel (PP) | |
| Parallelism | Expert Parallel (EP) | |
| Parallelism | Data Parallel (DP) | |
| Parallelism | Context Parallel (CP) | |
| Parallelism | Sequence Parallel (IR-level) | |
| Parallelism | Async TP all-reduce | |
| Parallelism | Ulysses-style ring sequence parallel | |
| Parallelism | Elastic Expert Parallel | |
| MoE | Fused MoE kernel | |
| MoE | Grouped GEMM / Batched MoE | |
| MoE | Modular Kernel | |
| MoE | Topk-Softmax fused / Grouped Topk | |
| MoE | Permute / Unpermute ops | |
| MoE | MoE FP8 / INT8 / WNA16 | |
| MoE | Marlin MoE | |
| MoE | FlashInfer Cutlass MoE | |
| MoE | DeepGEMM | |
| MoE | DeepEP / pplx all-to-all | |
| MoE | Shared experts | |
| MoE | AITER shared-experts fusion | |
| Speculative Decoding | Draft-target spec decoding | |
| Speculative Decoding | EAGLE / EAGLE-3 | |
| Speculative Decoding | Medusa | |
| Speculative Decoding | N-gram proposer (CPU + GPU) | |
| Speculative Decoding | Lookahead decoding | |
| Speculative Decoding | Suffix tree decoding | |
| Speculative Decoding | DFlash verification kernel | |
| Speculative Decoding | MTP (multi-token prediction) | |
| Speculative Decoding | Rejection sampler | |
| Speculative Decoding | Draft model warmup | |
| Compilation | torch.compile / Inductor backend | |
| Compilation | CUDA Graphs (full + piecewise) | |
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
| Compilation | Fusion: RoPE + KV-cache update | |
| Compilation | Fusion: Collective + Compute | |
| Compilation | Fusion: ROCm AITER | |
| Compilation | Sequence-parallelism IR pass | |
| Compilation | Lowering pass | |
| Compilation | Utility passes (noop elim, split coalesce, scatter-split) | |
| Compilation | CUDAGraph GC | |
| Scheduling | Continuous batching | |
| Scheduling | Chunked prefill | |
| Scheduling | Prefix-cache-aware scheduling | |
| Scheduling | Priority scheduling (FCFS / priority) | |
| Scheduling | Multi-step scheduling | |
| Scheduling | Async scheduler | |
| Scheduling | Ubatching / micro-batching | |
| Scheduling | Disaggregated prefill / decode | |
| Scheduling | Encoder budget for multimodal | |
| Scheduling | Output batching | |
| Communication | Custom all-reduce | |
| Communication | Quick all-reduce (low-precision) | |
| Communication | NVIDIA Symmetric Memory all-reduce | |
| Communication | FlashInfer all-reduce | |
| Communication | PyNCCL + tuning | |
| Communication | Shared-memory broadcast / object storage | |
| Communication | All-to-all kernels (DeepEP, pplx) | |
| Communication | NIXL connector | |
| Communication | LMCache connector | |
| Communication | MoonCake connector | |
| Communication | MoriIO connector | |
| Communication | P2P NCCL connector | |
| Communication | Simple CPU offload connector | |
| Communication | EC (expert collection) transfer | |
| Sampling | FlashInfer sampling | |
| Sampling | Triton top-k / top-p | |
| Sampling | Fused top-k / top-p sampler | |
| Sampling | Persistent topk kernel | |
| Sampling | Logits processors | |
| Sampling | Frequency / repetition penalties (Triton) | |
| Sampling | Bad-words masking | |
| Sampling | Structured output: xGrammar (DFA cache) | |
| Sampling | Structured output: Outlines | |
| Sampling | Structured output: Guidance | |
| Sampling | Rejection sampling | |
| Sampling | Logprobs caching | |
| Memory & Runtime | Block pool | |
| Memory & Runtime | Custom cumem allocator | |
| Memory & Runtime | Pinned host memory | |
| Memory & Runtime | Prefix-cache eviction (LRU) | |
| Memory & Runtime | UVA oversubscription | |
| Memory & Runtime | Weight offloading | |
| Memory & Runtime | Prefetch manager | |
| Memory & Runtime | Workspace manager | |
| Memory & Runtime | Memory profiler (CUDA-graph estimation) | |
| Model-specific Kernels | Fused RMSNorm + residual | |
| Model-specific Kernels | Fused activations (SiLU / GELU / Tanh) | |
| Model-specific Kernels | Minimax reduce-RMS kernel | |
| Model-specific Kernels | MLA latent attention | |
| Model-specific Kernels | Mamba / Mamba2 SSM kernels | |
| Model-specific Kernels | Linear attention (Mamba2-style) | |
| Model-specific Kernels | GDN linear attention | |
| Model-specific Kernels | Short-conv (Mamba local context) | |
| Model-specific Kernels | FLA packed recurrent decode | |
| Model-specific Kernels | KDA (kernel-dimension awareness) | |
| Model-specific Kernels | Batch-invariant mode (deterministic, SM90+) | |
| Model-specific Kernels | RoPE: linear scaling | |
| Model-specific Kernels | RoPE: NTK | |
| Model-specific Kernels | RoPE: YaRN | |
| Model-specific Kernels | RoPE: Llama3 | |
| Model-specific Kernels | RoPE: Llama4-vision | |
| Model-specific Kernels | RoPE: DeepSeek scaling | |
| Model-specific Kernels | RoPE: MRoPE | |
| Model-specific Kernels | RoPE: MRoPE interleaved | |
| Model-specific Kernels | RoPE: FoPE | |
| Model-specific Kernels | RoPE: XDRoPE | |
| Model-specific Kernels | RoPE: Phi3-long | |
| LoRA | Punica kernels | |
| LoRA | Multi-LoRA batching | |
| LoRA | LoRA dual-stream execution | |
| LoRA | PDL projection optimization | |
| LoRA | LoRA model manager | |
| LoRA | LoRA weight resolver | |
| Multi-Modal | Encoder output cache | |
| Multi-Modal | Image-embedding hash cache | |
| Multi-Modal | Video chunking | |
| Multi-Modal | Multimodal prefix caching | |
| Multi-Modal | Media cache | |
| Multi-Modal | Media hasher | |
| Multi-Modal | Multimodal registry | |
| Distributed / Disaggregated | Disaggregated prefill / decode (KV transfer) | |
| Distributed / Disaggregated | KV-transfer connectors (umbrella) | |
| Distributed / Disaggregated | KV events lifecycle tracking | |
| Distributed / Disaggregated | Stateless coordinator | |
| Distributed / Disaggregated | Elastic EP (dynamic expert scaling) | |
| ROCm (AITER) | AITER paged attention | |
| ROCm (AITER) | AITER MHA / Unified attention | |
| ROCm (AITER) | AITER Triton RoPE | |
| ROCm (AITER) | AITER RMSNorm | |
| ROCm (AITER) | AITER linear / Triton GEMM | |
| ROCm (AITER) | AITER MoE | |
| ROCm (AITER) | AITER FP4 ASM GEMM | |
| ROCm (AITER) | AITER FP8 / FP4 BMM | |
| ROCm (AITER) | AITER shared-experts fusion | |
| ROCm (AITER) | Quick-reduce quantization for all-reduce | |
| ROCm (AITER) | Skinny GEMM | |
| ROCm (AITER) | FP8 padding / MoE padding | |
| ROCm (AITER) | KV cache shuffle layout | |
| Misc | OINK fused custom ops | |
| Misc | Weight streaming between phases | |
| Misc | Host-device transfer overlap | |
| Misc | Custom AR for small messages | |
| Misc | Attention metadata caching across steps | |
| Misc | Triton custom kernels (broadly) | |
| Misc | Profiling scopes (NVTX, custom) | |
| Misc | MFU debug metrics | |
| Misc | Dry-run / dummy forward for memory profiling | |

---

## Detailed catalog

The detailed, path-annotated catalog organized by category follows below.

## 1. Attention Algorithms & Kernels

- **PagedAttention v1/v2** — Block-based KV cache access (`csrc/attention/paged_attention_v{1,2}.cu`)
- **FlashAttention v2 / v3** — IO-aware tiled attention (`vllm/v1/attention/backends/flash_attn.py`)
- **FlashAttention DiffKV** — Heterogeneous QK/V dimension support
- **FlashInfer** — Custom decode/prefill kernels (`backends/flashinfer.py`)
- **Triton Attention** — Triton-based flexible backend
- **Flex Attention** — Torch-native flexible masking via `torch.compile`
- **Tree Attention** — For tree-structured spec decoding sequences
- **TurboQuant Attention** — Reduced-precision attention
- **MLA (Multi-head Latent Attention)** — DeepSeek's compressed-KV attention with FlashInfer/Triton/Cutlass/FlashMLA/FlashAttn variants (`backends/mla/`)
- **MLA Sparse** — Sparse-indexed MLA
- **Sliding Window Attention** — Local-window (Mistral-style)
- **Chunked Local Attention** — Local-chunk attention
- **Static Sink Attention** — Persistent first-token sinks
- **Cross-Attention** — Encoder-decoder
- **MM Encoder Attention** — For multimodal encoders
- **ROCm AITER Attention** / **AITER Unified** / **ROCm standard** — AMD-tuned backends
- **CPU Attention** — CPU fallback
- **Vertical Slash Index** — Optimized metadata indexing
- **Merge Attn States** — Fused merging of partial attention outputs
- **Fused QK-Norm + RoPE** (`csrc/fused_qknorm_rope_kernel.cu`)

## 2. KV Cache Management

- **Paged KV cache** with logical→physical block mapping
- **Hash-based prefix caching** with cross-request block reuse
- **Chunked prefill** (interleave prefill/decode)
- **KV cache quantization** (FP8, INT8, NVFP4, MXFP4)
- **NVFP4 KV-cache kernels** (`csrc/nvfp4_kv_cache_kernels.cu`)
- **Copy-on-write block sharing**
- **CPU KV-cache offload** (`vllm/v1/kv_offload/`)
- **KV-cache coordinator** for distributed allocation
- **Single-type KV manager** (homogeneous-model fast path)
- **Encoder cache manager** (multimodal embedding reuse)
- **KV events** for disaggregated coordination
- **Hybrid Memory Allocator (HMA)** — CPU/GPU tiered
- **KV cache layout selection** (NHD vs HND, ROCm shuffle)

## 3. Quantization

- **FP8 (E4M3/E5M2)** weights/activations
- **FBGEMM FP8**
- **W8A8** (per-token / per-channel / per-tensor / per-group)
- **AWQ** + **AWQ-Marlin**
- **GPTQ** + **GPTQ-Marlin**
- **GGUF** multi-bit
- **Marlin / Machete** CUTLASS kernels
- **Compressed-tensors** (sparsity + structured)
- **MXFP4** / **NVFP4** 4-bit float
- **Bitsandbytes**
- **TorchAO**
- **Quark**
- **Humming** (MoE PTQ)
- **SmoothQuant**
- **Experts INT8** / **MoE WNA16**
- **Online (dynamic) quantization**
- **Input FP8 quant** (activation quant in attention)
- **Fused activation+quant** (`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`)
- **Fused layernorm+quant** (per-token dynamic)
- **KV cache FP8 / INT8 / FP4**

## 4. Parallelism

- **Tensor Parallel (TP)**
- **Pipeline Parallel (PP)**
- **Expert Parallel (EP)**
- **Data Parallel (DP)**
- **Context Parallel (CP)** — sequence sharding
- **Sequence Parallel** — IR-level (`compilation/passes/fusion/sequence_parallelism.py`)
- **Async TP all-reduce**
- **Ulysses-style ring sequence parallel**
- **Elastic Expert Parallel** — dynamic expert rebalancing

## 5. MoE Optimizations

- **Fused MoE kernel** (router + GEMM)
- **Grouped GEMM** / **Batched MoE**
- **Modular Kernel** (pluggable expert + topk)
- **Topk-Softmax fused** / **Grouped Topk**
- **Permute/Unpermute** ops
- **MoE FP8 / INT8 / WNA16**
- **Marlin MoE** (quantized)
- **FlashInfer Cutlass MoE**
- **DeepGEMM** (`VLLM_USE_DEEP_GEMM`)
- **DeepEP / pplx all-to-all** for experts
- **Shared experts** (parallel with routed)
- **AITER shared-experts fusion** (ROCm)

## 6. Speculative Decoding

- **Draft-target spec decoding**
- **EAGLE / EAGLE-3** with custom attention metadata
- **Medusa** multi-head drafting
- **N-gram proposer** (CPU + GPU variants)
- **Lookahead decoding**
- **Suffix tree decoding**
- **DFlash** verification kernel
- **MTP** (multi-token prediction)
- **Rejection sampler** for verification
- **Draft model warmup** for CUDA graph stability

## 7. Compilation & Graph Optimizations

- **torch.compile / Inductor backend**
- **CUDA Graphs** (full + piecewise)
- **AOT compile** + **mega AOT artifact**
- **Custom Inductor passes**
- **Compile cache** (avoid recompilation)
- **Inductor max-autotune** + **coordinate-descent tuning**
- **Fusion passes**:
  - RMSNorm + Quant
  - All-Reduce + RMSNorm
  - Attention + Quant
  - QK-Norm + RoPE
  - Activation + Quant
  - MLA Attention + Quant
  - RoPE + KV-cache update
  - Collective + Compute fusion
  - ROCm AITER fusion
- **Sequence-parallelism IR pass**
- **Lowering pass** (high→low IR)
- **Utility passes** (noop elim, split coalesce, scatter-split)
- **CUDAGraph GC**

## 8. Scheduling & Batching

- **Continuous batching**
- **Chunked prefill**
- **Prefix-cache-aware scheduling**
- **Priority scheduling** (FCFS / priority)
- **Multi-step scheduling**
- **Async scheduler**
- **Ubatching / micro-batching** (`vllm/v1/worker/ubatching.py`)
- **Disaggregated prefill/decode**
- **Encoder budget** for multimodal
- **Output batching**

## 9. Communication

- **Custom all-reduce** (`csrc/custom_all_reduce.cu`)
- **Quick all-reduce** (low-precision)
- **NVIDIA Symmetric Memory all-reduce**
- **FlashInfer all-reduce**
- **PyNCCL** + tuning (toggleable)
- **Shared-memory broadcast / object storage**
- **All-to-all kernels** (DeepEP, pplx)
- **NIXL connector** (network KV transfer)
- **LMCache** / **MoonCake** / **MoriIO** connectors
- **P2P NCCL connector**
- **Simple CPU offload connector**
- **EC (expert collection) transfer** for distributed MoE

## 10. Sampling & Output

- **FlashInfer sampling**
- **Triton top-k / top-p**
- **Fused top-k/top-p sampler**
- **Persistent topk** kernel (`csrc/persistent_topk.cuh`)
- **Logits processors**
- **Frequency / repetition penalties** (Triton)
- **Bad-words masking**
- **Structured output**: xGrammar (with DFA cache), Outlines, Guidance
- **Rejection sampling**
- **Logprobs caching**

## 11. Memory & Runtime

- **Block pool** (pre-allocated blocks)
- **Custom cumem allocator** (`csrc/cumem_allocator.cpp`)
- **Pinned host memory**
- **Prefix-cache eviction** (LRU)
- **UVA oversubscription**
- **Weight offloading** + **prefetch manager**
- **Workspace manager** for kernel scratch
- **Memory profiler** (CUDA-graph estimation)

## 12. Model-Architecture-Specific Kernels

- **Fused RMSNorm + residual**
- **Fused activations** (SiLU, GELU, Tanh)
- **Minimax reduce-RMS kernel**
- **MLA latent attention** (multiple variants)
- **Mamba/Mamba2 SSM kernels** (`csrc/mamba/mamba_ssm/`)
- **Linear attention** (Mamba2-style)
- **GDN linear attention**
- **Short-conv** (Mamba local context)
- **FLA packed recurrent decode**
- **KDA** (kernel-dimension awareness)
- **Batch-invariant mode** (deterministic outputs, SM90+)
- **RoPE variants**: linear, NTK, YaRN, Llama3, Llama4-vision, DeepSeek scaling, MRoPE, MRoPE interleaved, FoPE, XDRoPE, Phi3-long

## 13. LoRA / Multi-LoRA

- **Punica kernels** for batched LoRA projection
- **Multi-LoRA batching** in scheduler
- **LoRA dual-stream execution** (`VLLM_LORA_ENABLE_DUAL_STREAM`)
- **PDL projection optimization**
- **LoRA model manager** (lifecycle + warmup)
- **LoRA weight resolver** (multi-source loading)

## 14. Multi-Modal

- **Encoder output cache** (vision/audio embeddings)
- **Image-embedding hash cache**
- **Video chunking**
- **Multimodal prefix caching**
- **Media cache** (`VLLM_MEDIA_CACHE`)
- **Media hasher** for cache keys
- **Multimodal registry** (extensible processors)

## 15. Distributed / Disaggregated Serving

- **Disaggregated prefill/decode** with KV transfer
- **Connectors**: NIXL, LMCache, MoonCake, MoriIO, P2P-NCCL, simple-CPU offload
- **KV events** lifecycle tracking
- **Stateless coordinator** (no persistent server)
- **Elastic EP** (dynamic expert scaling)

## 16. ROCm-Specific (AITER)

- AITER paged attention, MHA, unified attention, Triton RoPE, RMSNorm, linear, MoE, FP4 ASM GEMM, FP8/FP4 BMM, Triton GEMM, shared-experts fusion
- **Quick-reduce quantization** for all-reduce
- **Skinny GEMM**
- **FP8 padding** / **MoE padding**
- **KV cache shuffle layout**

## 17. Miscellaneous

- **OINK fused custom ops** (`VLLM_USE_OINK_OPS`)
- **Weight streaming** between phases
- **Host-device transfer overlap**
- **Custom AR for small messages**
- **Attention metadata caching** across steps
- **Triton custom kernels** throughout
- **Profiling scopes** (NVTX, custom)
- **MFU debug metrics**
- **Dry-run / dummy forward** for memory profiling
