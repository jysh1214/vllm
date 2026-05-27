# Fusion: RoPE + KV Cache Update

**Core idea: collapse the two micro-ops at the start of every attention layer — applying rotary embeddings to Q and K, then writing K and V into the paged KV cache — into a single kernel, eliminating one launch and one HBM round-trip per layer per step.**

## The problem it solves

Every transformer attention layer, before computing attention proper, does two small but unavoidable bookkeeping ops:

1. **RoPE** applies rotary position embeddings to `Q` and `K`.
2. **KV cache update** writes the rotated `K` and the unrotated `V` into the paged KV cache at the slots assigned to this batch.

In their standard form these are two separate kernel launches:

```
q, k = rotary_embedding(positions, q, k, cos_sin_cache, is_neox)
unified_kv_cache_update(k, v, layer_name)
```

Each op:

- launches a CUDA / Triton kernel (low single-digit µs of launch overhead)
- reads its inputs from HBM
- writes its outputs back to HBM

In decode, `seq_q` is 1 (or a small `N` for speculative verify). Each of these kernels processes a handful of tokens, so the launch overhead and HBM round-trip dominate the actual arithmetic. Per-layer cost is set by **how often you cross HBM**, not by the math.

For a 70B-class model with 80 attention layers, that is 160 small kernels per decode step just for the rotate-and-cache-write phase. The fusion compresses it to 80.

## What the fusion does

Pattern matched at the FX-graph level inside torch.compile's pass pipeline:

```
# Before
q, k = rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox)
kv_cache_dummy = unified_kv_cache_update(k, v, layer_name)

# After
kv_cache_dummy = fused_rope_and_unified_kv_cache_update(
    q, k, v, positions, cos_sin_cache, is_neox, layer_name
)
```

The fused kernel reads `Q`, `K`, `V`, `positions`, and the precomputed `cos_sin_cache` from HBM once. It applies RoPE to `Q` and `K` in registers / shared memory, then writes:

- the rotated `Q` back in place
- the rotated `K` and unrotated `V` **directly into the paged KV cache slots**

`Q` is mutated in place — the attention call reads it next anyway. `K` and `V` never go through an intermediate HBM tensor; they flow rotation → cache slot in a single sweep.

## Why it is decode-specific

The pass has a config knob:

```python
rope_kvcache_fusion_max_token_num: int = 256
```

The pass is only registered for compile shapes whose batch size is `≤ 256`. Above that, the pass disables itself and the unfused two-kernel path stays in the graph.

The reasoning is the same trade-off that runs through every decode-vs-prefill optimization in vLLM:

| Regime | seq_q | Bottleneck for these tiny ops | Best strategy |
|---|---|---|---|
| Decode / small batch | ≤ 256 | kernel launch + HBM round-trip | **fuse** |
| Prefill / large batch | thousands | actual arithmetic | leave separate — two specialized kernels each tuned for its own access pattern beat one compromise kernel |

In the compute-bound regime, the saved launch and the saved HBM trip are negligible relative to the math, and the fused kernel cannot be tuned as aggressively as two specialized ones (RoPE has a regular per-position stride; the cache write follows the slot-mapping permutation — different optimal layouts).

## Backend support

Not every attention backend implements the fused path. The pass calls `impl.fused_rope_kvcache_supported()` per layer at registration time and only fuses where the backend says yes. Today the fused path is primarily ROCm / AITER.

| Backend | Fused path | Underlying kernel |
|---|---|---|
| Triton attention | yes (when AITER enabled) | `rocm_aiter_ops.triton_rope_and_cache` |
| ROCm attention | yes (when AITER enabled) | `rocm_aiter_ops.triton_rope_and_cache` |
| ROCm AITER unified | yes | `rocm_aiter_ops.triton_rope_and_cache` |
| ROCm AITER Flash | yes | `rocm_aiter_ops.triton_rope_and_cache` |
| FlashAttention (CUDA) | not yet | — |
| FlashInfer | not yet | — |

For backends without a fused implementation the pass simply skips registration; the unfused two-kernel path remains.

## Why this is a compile pass and not a hand call

Two reasons it lives in `vllm/compilation/passes/`:

1. **Per-layer applicability.** Whether the fusion can apply depends on the layer's attention impl (Triton vs FlashInfer vs MLA, plus quantization options like FP8 KV cache or per-head per-token quant). A compile pass walks the FX graph for every layer and fuses only where the impl reports support — hard to do cleanly inline in model code.

2. **Side-effect ordering.** The custom op declares `mutates_args=["query", "key"]` and returns a dummy tensor that downstream attention consumes. That dummy tensor is what makes torch.compile preserve the ordering between "rotate-and-cache" and "attention" — without it, Inductor would be free to reorder the cache write past the attention call. The fake / meta impl returns an empty tensor of the right device and dtype so compile-time tracing has something to bind.

## What it buys you

For a small-batch decode step on a model with `L` attention layers:

- Kernel launches per step: `2L` → `L`.
- HBM round-trips for `K`: 2 (RoPE writes `K`, cache update reads `K` again) → 1.
- Wall-clock: a small per-layer win on the order of single-digit µs, but it compounds over layers, decode steps, and concurrent requests.

The improvement is most visible when launch overhead is a meaningful fraction of decode latency — exactly the small-batch regime the `≤ 256` threshold gates for.

## Where it lives in vLLM

- `vllm/compilation/passes/fusion/rope_kvcache_fusion.py` — the FX pattern-matcher pass (`RopeKVCacheFusionPass`) and the registration of `torch.ops.vllm.fused_rope_and_unified_kv_cache_update`
- `vllm/v1/attention/backend.py` — `fused_rope_kvcache_supported()` and `do_rope_and_kv_cache_update()` on the `AttentionImpl` base
- `vllm/v1/attention/backends/{triton_attn,rocm_attn,rocm_aiter_unified_attn,rocm_aiter_fa}.py` — backends that implement the fused path
- `vllm/_aiter_ops.py::triton_rope_and_cache` — the underlying ROCm / AITER fused Triton kernel
- `vllm/config/compilation.py::PassConfig.fuse_rope_kvcache` — enable / disable
- `vllm/config/compilation.py::PassConfig.rope_kvcache_fusion_max_token_num` (default `256`) — batch-size threshold above which the pass declines to apply
