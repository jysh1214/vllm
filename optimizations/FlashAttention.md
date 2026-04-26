# FlashAttention

**Core idea: an IO-aware attention algorithm that uses tiling to keep all intermediate matrices on the GPU's on-chip SRAM, eliminating the HBM round-trip for the `[N, N]` score matrix.**

## The problem it solves

Standard attention computes and materializes two large intermediates:

```
S = Q @ K^T          shape: [seq_len, seq_len]
P = softmax(S)       shape: [seq_len, seq_len]
O = P @ V            shape: [seq_len, head_dim]
```

At `seq_len = 8192`, each of `S` and `P` has ~67 million entries. Both get written to HBM and read back. The result is that **HBM bandwidth — not arithmetic — becomes the bottleneck**.

GPU memory hierarchy (rough H100 numbers):

| Level | Size | Bandwidth |
|---|---|---|
| Registers | ~256 KB / SM | ~30 TB/s |
| **SRAM** (shared memory / L1) | ~228 KB / SM | ~19 TB/s |
| L2 cache | ~50 MB | ~5 TB/s |
| **HBM** (global memory) | 80 GB | ~3 TB/s |

SRAM is roughly 6× faster than HBM but ~350,000× smaller. Standard attention squanders the algorithmic surface that this asymmetry rewards.

## What "SRAM" means in this context

In CUDA terminology:

- **HBM** = global memory.
- **SRAM**, in the FlashAttention paper and most kernel-engineering writing, refers to **shared memory / L1** — the per-SM on-chip scratchpad you allocate explicitly with `__shared__`. The L2 cache is also physically SRAM, but it's chip-wide and hardware-managed; FlashAttention specifically wants the per-SM, programmer-controlled tier.

The pitch translates to: "keep Q/K/V tiles and softmax state in shared memory; only touch global memory to load inputs and write the final output."

## The algorithm

Tile Q, K, V so each tile fits in SRAM. Process one (K, V) tile against (a subset of) Q tiles entirely on-chip. The complete `[N, N]` score matrix is never assembled in HBM — only block-sized fragments live transiently in SRAM.

The single non-obvious part is **softmax**, which needs a *global* row max and a *global* row sum. You cannot softmax a single block in isolation. FlashAttention handles this with **online softmax**: it carries running statistics across blocks, and rescales prior work whenever the running max moves.

### State per Q row

```
m  : scalar       running max
l  : scalar       running denominator   Σ exp(s_i − m)
O  : vector[d]    running numerator     Σ exp(s_i − m) · v_i
```

Three things — that's the entire per-row state. The `[N, N]` score matrix has been compressed into O(d) state per row by exploiting softmax's max-shift invariance.

### The loop

```
m = -inf, l = 0, O = 0
for each (K_block, V_block):
    S_block = Q_block @ K_block^T               # tiled GEMM #1, in SRAM
    m_new   = max(m, max(S_block))              # update running max
    α       = exp(m - m_new)                    # rescale factor for past terms
    P_block = exp(S_block - m_new)              # local UNNORMALIZED exponentials
    l       = α * l + sum(P_block)              # update denominator
    O       = α * O + P_block @ V_block         # tiled GEMM #2, in SRAM
    m       = m_new
return O / l                                    # one division at the very end
```

(FlashAttention v2 puts Q on the outer loop and K/V on the inner loop so that one thread block owns one Q tile end-to-end and avoids cross-block synchronization on `O`.)

### Why the rescaling step is needed

When the running max jumps from `m_old` to `m_new`, every accumulated exponential `exp(s_i − m_old)` is now using the wrong reference. Multiplying both `O` and `l` by `α = exp(m_old − m_new)` retroactively re-bases past contributions:

```
α · exp(s_i - m_old) = exp(m_old - m_new) · exp(s_i - m_old)
                     = exp(s_i - m_new)
```

After the final block, `O` equals exactly `Σ exp(s_i − m_final) · v_i` and `l` equals exactly `Σ exp(s_i − m_final)`, independent of block ordering. `O / l` then matches standard attention bit-for-bit (modulo floating-point rounding).

### "softmax_partial" is not a real softmax

In casual pseudocode you might see `O += softmax_partial(S) @ V`. That's shorthand. There's no per-block softmax — there can't be, because softmax needs a global row sum. What's actually being accumulated is the *unnormalized* weighted sum of V; the single division by `l` at the very end produces the true softmax-weighted output.

Inside the loop only `O`, `l`, and `m` are mutated. `S_block` and `P_block` are scratch that lives in registers/SRAM and is discarded as soon as the block is consumed.

## What it buys you

1. **Speed.** Prefill is typically **2–4×** faster than standard attention; the speedup grows with sequence length because the saved HBM traffic scales as `O(N²)` while the added arithmetic scales as a small constant per element.
2. **Memory.** Attention's auxiliary memory drops from `O(N²)` to `O(N)`, enabling much longer contexts.
3. **Mathematically equivalent.** Not an approximation — same result as standard attention up to floating-point rounding.

## Why this works at all

FlashAttention isn't clever because the math is novel — it's clever because the **per-row state shrinks from O(N) to O(d)** by exploiting softmax's specific algebraic structure: max-shift invariance plus `exp(a) · exp(b) = exp(a+b)`, which lets partial sums compose correctly.

Operations that lack this structure cannot tile this way. You'd still get an I/O win on `Q @ K^T`, but you could not avoid materializing the full row to normalize. That's the reason "linear attention" variants (Performer, RetNet, Mamba) exist as a separate family — they chase the same I/O property through a different algebraic door.

## Prefill vs. decode

The `[N, N]` problem is a **prefill** problem. During decode:

- Only **one** new query token is processed per step. `Q` has shape `[1, head_dim]`.
- The `Q @ K^T` is `[1, context_len]` — already small. No quadratic intermediate exists.
- The bottleneck shifts to **streaming the entire KV cache from HBM** every step.

So FlashAttention's headline win is in prefill (and long-context training). Decode is memory-bandwidth-bound for a different reason — KV-cache reads — and is addressed by KV-cache quantization, paged layout, FlashInfer's decode kernels, etc.

## Version timeline

- **v1 (2022)** — tiling + online softmax.
- **v2 (2023)** — better warp-level parallelism, fewer non-matmul ops, ~2× over v1.
- **v3 (2024)** — Hopper-specific: TMA async copies, warp specialization, FP8.

## Where it lives in vLLM

- `vllm/v1/attention/backends/flash_attn.py` — Flash v2/v3 backend
- `vllm/v1/attention/backends/flashinfer.py` — FlashInfer's variant (decode-specialized kernels)
- `vllm/v1/attention/backends/triton_attn.py` — Triton implementation in the same family
- `vllm/v1/attention/backends/mla/` — MLA-flavored Flash variants for DeepSeek-style models
