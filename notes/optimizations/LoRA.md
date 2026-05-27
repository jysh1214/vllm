# LoRA (Low-Rank Adaptation)

**Core idea: freeze the pretrained weights and represent each task-specific weight update as a product of two low-rank matrices. Training touches ~0.1–1% of the parameters, and at inference time many adapters can be served simultaneously on a single base model.**

## The problem it solves

Full fine-tuning of a large model produces another full-size model. To serve `N` customer-specific fine-tunes you would need `N` copies of the base weights in GPU memory — for a 70B model, hundreds of gigabytes per customer. This makes "one fine-tune per customer" economically impossible at scale.

LoRA (Hu et al. 2021) starts from an empirical observation: when fine-tuning a pretrained model, the **effective weight update `ΔW` tends to be low rank**. So instead of learning `ΔW` directly, learn two thin matrices whose product approximates it.

## The decomposition

For every linear projection `y = W · x` you want to adapt:

```
original:   y = W · x                                 W ∈ R^(d_out × d_in)
LoRA:       y = W · x + (α / r) · B · A · x          A ∈ R^(r × d_in)
                                                      B ∈ R^(d_out × r)
                                                      r ≪ min(d_in, d_out)
```

- `W` stays **frozen**. Only `A` and `B` are trained.
- `r` (the **rank**) is typically 8, 16, 32, or 64 — far smaller than the projection's input or output dimension.
- `α / r` is a fixed scaling factor; only `α` is a hyperparameter, `r` and `α` together control the magnitude of `BA`.
- `A` is initialized random, `B` is initialized to **zero**. The product `BA` starts at exactly zero, so the adapted model behaves identically to the base model at step 0 of training.

Parameter-count comparison for a single projection with `d_in = d_out = 4096` and `r = 16`:

| | Trainable parameters |
|---|---|
| Full fine-tune | `d_in · d_out = 16,777,216` |
| LoRA | `d_in · r + r · d_out = 131,072` |

A **128× reduction** for that one layer. Across a 70B model with adapters on every linear projection, the total adapter is typically 10–100 MB, vs 140 GB for a full fine-tune.

## Where LoRA attaches in a transformer

LoRA only adapts **linear projection layers**. Every other component is untouched.

```
                      ┌─────────────────────────────┐
                      │         Block input         │
                      └─────────────────────────────┘
                                    │
                            RMSNorm / LayerNorm
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                Attention                      │
            │   x ──► [Q proj] ──► Q                      ★ │
            │   x ──► [K proj] ──► K                      ★ │
            │   x ──► [V proj] ──► V                      ★ │
            │   softmax(QK^T / √d) · V                      │
            │                ──► [O proj] ──► attn_out    ★ │
            └───────────────────────┬───────────────────────┘
                                    │
                            RMSNorm / LayerNorm
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                   MLP                         │
            │   h ──► [gate_proj] ──► gate                ★ │
            │   h ──► [up_proj]   ──► up                  ★ │
            │   silu(gate) · up                             │
            │              ──► [down_proj] ──► mlp_out    ★ │
            └───────────────────────┬───────────────────────┘
                                    │
                                Block output
```

★ marks every projection that LoRA can attach to. Each ★ becomes `Wx + (α/r) · B · A · x`.

| Layer | LoRA-able? | Notes |
|---|---|---|
| Q / K / V / O projections | yes | classic targets |
| MLP gate / up / down | yes | now standard in modern PEFT setups |
| RMSNorm / LayerNorm | no | tiny parameter count; not worth a decomposition |
| Attention scores / softmax | no | no learnable matrix |
| Position embeddings (RoPE) | no | not a matmul |
| Activation functions | no | pointwise |
| Token embedding / LM head | rare | only adapt when changing vocabulary |

The original LoRA paper applied it only to Q and V; modern practice usually attaches to all seven projection types.

## Merge or stay unmerged

LoRA can be deployed two ways:

| Mode | Formula | Pros | Cons |
|---|---|---|---|
| **Merged** | `W' = W + (α/r) · BA`, then run `y = W' · x` | Zero runtime overhead | One adapter per replica — loses multi-tenancy |
| **Unmerged** | `y = W · x + (α/r) · B · (A · x)` at every forward | Supports many adapters at once | Two extra matmuls per projection |

For multi-tenant serving (the regime vLLM targets) you **must** stay unmerged. Merging collapses the base+adapter back into one matrix, eliminating the ability to switch adapters per request.

## The multi-tenant serving problem

Suppose 100 customers each have their own LoRA adapter for the same 70B base. A request batch can contain 5 requests from 5 different customers. The naive options are:

| Approach | Why it breaks |
|---|---|
| One forward pass per request, swap adapter each time | Destroys batching → throughput collapses to 1/B |
| Pre-merge each adapter into a separate `W'`, hold all in memory | 100 copies of 140 GB = no |
| Only one adapter active at a time | Defeats the point — customers wait in line |

The right answer: **batch requests with mixed adapters into one forward pass**, and let a custom kernel apply each row's adapter without launching N separate matmuls. This is what Punica's BGMV / SGMV kernels are for.

## BGMV — batched grouped matrix-vector

`vllm/lora/punica_wrapper/` implements **Batched Grouped Matrix-Vector** multiply.

Standard GEMV: `y = W · x`, one weight matrix for the whole batch.
BGMV: `y[i] = W[adapter_id[i]] · x[i]`, **each row of the batch indexes its own weight matrix**.

Implementation sketch:

1. All adapters' `A` and `B` matrices are packed into one big tensor in HBM.
2. An `adapter_id → offset` table maps each request to its slice.
3. A custom CUDA / Triton kernel reads the right slice per row and does the matmul, all inside one launch.

The result: **a batch of 5 requests with 5 different adapters costs roughly the same as one ordinary GEMV** — instead of 5 separate kernel launches.

## SGMV — handling LoRA's "skinny-of-skinny" shape

LoRA's `A` and `B` matrices are already very skinny: one of their dimensions is `r ≈ 8..64`. So the LoRA path is:

```
A · x:  [r, d_in] @ [d_in, B]    →  [r, B]    intermediate, r tiny
B · _:  [d_out, r] @ [r, B]      →  [d_out, B]   contraction over tiny r
```

Tensor Cores are tuned for matmuls where every dim is at least ~16. They lose efficiency badly when the contraction dimension is `r = 8`. Standard cuBLAS GEMM is the wrong tool here.

**SGMV (Segmented Gather Matrix-Vector)** is Punica's answer:

- The `r`-dim reduction is done in **shared memory** with regular CUDA cores instead of Tensor Cores.
- Multiple adapters' computations are packed into one kernel invocation, segmented by `adapter_id`.
- Memory layout is chosen so each thread block reads a contiguous slice for its assigned adapter.

This is the same family of optimizations as Marlin for W4A16 GEMM — small, awkward shapes that general-purpose kernels can't handle well, made fast by hand-written specialization.

## Other LoRA optimizations in vLLM

| Optimization | Effect |
|---|---|
| **Dual-stream LoRA** (`VLLM_LORA_ENABLE_DUAL_STREAM`) | Run the LoRA path on a separate CUDA stream concurrently with the base `W·x`, so the LoRA overhead overlaps with base compute instead of stacking onto it |
| **PDL** (parameter-dimension-level) projection | Tuned strides / packing for the very-skinny `A` and `B` shapes; toggleable via `VLLM_LORA_DISABLE_PDL` |
| **LoRA model manager** | Lifecycle: load/unload adapters from HBM, warm them up, evict on memory pressure |
| **LoRA weight resolver** | Pull adapter weights from local paths, HuggingFace Hub, S3, etc., transparently to the request path |

## LoRA vs full fine-tuning

| Dimension | Full fine-tune | LoRA |
|---|---|---|
| Trainable parameters | 100% | ~0.1–1% |
| Training GPU memory | very large (gradients, optimizer state) | small |
| Training speed | slow | fast |
| Per-customer artifact size | 100s of GB | 10s of MB |
| Switch adapter at inference | impossible without reloading the whole model | microseconds (swap A and B) |
| Multi-tenant serving | one full model copy per customer | one base + many tiny adapters |
| Expressive ceiling | unconstrained | bounded by rank `r` |

The headline pitch in the paper is "training is cheap." The headline value in production is "**you can serve a thousand customer-specific models on a single base**" — a different and arguably bigger win.

## Where it lives in vLLM

- `vllm/lora/punica_wrapper/` — BGMV / SGMV kernel wrappers
- `vllm/lora/model_manager.py` — adapter lifecycle, warmup, eviction
- `vllm/lora/resolver.py` — weight loading from local / Hub / object storage
- `vllm/model_executor/layers/linear.py` — the `*ParallelLinear` classes with built-in LoRA hooks (this is where ★ becomes a real call site)
- `vllm/envs.py::VLLM_LORA_ENABLE_DUAL_STREAM`, `VLLM_LORA_DISABLE_PDL` — runtime toggles
