# Speculative Decoding

**Core idea: turn one sequential decode step into a parallel verification by having a cheap draft model fill in plausible guesses; the target model scores all of them in a single forward pass, and the longest correct prefix gets emitted.**

## The problem it solves

Decode is autoregressive — each forward pass produces exactly one new token, and each pass reads the entire model weights from HBM. So generating 100 tokens requires 100 full sweeps over multi-gigabyte weight tensors, even though the arithmetic per token is trivial. This makes decode **memory-bandwidth bound, not compute bound**: on an H100, a target forward pass that scores 1 candidate and one that scores 5 cost almost the same wall time, because the HBM read dominates either way.

Two facts about transformers, taken together, expose an opening:

| Fact | Consequence |
|---|---|
| Generation is sequential (1 token per FP) | Decode underutilizes compute |
| Causal attention can score N positions in parallel in one FP | Verifying N candidates costs ≈ verifying 1 |

Speculative decoding exploits the gap between them.

## The prefill connection

LLM inference is split into two strictly different phases:

| Phase | Process | Forward passes |
|---|---|---|
| **Prefill** | Process the prompt — compute every position in one shot | 1 |
| **Decode** | Generate new tokens — one position at a time | N (one per token) |

Prefill on a 1000-token prompt does the entire thing in **one** forward pass. The embedding layer turns 1000 tokens into 1000 vectors, every transformer layer does attention over all 1000 positions in parallel (causal mask ensures position `i` only sees `0..i`), and the LM head emits a logit at every position: `L_1, L_2, ..., L_1000`. Normally only `L_1000` is used (to sample the next token); `L_1..L_999` are computed and thrown away. There is no way to skip them — computing the last position's attention requires the K/V at every prior position, which means computing those positions' logits comes along for free.

Decode is the opposite. With the prompt already in the cache, generating one new token requires a forward pass that ingests one token, writes one K/V entry, and emits one logit. To generate 1000 new tokens, you do this 1000 times.

The amortization difference is dramatic:

| Goal | Forward passes | HBM weight reads | Logits / FP |
|---|---|---|---|
| Prefill on 1000-token prompt | 1 | 1 | 1000 |
| Decode 1000 tokens | 1000 | 1000 | 1 |

Both produce 1000 logits, but prefill reads the model weights once while decode reads them a thousand times. Each weight read is a multi-gigabyte HBM sweep, so **prefill's per-token cost is ~1000× lower than decode's** — not because the math is cheaper, but because the weight read amortizes across positions.

A speculative verify pass is structurally identical to prefill on a small chunk of tokens — but instead of discarding the intermediate logits, it uses them to verify draft candidates. Each `L_d_i` from the verify pass is exactly what prefill on `[context, d_1, ..., d_i]` would have produced for position `d_i`:

| Phase | Process | Logits produced | Logits used |
|---|---|---|---|
| Prefill | N prompt tokens in one FP | N | last 1 |
| Vanilla decode | 1 new token per FP | 1 | 1 |
| Speculative verify | N draft tokens in one FP | N+1 | up to N+1 |

The same comparison at the attention-tensor level — using `Q : (num_q_heads, seq_q, head_dim)` and `K, V : (num_kv_heads, seq_kv, head_dim)`:

| Phase | Q | K, V |
|---|---|---|
| Prefill (k-token prompt) | `(num_q_heads, k, head_dim)` | `(num_kv_heads, k, head_dim)` |
| Vanilla decode (context length k) | `(num_q_heads, 1, head_dim)` | `(num_kv_heads, k+1, head_dim)` |
| Speculative verify (context k, N drafts) | `(num_q_heads, N, head_dim)` | `(num_kv_heads, k+N, head_dim)` |

`seq_q` — how many positions get scored in this forward pass — is the only thing that varies between phases. `seq_kv` is always `context_so_far + seq_q`, because the new tokens being scored are written into the KV cache during the same pass. **Verify is just prefill at `seq_q = N` instead of `seq_q = k`.**

Same computation in all three cases — only the bookkeeping differs. **A verify pass is a mini-prefill, repurposed.**

## The algorithm

Three steps per round.

### Step 1 — Draft generates N candidates

A small, cheap model produces N autoregressive guesses (typical N = 3–8):

```
context = [t1, ..., tk]
for i in range(N):
    logits = draft(context + draft_tokens)
    draft_tokens.append(sample(logits))
# draft_tokens = [d1, d2, d3, d4, d5]
```

The draft is itself autoregressive, but its forward passes are far cheaper than the target's, so N drafts cost much less than one target FP.

### Step 2 — Target verifies all N in one parallel forward pass

Feed `[d1, d2, d3, d4, d5]` to the target. With prior context length `k` and `N = 5` draft tokens, the attention tensors have shape:

```
Q : (num_q_heads,  N,    head_dim)        ← N = 5 draft positions
K : (num_kv_heads, k+N,  head_dim)        ← prior context k + N drafts
V : (num_kv_heads, k+N,  head_dim)
```

Causal masking guarantees that each position's logit conditions only on the prefix up to that position:

```
L_pre  (already cached)   →  predicts d1   (verifies d1)
L_d1   (this verify)      →  predicts d2   (verifies d2 if d1 accepted)
L_d2                      →  predicts d3
L_d3                      →  predicts d4
L_d4                      →  predicts d5
L_d5                      →  predicts d6   (bonus — used only if all accepted)
```

Because each `L_d_i` sees exactly the same prefix it would have seen during sequential decode, **the parallel pass produces logits mathematically identical to N+1 sequential passes** — at the cost of one.

### Step 3 — Accept the longest matching prefix

Walk left to right. At the first rejection, sample a corrected token from that position's logit and stop:

- **Greedy** (T = 0): accept iff `argmax(L_i) == d_{i+1}`.
- **Stochastic** (T > 0): rejection-sampling test (see [Stochastic correctness](#stochastic-correctness) below) that makes the output distribution provably identical to running the target alone.

### Per-round outcomes

| Drafts accepted | Tokens emitted | Source of last token |
|---|---|---|
| 0 | 1 | corrected from `L_pre` |
| 1 | 2 | d_1 + corrected from `L_d1` |
| 2 | 3 | d_1, d_2 + corrected from `L_d2` |
| 3 | 4 | d_1, d_2, d_3 + corrected from `L_d3` |
| 4 | 5 | d_1..d_4 + corrected from `L_d4` |
| 5 | 6 | all drafts + bonus from `L_d5` |

Per-round cost is constant (1 target FP + N draft FPs); output ranges from 1 to N+1 tokens.

### Worked example

Target 70B at 100 ms / FP, draft 1B at 5 ms / FP, N = 5, context `"The cat sat on the"`.

Draft produces `[" mat", ".", " The", " cat", " was"]` in 25 ms. Target verifies in 100 ms, then the accept walk:

```
position 0: argmax(L_pre) = " mat"  == d1   ✓ accept
position 1: argmax(L_d1)  = "."     == d2   ✓ accept
position 2: argmax(L_d2)  = " The"  == d3   ✓ accept
position 3: argmax(L_d3)  = " A"    != d4   ✗ reject → emit " A"
```

Output: 4 tokens. Cost: 25 + 100 = 125 ms → **31 ms / token**. Vanilla decode of the same 4 tokens: 4 × 100 = 400 ms → **100 ms / token**. **~3.2× speedup.**

## What it buys you

1. **Speed.** 2–4× decode throughput is typical for a well-matched draft pair on memory-bandwidth-bound hardware. Above 6× requires very high accept rates.
2. **Mathematically exact.** Output distribution is provably identical to the target alone — both for greedy and stochastic sampling.
3. **Asymmetric upside.** Best case is N+1× the throughput for one target FP; worst case is a small constant overhead from the draft. Expected value is positive across most realistic accept rates.

## Why this works at all

The win comes from a structural quirk of decode, not from anything clever about attention:

> The cost of a target forward pass on N tokens equals the cost on 1 token, in a memory-bandwidth-bound regime.

In a compute-bound regime (huge batches, prefill of long sequences) this is no longer true — verifying 5 positions actually costs ~5× more, and speculative decoding stops paying. So speculative decoding is a *decode-specific* optimization: it converts decode's wasted compute into useful work, and only there.

This is also why all the variants below converge on the same problem statement: find a way to produce candidate tokens cheaper than a target forward pass.

### Speculative decoding disguises decode as prefill

In ordinary decode, position `t_{k+2}` cannot be computed alongside `t_{k+1}` because `t_{k+1}` is unknown until it has been sampled. The dependency chain forces serialization. The draft model breaks the chain by **guessing** the missing tokens, letting the target run a forward pass that *looks like* prefill — N tokens in, N+1 logits out — even though those N tokens were never actually observed.

If the guesses are right, the target produced N+1 logits worth of useful work for the price of one forward pass. If they are wrong, the speculation collapses back to one usable logit (`L_pre`), and the round degrades into a slightly-padded vanilla decode.

> Speculative decoding turns decode's serial dependency chain into prefill's parallel one, by paying a draft model to fill the slots that prefill would have filled with prompt tokens.

### The training connection

Training is entirely prefill. Under teacher forcing, every position in a sequence is fed its ground-truth predecessor as input, so all positions can be computed in parallel — one forward pass produces a logit at every position, and every one of those logits contributes to the loss. Nothing is wasted.

Inference loses this property because the "true predecessor" is exactly what you are trying to generate. Speculative decoding **restores teacher forcing at inference time**, with the draft model standing in for ground truth. When the draft is correct, the verify pass is just teacher forcing on the next N tokens. When the draft is wrong, the rejection-sampling rule (see [Stochastic correctness](#stochastic-correctness)) patches things up so the output distribution remains exact.

## Why a draft model is required

If the verify pass already produces logits at every position in parallel, can we just feed any N tokens — random garbage — and harvest those logits?

No. Each `L_d_i` is `P(next | context, d_1, ..., d_i)`. It is only useful if the conditioning prefix matches a plausible future:

```
Feed [garbage, garbage, garbage, garbage, garbage]:
   L_pre               →  useful  (predicts position k)
   L_g1, ..., L_g5     →  useless (conditioned on a fictional past)
```

Without a draft you can extract exactly 1 token per target FP. The draft's whole job is to **make the conditioning prefixes credible enough that the parallel logits become usable**. When the draft guesses well, those logits are the right distributions to sample from. When it guesses badly, they get discarded.

## How to draft cheaply

All variants solve the same constraint: drafting must cost less than a target FP, or there is no win.

| Method | How it drafts |
|---|---|
| Separate small model (classic) | A 1B autoregressive model alongside the 70B target |
| **EAGLE / EAGLE-2 / EAGLE-3** | One tiny extra autoregressive transformer layer fed the target's hidden states |
| **Medusa** | Multiple extra heads on the target's last hidden state, predicting +1, +2, ... in one target FP |
| **MTP** | Target itself is pretrained to emit multiple future tokens per position |
| **Lookahead decoding** | Jacobi iteration on the target, surfacing parallel guesses without a separate model |
| **N-gram / prompt-lookup** | String search in the prompt or prior generation — zero parameters |
| **Suffix decoding** | Suffix-tree lookup over global history — a generalization of n-gram |
| **Self-speculative / LayerSkip** | Run only the first L layers of the target as the draft |

## Worst case and win conditions

Even when every draft is rejected:

- Target FP cost: same as vanilla decode (1 FP).
- Draft overhead: N × draft FP (small).
- Output: 1 token, corrected from `L_pre` — which the verify pass has already computed.
- No fallback decode needed; the verify pass *is* the decode for that position.

Worst case: vanilla decode plus draft overhead, typically 10–25% slower. Best case: N+1 tokens for one target FP, typically 4–6× faster.

Speculative decoding **wins when**:
- `target_cost ≫ draft_cost` (e.g. 70B target with a 1B draft).
- Accept rate is high — draft and target agree often, usually because they share a family, tokenizer, or training distribution.
- The regime is memory-bandwidth bound (decode), so a target FP scoring N positions costs ≈ a target FP scoring 1.

It **loses when**:
- The draft is too large — overhead eats the savings.
- Style or distribution mismatch — accept rate collapses, draft work is wasted.
- The regime is compute-bound (very large batches, long prefill) — the parallel verify pass becomes proportionally expensive.

## KV cache discipline

Sampling is a pure tensor op on a logit vector — it picks a token but **does not run the model and does not update the KV cache**. Every K/V entry is produced by the forward pass that *consumes* a token as input, not by the sampling step that produces it.

That gives a precise timeline across phases:

```
Prefill FP on [t_1, ..., t_n]
    KV cache: writes positions 1..n            ✓
    Output:   logits L_1..L_n

Sample t_{n+1} from L_n
    KV cache: unchanged                        ✗
    (position n+1 is known, but has no K/V yet)

First decode FP on [t_{n+1}]
    KV cache: writes position n+1              ✓
    Output:   logits L_{n+1}

Sample t_{n+2} from L_{n+1}
    KV cache: unchanged                        ✗
    ...
```

The same pattern holds inside speculative decoding. The verify pass writes K/V for all N draft positions assuming they will be accepted. When position `i` is rejected, the cache must be repaired:

| Position range | What was written | Action |
|---|---|---|
| Original context | correct prior tokens | keep |
| 0 .. i−1 | accepted drafts | keep |
| i | rejected draft d_{i+1} | discard |
| i+1 .. N−1 | rejected drafts | discard |

The corrected token `x` sampled from `L_i` has **no K/V computed during recovery** — sampling alone never writes to the cache. It gets computed for free on the next round's verify pass, where `x` becomes the first input of `[x, new_d1, ..., new_d5]`. That forward pass consumes `x` as input, so it produces `x`'s K/V as a side effect of doing the verify it would have done anyway.

The unified invariant across prefill, vanilla decode, speculative verify, and rejection recovery:

> Sampling produces a token but not its K/V. The K/V is computed on the next forward pass that consumes that token as input.

## Stochastic correctness

For draft distribution `q` and target distribution `p` at each position:

```
accept d   with probability   min(1, p(d) / q(d))
on reject, sample from        max(0, p − q),  renormalized
```

This is classical rejection sampling. The proof (Leviathan et al. 2023; Chen et al. 2023) shows the resulting output distribution equals `p` exactly — independent of the choice of `q`. A bad draft hurts speed, never quality.

## Where it lives in vLLM

- `vllm/v1/spec_decode/llm_base_proposer.py` — base proposer abstraction
- `vllm/v1/spec_decode/eagle.py` — EAGLE / EAGLE-3
- `vllm/v1/spec_decode/medusa.py` — Medusa
- `vllm/v1/spec_decode/ngram_proposer.py`, `ngram_proposer_gpu.py` — N-gram / prompt-lookup (CPU and GPU variants)
- `vllm/v1/spec_decode/suffix_decoding.py` — suffix-tree drafting
- `vllm/v1/spec_decode/dflash.py` — verification-specialized attention kernel
- `vllm/v1/spec_decode/metadata.py` — per-request spec-decode metadata
- `vllm/v1/sample/rejection_sampler.py` — stochastic accept/reject implementation
