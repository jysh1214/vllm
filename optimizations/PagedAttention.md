# PagedAttention

**Core idea: bring the operating system's virtual-memory / paging abstraction to the LLM KV cache.**

## The problem it solves

Traditional LLM inference allocates a single contiguous KV-cache buffer per request, sized to the *maximum* possible sequence length. This causes:

- **Internal fragmentation** — actual generated length is usually far shorter than the max, so the reservation is mostly wasted.
- **External fragmentation** — different request lengths leave the global pool with awkward gaps that no new request fits cleanly into.
- **No sharing** — beam search, parallel sampling, and shared system prompts force duplicate copies of identical KV tensors.

In practice, naive allocation pushes effective KV-cache memory utilization down to **20–40%**.

## How it works

The KV cache is split into fixed-size **blocks** (commonly 16 tokens per block). Blocks are stored in one big preallocated pool in GPU HBM, and they do **not** need to be physically contiguous within a request. Each request keeps a **block table** — analogous to an OS page table — that maps logical token positions to physical block indices.

```
HBM (GPU)
┌────────────────────────────────────────────────┐
│  KV cache pool: [block 0][block 1][block 2]... │   one big tensor
└────────────────────────────────────────────────┘
            ↑           ↑           ↑
   request A's table:  [7, 3, 12]    logical → physical
   request B's table:  [3, 5]        request B reuses block 3 with A
```

Logical view:

```
logical seq:  [tok0..tok15][tok16..tok31][tok32..tok47] ...
                  ↓             ↓              ↓
block table:  [block 7]    [block 3]      [block 12]
```

The attention kernel reads through the block table indirectly, treating the scattered physical blocks as though they were a contiguous logical sequence.

## What it enables

1. **Near-zero waste.** Blocks are allocated on demand; KV utilization climbs to **~96%+**.
2. **Prefix caching.** A shared system prompt is stored once. Multiple requests' block tables point at the same physical blocks.
3. **Copy-on-write.** Beam-search forks share blocks until one branch writes; only then is a block copied.
4. **Larger effective batch size.** More requests fit in the same HBM, which directly lifts throughput.

## Where blocks live

By default, blocks live in **GPU HBM only**. PagedAttention is *not* a swap-to-disk mechanism — the "paging" name refers to the virtual-memory abstraction, not to demand paging from secondary storage.

Optional, layered features can move blocks off the GPU:

| Feature | Block destination | When |
|---|---|---|
| **PagedAttention** (default) | GPU HBM only | Always |
| **CPU offload** (`vllm/v1/kv_offload/`) | Host RAM | Idle requests, GPU pressure |
| **KV-transfer connectors** (NIXL, LMCache, MoonCake, MoriIO, P2P-NCCL) | Other GPUs over network, or remote storage | Disaggregated prefill/decode, cross-replica prefix reuse |
| **LMCache** (specifically) | Can spill to local disk | Long-tail prefix reuse |

So disk-resident KV cache is achievable in vLLM, but it's a layer **above** PagedAttention, not part of it.

## Mental model

- **PagedAttention** = virtual memory **inside the GPU**. Block table = page table. "Pages" = chunks of HBM.
- **Offload / KV transfer** = swap **out of the GPU**. Host RAM, disk, and remote nodes act as the swap tiers.

The two are composable but independent. A request can target a paged GPU block; if the block has been evicted, an offload connector can fault it back into HBM before the kernel runs.

## Where it lives in vLLM

- `vllm/v1/core/kv_cache_manager.py` — block allocation, prefix-hash bookkeeping
- `vllm/v1/core/block_pool.py` — the underlying free-block pool
- `vllm/v1/core/kv_cache_coordinator.py` — distributed allocation
- `csrc/attention/paged_attention_v1.cu`, `paged_attention_v2.cu` — CUDA kernels that read through block tables
