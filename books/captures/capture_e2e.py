#!/usr/bin/env python
"""End-to-end capture: prefix caching in action on a real model (RTX 4090).

Send the same long prompt twice; the second request's num_cached_tokens
shows the prefix-cache hit, block-aligned, with the last-token rule visible.
"""
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    gpu_memory_utilization=0.45,
    enforce_eager=True,  # skip CUDA-graph capture; scheduler/KV behavior unchanged
)

cfg = llm.llm_engine.vllm_config
print(f"\n[E0] block_size = {cfg.cache_config.block_size}")

# A long shared prefix (simulates a shared system prompt), then two variants.
prefix = (
    "You are a meticulous assistant for the vLLM project. "
    "Answer questions about schedulers, paged KV caches, prefix caching, "
    "tensor parallelism and speculative decoding precisely and concisely. "
) * 4
prompts_round1 = [prefix + "Question: what is a KV cache block?"]
prompts_round2 = [prefix + "Question: what is preemption?"]  # same prefix, new tail

sp = SamplingParams(temperature=0.0, max_tokens=8)

out1 = llm.generate(prompts_round1, sp)
out2 = llm.generate(prompts_round2, sp)

for tag, outs in (("round 1 (cold)", out1), ("round 2 (shared prefix)", out2)):
    for o in outs:
        n_prompt = len(o.prompt_token_ids)
        print(
            f"[E1] {tag}: prompt_tokens={n_prompt}, "
            f"num_cached_tokens={o.num_cached_tokens}"
        )

# Same *exact* prompt as round 1 -> full hit minus last-token rule
out3 = llm.generate(prompts_round1, sp)
for o in out3:
    n_prompt = len(o.prompt_token_ids)
    print(
        f"[E1] round 3 (identical to round 1): prompt_tokens={n_prompt}, "
        f"num_cached_tokens={o.num_cached_tokens}"
    )
