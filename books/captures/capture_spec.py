#!/usr/bin/env python
"""Capture: n-gram speculative decoding vs baseline, wall-clock (batch=1).

Task is repetition-heavy so prompt-lookup drafting gets high acceptance:
the model is asked to reproduce a paragraph verbatim.
Run:  capture_spec.py base   |   capture_spec.py ngram
"""
import sys
import time

from vllm import LLM, SamplingParams

mode = sys.argv[1] if len(sys.argv) > 1 else "base"

para = (
    "PagedAttention splits the KV cache into fixed-size blocks managed like "
    "virtual-memory pages, so fragmentation disappears and prefixes can be "
    "shared between requests. "
)
prompt = (
    "Here is a paragraph:\n\n" + para * 3 +
    "\n\nRepeat the paragraph above exactly, word for word, three times."
)

kwargs = dict(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    gpu_memory_utilization=0.45,
    enforce_eager=True,
    enable_prefix_caching=False,
)
if mode == "ngram":
    kwargs["speculative_config"] = {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }

llm = LLM(**kwargs)
sp = SamplingParams(temperature=0.0, max_tokens=256)

llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))  # warmup

t0 = time.perf_counter()
out = llm.generate([prompt], sp)
dt = time.perf_counter() - t0

n = len(out[0].outputs[0].token_ids)
print(f"[S-{mode}] {n} tokens in {dt*1e3:.0f} ms -> {n/dt:.1f} tok/s")
