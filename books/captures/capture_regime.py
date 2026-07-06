#!/usr/bin/env python
"""Capture: prefill vs decode per-token cost on one GPU (batch=1).

Method: prefix caching OFF so each run really prefills.
  T_prefill  = time of generate(promptA, max_tokens=1)   ~= prefill cost
  T_full     = time of generate(promptB, max_tokens=129) ~= prefill + 128 decode steps
  decode/token ~= (T_full - T_prefill_B) / 128, with T_prefill_B ~= T_prefill (same length)
"""
import time

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    gpu_memory_utilization=0.45,
    enforce_eager=True,
    enable_prefix_caching=False,
)

# two prompts of identical token length (~1k tokens), different content
base_a = "The quick brown fox jumps over the lazy dog near the river bank today. "
base_b = "A slow green turtle walks under the busy bridge past the market stall. "
prompt_a = base_a * 64
prompt_b = base_b * 64

sp1 = SamplingParams(temperature=0.0, max_tokens=1)
sp129 = SamplingParams(temperature=0.0, max_tokens=129)

# warmup (compile-free eager, but warm allocator/graphs)
llm.generate([prompt_b], sp1)

t0 = time.perf_counter()
out_a = llm.generate([prompt_a], sp1)
t_prefill = time.perf_counter() - t0

t0 = time.perf_counter()
out_b = llm.generate([prompt_b], sp129)
t_full = time.perf_counter() - t0

n_a = len(out_a[0].prompt_token_ids)
n_b = len(out_b[0].prompt_token_ids)
decode_steps = 128
prefill_per_tok_us = t_prefill / n_a * 1e6
decode_per_tok_ms = (t_full - t_prefill) / decode_steps * 1e3

print(f"[R0] prompt_a={n_a} tok, prompt_b={n_b} tok, decode_steps={decode_steps}")
print(f"[R1] prefill: {t_prefill*1e3:.1f} ms total -> {prefill_per_tok_us:.1f} us/token")
print(f"[R2] decode : {(t_full - t_prefill)*1e3:.1f} ms for {decode_steps} steps "
      f"-> {decode_per_tok_ms:.2f} ms/token")
print(f"[R3] per-token cost ratio decode/prefill = "
      f"{decode_per_tok_ms*1e3/prefill_per_tok_us:.0f}x")
