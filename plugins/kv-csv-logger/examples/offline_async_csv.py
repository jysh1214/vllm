"""Offline example that exercises the kv_csv plugin with a real model.

Run with the plugin enabled:

    VLLM_PLUGINS=kv_csv VLLM_KV_CSV_PATH=/tmp/kv_run.csv \
        python examples/offline_async_csv.py

A CSV (e.g. /tmp/kv_run.0.csv) is written with one row per scheduler step.

We drive AsyncLLM directly because stat-logger plugins are only auto-loaded on
the AsyncLLM path (the sync `vllm.LLM` class does not load them and disables
stats by default).
"""

import asyncio
import os
import uuid

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = os.environ.get("KV_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PROMPTS = [
    "The capital of France is",
    "Write a one-sentence summary of photosynthesis:",
    "List three primary colors:",
    "Explain gravity to a five year old in two sentences:",
    "Translate 'good morning' into Spanish:",
    "What is 17 multiplied by 23?",
]


async def _run_one(engine: AsyncLLM, prompt: str) -> str:
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    final = None
    async for out in engine.generate(prompt, sp, str(uuid.uuid4())):
        final = out
    return final.outputs[0].text if final else ""


async def main() -> None:
    args = AsyncEngineArgs(
        model=MODEL,
        gpu_memory_utilization=0.55,
        max_model_len=2048,
        enforce_eager=True,
    )
    engine = AsyncLLM.from_engine_args(args)
    try:
        # Fire prompts concurrently so the scheduler batches them across many
        # steps -> many CSV rows.
        results = await asyncio.gather(*(_run_one(engine, p) for p in PROMPTS))
        for prompt, text in zip(PROMPTS, results):
            print(f"[{prompt!r}] -> {text.strip()[:60]!r}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
