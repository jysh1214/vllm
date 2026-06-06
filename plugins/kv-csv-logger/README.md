# kv-csv-logger

A standalone vLLM **stat-logger plugin** that dumps v1 KV-cache and scheduler
stats to a CSV file — one row per scheduler step — for offline analysis
(pandas / spreadsheets / plotting).

It plugs into vLLM through the supported `vllm.stat_logger_plugins` entry point,
so **nothing in the `vllm/` tree is modified**. vLLM calls the plugin's
`record()` once per scheduler step and we append a row.

> Scope: aggregate ("Tier A") stats only — KV cache usage, prefix-cache reuse,
> queue depths, throughput/latency. Per-block heatmaps and per-request block
> tables are *not* available through this hook.

## Install

```bash
# from this directory (vLLM must already be installed in the same env)
uv pip install -e .
```

This does not depend on / reinstall vLLM; it only registers the entry point.

## Use

Enable the plugin by name via `VLLM_PLUGINS`. Stat-logger plugins are
auto-discovered on the **`AsyncLLM`** path — i.e. the OpenAI API server and any
script that drives `AsyncLLM` directly:

```bash
VLLM_PLUGINS=kv_csv vllm serve <model>
```

```bash
VLLM_PLUGINS=kv_csv python examples/offline_async_csv.py   # uses AsyncLLM
```

A CSV is written for each engine. With the default path and a single engine you
get `vllm_kv_cache_stats.0.csv`.

> **Note — the offline `LLM` class does not pick this up.** `vllm.LLM` (a) sets
> `disable_log_stats=True` by default and (b) runs the synchronous `LLMEngine`,
> which never calls `load_stat_logger_plugin_factories()`. Only `AsyncLLM`
> auto-loads stat-logger plugins. For offline use, drive `AsyncLLM` directly —
> see `examples/offline_async_csv.py`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_KV_CSV_PATH` | `vllm_kv_cache_stats.csv` | Output path. The engine index is inserted before the extension, e.g. `stats.csv` → `stats.0.csv`. |
| `VLLM_KV_CSV_FLUSH_EVERY` | `50` | Flush the file every N rows (keeps `record()` cheap on the engine hot path). |

## CSV columns

One row per `record()` call (one scheduler step). All values are **raw per-step**
quantities — compute cumulative sums / hit rates downstream.

| Column | Source | Notes |
|--------|--------|-------|
| `wall_time` | `time.time()` | Wall-clock at row write |
| `engine_idx` | `record()` arg | Engine that produced the row |
| `step_counter`, `current_wave` | `SchedulerStats` | DP scheduling bookkeeping |
| `num_running_reqs` | `SchedulerStats` | Running queue depth (batch size) |
| `num_waiting_reqs`, `num_skipped_waiting_reqs` | `SchedulerStats` | Waiting / skipped-waiting depth |
| `kv_cache_usage` | `SchedulerStats` | Block-pool occupancy, 0.0–1.0 |
| `num_eviction_events` | `SchedulerStats.kv_cache_eviction_events` | Sampled evictions this step |
| `prefix_requests/queries/hits` | `PrefixCacheStats` | Prefix-cache deltas this step |
| `prefix_preempted_requests/queries/hits` | `PrefixCacheStats` | Same, for previously preempted requests |
| `connector_prefix_requests/queries/hits` | `connector_prefix_cache_stats` | External KV-connector cache (blank if none) |
| `num_running_lora`, `num_waiting_lora` | `SchedulerStats` | Active LoRA adapter counts |
| `iter_timestamp` | `IterationStats` | Iteration wall-clock |
| `num_generation_tokens`, `num_prompt_tokens` | `IterationStats` | Tokens this iteration |
| `prompt_computed` | `PromptTokenStats` | Prompt tokens actually computed |
| `prompt_local_cache_hit` | `PromptTokenStats` | Prompt tokens from local prefix cache |
| `prompt_external_kv_transfer` | `PromptTokenStats` | Prompt tokens from external KV transfer |
| `prompt_cached_tokens` | `PromptTokenStats` | Prompt tokens skipped during prefill |
| `num_preempted_reqs`, `num_corrupted_reqs`, `num_finished_reqs` | `IterationStats` | Per-iteration counts |
| `ttft_count`, `ttft_sum` | `IterationStats.time_to_first_tokens_iter` | TTFT samples summarized as count + sum |
| `itl_count`, `itl_sum` | `IterationStats.inter_token_latencies_iter` | Inter-token-latency samples, count + sum |
| `mm_requests/queries/hits` | `MultiModalCacheStats` | Multimodal cache (blank for text-only) |

Cells are left blank when the corresponding stats object is absent for that step.
