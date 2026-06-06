"""CSV stat-logger plugin for vLLM v1.

Writes one CSV row per scheduler step from the aggregate ``SchedulerStats`` /
``IterationStats`` that vLLM hands to ``StatLoggerBase.record()``. Intended for
offline analysis (pandas / spreadsheets), so all fields are raw per-step values
— compute cumulative sums / hit rates downstream.

Only the aggregate stats are available through this hook; per-block /
per-request KV detail is not forwarded here.

Configuration (environment variables):
    VLLM_KV_CSV_PATH         Output path. The engine index is inserted before the
                             extension, e.g. ``stats.csv`` -> ``stats.0.csv``.
                             Default: ``vllm_kv_cache_stats.csv``.
    VLLM_KV_CSV_FLUSH_EVERY  Flush the file every N rows. Default: ``50``.
"""

import atexit
import csv
import os

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import (
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
)

logger = init_logger(__name__)

DEFAULT_PATH = "vllm_kv_cache_stats.csv"
DEFAULT_FLUSH_EVERY = 50

# CSV columns, in order. Keep this list and the row dict in record() in sync.
COLUMNS: tuple[str, ...] = (
    # bookkeeping
    "wall_time",
    "engine_idx",
    # SchedulerStats: scheduler / queues
    "step_counter",
    "current_wave",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_skipped_waiting_reqs",
    # SchedulerStats: KV cache
    "kv_cache_usage",
    "num_eviction_events",
    # SchedulerStats: prefix cache (per-step deltas)
    "prefix_requests",
    "prefix_queries",
    "prefix_hits",
    "prefix_preempted_requests",
    "prefix_preempted_queries",
    "prefix_preempted_hits",
    # SchedulerStats: external (KV-connector) prefix cache
    "connector_prefix_requests",
    "connector_prefix_queries",
    "connector_prefix_hits",
    # SchedulerStats: LoRA
    "num_running_lora",
    "num_waiting_lora",
    # IterationStats: throughput / tokens
    "iter_timestamp",
    "num_generation_tokens",
    "num_prompt_tokens",
    "prompt_computed",
    "prompt_local_cache_hit",
    "prompt_external_kv_transfer",
    "prompt_cached_tokens",
    "num_preempted_reqs",
    "num_corrupted_reqs",
    "num_finished_reqs",
    # IterationStats: latency (variable-length lists summarized as count + sum)
    "ttft_count",
    "ttft_sum",
    "itl_count",
    "itl_sum",
    # MultiModalCacheStats
    "mm_requests",
    "mm_queries",
    "mm_hits",
)


def _engine_path(base: str, engine_index: int) -> str:
    """Insert the engine index before the file extension."""
    root, ext = os.path.splitext(base)
    return f"{root}.{engine_index}{ext or '.csv'}"


class KVCsvLogger(StatLoggerBase):
    """Stat logger that appends one CSV row per scheduler step."""

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index

        base = os.environ.get("VLLM_KV_CSV_PATH", DEFAULT_PATH)
        self.path = _engine_path(base, engine_index)
        try:
            self.flush_every = max(
                1, int(os.environ.get("VLLM_KV_CSV_FLUSH_EVERY", DEFAULT_FLUSH_EVERY))
            )
        except ValueError:
            self.flush_every = DEFAULT_FLUSH_EVERY

        self._rows_since_flush = 0
        # newline="" per the csv module's recommendation.
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        self._file.flush()
        atexit.register(self._close)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        row: dict[str, object] = {col: "" for col in COLUMNS}
        row["wall_time"] = _time()
        row["engine_idx"] = engine_idx

        if scheduler_stats is not None:
            s = scheduler_stats
            row.update(
                step_counter=s.step_counter,
                current_wave=s.current_wave,
                num_running_reqs=s.num_running_reqs,
                num_waiting_reqs=s.num_waiting_reqs,
                num_skipped_waiting_reqs=s.num_skipped_waiting_reqs,
                kv_cache_usage=s.kv_cache_usage,
                num_eviction_events=len(s.kv_cache_eviction_events),
                num_running_lora=len(s.running_lora_adapters),
                num_waiting_lora=len(s.waiting_lora_adapters),
            )
            p = s.prefix_cache_stats
            if p is not None:
                row.update(
                    prefix_requests=p.requests,
                    prefix_queries=p.queries,
                    prefix_hits=p.hits,
                    prefix_preempted_requests=p.preempted_requests,
                    prefix_preempted_queries=p.preempted_queries,
                    prefix_preempted_hits=p.preempted_hits,
                )
            c = s.connector_prefix_cache_stats
            if c is not None:
                row.update(
                    connector_prefix_requests=c.requests,
                    connector_prefix_queries=c.queries,
                    connector_prefix_hits=c.hits,
                )

        if iteration_stats is not None:
            it = iteration_stats
            pts = it.prompt_token_stats
            ttft = it.time_to_first_tokens_iter
            itl = it.inter_token_latencies_iter
            row.update(
                iter_timestamp=it.iteration_timestamp,
                num_generation_tokens=it.num_generation_tokens,
                num_prompt_tokens=it.num_prompt_tokens,
                prompt_computed=pts.computed,
                prompt_local_cache_hit=pts.local_cache_hit,
                prompt_external_kv_transfer=pts.external_kv_transfer,
                prompt_cached_tokens=pts.cached_tokens,
                num_preempted_reqs=it.num_preempted_reqs,
                num_corrupted_reqs=it.num_corrupted_reqs,
                num_finished_reqs=len(it.finished_requests),
                ttft_count=len(ttft),
                ttft_sum=sum(ttft),
                itl_count=len(itl),
                itl_sum=sum(itl),
            )

        if mm_cache_stats is not None:
            m = mm_cache_stats
            row.update(
                mm_requests=m.requests,
                mm_queries=m.queries,
                mm_hits=m.hits,
            )

        self._writer.writerow([row[col] for col in COLUMNS])
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every:
            self._file.flush()
            self._rows_since_flush = 0

    def log_engine_initialized(self):
        logger.info(
            "KVCsvLogger (engine %d) writing KV-cache stats to %s",
            self.engine_index,
            self.path,
        )

    def _close(self):
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()


def _time() -> float:
    # Wrapped so the import stays local to where it is used / easy to stub.
    import time

    return time.time()
