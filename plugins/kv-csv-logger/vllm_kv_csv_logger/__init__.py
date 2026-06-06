"""vLLM stat-logger plugin that dumps KV-cache / scheduler stats to CSV."""

from vllm_kv_csv_logger.logger import KVCsvLogger

__all__ = ["KVCsvLogger"]
