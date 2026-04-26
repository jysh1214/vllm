# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Build & Development Commands

**Never use system `python3` or bare `pip`.** All Python commands go through `uv` and `.venv/bin/python`.

```bash
# Environment setup
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install

# Install (Python-only changes)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Install (with C/C++/CUDA changes)
uv pip install -e . --torch-backend=auto

# Run a single test
uv pip install -r requirements/test/cuda.in
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Lint
pre-commit run --all-files              # all hooks
pre-commit run ruff-check --all-files   # just ruff
pre-commit run mypy-3.10 --all-files --hook-stage manual  # mypy as in CI
```

Pre-commit hooks include: ruff (lint + format), typos, clang-format (C++/CUDA), markdownlint, actionlint.

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. The codebase has moved to the **v1 engine** as the production path; legacy engine code in `vllm/engine/` is now thin re-export aliases into `vllm/v1/`.

### Request flow

```
Entrypoints (LLM class / OpenAI API server / CLI)
  -> EngineClient (protocol in vllm/engine/protocol.py)
    -> v1 EngineCore (vllm/v1/engine/core.py, ZMQ-based scheduler loop)
      -> Scheduler (vllm/v1/core/sched/scheduler.py)
      -> Executor (multiproc / Ray / uniproc)
        -> Worker (one per GPU, owns a ModelRunner)
          -> ModelRunner (forward pass, attention backends)
            -> Model (PyTorch impl in vllm/model_executor/models/)
```

### Key layers

- **Entrypoints** (`vllm/entrypoints/`): `llm.py` (offline batch inference), `openai/` (OpenAI-compatible HTTP server), `cli/` (vllm CLI), `grpc_server.py`.
- **v1 Engine** (`vllm/v1/engine/`): `llm_engine.py` (sync), `async_llm.py` (async), `core.py` (EngineCore — runs scheduler, communicates with workers over ZMQ).
- **Scheduler** (`vllm/v1/core/sched/`): Manages request queue, assigns KV cache blocks via `KVCacheManager`, emits `SchedulerOutput` each step.
- **Executor/Worker** (`vllm/v1/executor/`, `vllm/v1/worker/`): Executors manage worker pools. Workers (`gpu_worker.py`, `cpu_worker.py`) each own a model runner. Multi-GPU parallelism (tensor/pipeline/expert) coordinated via `vllm/distributed/`.
- **Model Executor** (`vllm/model_executor/`): `models/` has per-architecture PyTorch implementations. `layers/` has reusable components (attention, fused MoE, quantization). `model_loader/` handles weight loading with strategy classes.
- **Attention / KV Cache** (`vllm/v1/attention/`, `vllm/v1/kv_cache_interface.py`): Paged KV cache with prefix-hash caching. Multiple attention backends (FlashAttention, FlashInfer, etc.) selected via `vllm/v1/attention/selector.py`.
- **Configuration** (`vllm/config/`): Dataclass per concern (`model.py`, `cache.py`, `parallel.py`, `compilation.py`, etc.), aggregated into `VllmConfig` passed throughout the system.
- **C++/CUDA kernels** (`csrc/`): Custom CUDA kernels for attention, cache management, quantization, all-reduce, activation, layernorm, and more. Built via CMake (`CMakeLists.txt`).

### Adding a new model

Model implementations live in `vllm/model_executor/models/`. Each file registers architectures via a registry. Models compose layers from `vllm/model_executor/layers/` (linear, attention, quantization, etc.).

### Multi-modal support

`vllm/multimodal/` handles image/video/audio inputs. Multi-modal models combine vision encoders with language models, with processing pipelines defined per model.

### Platform support

`vllm/platforms/` abstracts hardware differences. Primary target is NVIDIA CUDA; also supports AMD ROCm, CPU, XPU, TPU, and others via plugins.
