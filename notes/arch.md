# vLLM Architecture

Diagrams of how vLLM is structured and how it sits on top of PyTorch. The
codebase runs on the **v1 engine** as the production path; legacy
`vllm/engine/` is mostly thin re-export aliases into `vllm/v1/`.

## 1. Request flow (top to bottom)

From a user request down to a GPU forward pass.

```mermaid
flowchart TD
    subgraph EP["Entrypoints — vllm/entrypoints/"]
        LLM["LLM class<br/>(offline batch)"]
        API["OpenAI API server<br/>(openai/)"]
        CLI["vllm CLI"]
    end

    EC["EngineClient<br/>(protocol in vllm/engine/protocol.py)"]

    subgraph CORE["v1 EngineCore — vllm/v1/engine/core.py (ZMQ scheduler loop)"]
        SCHED["Scheduler<br/>vllm/v1/core/sched/scheduler.py"]
        KVM["KVCacheManager<br/>(paged blocks + prefix hashing)"]
    end

    EXEC["Executor<br/>multiproc / Ray / uniproc<br/>vllm/v1/executor/"]

    subgraph GPU["Per-GPU (xN)"]
        WORKER["Worker<br/>vllm/v1/worker/gpu_worker.py"]
        MR["ModelRunner<br/>(forward pass, attention backends)"]
        MODEL["Model — nn.Module<br/>vllm/model_executor/models/"]
    end

    LLM --> EC
    API --> EC
    CLI --> EC
    EC --> CORE
    SCHED <--> KVM
    SCHED -->|SchedulerOutput per step| EXEC
    EXEC --> WORKER
    WORKER --> MR
    MR --> MODEL
    MODEL -.->|logits / sampled tokens| MR
    MR -.->|ModelRunnerOutput| EXEC
    EXEC -.-> CORE
    CORE -.-> EC
```

## 2. Component / module map

The major packages and what each owns.

```mermaid
graph LR
    subgraph SERVE["Serving layer (vLLM-specific)"]
        E["entrypoints/"]
        V1["v1/engine/<br/>(sync + async)"]
        S["v1/core/sched/<br/>scheduler + KV manager"]
        X["v1/executor/ + v1/worker/"]
    end

    subgraph EXECML["Model execution"]
        ME["model_executor/models/<br/>(per-architecture nn.Module)"]
        LAYERS["model_executor/layers/<br/>(attention, fused MoE,<br/>quantization, linear)"]
        LOADER["model_executor/model_loader/<br/>(weight loading strategies)"]
    end

    subgraph LOWLVL["Low level"]
        ATTN["v1/attention/<br/>(backend selector)"]
        KV["paged KV cache<br/>(prefix-hash blocks)"]
        CSRC["csrc/<br/>(CUDA / C++ kernels)"]
    end

    subgraph CROSS["Cross-cutting"]
        CFG["config/<br/>→ VllmConfig"]
        DIST["distributed/<br/>(TP / PP / EP)"]
        COMP["compilation/<br/>(VllmBackend for torch.compile)"]
    end

    E --> V1 --> S --> X --> ME
    ME --> LAYERS --> ATTN --> KV
    LAYERS --> CSRC
    LOADER --> ME
    CFG -.->|threaded everywhere| V1
    CFG -.-> X
    DIST -.-> X
    COMP -.-> ME
```

## 3. The EngineCore step loop

`EngineCore` runs a continuous-batching loop: each step the scheduler
selects a batch, the executor runs one forward pass, and finished/streaming
tokens flow back. Requests join and leave the running batch at any step —
they do not wait for a whole batch to finish.

```mermaid
sequenceDiagram
    participant C as EngineClient
    participant Core as EngineCore
    participant Sched as Scheduler
    participant Exec as Executor
    participant W as Worker / ModelRunner

    C->>Core: add_request (ZMQ)
    loop every step
        Core->>Sched: schedule()
        Sched->>Sched: pick running batch,<br/>allocate KV blocks
        Sched-->>Core: SchedulerOutput
        Core->>Exec: execute_model(SchedulerOutput)
        Exec->>W: forward pass (1 step)
        W-->>Exec: ModelRunnerOutput<br/>(sampled tokens)
        Exec-->>Core: outputs
        Core->>Sched: update_from_output()<br/>(free finished, repair KV)
        Core-->>C: stream tokens / finished reqs
    end
```

## 4. vLLM on top of PyTorch

vLLM is a framework built **on top of** PyTorch, not a replacement. PyTorch
provides the tensor/kernel/compiler substrate; vLLM adds scheduling, paged
memory management, and serving. The two connect at four well-defined seams.

```mermaid
flowchart TB
    subgraph VLLM["vLLM (serving engine)"]
        SCHEDV["Scheduler + continuous batching"]
        PAGED["Paged KV cache"]
        MODELS["Models as nn.Module<br/>(model_executor/models/)"]
        VBACK["VllmBackend<br/>(vllm/compilation/)"]
        OPS["_custom_ops.py wrappers"]
    end

    subgraph TORCH["PyTorch (substrate)"]
        NNMOD["torch.nn.Module"]
        TLIB["torch.ops / torch.library<br/>(custom op registration)"]
        COMPILE["torch.compile / Dynamo / FX"]
        RT["tensors, dtypes, CUDA streams,<br/>torch.distributed"]
    end

    KERNELS["csrc/ CUDA kernels<br/>compiled into a torch extension"]

    MODELS -->|subclass| NNMOD
    OPS -->|"torch.ops._C.paged_attention_v1(...)"| TLIB
    KERNELS -->|registered as ops| TLIB
    VBACK -->|"plugs in as a backend"| COMPILE
    SCHEDV -.->|uses| RT
    PAGED -.->|uses| RT
    MODELS -.->|uses| RT
```

### Who owns what

| Concern | Owner |
|---|---|
| Tensor ops, dtypes, CUDA streams, distributed comm | **PyTorch** |
| Model definitions (`nn.Module`), forward-pass graph | **Both** — PyTorch modules, vLLM-specific layers |
| Custom kernels (paged attention, fused MoE, quant) | **vLLM**, exposed *as* PyTorch ops via `torch.ops` / `torch.library` |
| Graph compilation | **vLLM backend** plugged into **PyTorch's** `torch.compile` |
| Scheduling, continuous batching, paged KV cache, serving | **vLLM** |

> **Mental model:** PyTorch gives vLLM the tensor/kernel/compiler machinery;
> vLLM uses it to build a serving engine that PyTorch alone doesn't provide.
> vLLM extends PyTorch through its own extension points — custom ops, compile
> backends, and `nn.Module` — rather than forking it.

## Pointers into the code

- `vllm/entrypoints/` — `llm.py`, `openai/`, `cli/`
- `vllm/v1/engine/core.py` — EngineCore, the ZMQ scheduler loop
- `vllm/v1/core/sched/scheduler.py` — scheduler + `KVCacheManager`
- `vllm/v1/executor/`, `vllm/v1/worker/` — executors and workers
- `vllm/model_executor/models/` — per-architecture `nn.Module` impls
- `vllm/model_executor/layers/` — reusable attention / MoE / quant layers
- `vllm/_custom_ops.py` — Python wrappers over `torch.ops._C.*` kernels
- `vllm/compilation/backends.py` — `VllmBackend` for `torch.compile`
- `csrc/` — CUDA / C++ kernel sources
- `vllm/config/` — per-concern dataclasses aggregated into `VllmConfig`
