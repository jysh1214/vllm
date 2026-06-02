# `aten::mm` lowering chains — CPU target

`aten::mm` (and its bias-fused sibling `aten::addmm`) is the native ATen GEMM
that `nn.Linear` / `F.linear` decomposes into. In the lowering-path taxonomy
(`notes/lowering-paths.txt`, `notes/lowering-explorer.html`) this is the
**`linear` (Linear / GEMM)** op.

Decomposition prefix shared by every chain below:

```
nn.Linear.forward → F.linear → aten::linear → aten::addmm / aten::mm
```

## Chains that pass through `aten::mm` (source = Native / ATen)

CPU rows #579–585 in `notes/lowering-paths.txt`:

```
#    PLATFORM  OP      COMPMODE     TCBACKEND  SOURCE   MECHANISM
579  cpu       linear  none         na         native   aten_cpu       (eager)
581  cpu       linear  stock        eager      native   aten_cpu
583  cpu       linear  trace_once   eager      native   aten_cpu
585  cpu       linear  vllm         eager      native   aten_cpu
580  cpu       linear  stock        inductor   native   inductor_cpp
582  cpu       linear  trace_once   inductor   native   inductor_cpp
584  cpu       linear  vllm         inductor   native   inductor_cpp
```

Collapsing over compile mode, these are **two distinct terminal chains**:

### 1. ATen-CPU dispatch (eager)

Triggered by `compmode = NONE`, or any compile mode with `backend = eager`.

```
... → aten::addmm / aten::mm
    → ATen-CPU dispatch (torch vec C++)
    → fp32 GEMM → BLAS (MKL on x86 / OpenBLAS on ARM)
      bf16 / int8 GEMM → oneDNN
```

### 2. Inductor → C++/OpenMP (compiled)

Triggered by `backend = inductor`, in any of the `stock` / `trace_once` /
`vllm` compile modes.

```
... → aten::mm  (captured into FX graph)
    → Dynamo / Inductor codegen
    → Inductor → C++/OpenMP
      (vllm/compilation/backends.py:798 — VllmBackend → Inductor → C++/OpenMP;
       Inductor may still emit an extern GEMM template call)
```

## The oneDNN bypass (source = External) — does NOT go through `aten::mm`

CPU rows #636–642. vLLM skips ATen and calls oneDNN directly, so `aten::mm`
never appears. Collapses to **one terminal chain**, invariant across compile
modes since custom/external kernels are opaque to `torch.compile`:

```
636–642  cpu  linear  {none/stock/trace_once/vllm} × {eager/inductor}  external  onednn
    → External: oneDNN (vLLM csrc/cpu/dnnl_kernels.cpp)
```

## Summary

| Source   | Terminal mechanism                         | Dispatched via            | Through `aten::mm`?     | Config variants    |
|----------|--------------------------------------------|---------------------------|-------------------------|--------------------|
| native   | `aten_cpu` (ATen-CPU dispatch → BLAS/oneDNN)| `aten::*` (built-in)      | yes                     | eager paths (4)    |
| native   | `inductor_cpp` (Inductor C++/OpenMP)        | `aten::*` → Inductor      | captured, then codegen'd| inductor paths (3) |
| external | `onednn` (`csrc/cpu/dnnl_kernels.cpp`)       | `torch.ops._C.onednn_mm`  | no (bypassed)           | all 7 rows         |
| custom   | `cpp` SGL packed (`csrc/cpu/sgl-kernels/gemm.cpp`) | `torch.ops._C.weight_packed_linear` | no (bypassed) | 7 rows (Chain 4) |

(The Zen CPU binding — `torch.ops.zentorch.zentorch_linear_unary` — is a fifth,
external-lib custom-op path not yet modeled in the taxonomy.)

Strictly for `aten::mm` on CPU there are **2 distinct lowering chains**
(ATen-CPU dispatch, and Inductor → C++/OpenMP), spanning 7 enumerated config
combinations. The oneDNN and SGL-packed paths are CPU GEMM chains that do *not*
use `aten::mm` — both reach their kernel through a registered `torch.ops._C`
**custom op**, not the built-in ATen dispatcher.

## Real code paths

All three CPU chains share the same layer plumbing. The kernel binding is chosen
**once at weight-load time** and stashed on the layer as `layer.cpu_linear`; the
forward pass is a thin indirection through it.

Shared entry/exit (unquantized linear, `vllm/model_executor/layers/linear.py`):

```
UnquantizedLinearMethod.process_weights_after_loading()      # linear.py:214  (CPU only)
  → dispatch_cpu_unquantized_gemm(layer, remove_weight=True)  # utils.py:222  ← picks the branch
       sets layer.cpu_linear = <one of the bindings below>

UnquantizedLinearMethod.apply()                              # linear.py:220
  → dispatch_unquantized_gemm()        # utils.py:302 → returns cpu_unquantized_gemm on CPU
  → cpu_unquantized_gemm(layer, x, w, bias)                  # utils.py:293
  → layer.cpu_linear(x, weight, bias)                        # utils.py:299
```

`dispatch_cpu_unquantized_gemm` (`vllm/model_executor/layers/utils.py:222`) picks,
in priority order:

1. **Zen CPU** (`is_zen_cpu()`, :236) → `torch.ops.zentorch.zentorch_linear_unary`
   — AMD ZenDNN custom op (external lib via the `zentorch` dispatcher).
2. **SGL packed** (`VLLM_CPU_SGL_KERNEL` + AMX + bf16/int8, :258) →
   `torch.ops._C.weight_packed_linear` — prebuilt CPU custom op (Chain 4 below).
3. **oneDNN** (:270) → `torch.ops._C.onednn_mm` — external lib via custom op (Chain 3).
4. **`F.linear` fallback** (:288) — native ATen (Chains 1 & 2).

Three of the four bindings dispatch through a registered **torch custom op**
(`torch.ops._C.*` or `torch.ops.zentorch.*`); only the `F.linear` fallback uses
the built-in `aten::*` ops. So the only chains that actually execute PyTorch's
own `aten::mm` kernel are Chains 1 & 2 — every other binding bypasses it.

### Chain 1 — ATen-CPU dispatch (native, eager): the `F.linear` fallback

Bound when oneDNN is unavailable/fails and SGL is off (`utils.py:288`):

```python
layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(x, weight, bias)
```

Lowering at runtime (no torch.compile, or compile with `backend=eager`):

```
layer.cpu_linear → F.linear → aten::linear → aten::addmm / aten::mm
  → ATen-CPU dispatch (torch vec C++)
  → fp32 GEMM → BLAS (MKL x86 / OpenBLAS ARM);  bf16 / int8 GEMM → oneDNN (inside ATen)
```

This is the only chain where PyTorch's own `aten::mm` CPU kernel runs the GEMM.

### Chain 2 — Inductor → C++/OpenMP (native, compiled)

Same `F.linear` binding as Chain 1, but the enclosing module is wrapped by
torch.compile, so Dynamo captures `aten::mm`/`aten::addmm` into an FX graph that
Inductor codegens to C++/OpenMP instead of dispatching to ATen.

```
@support_torch_compile                                  # decorators.py:115 → _support_torch_compile:325
  → torch.compile(..., backend=backend)                 # wrapper.py:164
       backend = compilation_config.init_backend(...)   # wrapper.py:96
  CPU forces DYNAMO_TRACE_ONCE + backend="inductor"      # platforms/cpu.py:170–186  ("eager" if VLLM_CPU_CI_ENV)
  → VllmBackend (splits graph, adds post-grad passes)    # backends.py:797
  → InductorStandaloneAdaptor.compile                    # compiler_interface.py:247
  → torch._inductor.compile_fx / standalone_compile      # compiler_interface.py:12, :51
  → Inductor → C++/OpenMP codegen (CPU)
```

**Key nuance — why the binding decides the terminal, not the compile mode.**
`dispatch_cpu_unquantized_gemm` runs at load time, *before* Dynamo traces. If it
bound `onednn_mm` (a `torch.ops._C` custom op), the traced graph contains an
**opaque** custom-op node that Inductor cannot codegen — it stays oneDNN
regardless of compile mode. Inductor only produces a C++/OpenMP GEMM when the
**native `F.linear`** binding (Chain 1) was the one captured. So `backend=eager`
→ Chain 1 (ATen), `backend=inductor` → Chain 2 (Inductor C++), but both require
the native binding to begin with.

### Chain 3 — External: oneDNN (direct, bypasses `aten::mm`)

Bound by default on CPU when supported and arch ≠ POWERPC
(`vllm/model_executor/layers/utils.py:270–285`):

```python
handler = ops.create_onednn_mm(origin_weight.t(), 32)                     # utils.py:276
layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias) # utils.py:277
```

**This is a torch custom op, not a raw FFI call.** Despite the "external" source
label (which describes only what the C++ *body* does — wrap the oneDNN library),
`onednn_mm` is registered in the `_C` library exactly like a prebuilt kernel
(`TORCH_LIBRARY_EXPAND(_C, ops)`, `csrc/cpu/torch_bindings.cpp:201`) and is
invoked through the **torch dispatcher** as `torch.ops._C.onednn_mm`. So its
launch *mechanism* is identical to a `custom`/`cpp` op; the taxonomy's
`external` vs `custom` split only reflects vendor-library wrapper vs.
hand-written kernel inside the registered op. No fake/meta impl is registered
(`vllm/_custom_ops.py`), which is why Dynamo treats it as opaque and the terminal
is invariant across compile modes.

The oneDNN matmul primitive is built **once** from the transposed weight and
captured in the closure (`remove_weight=True` then frees the original weight).
The hot path is just `onednn_mm(handler, x, bias)`. Full descent:

```
ops.onednn_mm(handler, x, bias)                          # _custom_ops.py:3234
  → torch.ops._C.onednn_mm(handler, a, bias, ...)        # _custom_ops.py:3240
  → C++ op (registered)                                  # torch_bindings.cpp:284–286 (impl, torch::kCPU)
  → onednn_mm()                                          # dnnl_kernels.cpp:519
       handler->execute(...)                             # MatMulPrimitiveHandler
  → dnnl::matmul::primitive_desc → dnnl::matmul          # dnnl_helper.h:169, get_matmul_cache
  → matmul.execute(default_stream(), memory_cache_)      # dnnl_helper.cpp:284
       singleton dnnl::engine(cpu) / dnnl::stream        # dnnl_helper.cpp:10–16
       #include "oneapi/dnnl/dnnl.hpp"                    # dnnl_helper.h:7

handler creation:
ops.create_onednn_mm(weight.t(), 32)                     # _custom_ops.py:3220
  → torch.ops._C.create_onednn_mm_handler(...)           # _custom_ops.py:3228
  → create_onednn_mm_handler()                           # dnnl_kernels.cpp:497
  → new MatMulPrimitiveHandler(args)                      # dnnl_kernels.cpp:516
```

Notes:
- The real dispatch decision (oneDNN vs `F.linear`) lives in
  `dispatch_cpu_unquantized_gemm`, not in `dnnl_kernels.cpp`.
- This is the **unquantized** (fp32/bf16) path. The quantized w8a8 path is
  parallel: `vllm/model_executor/kernels/linear/scaled_mm/cpu.py:127,199`
  → `ops.create_onednn_scaled_mm` / `onednn_scaled_mm`
  → `W8A8MatMulPrimitiveHandler` in the same `dnnl_helper`.
- The `FusedMoE` CPU path reuses the same op: `cpu_fused_moe.py:319–353`
  (`ops.create_onednn_mm` / `onednn_mm`, gated by `is_onednn_acl_supported()`).

### Chain 4 — Prebuilt CPU op: SGL packed GEMM (custom / `cpp`, bypasses `aten::mm`)

The highest-priority `_C` binding, taken when `VLLM_CPU_SGL_KERNEL` is set and
the shape qualifies for the AMX kernel (`check_cpu_sgl_kernel`: AMX-tile
support, dtype ∈ {bf16, int8}, `K % 32 == 0`, `N % 16 == 0`,
`vllm/model_executor/layers/utils.py:213`). Bound at `utils.py:258–268`:

```python
packed_weight = torch.ops._C.convert_weight_packed(layer.weight)            # utils.py:259
layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear( # :264
    x, packed_weight, bias_f32 if bias is not None else None, True
)
```

This is a hand-written prebuilt CPU op (`custom` source, `cpp` mechanism), not
oneDNN. Full descent:

```
torch.ops._C.weight_packed_linear(x, packed_weight, bias, True)
  → C++ op (registered)                       # torch_bindings.cpp:346–348 (impl, torch::kCPU)
  → weight_packed_linear()                     # csrc/cpu/sgl-kernels/gemm.cpp
  → hand-written AMX micro-kernel (bf16 / int8)
weight prepack (once, at load):
torch.ops._C.convert_weight_packed(weight)     # torch_bindings.cpp:349–350 → sgl-kernels/gemm.cpp
```

Like oneDNN it is opaque to `torch.compile` (custom op, no fake impl), so the
terminal is invariant across compile modes.

**Taxonomy gap fixed:** `SUPPORT.linear` in `notes/lowering-explorer.html`
previously listed `custom: ["cuda"]` (CUDA-only), so this CPU `custom`/`cpp`
chain was missing from the enumeration. `SUPPORT.linear.custom` now includes
`"cpu"`, and `notes/lowering-paths.txt` has been regenerated: the 7 new CPU rows
are **#601–607** (none/eager, {stock,trace_once,vllm}×{inductor,eager}), bumping
the `linear` section from 78 → 85 chains and the file total from 1261 → 1268.

## Enabling a RISC-V RVV Kernel

"RVV" = the RISC-V Vector extension. The goal is to run the GEMM (ideally the
whole model) with RVV-vectorized kernels on a RISC-V CPU. There is no single
"enable RVV" switch — *where* you add RVV depends on which of the four chains
above you target, and each has very different effort / coverage / control
trade-offs.

Two facts about what vLLM already ships shape every option below:

- **RVV vec abstraction** — `csrc/cpu/cpu_types_riscv*.hpp`, selected via
  `#elif defined(__riscv_v)` in `csrc/cpu/cpu_types.hpp:16`. This is vLLM's own
  `at::vec`-equivalent that the custom CPU ops (activation, layernorm,
  attention, …) build on.
- **Generic ISA-portable micro-GEMM** — `MicroGemm<ISA::VEC, scalar_t>` in
  `csrc/cpu/micro_gemm/cpu_micro_gemm_vec.hpp`, today consumed by
  `cpu_fused_moe.cpp` and `cpu_wna16.cpp`. Templated on the vec abstraction, so
  on a `__riscv_v` build it already picks up the RVV vec types.

By contrast the SGL packed kernel (`sgl-kernels/gemm.cpp`, Chain 4) is **x86
only** (AVX512-bf16 / AMX), gated by `torch.cpu._is_amx_tile_supported()`.

See also `notes/rvv-cpu-chains.txt` for the collapsed CPU/RISC-V chain list.

### Path A — Integrate RVV into OpenBLAS (Chain 1: ATen-CPU dispatch `aten::mm` → BLAS / oneDNN)

The RVV kernel is integrated into **OpenBLAS**, then PyTorch is built for RISC-V
and linked against it; fp32 `aten::mm` then runs the RVV BLAS GEMM. No
vLLM/PyTorch *source* changes.

**Where the RVV kernel actually lives (not in PyTorch, for fp32).** `aten::mm`
does *not* implement the GEMM itself — `at::native::cpublas::gemm` delegates to
`cblas_sgemm` in the linked BLAS, so the fp32 RVV microkernel is **OpenBLAS's**.
"Build PyTorch" here means *build for RISC-V + link the RVV BLAS*, not author an
`aten::mm` RVV kernel. The only genuinely PyTorch-internal RVV lever in this
chain is `at::vec` (ATen's SIMD abstraction — has AVX2/AVX512/NEON/VSX/ZVECTOR
backends but **no RVV** upstream yet), and it only bites two edge subpaths:
(a) bf16/fp16 `mm`, which may use an `at::vec`-based reduced-precision kernel in
`ATen/native/cpu/BlasKernel.cpp` instead of BLAS, and (b) the slow no-BLAS
reference fallback. Adding `at::vec` RVV is the same upstream effort as Path B.
The bf16/int8 branch that routes to oneDNN (ideep) is Path C, not a PyTorch
kernel.

**"Integrate" is usually select/extend, not author from scratch.** OpenBLAS
already ships RVV intrinsic GEMM microkernels under `kernel/riscv64/` for several
targets — `C910V` (RVV 0.7.1), `RISCV64_ZVL128B` / `RISCV64_ZVL256B` (RVV 1.0,
VLEN-specific), `x280` (SiFive). So the realistic work is: (1) build OpenBLAS for
the right `TARGET` with an RVV toolchain (`-march=rv64gcv…` / matching `zvl*`);
(2) verify `sgemm` actually dispatches to the RVV kernel for your core/VLEN and
isn't silently hitting the generic C fallback (RISC-V `DYNAMIC_ARCH` coverage is
limited); (3) only if your VLEN is uncovered or slow, contribute/tune an RVV
microkernel — *that* is the kernel-authoring case; (4) link PyTorch (`BLAS=OpenBLAS`).

- **Pros:**
  - **No PyTorch/vLLM source change** — reuse the existing ATen→BLAS lowering
    (`addmm_impl_cpu_` → `cpublas::gemm` → `cblas_sgemm`); just swap in an
    RVV-enabled OpenBLAS and rebuild/link. Lowest effort of the four paths.
  - **Reuses a mature, autotuned BLAS** — and OpenBLAS often already ships RVV
    GEMM microkernels for several RISC-V targets, so it's usually *select*, not
    *author*.
  - **Broad compile-mode coverage — not eager-only.** The matmul reaches BLAS in
    eager, `backend=eager`, *and* default `backend=inductor` (Inductor defers
    `aten.mm` to the extern `at::mm_out` choice, `kernel/mm.py:130`). Only
    max-autotune's CPP GEMM template (Path B) bypasses it.
  - Benefits *every* `aten::mm` in the process, not just `nn.Linear`.
- **Cons:**
  - **fp32/fp64 only in practice** — OpenBLAS reduced-precision RVV coverage is
    thin (`sbgemm`/bf16 partial, int8 not its domain), so the **bf16/int8 GEMM
    that dominates LLM inference is not helped**; those route to oneDNN (Chain 1's
    other branch / Path C) or vLLM's custom ops, never BLAS.
  - **In vLLM, only active on the `F.linear` fallback binding.** When
    `onednn_mm`/`weight_packed_linear` is bound at load time (the default),
    `aten::mm` never appears and Path A is moot — regardless of compile mode. So
    it rarely sits on vLLM's hot path.
  - **Bounded by OpenBLAS RVV maturity** — RVV GEMM kernels are young; perf /
    dtype / VLEN coverage lags x86, with a real risk of silently falling back to
    the generic C kernel on a TARGET/VLEN mismatch.
  - **No control / no tuning** — you don't own the kernel and can't tune it for
    LLM shapes.
  - **GEMM only** — elementwise / norm / activation aten ops still need `at::vec`
    RVV (Path B) or run scalar.

### Path B — Integrate RVV into Inductor + `at::vec` (Chain 2: Inductor → C++/OpenMP)

This is the example in the task. Inductor's CPU backend emits C++ over
`at::vec::Vectorized<T>` and selects an ISA via a `VecISA` class in
`torch._inductor.codegen.cpp` (`VecAVX512`, `VecAVX2`, `VecNEON`,
`VecZVECTOR`, …). **There is no `VecRVV`.** Enabling RVV here means upstream
PyTorch work: (1) add a `VecRVV` ISA (detection, `-march=…zvl*` flags, vector
width) to Inductor's cpp codegen, and (2) provide an `at::vec` RVV backend so the
emitted `at::vec` calls resolve to RVV. (The "MLIR lowering path" framing applies
to the *experimental* MLIR / triton-cpu Inductor backends; mainline Inductor's
CPU path is C++/`at::vec`, not MLIR — but either way it is PyTorch-core codegen
work, not a vLLM change.)

**This is the only path that lives *inside* PyTorch.** Unlike Path A (kernel in
OpenBLAS) and Path C (kernel in vLLM's fetched oneDNN), here the RVV work is
authored in **PyTorch core** in two places: `torch/_inductor/codegen/cpp.py` (the
`VecRVV` ISA + codegen) and `aten/src/ATen/cpu/vec/` (the `at::vec` RVV backend).
vLLM contributes nothing and changes nothing — once a PyTorch build ships these,
vLLM's default CPU compile path picks RVV up transparently. The flip side: you
have no in-repo lever, and you inherit PyTorch's release/upstreaming cadence.

- **Pros:** Most general — once `VecRVV` + `at::vec` RVV exist, **all** compiled
  native ops get RVV (fusions, elementwise, norms, activations), not just GEMM.
  Biggest end-to-end win, and it matches vLLM's default CPU compile mode
  (`DYNAMO_TRACE_ONCE` + inductor, `platforms/cpu.py:170`). Stays `native`/fusable.
- **Cons:** Large, ongoing **upstream** effort; must track Inductor churn. Does
  *not* by itself give a fast matmul — Inductor defers GEMM to an extern call
  (ATen/MKL) or its CPP GEMM template, so `aten::mm` perf still leans on Path A's
  BLAS or Path D's kernel even after RVV codegen lands. The cheap alternative —
  just pass `-march=rv64gcv` and rely on gcc/clang auto-vectorization of the
  generated scalar C++ — is far less reliable than explicit intrinsics.

### Path C — Integrate RVV into oneDNN (Chain 3: External oneDNN `torch.ops._C.onednn_mm`)

Add RVV GEMM kernels to **oneDNN** (the library vLLM's custom op wraps).

**This is vLLM's oneDNN, not PyTorch's — no PyTorch integration needed.** vLLM
fetches and builds its *own* oneDNN and links it into the custom op:

```cmake
GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git   # cpu_extension.cmake:294 (pinned tag)
FetchContent_MakeAvailable(oneDNN)                         # :334
target_link_libraries(dnnl_ext dnnl torch)                 # :343
```

Chain 3's GEMM goes `torch.ops._C.onednn_mm` → `dnnl_helper.cpp` → **vLLM's
fetched oneDNN** (`dnnl`); the linked `torch` is only for tensor metadata / the
`torch.ops._C` registration boundary — **the matmul compute never enters ATen**
(that is what makes it the `external` chain). So the RVV work lands in (1) the
oneDNN library itself — upstream `oneapi-src/oneDNN`, or a local RVV fork via the
`FETCHCONTENT_SOURCE_DIR_ONEDNN` hook (`cpu_extension.cmake:280`) — and (2) vLLM's
CPU CMake building it for the RISC-V/RVV target. No PyTorch rebuild or source
change.

(There are *two* oneDNN copies in the system: vLLM's, above — Path C; and
PyTorch's bundled oneDNN/ideep, used only by the bf16/int8 branch of Chain 1's
`aten::mm`. Making *that* use RVV is a Chain 1 / Path A concern — a PyTorch
submodule rebuild — not Path C.)

- **Pros:** vLLM already routes CPU linear, w8a8 `scaled_mm`, and MoE through this
  op by default, so RVV in oneDNN would be picked up "for free" on rebuild — the
  handler/primitive-cache infra is already in place. Targets exactly the
  bf16/int8 GEMM LLM inference cares about. Self-contained: the RVV oneDNN is
  vLLM's own fetched copy, so it doesn't depend on (or touch) PyTorch.
- **Cons:** oneDNN is x86-JIT-centric (Xbyak); RISC-V/RVV support is
  experimental/minimal (Xbyak_riscv coverage is partial), and ACL is ARM-only.
  A large, external, not-near-term effort outside your control; even reference
  oneDNN on RISC-V is slow. Heaviest dependency, least control.

### Path D — Integrate an RVV kernel into vLLM `csrc/cpu` (Chain 4: prebuilt custom op)

Write the RVV GEMM **directly in vLLM's `csrc/cpu`**, reusing the RVV vec types
already in-tree. Two sub-targets:

- Add an RVV micro-kernel + RVV gate to the SGL path (`sgl-kernels/gemm.cpp`,
  `check_cpu_sgl_kernel`) — but that file is built around x86 AVX512/AMX.
- **More natural:** the generic `MicroGemm<ISA::VEC, scalar_t>`
  (`micro_gemm/cpu_micro_gemm_vec.hpp`) is already templated on the vec
  abstraction. On a `__riscv_v` build it auto-selects `cpu_types_riscv*.hpp`;
  you mostly ensure the RVV vec type implements the ops it uses, then add a CPU
  dispatch entry so unquantized linear can select it.

**Entirely in vLLM `csrc/cpu` — no PyTorch or upstream-library integration.**
The kernel, the RVV vec abstraction (`cpu_types_riscv*.hpp`), and the micro-GEMM
scaffolding all live in this repo and are built by vLLM's own CPU CMake. It links
`torch` only for the `torch.ops._C` registration boundary and tensor metadata —
the GEMM compute never enters ATen, OpenBLAS, or oneDNN. So unlike Path A
(OpenBLAS), Path B (PyTorch core), and Path C (vLLM's fetched oneDNN), every line
of the RVV work is in-repo and under vLLM's control; the cost is that you own the
kernel and its maintenance outright.

- **Pros:** Fully under vLLM's control; reuses the existing RVV vec abstraction
  and micro-GEMM scaffolding — **no PyTorch/oneDNN upstream dependency**. Tunable
  for LLM bf16/int8 shapes. Lands as a `torch.ops._C` custom op exactly like the
  AMX path (this is the `custom`/`cpp` chain we added — rows #601–607), opaque to
  torch.compile, so it works in every compile mode. Smallest blast radius and the
  fastest route to a *working* RISC-V GEMM given what is already in-tree.
- **Cons:** You must implement and maintain a real GEMM (packing, blocking,
  micro-kernel, tuning) plus a dispatch gate; matching BLAS perf is hard. Covers
  only the ops you write (linear, MoE, wna16) — every other op still needs RVV
  via Path B or its own custom RVV op. New dtype/shape constraints to define
  (analogous to AMX's `K % 32 == 0`, `N % 16 == 0`).

### Recommendation

| Path | Where the work lands | Effort | Coverage | Control |
|------|----------------------|--------|----------|---------|
| A — OpenBLAS-RVV (Chain 1) | build/dep only | low | fp32 GEMM only | none |
| B — Inductor `VecRVV` + `at::vec` (Chain 2) | upstream PyTorch | very high | **all native ops** (but defers GEMM) | low |
| C — oneDNN-RVV (Chain 3) | upstream oneDNN | very high | bf16/int8 GEMM | none |
| D — vLLM csrc RVV micro-GEMM (Chain 4) | vLLM `csrc/cpu` | medium | the ops you write | **full** |

- **Fastest working RVV GEMM in vLLM today: Path D** via the in-tree
  `MicroGemm<ISA::VEC>` + `cpu_types_riscv*.hpp`, wired into
  `dispatch_cpu_unquantized_gemm` as a new `custom`/`cpp` binding (the chain at
  rows #601–607).
- **Best whole-model coverage: Path B** (RVV in Inductor + `at::vec`), upstream —
  the only path that vectorizes everything, but it still needs A or D for the
  matmul itself.
- **Path A** is a reasonable zero-code fp32 stopgap; **Path C** is impractical
  near-term.

## Sources

- `notes/lowering-paths.txt` — `linear` section, lines 590–675 (85 chains).
- `notes/lowering-explorer.html` — rule engine: `mechOf()` (line 338),
  `SUPPORT.linear` (line 407), mechanism labels (lines 280–300).
- `csrc/cpu/dnnl_kernels.cpp` — oneDNN GEMM integration.
- `vllm/compilation/backends.py:798` — VllmBackend → Inductor → C++/OpenMP.
