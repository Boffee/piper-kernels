# ConvRot INT8 implementation boundaries

`dispatch.py` and `_update.py` validate shared operation contracts and retain the
portable reference fallbacks. Optimized execution enters `_ops.py`, which owns
the existing custom-op names, schemas, mutation declarations, and fake tensor
implementations. Its runtime bodies select implementations through `_backend.py`.
Compiler rewrites emit these same ops, so preparation sharing and activated
preparation use the same selection boundary as eager execution. Fake propagation
does not query hardware or select an implementation.

`_interfaces.LinearBackend` defines ordinary linear, input preparation, and
prepared projection. Prepared storage is contiguous rowwise INT8 data of shape
`[..., K]` plus FP32 scales of shape `[...]`. The logical output dtype is passed
separately. Preparation must not depend on weight identity or output width, so
multiple projections can consume the same prepared input. Implementations must
preserve the existing quantization, bias, and logical-dtype rounding semantics.

The interface also covers caller-owned preparation buffers and column-contiguous,
row-strided projection outputs. A second projection uses equal-shaped independent
weights and writes adjacent columns. Supplied output buffers are returned by
identity, and writes must stay inside the views.

Dense updates, low-rank updates, GGUF conversion, and prepared-input means have
independent selectors. Adding a linear implementation does not enable these
operations. Unsupported updates retain the reference path; unsupported GGUF
conversion and explicitly invoked optimized ops report an error.

`_nvidia/triton.py` owns NVIDIA kernels and launchers, including preparation
strategy and compiler launch options. `_nvidia/policy.py` owns target support,
production tuning, and the `NvidiaExecutionPlan` limits used by the tuner. Shared
`_plan.LinearExecutionPlan` fields only impose structural requirements; another
implementation can use different launch limits and its own compiler options.
The current linear/update gate remains SM75+, while preparation-only GGUF
conversion retains its separate CUDA gate.

`triton.py` keeps existing import paths for the shared custom ops and NVIDIA
kernel utilities used by fusions and benchmarks. `_policy.py` retains the NVIDIA
planning imports used by existing tooling. Offline planning requires an explicit
target for CPU/meta weight tensors. These utility imports do not imply that a
fusion is portable: FFN and attention fusion matching remain NVIDIA-specific.

To integrate another accelerator, implement the linear interface in a separate
package and add its support rule to `_backend.select_linear_backend`. Keep its
preparation kernels, schedules, and compiler options with that implementation.
Add other operation selectors only when those operations are supported. The
shared custom ops and base compiler passes should not need accelerator branches.
Future fusion integration must separately establish consumer compatibility.

Validation includes a reference-backed test implementation exercised through
eager dispatch and real Inductor preparation sharing, plus fake schemas, optional
Triton imports, buffer forwarding, and offline NVIDIA compilation. SM75 paired
INT8 compilation fails in Triton 3.7.1 with an `arith.extf` error; the same failure
was reproduced on untouched upstream `f0223c8` and is recorded as an expected
failure. The support gate is preserved by this refactor. Device correctness and
performance still require GPU validation.
