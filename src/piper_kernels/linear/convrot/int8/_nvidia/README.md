# NVIDIA ConvRot INT8 implementation

For public operators and weight conversion, see
[weights](../../../../../../docs/weights.md). Policy, orchestration, and launchers
have separate ownership:

| Module | Responsibility |
|---|---|
| [policy.py](policy.py) | Select preparation and SM8x/SM120 GEMM schedules from target/shape metadata. |
| [_plan.py](_plan.py) | Immutable plans/schedules, configuration validation, effective-choice reporting. |
| [dispatch.py](dispatch.py) | Prepare shared operands, allocate/reuse outputs, dispatch GEMM. |
| [triton.py](triton.py) | Shared preparation and portable Triton GEMM launchers. |
| [gluon_async_copy.py](gluon_async_copy.py) | `cp.async`/MMAv2 GEMM, alignment requirements, launch. |
| [shared kernels](../_kernels/triton.py) | NVIDIA/AMD INT8 arithmetic and reusable projection tile. |

Preparation depends on target family and input width, independently of output
width so projections can share it. GEMM selection returns a complete
`MatmulSchedule`; both choices form one `NvidiaExecutionPlan`. Production does
not construct and rewrite intermediate plans.

`matmul_kernel` explicitly selects `triton` or `gluon_async_copy`. Replacing a
tile with `dataclasses.replace` never changes the implementation. Plans validate
supported tile geometry; the offline tuner records unsupported combinations.
Kernel compatibility alone does not select production policy: an implementation
can be valid on a target without being the measured choice there.

`matmul_group_m` controls cache grouping, with zero meaning ungrouped.
`triton_specialize_m` decides whether row alignment enters the JIT cache key;
dynamic-M launches branch per tile. `triton_explicit_bias_fma` preserves
scale/bias rounding, independently of tail scheduling. Gluon and its unaligned
Triton fallback use dynamic M and explicit bias FMAs to retain output bits.

The read-only `matmul_specialize_m` and `matmul_explicit_bias_fma` properties,
and `as_dict()`, report the selected implementation's effective behavior.
Switching implementations retains stored Triton options; the alignment fallback
selects its own rounding-compatible options.

Async-copy launches require 16-byte-aligned INT8 rows. Dispatch checks pointer
and stride metadata and uses the grouped 128x64 Triton fallback when needed,
retaining preparation choices. Neither implementation scans tensor contents;
follow the [validation contract](../../../../../../docs/development.md#validation-contract).
For candidate comparisons and workload coverage, see the
[benchmark guide](../../../../../../benchmarks/README.md#workloads).

## Residual epilogues

Dense/sparse attention output projections and GELU/SwiGLU FFN down projections
pass a validated [`ResidualAdd`](../../../../_residual/__init__.py) to
[`LinearBackend.linear_prepared_residual`](../_interfaces.py), whose docstrings own
the operand, storage, and chunking contract. Ordinary and paired projections retain
their separate `linear_prepared` operation.

For FP16/BF16 outputs on SM120, the Triton GEMM adds the residual before storing
the final output. There is no materialized projection or separate residual launch.
The projection rounds to its logical dtype in registers before FP32
residual arithmetic, preserving the materialized projection's rounding boundary.
FP32 multiply-add contraction may still change last bits. The final store rounds
to the residual's dtype; no full-size FP32 tensor is allocated.

Retaining the projection in FP32 increased register pressure enough to spill in
the current 128x256 SM120 tile. RTX 5090 measurements found full attention blocks
up to about 1.6% slower and 1–5% slower FFNs versus the rounded epilogue. Splitting the
epilogue into smaller pieces removed spills but remained slower. Keep the rounding
until a different schedule demonstrates a benefit across representative workloads.

Plain and residual kernels share projection and store helpers. Their separate
entry points keep residual arguments out of ordinary GEMM launches. Backend policy
owns the epilogue choice. FP32 outputs, other NVIDIA targets, and AMD retain the
materialized projection plus standalone residual kernel, and NVFP4 fusions retain
their existing path. The same [Triton residual arithmetic](../../../../_residual/triton.py)
serves native and standalone epilogues; fusion code does not select a GEMM kernel
or reproduce its arithmetic. The native path has been validated on RTX 5090.
