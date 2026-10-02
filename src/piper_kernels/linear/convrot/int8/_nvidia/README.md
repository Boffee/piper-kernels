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
