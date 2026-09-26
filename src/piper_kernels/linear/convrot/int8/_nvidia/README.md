# NVIDIA ConvRot INT8 implementation

The organization follows sparse Piper attention's separation of target policy, common
orchestration, and implementation-owned launchers:

| Module | Responsibility |
|---|---|
| `policy.py` | Validate NVIDIA plans and select measured SM8x/SM120 schedules from host metadata. |
| `dispatch.py` | Prepare shared operands, allocate or reuse outputs, and dispatch the selected GEMM. |
| `triton.py` | Launch shared preparation kernels and the portable Triton GEMM. |
| `gluon_async_copy.py` | Own the `cp.async`/MMAv2 GEMM, alignment requirements, and launch. |
| `../_kernels/triton.py` | Shared NVIDIA/AMD INT8 arithmetic and reusable projection tile. |

Both target policies return the same `NvidiaExecutionPlan`. Kernel selection is explicit:
`matmul_kernel="triton"` or `"gluon_async_copy"`. Changing a tile with `dataclasses.replace`
never changes its implementation. Plan validation checks the selected kernel's supported
tile geometry; unsupported tuning combinations are reported and skipped by the offline tuner.

`matmul_group_m` controls cache grouping, with zero meaning ungrouped. Triton's
`matmul_specialize_m` controls whether row alignment enters the JIT cache key. Dynamic-M
launches branch per tile instead. `matmul_explicit_bias_fma` preserves the scale/bias rounding
when the compiler would otherwise separate those operations across a tail branch. It does
not control tail scheduling. Gluon and its unaligned-operand Triton fallback always use
dynamic M and explicit bias FMAs to retain the same output bits.

SM8x selects its measured five configurations, dynamic-M launches, and explicit FMAs.
SM120 retains its three Triton schedules, grouping, row specialization, and preparation
defaults. The async-copy implementation is reusable on both architectures; correctness tests
exercise the SM8x schedules on SM120 as well. Tuning results, rather than kernel compatibility,
determine production selection.

An async-copy launch requires 16-byte-aligned INT8 rows. Dispatch checks pointer/stride metadata
and selects the grouped 128x64 Triton fallback when necessary. Neither implementation performs
tensor-content validation; both obey the library's [validation contract](../../../../../../README.md#validation-contract).
