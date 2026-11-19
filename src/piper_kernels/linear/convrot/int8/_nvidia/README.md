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
| [gluon_tma.py](gluon_tma.py) | SM120 TMA/MMAv2 GEMM and chunked residual epilogue, alignment requirements, launch. |
| [shared kernels](../_kernels/triton.py) | NVIDIA/AMD INT8 arithmetic and reusable projection tile. |

Preparation depends on target family and input width, independently of output
width so projections can share it. GEMM selection returns a complete
`MatmulSchedule`; both choices form one `NvidiaExecutionPlan`. Production does
not construct and rewrite intermediate plans.

`matmul_kernel` explicitly selects `triton`, `gluon_async_copy`, or `gluon_tma`.
Replacing a tile with `dataclasses.replace` never changes the implementation.
Plans validate supported tile geometry; the offline tuner records unsupported
combinations. Kernel compatibility alone does not select production policy: an
implementation can be valid on a target without being the measured choice there.
SM120 selects `gluon_tma` (`128x256x64`, 8 warps, 3 stages, `group_m=16`) for
large tiles while keeping Triton for small/medium tiles; other supported targets
outside SM8x retain the `triton` baseline schedule.

`matmul_group_m` controls cache grouping, with zero meaning ungrouped.
`triton_specialize_m` decides whether row alignment enters the JIT cache key;
dynamic-M launches branch per tile. `triton_explicit_bias_fma` preserves
scale/bias rounding, independently of tail scheduling. Gluon implementations
(`gluon_async_copy` and `gluon_tma`) and their unaligned Triton fallback use
dynamic M and explicit bias FMAs to retain output bits.

The read-only `matmul_specialize_m` and `matmul_explicit_bias_fma` properties,
and `as_dict()`, report the selected implementation's effective behavior.
Switching implementations retains stored Triton options; the alignment fallback
selects its own rounding-compatible options.

Async-copy launches require 16-byte-aligned INT8 row starts and strides. TMA
launches require contiguous 2-D INT8 operands whose pointers and feature width
`K` are positive 16-byte multiples (TMA hardware zero-fills out-of-bounds M, N,
and K). Dispatch checks pointer, shape, and layout metadata and uses the
grouped 128x64 Triton fallback for both plain/paired and fused-residual paths
when operands do not satisfy the selected Gluon kernel's contract, retaining
preparation choices. Neither implementation scans tensor contents;
follow the [validation contract](../../../../../../docs/development.md#validation-contract).
For candidate comparisons and workload coverage, see the
[benchmark guide](../../../../../../benchmarks/README.md#workloads).

## Residual epilogues

Dense/sparse attention output projections and GELU/SwiGLU FFN down projections
pass a validated [`ResidualAdd`](../../../../_residual/__init__.py) to
[`LinearBackend.linear_prepared_residual`](../_interfaces.py), whose docstrings own
the operand, storage, and chunking contract. Ordinary and paired projections retain
their separate `linear_prepared` operation.

For FP16/BF16 outputs on SM120, both the small/medium Triton GEMM and the large
`gluon_tma` GEMM add the residual before storing the final output. There is no
materialized projection or separate residual launch. In `gluon_tma`, the
rounded 16-bit projection tile is staged through shared memory (reusing the TMA
ring allocation) and processed in 32-row chunks so single- and two-update
(`ReusableUpdate`) epilogues stay free of register spills. Both paths round the
projection to its logical dtype before FP32 residual arithmetic, preserving the
materialized projection's rounding boundary. FP32 multiply-add contraction may
still change last bits. The final store rounds to the residual's dtype; no
full-size FP32 tensor is allocated.

Retaining the projection in FP32 increased register pressure enough to spill in
the SM120 `128x256` Triton tile. RTX 5090 measurements found full attention blocks
up to about 1.6% slower and 1–5% slower FFNs versus the rounded epilogue. Splitting the
epilogue into smaller pieces in Triton removed spills but remained slower. Keep the
rounding until a different schedule demonstrates a benefit across representative
workloads.

Plain and residual kernels share projection and store helpers. Their separate
entry points keep residual arguments out of ordinary GEMM launches. Backend policy
owns the epilogue choice: SM120 fuses 16-bit projections, RDNA4 `gfx1201` (see the
[AMD README](../_amd/README.md#residual-epilogues)) fuses aligned 16-bit
projections, and FP32 outputs, other targets, and unaligned shapes retain the
materialized projection plus standalone residual kernel (while NVFP4 fusions
retain their existing path). Triton native and standalone epilogues share
[Triton residual arithmetic](../../../../_residual/triton.py), and
`gluon_tma.py` mirrors the same FP32 arithmetic in Gluon; fusion code does not
select a GEMM kernel or reproduce its arithmetic. The native paths have been
validated on RTX 5090 and RX 9070 XT.
