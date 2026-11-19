# AMD ConvRot INT8 implementation

For public operators and weight conversion, see
[weights](../../../../../../docs/weights.md). Policy, plans, and launchers have
separate ownership:

| Module | Responsibility |
|---|---|
| [policy.py](policy.py) | Select preparation, GEMM tiles, and fast paths from target/shape metadata. |
| [_plan.py](_plan.py) | Immutable AMD execution plans and wave32/tile constraints. |
| [triton.py](triton.py) | Preparation, single/paired GEMMs, and RDNA4 residual epilogue launchers. |
| [shared kernels](../_kernels/triton.py) | Cross-backend INT8 GEMM arithmetic and reusable projection tile. |

Support targets HIP architectures covered by the offline compile tests on Linux
and Windows (`gfx942`, `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`), with hardware
correctness and performance measurements on RX 9070 XT (`gfx1201`). Unknown
targets use the reference path. All paths follow the
[validation contract](../../../../../../docs/development.md#validation-contract).

## Preparation and GEMM schedules

Preparation chooses up to three group-aligned power-of-two segments, fusing
rotation and quantization through 16,384 input features before falling back to a
split path.

On `gfx1201`, baseline GEMMs use `(128, 256, 64)` tiles (8 warps, 2 stages,
`group_m = 8`), switching to `(128, 128, 128)` when `in_features` is a multiple
of 8,192 and the launch fills at least two workgroups per WGP (`64` tiles). For
reductions of 4,096+ features with at least 64 output tiles, the GEMM seeds INT32
accumulators with `0x155555` to avoid signed zero-crossing clock drops and
subtracts the seed before FP32 scaling. Matrix launches also pass LLVM attributes
from [the AMD compiler workarounds](../../../../_triton/amd_compiler_workarounds.py)
so LDS barriers omit redundant `global_inv` fences.

## Paired projections

When projecting one prepared activation into two equal-width weights
(`LinearBackend.linear_prepared(..., second_projection=...)`),
`supports_paired_matmul` selects `_paired_int8_matmul_kernel` on aligned
`gfx1201` `(128, 256, 64)` shapes with `rows > 64` and signed 32-bit element/byte
offsets for SRD buffer addressing. The kernel reuses each `128x64` activation
tile across both `128`-column accumulators.

## Residual epilogues

Aligned `gfx1201` 16-bit projections with `rows > 64` and signed 32-bit SRD
offsets fuse the validated [`ResidualAdd`](../../../../_residual/__init__.py)
epilogue into `_amd_int8_residual_matmul_kernel`. Like SM120 (see
[NVIDIA README](../_nvidia/README.md#residual-epilogues)), the kernel rounds
each projection half to its logical dtype in registers before calling the shared
[Triton residual arithmetic](../../../../_residual/triton.py).

For `256`-column tiles, splitting the epilogue into two sequenced `128`-column
halves (`_store_amd_residual_half`) separated by `_fence_register_value` and
`tl.debug_barrier()` prevents AMDGPU from hoisting residual global loads across
both live `128x256` INT32 accumulators, keeping `128x128` and non-reusable
`256x64` kernels spill-free while keeping reusable-update `256x64` spills
bounded. FP32 outputs, short batches, unaligned dimensions, and oversized
offsets fall back to materialized projection plus `apply_residual_add`.
