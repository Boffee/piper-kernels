"""Shared launch-plan values for ConvRot INT8 and ConvRot Trellis implementations."""

from dataclasses import asdict, dataclass, fields

from piper_kernels.linear._splitk import select_split_k

TRELLIS_A16_MAX_BLOCK_M = 8
TRELLIS_DEFAULT_GEMV_MAX_ROWS = 4
TRELLIS_DEFAULT_SCRATCH_DECODE_MAX_BYTES = 32 * 1024 * 1024
TRELLIS_LARGE_SCRATCH_DECODE_MAX_BYTES = 64 * 1024 * 1024
TRELLIS_LARGE_SCRATCH_MIN_ROWS = 1024
TRELLIS_SPLITK_MIN_WEIGHT_BYTES = 2 * 1024 * 1024
_TRELLIS_SPLITK_MAX = 8


@dataclass(frozen=True, slots=True)
class TrellisLinearSchedule:
    """Resolved ConvRot Trellis A16 GEMV execution schedule for one target and workload."""

    block_m: int
    num_warps: int
    split_k: int
    unroll_k: int
    use_dp4a: bool
    block_n: int = 1
    use_sad_u8: bool = False
    split_x_loads: bool = False
    num_stages: int = 1


def trellis_block_m(rows: int) -> int:
    """Return the padded GEMV row tile covering ``rows``."""
    if rows == 1:
        return 1
    if rows == 2:
        return 2
    if rows <= 4:
        return 4
    return TRELLIS_A16_MAX_BLOCK_M


def select_generic_trellis_split_k(
    num_sms: int,
    rows: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
    block_m: int,
    *,
    block_n: int = 1,
) -> int:
    """Compute generic Split-K factor for bounded small-batch Trellis A16 GEMV."""
    if rows > TRELLIS_A16_MAX_BLOCK_M:
        # Large-M fallback (devices without an optimized W8A8 backend) already fills SMs on M x N.
        return 1
    output_tiles = ((out_features + block_n - 1) // block_n) * ((rows + block_m - 1) // block_m)
    return select_split_k(
        num_sms,
        output_tiles=output_tiles,
        k_tiles=blocks_per_row,
        weight_bytes=weight_bytes,
        min_weight_bytes=TRELLIS_SPLITK_MIN_WEIGHT_BYTES,
        max_split_k=_TRELLIS_SPLITK_MAX,
        cta_multiplier=4,
        min_k_tiles=4,
        min_tiles_per_split=2,
    )


def select_default_trellis_schedule(
    num_sms: int,
    rows: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
    *,
    use_dp4a: bool = False,
) -> TrellisLinearSchedule:
    """Build the default conservative Trellis A16 GEMV schedule."""
    block_m = trellis_block_m(rows)
    split_k = select_generic_trellis_split_k(
        num_sms, rows, out_features, blocks_per_row, weight_bytes, block_m
    )
    return TrellisLinearSchedule(
        block_m=block_m,
        num_warps=1,
        split_k=split_k,
        unroll_k=1,
        use_dp4a=use_dp4a,
    )


def default_trellis_w8a8_scratch_budget(rows: int) -> int:
    """Return default bounded INT8 scratch decode budget for ``rows``."""
    return (
        TRELLIS_LARGE_SCRATCH_DECODE_MAX_BYTES
        if rows >= TRELLIS_LARGE_SCRATCH_MIN_ROWS
        else TRELLIS_DEFAULT_SCRATCH_DECODE_MAX_BYTES
    )


@dataclass(frozen=True, slots=True)
class LinearExecutionPlan:
    """Host-side preparation and GEMM choices for one ConvRot INT8 invocation."""

    fuse_rotation_quantization: bool
    fused_num_warps: int
    rotation_num_warps: int
    quantization_num_warps: int
    matmul_block_m: int
    matmul_block_n: int
    matmul_block_k: int
    matmul_num_warps: int = 4
    matmul_num_stages: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.fuse_rotation_quantization, bool):
            raise ValueError("ConvRot preparation fusion choice must be boolean")
        for field in fields(LinearExecutionPlan):
            name, value = field.name, getattr(self, field.name)
            if name == "fuse_rotation_quantization":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ConvRot launch dimension {name} must be a positive integer")

    def as_dict(self) -> dict[str, int | bool | str]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)
