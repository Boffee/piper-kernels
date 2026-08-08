"""Benchmark folding H3 gated residual updates into ConvRot INT8 GEMM."""

# The experimental JIT kernel mirrors Triton's launcher-style production
# signature, whose compile-time arguments are not represented to type checkers.
# pyright: reportCallIssue=false
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from lib import Timing, triton_benchmark

from piper_kernels.convrot.int8.backends import triton as baseline


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    m: int
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    split_epilogue: bool = False

    def display(self) -> str:
        suffix = ",split" if self.split_epilogue else ""
        return (
            f"{self.block_m}x{self.block_n}x{self.block_k},"
            f"s{self.num_stages},w{self.num_warps}{suffix}"
        )


_SHAPES = {
    shape.name: shape
    for shape in (
        Shape("attention-out", 37_710, 5_376, 7_168),
        Shape("mlp-fc2", 37_710, 5_376, 14_336),
    )
}

# The first entry consumes the full epilogue tile. The second consumes two
# N-halves serially to trade a layout conversion for lower peak liveness.
_CONFIGS = (
    Config(128, 256, 128, 3, 8),
    Config(128, 256, 128, 3, 8, split_epilogue=True),
)


@triton.jit
def _store_gated_residual_half(
    accumulator,
    activation_scale,
    weight_scale_ptr,
    residual_ptr,
    gate_ptr,
    row_ids_ptr,
    output_ptr,
    offsets_m,
    pid_n,
    m,
    n,
    stride_om,
    stride_on,
    stride_rm,
    stride_rn,
    stride_gr,
    stride_gn,
    half: tl.constexpr,
    block_n: tl.constexpr,
):
    """Consume half an accumulator tile to reduce peak epilogue liveness."""
    half_n: tl.constexpr = block_n // 2
    offsets_n = pid_n * block_n + half * half_n + tl.arange(0, half_n)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    row_mask = offsets_m < m
    mask = row_mask[:, None] & (offsets_n[None, :] < n)
    weight_scale = tl.load(
        weight_scale_ptr + offsets_n,
        mask=offsets_n < n,
        other=0.0,
    )
    # Preserve the materialized projection boundary before addcmul.
    projection = (
        accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
    ).to(tl.bfloat16)
    residual_pointers = (
        residual_ptr + offsets_m_i64[:, None] * stride_rm + offsets_n_i64[None, :] * stride_rn
    )
    residual = tl.load(residual_pointers, mask=mask, other=0.0).to(tl.float32)
    gate_rows = tl.load(row_ids_ptr + offsets_m, mask=row_mask, other=0).to(tl.int64)
    gate = tl.load(
        gate_ptr + gate_rows[:, None] * stride_gr + offsets_n_i64[None, :] * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = residual + projection.to(tl.float32) * gate
    output_pointers = (
        output_ptr + offsets_m_i64[:, None] * stride_om + offsets_n_i64[None, :] * stride_on
    )
    tl.store(output_pointers, result.to(tl.bfloat16), mask=mask)


@triton.jit
def _int8_matmul_gated_residual_kernel(  # noqa: PLR0912, PLR0915
    activation_ptr,
    weight_ptr,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    residual_ptr,
    gate_ptr,
    row_ids_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_rm,
    stride_rn,
    stride_gr,
    stride_gn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    even_m: tl.constexpr,
    even_n: tl.constexpr,
    even_k: tl.constexpr,
    group_m: tl.constexpr,
    gated_residual: tl.constexpr,
    split_epilogue: tl.constexpr,
):
    """Mirror production GEMM and optionally apply its exact BF16 epilogue."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    actual_group_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % num_pid_in_group) % actual_group_m
    pid_n = (pid % num_pid_in_group) // actual_group_m
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)

    activation_pointers = (
        activation_ptr + offsets_m_i64[:, None] * stride_am + offsets_k_i64[None, :] * stride_ak
    )
    weight_pointers = (
        weight_ptr + offsets_n_i64[None, :] * stride_wn + offsets_k_i64[:, None] * stride_wk
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k_offset in range(tl.cdiv(k, block_k)):
        if even_m and even_k:
            activation = tl.load(activation_pointers)
        else:
            activation = tl.load(
                activation_pointers,
                mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k - k_offset * block_k),
                other=0,
            )
        if even_n and even_k:
            weight = tl.load(weight_pointers)
        else:
            weight = tl.load(
                weight_pointers,
                mask=(offsets_n[None, :] < n) & (offsets_k[:, None] < k - k_offset * block_k),
                other=0,
            )
        accumulator += tl.dot(activation, weight)
        activation_pointers += block_k * stride_ak
        weight_pointers += block_k * stride_wk

    if even_m:
        activation_scale = tl.load(activation_scale_ptr + offsets_m)
    else:
        activation_scale = tl.load(
            activation_scale_ptr + offsets_m,
            mask=offsets_m < m,
            other=0.0,
        )
    if gated_residual and split_epilogue:
        grouped = tl.reshape(accumulator, (block_m, 2, block_n // 2))
        paired = tl.permute(grouped, (0, 2, 1))
        first_half, second_half = tl.split(paired)
        _store_gated_residual_half(
            first_half,
            activation_scale,
            weight_scale_ptr,
            residual_ptr,
            gate_ptr,
            row_ids_ptr,
            output_ptr,
            offsets_m,
            pid_n,
            m,
            n,
            stride_om,
            stride_on,
            stride_rm,
            stride_rn,
            stride_gr,
            stride_gn,
            0,
            block_n,
        )
        _store_gated_residual_half(
            second_half,
            activation_scale,
            weight_scale_ptr,
            residual_ptr,
            gate_ptr,
            row_ids_ptr,
            output_ptr,
            offsets_m,
            pid_n,
            m,
            n,
            stride_om,
            stride_on,
            stride_rm,
            stride_rn,
            stride_gr,
            stride_gn,
            1,
            block_n,
        )
        return
    if even_n:
        weight_scale = tl.load(weight_scale_ptr + offsets_n)
    else:
        weight_scale = tl.load(
            weight_scale_ptr + offsets_n,
            mask=offsets_n < n,
            other=0.0,
        )
    result = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]

    output_pointers = (
        output_ptr + offsets_m_i64[:, None] * stride_om + offsets_n_i64[None, :] * stride_on
    )
    mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
    if gated_residual:
        # Preserve the materialized projection boundary exactly: production
        # first stores the scaled projection as BF16, and the following op
        # promotes that BF16 value for its multiply-add.
        result = result.to(tl.bfloat16)
        residual_pointers = (
            residual_ptr + offsets_m_i64[:, None] * stride_rm + offsets_n_i64[None, :] * stride_rn
        )
        residual = tl.load(residual_pointers, mask=mask, other=0.0).to(tl.float32)
        gate_rows = tl.load(
            row_ids_ptr + offsets_m,
            mask=offsets_m < m,
            other=0,
        ).to(tl.int64)
        gate = tl.load(
            gate_ptr + gate_rows[:, None] * stride_gr + offsets_n_i64[None, :] * stride_gn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        result = residual + result.to(tl.float32) * gate

    if even_m and even_n:
        tl.store(output_pointers, result.to(tl.bfloat16))
    else:
        tl.store(output_pointers, result.to(tl.bfloat16), mask=mask)


@triton.jit
def _gated_residual_kernel(
    projection_ptr,
    residual_ptr,
    gate_ptr,
    row_ids_ptr,
    output_ptr,
    n,
    numel,
    stride_gr,
    stride_gn,
    block_size: tl.constexpr,
):
    """Materialized BF16 equivalent of residual + projection * gate[:, None]."""
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    offsets_i64 = offsets.to(tl.int64)
    rows = offsets // n
    columns = offsets - rows * n
    projection = tl.load(projection_ptr + offsets_i64, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets_i64, mask=mask, other=0.0).to(tl.float32)
    gate_rows = tl.load(row_ids_ptr + rows, mask=mask, other=0).to(tl.int64)
    gate = tl.load(
        gate_ptr + gate_rows * stride_gr + columns * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = residual + projection * gate
    tl.store(output_ptr + offsets_i64, result.to(tl.bfloat16), mask=mask)


def _production_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> Callable[[], None]:
    m, k = activation.shape
    n = weight.shape[0]
    block_m, block_n, block_k, stages, warps = baseline._int8_matmul_config(
        m,
        n,
        k,
        True,
    )
    num_m_tiles = triton.cdiv(m, block_m)
    num_n_tiles = triton.cdiv(n, block_n)
    group_m = 1 if num_n_tiles <= 32 else 64

    def launch() -> None:
        baseline._int8_matmul_kernel[(num_m_tiles * num_n_tiles,)](
            activation,
            weight,
            output,
            activation_scale,
            weight_scale,
            output,
            m,
            n,
            k,
            activation.stride(0),
            activation.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            has_bias=False,
            even_m=m % block_m == 0,
            even_n=n % block_n == 0,
            even_k=k % block_k == 0,
            group_m=group_m,
            num_stages=stages,
            num_warps=warps,
            num_ctas=1,
        )

    return launch


def _standalone_launcher(
    projection: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    row_ids: torch.Tensor,
    output: torch.Tensor,
) -> Callable[[], None]:
    _, n = projection.shape
    numel = projection.numel()
    block_size = 1_024

    def launch() -> None:
        _gated_residual_kernel[(triton.cdiv(numel, block_size),)](
            projection,
            residual,
            gate,
            row_ids,
            output,
            n,
            numel,
            gate.stride(0),
            gate.stride(1),
            block_size=block_size,
            num_warps=8,
        )

    return launch


def _fused_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    row_ids: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config: Config,
    *,
    gated_residual: bool = True,
) -> Callable[[], None]:
    m, k = activation.shape
    n = weight.shape[0]
    num_m_tiles = triton.cdiv(m, config.block_m)
    num_n_tiles = triton.cdiv(n, config.block_n)
    group_m = 1 if num_n_tiles <= 32 else 64

    def launch() -> None:
        _int8_matmul_gated_residual_kernel[(num_m_tiles * num_n_tiles,)](
            activation,
            weight,
            output,
            activation_scale,
            weight_scale,
            residual,
            gate,
            row_ids,
            m,
            n,
            k,
            activation.stride(0),
            activation.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            residual.stride(0),
            residual.stride(1),
            gate.stride(0),
            gate.stride(1),
            block_m=config.block_m,
            block_n=config.block_n,
            block_k=config.block_k,
            even_m=m % config.block_m == 0,
            even_n=n % config.block_n == 0,
            even_k=k % config.block_k == 0,
            group_m=group_m,
            gated_residual=gated_residual,
            split_epilogue=config.split_epilogue,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            num_ctas=1,
        )

    return launch


def _composite_launcher(
    production: Callable[[], None],
    standalone: Callable[[], None],
) -> Callable[[], None]:
    def launch() -> None:
        production()
        standalone()

    return launch


def _exact_validation(configs: Sequence[Config]) -> None:
    """Check both copied GEMM and fused epilogue against materialized kernels."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    m, n, k = 257, 512, 256
    activation = torch.randint(
        -16,
        17,
        (m, k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -16,
        17,
        (n, k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    activation_scale = torch.rand(m, device="cuda", dtype=torch.float32, generator=generator)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32, generator=generator)
    residual = torch.randn(
        (m, n),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = torch.randn((3, n), device="cuda", dtype=torch.bfloat16, generator=generator)
    row_ids = torch.randint(0, 3, (m,), device="cuda", dtype=torch.int32, generator=generator)
    projection = torch.empty_like(residual)
    expected = torch.empty_like(residual)
    actual = torch.empty_like(residual)
    _production_launcher(
        activation,
        weight,
        projection,
        activation_scale,
        weight_scale,
    )()
    _standalone_launcher(projection, residual, gate, row_ids, expected)()
    for config in configs:
        _fused_launcher(
            activation,
            weight,
            residual,
            gate,
            row_ids,
            actual,
            activation_scale,
            weight_scale,
            config,
        )()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        _fused_launcher(
            activation,
            weight,
            residual,
            gate,
            row_ids,
            actual,
            activation_scale,
            weight_scale,
            config,
            gated_residual=False,
        )()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, projection, rtol=0, atol=0)


def _tops(shape: Shape, timing: Timing) -> float:
    return 2 * shape.m * shape.n * shape.k / timing.median_ms / 1e9


def _benchmark_shape(
    shape: Shape,
    configs: Sequence[Config],
    warmup_ms: int,
    measurement_time_ms: int,
    reverse_provider_order: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randint(
        -127,
        128,
        (shape.m, shape.k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -127,
        128,
        (shape.n, shape.k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    activation_scale = torch.full(
        (shape.m,),
        1e-4,
        device="cuda",
        dtype=torch.float32,
    )
    weight_scale = torch.full(
        (shape.n,),
        1e-4,
        device="cuda",
        dtype=torch.float32,
    )
    residual = torch.randn(
        (shape.m, shape.n),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = torch.randn(
        (3, shape.n),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    row_ids = torch.randint(
        0,
        3,
        (shape.m,),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    projection = torch.empty_like(residual)
    expected = torch.empty_like(residual)
    actual = torch.empty_like(residual)

    production = _production_launcher(
        activation,
        weight,
        projection,
        activation_scale,
        weight_scale,
    )
    standalone = _standalone_launcher(projection, residual, gate, row_ids, expected)
    composite = _composite_launcher(production, standalone)
    composite()
    torch.cuda.synchronize()

    providers: list[tuple[str, Callable[[], None]]] = [
        ("production GEMM", production),
        ("standalone gated residual", standalone),
        ("production + standalone", composite),
    ]
    for config in configs:
        fused = _fused_launcher(
            activation,
            weight,
            residual,
            gate,
            row_ids,
            actual,
            activation_scale,
            weight_scale,
            config,
        )
        fused()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        providers.append((f"fused {config.display()}", fused))
    if reverse_provider_order:
        providers.reverse()

    print(f"\n{shape.name}: M={shape.m} N={shape.n} K={shape.k}")
    print("gate shape: [3, N] selected by row_ids[M]; tensors: BF16 residual/output")
    print("| provider | p50 [p20, p80] (ms) | GEMM TOP/s | vs composite |")
    print("|:---|---:|---:|---:|")
    measurements = [
        (
            name,
            triton_benchmark(launch, warmup_ms, measurement_time_ms),
        )
        for name, launch in providers
    ]
    composite_ms = next(
        timing.median_ms for name, timing in measurements if name == "production + standalone"
    )
    for name, timing in measurements:
        effective_tops = (
            "-" if name == "standalone gated residual" else f"{_tops(shape, timing):.1f}"
        )
        print(
            f"| {name} | {timing.display(4)} | {effective_tops} | "
            f"{composite_ms / timing.median_ms:.3f}x |"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        choices=tuple(_SHAPES),
        nargs="+",
        default=list(_SHAPES),
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--rows", type=int, default=37_710)
    parser.add_argument("--reverse-provider-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        raise SystemExit("gated-residual experiments require an NVIDIA Blackwell GPU")
    if arguments.rows <= 0:
        raise SystemExit("--rows must be positive")
    if arguments.warmup_ms < 0 or arguments.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")

    _exact_validation(_CONFIGS)
    print("exact small-shape validation: PASS")
    eliminated_bytes = 4 * arguments.rows * _SHAPES["attention-out"].n
    print(f"eliminated projection round trip: {eliminated_bytes / 1e9:.3f} GB per call")
    for name in arguments.cases:
        base_shape = _SHAPES[name]
        _benchmark_shape(
            Shape(base_shape.name, arguments.rows, base_shape.n, base_shape.k),
            _CONFIGS,
            arguments.warmup_ms,
            arguments.measurement_time_ms,
            arguments.reverse_provider_order,
        )


if __name__ == "__main__":
    main()
