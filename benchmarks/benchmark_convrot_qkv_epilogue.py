"""Benchmark folding H3 Q/K RMSNorm and RoPE into the ConvRot GEMM epilogue."""

# The experimental JIT kernels intentionally use Triton's launcher-style
# signatures and have a high argument count.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from lib import Timing, triton_benchmark
from torch.nn import functional

from piper_kernels.convrot.int8.backends import triton as baseline

try:
    import comfy_kitchen
except ImportError:
    comfy_kitchen = None

K = 5_376
HEADS = 56
HEAD_DIM = 128
ROT_DIM = 96
QKV_WIDTH = 3 * HEADS * HEAD_DIM
SEGMENT_WIDTH = HEADS * HEAD_DIM


@dataclass(frozen=True, slots=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    group_m: int = 0

    def display(self) -> str:
        group = f",g{self.group_m}" if self.group_m else ""
        return (
            f"{self.block_m}x{self.block_n}x{self.block_k},"
            f"s{self.num_stages},w{self.num_warps}{group}"
        )


@triton.jit
def _apply_rms_rope(
    values,
    norm_ptr,
    cos_ptr,
    sin_ptr,
    row_offsets,
    m,
    epsilon: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    rot_dim: tl.constexpr,
):
    """Apply eager-boundary RMSNorm and partial split-half RoPE per head."""
    heads_per_tile: tl.constexpr = block_n // head_dim
    # The existing graph first materializes the BF16 projection. Preserve that
    # logical boundary before the FP32 normalization arithmetic.
    values = values.to(tl.bfloat16).to(tl.float32)
    heads = tl.reshape(values, (block_m, heads_per_tile, head_dim))
    mean_square = tl.sum(heads * heads, axis=2, keep_dims=True) / head_dim
    normalized = heads * tl.rsqrt(mean_square + epsilon)

    dim_offsets = tl.arange(0, head_dim)
    norm = tl.load(norm_ptr + dim_offsets).to(tl.float32)
    # RMSNorm is a distinct BF16 operation in the H3 graph.  Round its
    # weighted output before RoPE rather than carrying hidden FP32 precision.
    normalized = (normalized * norm[None, None, :]).to(tl.bfloat16).to(tl.float32)

    half_rot: tl.constexpr = rot_dim // 2
    partner_offsets = tl.where(
        dim_offsets < half_rot,
        dim_offsets + half_rot,
        tl.where(dim_offsets < rot_dim, dim_offsets - half_rot, dim_offsets),
    )
    partner_indices = (
        tl.zeros((block_m, heads_per_tile, 1), tl.int32) + partner_offsets[None, None, :]
    )
    partner = tl.gather(normalized, partner_indices, axis=2)

    pair_offsets = tl.where(dim_offsets < half_rot, dim_offsets, dim_offsets - half_rot)
    row_mask = row_offsets < m
    pair_mask = dim_offsets < rot_dim
    trig_offsets = row_offsets[:, None] * half_rot + pair_offsets[None, :]
    cosine = tl.load(
        cos_ptr + trig_offsets,
        mask=row_mask[:, None] & pair_mask[None, :],
        other=1.0,
    )
    sine = tl.load(
        sin_ptr + trig_offsets,
        mask=row_mask[:, None] & pair_mask[None, :],
        other=0.0,
    )
    signed_sine = tl.where(dim_offsets < half_rot, -sine, sine)
    rotated = normalized * cosine[:, None, :] + partner * signed_sine[:, None, :]
    return tl.reshape(tl.where(pair_mask[None, None, :], rotated, normalized), (block_m, block_n))


@triton.jit
def _int8_qkv_segment_kernel(
    activation_ptr,
    weight_ptr,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    norm_ptr,
    cos_ptr,
    sin_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    apply_norm_rope: tl.constexpr,
    epsilon: tl.constexpr,
    head_dim: tl.constexpr,
    rot_dim: tl.constexpr,
):
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
    activation_ptrs = (
        activation_ptr + offsets_m_i64[:, None] * stride_am + offsets_k_i64[None, :] * stride_ak
    )
    weight_ptrs = (
        weight_ptr + offsets_n_i64[None, :] * stride_wn + offsets_k_i64[:, None] * stride_wk
    )
    accumulator = tl.zeros((block_m, block_n), tl.int32)
    for k_offset in range(tl.cdiv(k, block_k)):
        activation = tl.load(
            activation_ptrs,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k - k_offset * block_k),
            other=0,
        )
        weight = tl.load(weight_ptrs)
        accumulator += tl.dot(activation, weight)
        activation_ptrs += block_k * stride_ak
        weight_ptrs += block_k * stride_wk

    activation_scale = tl.load(
        activation_scale_ptr + offsets_m,
        mask=offsets_m < m,
        other=0.0,
    )
    weight_scale = tl.load(weight_scale_ptr + offsets_n)
    result = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
    if apply_norm_rope:
        result = _apply_rms_rope(
            result,
            norm_ptr,
            cos_ptr,
            sin_ptr,
            offsets_m,
            m,
            epsilon,
            block_m,
            block_n,
            head_dim,
            rot_dim,
        )

    output_ptrs = (
        output_ptr + offsets_m_i64[:, None] * stride_om + offsets_n_i64[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        result,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


@triton.jit
def _standalone_rms_rope_kernel(
    qkv_ptr,
    q_norm_ptr,
    k_norm_ptr,
    cos_ptr,
    sin_ptr,
    m,
    stride_m,
    heads: tl.constexpr,
    segment_width: tl.constexpr,
    block_m: tl.constexpr,
    head_dim: tl.constexpr,
    rot_dim: tl.constexpr,
    epsilon: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    head = tl.program_id(1) % heads
    segment = tl.program_id(1) // heads
    dim_offsets = tl.arange(0, head_dim)
    base = row_offsets.to(tl.int64)[:, None] * stride_m
    base += segment * segment_width + head * head_dim
    pointers = qkv_ptr + base + dim_offsets[None, :]
    values = tl.load(
        pointers,
        mask=row_offsets[:, None] < m,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(values * values, axis=1, keep_dims=True) / head_dim
    normalized = values * tl.rsqrt(mean_square + epsilon)
    norm_ptr = tl.where(segment == 0, q_norm_ptr, k_norm_ptr)
    normalized = (
        (normalized * tl.load(norm_ptr + dim_offsets)[None, :].to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )

    half_rot: tl.constexpr = rot_dim // 2
    partner_offsets = tl.where(
        dim_offsets < half_rot,
        dim_offsets + half_rot,
        tl.where(dim_offsets < rot_dim, dim_offsets - half_rot, dim_offsets),
    )
    partner_indices = tl.zeros((block_m, 1), tl.int32) + partner_offsets[None, :]
    partner = tl.gather(normalized, partner_indices, axis=1)
    pair_offsets = tl.where(dim_offsets < half_rot, dim_offsets, dim_offsets - half_rot)
    pair_mask = dim_offsets < rot_dim
    trig_offsets = row_offsets[:, None] * half_rot + pair_offsets[None, :]
    cosine = tl.load(
        cos_ptr + trig_offsets,
        mask=(row_offsets[:, None] < m) & pair_mask[None, :],
        other=1.0,
    )
    sine = tl.load(
        sin_ptr + trig_offsets,
        mask=(row_offsets[:, None] < m) & pair_mask[None, :],
        other=0.0,
    )
    signed_sine = tl.where(dim_offsets < half_rot, -sine, sine)
    rotated = normalized * cosine + partner * signed_sine
    result = tl.where(pair_mask[None, :], rotated, normalized)
    tl.store(pointers, result, mask=row_offsets[:, None] < m)


def _production_gemm_launcher(
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
    output: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    m: int,
) -> Callable[[], None]:
    block_m = 8

    def launch() -> None:
        _standalone_rms_rope_kernel[(triton.cdiv(m, block_m), 2 * HEADS)](
            output,
            q_norm,
            k_norm,
            cosine,
            sine,
            m,
            output.stride(0),
            heads=HEADS,
            segment_width=SEGMENT_WIDTH,
            block_m=block_m,
            head_dim=HEAD_DIM,
            rot_dim=ROT_DIM,
            epsilon=1e-5,
            num_warps=4,
        )

    return launch


def _fused_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    config: Config,
    m: int,
) -> Callable[[], None]:
    num_m_tiles = triton.cdiv(m, config.block_m)
    num_n_tiles = triton.cdiv(SEGMENT_WIDTH, config.block_n)
    group_m = config.group_m or (1 if num_n_tiles <= 32 else 64)
    grid = (num_m_tiles * num_n_tiles,)
    segments = tuple(
        (
            weight[offset:],
            output[:, offset:],
            weight_scale[offset:],
            norm,
            apply_norm_rope,
        )
        for offset, norm, apply_norm_rope in (
            (0, q_norm, True),
            (SEGMENT_WIDTH, k_norm, True),
            (2 * SEGMENT_WIDTH, q_norm, False),
        )
    )

    def launch() -> None:
        for segment_weight, segment_output, segment_scale, norm, apply_norm_rope in segments:
            _int8_qkv_segment_kernel[grid](
                activation,
                segment_weight,
                segment_output,
                activation_scale,
                segment_scale,
                norm,
                cosine,
                sine,
                m,
                SEGMENT_WIDTH,
                K,
                activation.stride(0),
                activation.stride(1),
                weight.stride(0),
                weight.stride(1),
                output.stride(0),
                output.stride(1),
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                group_m=group_m,
                apply_norm_rope=apply_norm_rope,
                epsilon=1e-5,
                head_dim=HEAD_DIM,
                rot_dim=ROT_DIM,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
            )

    return launch


def _time_composite(
    first: Callable[[], None],
    second: Callable[[], None],
    warmup_ms: int,
    measurement_time_ms: int,
) -> Timing:
    def launch() -> None:
        first()
        second()

    return triton_benchmark(launch, warmup_ms, measurement_time_ms)


def _comfy_kitchen_launcher(
    output: torch.Tensor,
    rotation: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    m: int,
) -> Callable[[], None]:
    if comfy_kitchen is None:
        raise RuntimeError("Comfy Kitchen is not installed")
    q = output[:, :SEGMENT_WIDTH].view(1, m, HEADS, HEAD_DIM)
    k = output[:, SEGMENT_WIDTH : 2 * SEGMENT_WIDTH].view(1, m, HEADS, HEAD_DIM)

    def launch() -> None:
        assert comfy_kitchen is not None
        comfy_kitchen.rms_rope_split_half_(
            q,
            k,
            rotation,
            q_norm,
            k_norm,
            epsilon=1e-5,
            rot_dim=ROT_DIM,
        )

    return launch


def _eager_rms_rope_reference(
    qkv: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply H3's materialized BF16 RMSNorm and split-half RoPE boundaries."""
    result = qkv.clone()
    rows = qkv.shape[0]
    for segment, norm in ((0, q_norm), (1, k_norm)):
        offset = segment * SEGMENT_WIDTH
        values = qkv[:, offset : offset + SEGMENT_WIDTH].view(rows, HEADS, HEAD_DIM)
        normalized = functional.rms_norm(values, (HEAD_DIM,), norm, eps=1e-5)
        destination = result[:, offset : offset + SEGMENT_WIDTH].view(
            rows,
            HEADS,
            HEAD_DIM,
        )
        destination.copy_(normalized)

        first = normalized[..., : ROT_DIM // 2].float()
        second = normalized[..., ROT_DIM // 2 : ROT_DIM].float()
        cos = cosine.float()[:, None, :]
        sin = sine.float()[:, None, :]
        destination[..., :ROT_DIM].copy_(
            torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
        )
    return result


def _prepare_expected(
    activation: torch.Tensor,
    weight: torch.Tensor,
    expected: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    rotation: torch.Tensor,
    validation_rows: torch.Tensor,
    m: int,
) -> None:
    """Build the independent reference while checking both numerical boundaries."""
    production_expected = _production_gemm_launcher(
        activation,
        weight,
        expected,
        activation_scale,
        weight_scale,
    )
    standalone_expected = _standalone_launcher(expected, q_norm, k_norm, cosine, sine, m)
    production_expected()
    plain_validation = expected[validation_rows].clone()
    standalone_expected()
    torch.cuda.synchronize()
    eager_validation = _eager_rms_rope_reference(
        plain_validation,
        q_norm,
        k_norm,
        cosine[validation_rows],
        sine[validation_rows],
    )
    torch.testing.assert_close(
        expected[validation_rows],
        eager_validation,
        rtol=0,
        atol=1 / 64,
    )

    if comfy_kitchen is not None:
        production_expected()
        comfy_expected = _comfy_kitchen_launcher(
            expected,
            rotation,
            q_norm,
            k_norm,
            m,
        )
        comfy_expected()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            expected[validation_rows],
            eager_validation,
            rtol=0,
            atol=1 / 64,
        )


def _validation_rows(m: int, device: torch.device) -> torch.Tensor:
    sample_count = min(m, 257)
    if sample_count == 1:
        indices = {0}
    else:
        indices = {round(index * (m - 1) / (sample_count - 1)) for index in range(sample_count)}
    # Include rows around the first signed-int32 output-offset boundary.
    boundary = (2**31 - 1) // QKV_WIDTH
    indices.update(row for row in (boundary - 1, boundary, boundary + 1) if 0 <= row < m)
    return torch.tensor(sorted(indices), device=device, dtype=torch.int64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--rows", type=int, default=37_710)
    return parser.parse_args()


def _validate_args(arguments: argparse.Namespace) -> int:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        raise SystemExit("QKV epilogue experiments require an NVIDIA Blackwell GPU")
    if arguments.rows <= 0:
        raise SystemExit("--rows must be positive")
    if arguments.warmup_ms < 0 or arguments.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    return int(arguments.rows)


def main() -> None:
    arguments = _parse_args()
    m = _validate_args(arguments)

    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randint(
        -127,
        128,
        (m, K),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -127,
        128,
        (QKV_WIDTH, K),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    activation_scale = torch.full((m,), 1e-4, device="cuda", dtype=torch.float32)
    weight_scale = torch.full((QKV_WIDTH,), 1e-4, device="cuda", dtype=torch.float32)
    q_norm = torch.rand(HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    k_norm = torch.rand(HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=generator)
    angles = torch.rand(
        (m, ROT_DIM // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cosine = torch.cos(angles).to(torch.bfloat16).contiguous()
    sine = torch.sin(angles).to(torch.bfloat16).contiguous()
    rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1,
        m,
        1,
        ROT_DIM // 2,
        2,
        2,
    )
    output = torch.empty((m, QKV_WIDTH), device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)

    production = _production_gemm_launcher(
        activation,
        weight,
        output,
        activation_scale,
        weight_scale,
    )
    standalone = _standalone_launcher(output, q_norm, k_norm, cosine, sine, m)
    validation_rows = _validation_rows(m, output.device)
    _prepare_expected(
        activation,
        weight,
        expected,
        activation_scale,
        weight_scale,
        q_norm,
        k_norm,
        cosine,
        sine,
        rotation,
        validation_rows,
        m,
    )

    rows: list[tuple[str, Timing]] = []
    rows.append(
        (
            "production GEMM",
            triton_benchmark(
                production,
                arguments.warmup_ms,
                arguments.measurement_time_ms,
            ),
        )
    )
    rows.append(
        (
            "production + standalone Triton",
            _time_composite(
                production,
                standalone,
                arguments.warmup_ms,
                arguments.measurement_time_ms,
            ),
        )
    )
    if comfy_kitchen is not None:
        comfy = _comfy_kitchen_launcher(output, rotation, q_norm, k_norm, m)
        rows.append(
            (
                "production + Comfy Kitchen",
                _time_composite(
                    production,
                    comfy,
                    arguments.warmup_ms,
                    arguments.measurement_time_ms,
                ),
            )
        )

    configs = (
        Config(128, 256, 128, 3, 8),
        Config(128, 256, 128, 3, 8, group_m=64),
    )
    for config in configs:
        fused = _fused_launcher(
            activation,
            weight,
            output,
            activation_scale,
            weight_scale,
            q_norm,
            k_norm,
            cosine,
            sine,
            config,
            m,
        )
        try:
            fused()
            torch.cuda.synchronize()
        except (
            triton.compiler.errors.CompilationError,
            triton.runtime.errors.OutOfResources,
        ) as error:
            print(f"skipping fused {config.display()}: {error}")
            continue
        torch.testing.assert_close(
            output[validation_rows],
            expected[validation_rows],
            rtol=0,
            # The GEMM and one-head kernels use different FP32 reduction trees.
            atol=1 / 64,
        )
        sampled_output = output[validation_rows, : 2 * SEGMENT_WIDTH]
        sampled_expected = expected[validation_rows, : 2 * SEGMENT_WIDTH]
        mismatch_count = torch.count_nonzero(sampled_output != sampled_expected).item()
        max_abs = (sampled_output.float() - sampled_expected.float()).abs().max().item()
        print(
            f"validation fused {config.display()}: "
            f"{mismatch_count}/{sampled_output.numel()} Q/K values differ, "
            f"max_abs={max_abs:.8f}"
        )
        rows.append(
            (
                f"fused {config.display()}",
                triton_benchmark(
                    fused,
                    arguments.warmup_ms,
                    arguments.measurement_time_ms,
                ),
            )
        )

    current_timing = next(
        (timing for name, timing in rows if name == "production + Comfy Kitchen"),
        rows[1][1],
    )
    baseline_ms = current_timing.median_ms
    print(f"\nH3 QKV: M={m} N={QKV_WIDTH} K={K}, {HEADS} heads x {HEAD_DIM}")
    eliminated_bytes = 8 * m * SEGMENT_WIDTH
    print(f"eliminated Q/K read-write pass: {eliminated_bytes / 1e9:.3f} GB")
    print("| provider | p50 [p20, p80] (ms) | vs current composite |")
    print("|:---|---:|---:|")
    for name, timing in rows:
        print(f"| {name} | {timing.display(4)} | {baseline_ms / timing.median_ms:.3f}x |")


if __name__ == "__main__":
    main()
