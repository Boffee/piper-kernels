"""Benchmark fusing H3 RMSNorm and AdaLN into ConvRot INT8 preparation.

This is deliberately an isolated experiment.  It preserves the BF16 tensor
boundaries in MiniMax H3's::

    h = RMSNorm(x)
    h.mul_(1 + scale[row_ids])
    h.add_(shift[row_ids])

and compares materializing ``h`` before the production ConvRot preparation
kernel with carrying those rounded values directly into the H256 rotation and
rowwise INT8 quantizer.
"""

# Triton's JIT launcher accepts compile-time options that are not represented
# in its Python signature.
# pyright: reportCallIssue=false
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import torch
import triton
import triton.language as tl
from lib import Timing, triton_benchmark
from triton.language.extra import libdevice

from piper_kernels.convrot.int8.backends import triton as convrot

K = 5_376
GROUP_SIZE = 256
BLOCK_SIZE = 8_192
EPSILON = 1e-5
_WARPS = (4, 8)


@triton.jit
def _rms_adaln_values(
    x_ptr,
    norm_weight_ptr,
    shift_ptr,
    scale_ptr,
    row_ids_ptr,
    row,
    row_width,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    """Load one row and reproduce H3's three successive BF16 boundaries."""
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < row_width
    row_offset = row_i64 * row_width
    values = tl.load(x_ptr + row_offset + offsets_i64, mask=mask, other=0.0).to(tl.float32)
    norm_weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / row_width + epsilon)

    # torch.nn.functional.rms_norm computes BF16 input and weight in FP32 and
    # rounds the weighted result once.  MiniMax H3 then performs an in-place
    # BF16 multiply and in-place BF16 add, introducing two more boundaries.
    normalized = (values * inverse_rms * norm_weight).to(tl.bfloat16).to(tl.float32)
    table_row = tl.load(row_ids_ptr + row).to(tl.int64)
    table_offset = table_row * row_width + offsets_i64
    modulation_scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    multiplier = (1.0 + modulation_scale).to(tl.bfloat16).to(tl.float32)
    values = (normalized * multiplier).to(tl.bfloat16).to(tl.float32)
    shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    return (values + shift).to(tl.bfloat16)


@triton.jit
def _rms_adaln_kernel(
    x_ptr,
    norm_weight_ptr,
    shift_ptr,
    scale_ptr,
    row_ids_ptr,
    output_ptr,
    row_width,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    """Materialize exact BF16 RMSNorm plus featurewise AdaLN rows."""
    row = tl.program_id(0)
    values = _rms_adaln_values(
        x_ptr,
        norm_weight_ptr,
        shift_ptr,
        scale_ptr,
        row_ids_ptr,
        row,
        row_width,
        epsilon,
        block_size,
    )
    offsets = tl.arange(0, block_size)
    row_offset = row.to(tl.int64) * row_width
    tl.store(output_ptr + row_offset + offsets, values, mask=offsets < row_width)


@triton.jit
def _rms_adaln_rotate_quantize_kernel(
    x_ptr,
    norm_weight_ptr,
    shift_ptr,
    modulation_scale_ptr,
    row_ids_ptr,
    q_ptr,
    quant_scale_ptr,
    row_width,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
):
    """Fuse exact H3 RMSNorm/AdaLN boundaries with ConvRot preparation."""
    row = tl.program_id(0)
    values = _rms_adaln_values(
        x_ptr,
        norm_weight_ptr,
        shift_ptr,
        modulation_scale_ptr,
        row_ids_ptr,
        row,
        row_width,
        epsilon,
        block_size,
    ).to(tl.float32)
    values = convrot._rotate_hadamard_groups(values, block_size, group_size)
    values *= inverse_sqrt_group

    # Match the production preparation kernel's materialized BF16 rotation
    # boundary before it chooses the row-wide dynamic-quantization scale.
    values = values.to(tl.bfloat16)
    quant_scale = tl.maximum(
        tl.max(tl.abs(values).to(tl.float32), axis=0) / 127.0,
        1e-30,
    )
    scaled = convrot._normalize_for_int8(values, quant_scale, 2)
    quantized = tl.clamp(
        libdevice.rint(scaled.to(tl.float32)),
        -128.0,
        127.0,
    ).to(tl.int8)

    offsets = tl.arange(0, block_size)
    row_i64 = row.to(tl.int64)
    output_offset = row_i64 * row_width
    tl.store(q_ptr + output_offset + offsets, quantized, mask=offsets < row_width)
    tl.store(quant_scale_ptr + row_i64, quant_scale)


def _rms_adaln_launcher(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_ids: torch.Tensor,
    output: torch.Tensor,
    num_warps: int,
) -> Callable[[], None]:
    def launch() -> None:
        _rms_adaln_kernel[(x.shape[0],)](
            x,
            norm_weight,
            shift,
            scale,
            row_ids,
            output,
            K,
            epsilon=EPSILON,
            block_size=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return launch


def _fused_launcher(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_ids: torch.Tensor,
    qdata: torch.Tensor,
    quant_scale: torch.Tensor,
    num_warps: int,
) -> Callable[[], None]:
    def launch() -> None:
        _rms_adaln_rotate_quantize_kernel[(x.shape[0],)](
            x,
            norm_weight,
            shift,
            scale,
            row_ids,
            qdata,
            quant_scale,
            K,
            epsilon=EPSILON,
            block_size=BLOCK_SIZE,
            group_size=GROUP_SIZE,
            inverse_sqrt_group=GROUP_SIZE**-0.5,
            num_warps=num_warps,
        )

    return launch


def _preparation_launcher(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
) -> Callable[[], None]:
    def launch() -> None:
        convrot._fused_rotate_quantize_activations(
            activation,
            qdata,
            scale,
            GROUP_SIZE,
            2,
        )

    return launch


def _sample_rows(rows: int, count: int) -> torch.Tensor:
    """Select a deterministic spread, including both ends of the sequence."""
    return (
        torch.linspace(
            0,
            rows - 1,
            steps=min(rows, count),
            device="cuda",
            dtype=torch.float64,
        )
        .round()
        .to(torch.int64)
    )


def _validate_rms_adaln(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_ids: torch.Tensor,
    actual: torch.Tensor,
    sample_rows: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    source = x.index_select(0, sample_rows)
    sample_ids = row_ids.index_select(0, sample_rows).to(torch.int64)
    expected = torch.nn.functional.rms_norm(source, (K,), norm_weight, EPSILON)
    expected.mul_(1.0 + scale.index_select(0, sample_ids))
    expected.add_(shift.index_select(0, sample_ids))
    sampled_actual = actual.index_select(0, sample_rows)
    torch.testing.assert_close(
        sampled_actual,
        expected,
        rtol=torch.finfo(torch.bfloat16).eps,
        # Triton's row reduction tree can move the final RMS factor by one
        # BF16 quantum around zero even though every tensor boundary matches.
        atol=torch.finfo(torch.bfloat16).eps,
    )
    max_error = float((sampled_actual.float() - expected.float()).abs().max().item())
    return max_error, expected


def _eager_quantized_error(
    expected_qdata: torch.Tensor,
    expected_scale: torch.Tensor,
    actual_qdata: torch.Tensor,
    actual_scale: torch.Tensor,
    *,
    max_qdata_error: int,
) -> tuple[int, float]:
    """Compare against eager Torch, whose RMS reduction tree may differ."""
    expected_q = expected_qdata.to(torch.int16)
    actual_q = actual_qdata.to(torch.int16)
    qdata_error = int((expected_q - actual_q).abs().max().item())
    if qdata_error > max_qdata_error:
        raise AssertionError(f"fused qdata differs from separate path by {qdata_error}")
    torch.testing.assert_close(
        actual_scale,
        expected_scale,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=0,
    )
    relative_scale_error = float(
        ((actual_scale - expected_scale).abs() / expected_scale.abs().clamp_min(1e-30)).max().item()
    )
    return qdata_error, relative_scale_error


def _validate_quantized(
    expected_qdata: torch.Tensor,
    expected_scale: torch.Tensor,
    actual_qdata: torch.Tensor,
    actual_scale: torch.Tensor,
    sample_rows: torch.Tensor,
) -> tuple[int, float]:
    """Require exact agreement with the same-warp staged Triton path."""
    expected_q = expected_qdata.index_select(0, sample_rows)
    actual_q = actual_qdata.index_select(0, sample_rows)
    if not torch.equal(actual_q, expected_q):
        qdata_error = int(
            (expected_q.to(torch.int16) - actual_q.to(torch.int16)).abs().max().item()
        )
        mismatches = int(torch.count_nonzero(actual_q != expected_q).item())
        raise AssertionError(
            "fused qdata differs from the same-warp staged path: "
            f"max_abs={qdata_error}, mismatches={mismatches}"
        )

    expected_s = expected_scale.index_select(0, sample_rows)
    actual_s = actual_scale.index_select(0, sample_rows)
    if not torch.equal(actual_s, expected_s):
        scale_error = float((actual_s - expected_s).abs().max().item())
        mismatches = int(torch.count_nonzero(actual_s != expected_s).item())
        raise AssertionError(
            "fused scale differs from the same-warp staged path: "
            f"max_abs={scale_error}, mismatches={mismatches}"
        )
    return 0, 0.0


def _format_timing(timing: Timing) -> str:
    return timing.display(4)


@torch.inference_mode()
def _benchmark_rows(
    rows: int,
    table_rows: int,
    seed: int,
    validate_rows: int,
    warmup_ms: int,
    measurement_time_ms: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16, generator=generator)
    norm_weight = 1.0 + 0.05 * torch.randn(
        K, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    shift = 0.1 * torch.randn(
        table_rows,
        K,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    modulation_scale = 0.1 * torch.randn(
        table_rows,
        K,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    row_ids = torch.randint(
        0,
        table_rows,
        (rows,),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    materialized = torch.empty_like(x)
    separate_qdata = torch.empty_like(x, dtype=torch.int8)
    separate_scale = torch.empty(rows, device="cuda", dtype=torch.float32)
    fused_qdata = torch.empty_like(separate_qdata)
    fused_scale = torch.empty_like(separate_scale)
    samples = _sample_rows(rows, validate_rows)

    rms_launchers = {
        warps: _rms_adaln_launcher(
            x,
            norm_weight,
            shift,
            modulation_scale,
            row_ids,
            materialized,
            warps,
        )
        for warps in _WARPS
    }
    preparation = _preparation_launcher(materialized, separate_qdata, separate_scale)
    fused_launchers = {
        warps: _fused_launcher(
            x,
            norm_weight,
            shift,
            modulation_scale,
            row_ids,
            fused_qdata,
            fused_scale,
            warps,
        )
        for warps in _WARPS
    }

    # Compile and validate each launch shape before timing.  Eager Torch is a
    # numerical reference because its valid RMS reduction tree differs.  Each
    # fused kernel must additionally be bit-identical to a staged Triton path
    # using the same number of warps.
    max_h_error = 0.0
    eager_h: torch.Tensor | None = None
    for warps, launch in rms_launchers.items():
        launch()
        h_error, expected_h = _validate_rms_adaln(
            x,
            norm_weight,
            shift,
            modulation_scale,
            row_ids,
            materialized,
            samples,
        )
        max_h_error = max(max_h_error, h_error)
        if warps == 4:
            eager_h = expected_h
    if eager_h is None:
        raise AssertionError("the four-warp numerical reference was not launched")
    eager_qdata = torch.empty_like(eager_h, dtype=torch.int8)
    eager_scale = torch.empty(eager_h.shape[0], device="cuda", dtype=torch.float32)
    _preparation_launcher(eager_h, eager_qdata, eager_scale)()

    validation: dict[int, tuple[tuple[int, float], tuple[int, float]]] = {}
    for warps, fused in fused_launchers.items():
        # Refresh both staged outputs from the same-warp RMS/AdaLN kernel.  A
        # reference from another warp count may use a different valid
        # reduction tree and therefore cannot support an exact assertion.
        rms_launchers[warps]()
        preparation()
        fused()
        staged_error = _validate_quantized(
            separate_qdata,
            separate_scale,
            fused_qdata,
            fused_scale,
            samples,
        )
        eager_error = _eager_quantized_error(
            eager_qdata,
            eager_scale,
            fused_qdata.index_select(0, samples),
            fused_scale.index_select(0, samples),
            # The Torch and Triton RMS reductions use different valid trees;
            # their one-BF16-quantum RMS-factor difference can move a rounded
            # INT8 endpoint by two, while the exact staged Triton path above
            # remains bit-identical.
            max_qdata_error=2,
        )
        validation[warps] = staged_error, eager_error

    prep_timing = triton_benchmark(preparation, warmup_ms, measurement_time_ms)
    rms_timings = {
        warps: triton_benchmark(launch, warmup_ms, measurement_time_ms)
        for warps, launch in rms_launchers.items()
    }

    def separate(warps: int) -> None:
        rms_launchers[warps]()
        preparation()

    separate_timings = {
        warps: triton_benchmark(
            lambda warps=warps: separate(warps),
            warmup_ms,
            measurement_time_ms,
        )
        for warps in _WARPS
    }
    fused_timings = {
        warps: triton_benchmark(launch, warmup_ms, measurement_time_ms)
        for warps, launch in fused_launchers.items()
    }
    best_separate_warps = min(_WARPS, key=lambda warps: separate_timings[warps].median_ms)
    best_fused_warps = min(_WARPS, key=lambda warps: fused_timings[warps].median_ms)
    best_separate = separate_timings[best_separate_warps]
    best_fused = fused_timings[best_fused_warps]
    boundary_bytes = 2 * rows * K * torch.bfloat16.itemsize

    print(f"\nM={rows:,}, K={K:,}, group={GROUP_SIZE}, BF16, AdaLN rows={table_rows}")
    print("| path | config | device p50 [p20, p80] (ms) | separate / path |")
    print("|:---|:---|---:|---:|")
    print(f"| production rotate+quant only | w4 | {_format_timing(prep_timing)} | - |")
    for warps in _WARPS:
        print(f"| standalone RMSNorm+AdaLN | w{warps} | {_format_timing(rms_timings[warps])} | - |")
        print(
            f"| separate end-to-end | w{warps}+production-w4 | "
            f"{_format_timing(separate_timings[warps])} | "
            f"{best_separate.median_ms / separate_timings[warps].median_ms:.3f}x |"
        )
    for warps in _WARPS:
        print(
            f"| fused end-to-end | w{warps} | {_format_timing(fused_timings[warps])} | "
            f"{best_separate.median_ms / fused_timings[warps].median_ms:.3f}x |"
        )
    (q_error, scale_error), (eager_q_error, eager_scale_error) = validation[best_fused_warps]
    print(
        f"best: separate w{best_separate_warps} {best_separate.median_ms:.4f} ms; "
        f"fused w{best_fused_warps} {best_fused.median_ms:.4f} ms; "
        f"speedup {best_separate.median_ms / best_fused.median_ms:.3f}x"
    )
    print(
        f"avoided materialized-h write+read: {boundary_bytes / 1e9:.3f} GB; "
        f"validation: RMS/AdaLN max_abs={max_h_error:.6g}, "
        f"staged qdata max_abs={q_error}, scale max_rel={scale_error:.3g}; "
        f"eager qdata max_abs={eager_q_error}, scale max_rel={eager_scale_error:.3g}"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[37_710])
    parser.add_argument("--table-rows", type=int, default=9)
    parser.add_argument("--validate-rows", type=int, default=64)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        raise SystemExit("RMSNorm/AdaLN preparation experiments require an NVIDIA Blackwell GPU")
    if any(rows <= 0 for rows in args.rows):
        raise SystemExit("every --rows value must be positive")
    if args.table_rows <= 0 or args.validate_rows <= 0:
        raise SystemExit("--table-rows and --validate-rows must be positive")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark at one or more MiniMax H3 sequence lengths."""
    args = _parse_args(argv)
    _validate_args(args)
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    print(
        f"GPU: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]}); "
        f"Torch: {torch.__version__}; Triton: {triton.__version__}"
    )
    print(
        "experiment: exact BF16 RMSNorm -> mul_(1 + scale[row]) -> add_(shift[row]) "
        "-> H256 -> rowwise INT8"
    )
    for rows in args.rows:
        _benchmark_rows(
            rows,
            args.table_rows,
            args.seed,
            args.validate_rows,
            args.warmup_ms,
            args.measurement_time_ms,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
