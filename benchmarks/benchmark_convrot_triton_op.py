"""Test whether ``torch.library.triton_op`` removes a ConvRot boundary.

The traceable operator wraps the production one-pass H256 rotation and INT8
quantization kernel.  A compiled PyTorch RMSNorm/AdaLN producer is placed
immediately before it.  The comparison answers two separate questions:

1. Can ``torch.compile`` trace and run the wrapped Piper Triton kernel?
2. Does Inductor fuse the producer into that kernel and avoid materializing
   the BF16 activation passed to ConvRot?

The explicit-fusion reference reuses the benchmark-only RMSNorm/AdaLN plus
ConvRot kernel from ``benchmark_convrot_rms_adaln_preparation.py``.
"""

# Triton's JIT launcher accepts compile-time options that are not represented
# in its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import torch
from benchmark_convrot_rms_adaln_preparation import _fused_launcher
from lib import Timing, triton_benchmark
from torch.library import triton_op, wrap_triton
from torch.profiler import ProfilerActivity, profile

from piper_kernels.convrot.int8.backends import triton as convrot

K = 5_376
GROUP_SIZE = 256
BLOCK_SIZE = 8_192
EPSILON = 1e-5


@triton_op("piper_kernels_benchmarks::convrot_prepare", mutates_args={})
def _traceable_convrot_prepare(
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose production ConvRot preparation as a traceable two-output op."""
    rows, width = activation.shape
    qdata = torch.empty_like(activation, dtype=torch.int8)
    quant_scale = torch.empty(rows, device=activation.device, dtype=torch.float32)
    wrap_triton(convrot._rotate_quantize_rows_kernel)[lambda _meta: (rows,)](
        activation,
        qdata,
        quant_scale,
        width,
        block_size=BLOCK_SIZE,
        group_size=GROUP_SIZE,
        inverse_sqrt_group=GROUP_SIZE**-0.5,
        input_dtype_code=2,
        input_act_code=0,
        num_warps=4,
    )
    return qdata, quant_scale


def _compiled_boundary(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    shift: torch.Tensor,
    modulation_scale: torch.Tensor,
    row_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose PyTorch RMSNorm/AdaLN with traceable ConvRot preparation."""
    activation = torch.nn.functional.rms_norm(x, (K,), norm_weight, EPSILON)
    activation = activation * (1.0 + modulation_scale[row_ids])
    activation = activation + shift[row_ids]
    return _traceable_convrot_prepare(activation)


def _format_timing(timing: Timing) -> str:
    return timing.display(4)


def _profile_kernel_names(launch: Callable[[], object]) -> list[tuple[str, int]]:
    with profile(activities=[ProfilerActivity.CUDA]) as profiling:
        launch()
    torch.cuda.synchronize()
    return [
        (event.key, event.count)
        for event in profiling.key_averages()
        if event.self_device_time_total > 0
    ]


@torch.inference_mode()
def _benchmark(
    rows: int,
    table_rows: int,
    validate_rows: int,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16, generator=generator)
    norm_weight = 1.0 + 0.05 * torch.randn(
        K,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
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
        table_rows,
        (rows,),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    fused_qdata = torch.empty((rows, K), device="cuda", dtype=torch.int8)
    fused_scale = torch.empty(rows, device="cuda", dtype=torch.float32)
    explicit_fusion = _fused_launcher(
        x,
        norm_weight,
        shift,
        modulation_scale,
        row_ids,
        fused_qdata,
        fused_scale,
        4,
    )
    compiled = torch.compile(_compiled_boundary, fullgraph=True, dynamic=False)

    def compiled_launch() -> tuple[torch.Tensor, torch.Tensor]:
        return compiled(x, norm_weight, shift, modulation_scale, row_ids)

    compiled_qdata, compiled_scale = compiled_launch()
    explicit_fusion()
    torch.cuda.synchronize()

    sample_rows = (
        torch.linspace(
            0,
            rows - 1,
            steps=min(rows, validate_rows),
            device="cuda",
            dtype=torch.float64,
        )
        .round()
        .to(torch.int64)
    )
    sampled_compiled_qdata = compiled_qdata.index_select(0, sample_rows).to(torch.int16)
    sampled_fused_qdata = fused_qdata.index_select(0, sample_rows).to(torch.int16)
    qdata_error = int((sampled_compiled_qdata - sampled_fused_qdata).abs().max().item())
    if qdata_error > 2:
        raise AssertionError(f"compiled and explicitly fused qdata differ by {qdata_error}")
    sampled_compiled_scale = compiled_scale.index_select(0, sample_rows)
    sampled_fused_scale = fused_scale.index_select(0, sample_rows)
    scale_relative_error = float(
        (
            (sampled_compiled_scale - sampled_fused_scale).abs()
            / sampled_fused_scale.abs().clamp_min(1e-30)
        )
        .max()
        .item()
    )
    if scale_relative_error > 2 * torch.finfo(torch.bfloat16).eps:
        raise AssertionError(
            f"compiled and explicitly fused scales differ by {scale_relative_error:.6g} relative"
        )

    compiled_timing = triton_benchmark(compiled_launch, warmup_ms, measurement_time_ms)
    fused_timing = triton_benchmark(explicit_fusion, warmup_ms, measurement_time_ms)
    kernel_names = _profile_kernel_names(compiled_launch)

    print(f"M={rows:,}, K={K:,}, BF16, AdaLN rows={table_rows}")
    print("| path | device p50 [p20, p80] (ms) | compiled / path |")
    print("|:---|---:|---:|")
    print(f"| compiled PyTorch -> triton_op | {_format_timing(compiled_timing)} | 1.000x |")
    print(
        f"| explicit one-pass fusion | {_format_timing(fused_timing)} | "
        f"{compiled_timing.median_ms / fused_timing.median_ms:.3f}x |"
    )
    print(
        f"sampled validation: qdata max_abs={qdata_error}, scale max_rel={scale_relative_error:.6g}"
    )
    print(f"compiled CUDA launches ({sum(count for _, count in kernel_names)} total):")
    for name, count in kernel_names:
        print(f"  {count}x {name}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=37_710)
    parser.add_argument("--table-rows", type=int, default=9)
    parser.add_argument("--validate-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.rows <= 0 or args.table_rows <= 0 or args.validate_rows <= 0:
        raise ValueError("rows, table rows, and validation rows must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        raise RuntimeError("this SM120 research benchmark requires NVIDIA Blackwell")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    _benchmark(
        args.rows,
        args.table_rows,
        args.validate_rows,
        args.seed,
        args.warmup_ms,
        args.measurement_time_ms,
    )


if __name__ == "__main__":
    main()
