"""Measure ConvRot activation rotation and rowwise INT8 preparation phases."""

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import torch
from lib import (
    BenchmarkProvider,
    add_compiler_inspection_arguments,
    capture_environment,
    format_compiler_report,
    inspect_provider,
    output_target,
    triton_benchmark,
    write_records,
)

from piper_kernels.convrot.int8.backends import triton as triton_backend


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _minimum_global_bytes(
    phase: str,
    rows: int,
    in_features: int,
    element_size: int,
    input_act: str = "none",
) -> int:
    """Return algorithmic global traffic, excluding cache and transactions."""
    elements = rows * in_features
    scale_bytes = rows * torch.float32.itemsize
    rotate_bytes = elements * element_size * 2
    quantize_bytes = elements * (element_size + torch.int8.itemsize) + scale_bytes
    if phase == "rotate":
        return rotate_bytes
    if phase == "quantize":
        return quantize_bytes
    if phase == "split":
        return rotate_bytes + quantize_bytes
    if phase in {"fused", "comfy-kitchen"}:
        if input_act == "swiglu":
            return elements * (2 * element_size + torch.int8.itemsize) + scale_bytes
        return quantize_bytes
    raise ValueError(f"unknown preparation phase {phase!r}")


def _effective_tbps(byte_count: int, latency_ms: float) -> float:
    return byte_count / latency_ms / 1e9


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=37_710)
    parser.add_argument(
        "--in-features",
        type=int,
        nargs="+",
        default=[5_376, 7_168, 14_336],
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--input-act", choices=["none", "swiglu"], default="none")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-comfy-kitchen",
        action="store_true",
        help="include comfy-kitchen's fused CUDA preparation when installed",
    )
    add_compiler_inspection_arguments(parser)
    return parser.parse_args(argv)


def _load_comfy_kitchen() -> ModuleType:
    try:
        return importlib.import_module("comfy_kitchen.backends.cuda")
    except ImportError as error:
        raise SystemExit(
            "--compare-comfy-kitchen requires the optional comfy-kitchen package"
        ) from error


def _selected_rotation_kernel(rows: int) -> object:
    if rows >= 512 and triton_backend._is_blackwell(torch.device("cuda")):
        return triton_backend._rotate_groups_matmul_kernel
    return triton_backend._rotate_groups_kernel


def _comfy_preparation_launcher(
    comfy_kitchen: ModuleType,
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    input_act_code: int,
) -> Callable[[], None]:
    """Build a preallocated launch of Comfy Kitchen's pinned native kernel."""
    native = comfy_kitchen._C
    wrap_for_dlpack = comfy_kitchen._wrap_for_dlpack
    stream_ptr = torch.cuda.current_stream(activation.device).cuda_stream

    def launch() -> None:
        native.quantize_int8_rowwise_convrot64(
            wrap_for_dlpack(activation),
            wrap_for_dlpack(qdata),
            wrap_for_dlpack(scale),
            256,
            False,
            input_act_code,
            0,
            stream_ptr,
        )

    return launch


def _benchmark_width(
    rows: int,
    in_features: int,
    dtype: torch.dtype,
    dtype_code: int,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
    comfy_kitchen: ModuleType | None,
    input_act: str,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    raw_activation = torch.randn(
        rows,
        in_features * (2 if input_act == "swiglu" else 1),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    if input_act == "swiglu":
        gate, up = raw_activation.chunk(2, dim=-1)
        activation = torch.nn.functional.silu(gate) * up
    else:
        activation = raw_activation
    rotated = torch.empty_like(activation)
    split_qdata = torch.empty_like(activation, dtype=torch.int8)
    split_scale = torch.empty(rows, device="cuda", dtype=torch.float32)
    fused_qdata = torch.empty_like(split_qdata)
    fused_scale = torch.empty_like(split_scale)
    comfy_qdata = torch.empty_like(split_qdata)
    comfy_scale = torch.empty((rows, 1), device="cuda", dtype=torch.float32)

    def rotate() -> None:
        triton_backend._rotate_activations(activation, rotated, 256)

    def quantize() -> None:
        triton_backend._quantize_activations(rotated, split_qdata, split_scale, dtype_code)

    def split() -> None:
        rotate()
        quantize()

    def fused() -> None:
        triton_backend._fused_rotate_quantize_activations(
            raw_activation,
            fused_qdata,
            fused_scale,
            256,
            dtype_code,
            input_act_code=1 if input_act == "swiglu" else 0,
        )

    split()
    fused()
    sample_rows = min(rows, 64)
    qdata_error = (
        (split_qdata[:sample_rows].to(torch.int16) - fused_qdata[:sample_rows].to(torch.int16))
        .abs()
        .max()
    )
    if qdata_error.item() > 1:
        raise AssertionError(f"fused qdata differs from split path by {qdata_error.item()}")
    torch.testing.assert_close(
        fused_scale[:sample_rows],
        split_scale[:sample_rows],
        rtol=2 * torch.finfo(dtype).eps,
        atol=0,
    )

    phases: list[tuple[str, Callable[[], object]]] = [("fused", fused)]
    baseline_phase = "fused"
    if input_act == "none":
        phases[:0] = [
            ("rotate", rotate),
            ("quantize", quantize),
            ("split", split),
        ]
        baseline_phase = "split"
    if comfy_kitchen is not None:
        comfy_launch = _comfy_preparation_launcher(
            comfy_kitchen,
            raw_activation,
            comfy_qdata,
            comfy_scale,
            2 if input_act == "swiglu" else 0,
        )
        comfy_launch()
        sample = slice(0, sample_rows)
        comfy_qdata_error = (
            (comfy_qdata[sample].to(torch.int16) - fused_qdata[sample].to(torch.int16)).abs().max()
        )
        if comfy_qdata_error.item() > 2:
            raise AssertionError(
                f"comfy-kitchen qdata differs from Piper by {comfy_qdata_error.item()}"
            )
        torch.testing.assert_close(
            comfy_scale[sample, 0],
            fused_scale[sample],
            rtol=0.02,
            atol=0,
        )
        phases.append(("comfy-kitchen", comfy_launch))
    timings = {
        phase: triton_benchmark(launch, warmup_ms, measurement_time_ms) for phase, launch in phases
    }
    baseline_median = timings[baseline_phase].median_ms
    for phase, _launch in phases:
        timing = timings[phase]
        traffic = _minimum_global_bytes(
            phase,
            rows,
            in_features,
            dtype.itemsize,
            input_act,
        )
        print(
            f"| {in_features} | {phase} | {timing.display()} "
            f"| {traffic / 1e9:.3f} | {_effective_tbps(traffic, timing.median_ms):.3f} "
            f"| {baseline_median / timing.median_ms:.2f}x |"
        )


def _inspection_provider(args: argparse.Namespace) -> BenchmarkProvider[None, None]:
    return BenchmarkProvider(
        name="triton-convrot-preparation",
        prepare=lambda: None,
        run=lambda _prepared: None,
        configuration={
            "rows": args.rows,
            "in_features": args.in_features[0],
            "dtype": args.dtype,
            "group_size": 256,
            "input_act": args.input_act,
        },
        triton_jit_functions={
            "rotate": _selected_rotation_kernel(args.rows),
            "quantize": triton_backend._quantize_rows_kernel,
            "fused": triton_backend._rotate_quantize_rows_kernel,
        },
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot preparation benchmarking requires a CUDA GPU")
    if args.rows <= 0 or any(width <= 0 for width in args.in_features):
        raise SystemExit("--rows and every --in-features value must be positive")
    if any(width % 256 for width in args.in_features):
        raise SystemExit("every --in-features value must be divisible by 256")
    compiler_output = output_target(args, option_prefix="compiler")
    if (args.compiler_report or compiler_output is not None) and len(args.in_features) != 1:
        raise SystemExit("compiler inspection requires exactly one --in-features value")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """Run allocation-free phase timings for each requested activation width."""
    args = _parse_args(argv)
    _validate_args(args)
    dtype = _dtype(args.dtype)
    comfy_kitchen = _load_comfy_kitchen() if args.compare_comfy_kitchen else None
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name} ({environment.gpu_architecture}); "
        f"Torch: {environment.torch_version}; Triton: {environment.triton_version}"
    )
    print(f"rows={args.rows} dtype={args.dtype} group_size=256 input_act={args.input_act}\n")
    print(
        "| K | phase | device p50 [p20, p80] (ms) | minimum global traffic (GB) "
        "| effective minimum traffic (TB/s) | baseline / phase |"
    )
    print("|---:|:---|---:|---:|---:|---:|")
    for in_features in args.in_features:
        _benchmark_width(
            args.rows,
            in_features,
            dtype,
            1 if dtype is torch.float16 else 2,
            args.seed,
            args.warmup_ms,
            args.measurement_time_ms,
            comfy_kitchen,
            args.input_act,
        )

    compiler_output = output_target(args, option_prefix="compiler")
    if args.compiler_report or compiler_output is not None:
        report = inspect_provider(
            _inspection_provider(args),
            environment,
            include_sass=args.sass,
            nvdisasm=args.nvdisasm,
        )
        write_records([report], compiler_output)
        if args.compiler_report:
            print()
            print(format_compiler_report(report))


if __name__ == "__main__":
    main()
