"""Measure ConvRot activation rotation and rowwise INT8 preparation phases."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from lib import (
    BenchmarkProvider,
    BenchmarkRecord,
    EnvironmentInfo,
    PhaseTimings,
    Timing,
    add_compiler_inspection_arguments,
    add_output_arguments,
    capture_environment,
    format_compiler_report,
    inspect_provider,
    output_target,
    triton_benchmark,
    write_records,
)

from piper_kernels.convrot.int8 import triton as triton_backend


@dataclass(frozen=True, slots=True)
class PreparationPhaseResult:
    """One allocation-free preparation phase measurement."""

    phase: str
    provider: str
    operation_provenance: str
    timing: Timing
    minimum_global_bytes: int
    effective_minimum_tbps: float
    baseline_phase: str
    speedup_vs_baseline: float


_PHASE_PROVENANCE = {
    "rotate": "piper_kernels.convrot.int8.triton._rotate_activations",
    "quantize": "piper_kernels.convrot.int8.triton._quantize_activations",
    "split": "Piper _rotate_activations followed by _quantize_activations",
    "fused": "piper_kernels.convrot.int8.triton._fused_rotate_quantize_activations",
    "comfy-kitchen": "comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64",
}


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _parse_input_activation(value: str) -> str | None:
    """Map the CLI compatibility spelling ``none`` to Python ``None``."""
    if value == "none":
        return None
    if value == "swiglu":
        return value
    raise argparse.ArgumentTypeError("input activation must be 'none' or 'swiglu'")


def _raw_input_features(in_features: int, input_activation: str | None) -> int:
    """Return the source width before applying an optional input activation."""
    return in_features * (2 if input_activation == "swiglu" else 1)


def _baseline_phase(input_activation: str | None) -> str:
    """Return the Piper phase used as the relative-speed baseline."""
    return "fused" if input_activation == "swiglu" else "split"


def _minimum_global_bytes(
    phase: str,
    rows: int,
    in_features: int,
    element_size: int,
    input_activation: str | None = None,
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
        if input_activation == "swiglu":
            return elements * (2 * element_size + torch.int8.itemsize) + scale_bytes
        return quantize_bytes
    raise ValueError(f"unknown preparation phase {phase!r}")


def _effective_tbps(byte_count: int, latency_ms: float) -> float:
    """Normalize the algorithmic minimum byte count by device latency."""
    return byte_count / latency_ms / 1e9


def _apply_input_activation(
    activation: torch.Tensor,
    input_activation: str | None,
) -> torch.Tensor:
    if input_activation is None:
        return activation
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    up, gate = activation.chunk(2, dim=-1)
    return up * torch.nn.functional.silu(gate)


def _comfy_input(activation: torch.Tensor, input_activation: str | None) -> torch.Tensor:
    """Adapt Piper's [up|gate] input to Comfy Kitchen 0.2.x [gate|up]."""
    if input_activation is None:
        return activation
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    up, gate = activation.chunk(2, dim=-1)
    return torch.cat((gate, up), dim=-1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=37_710,
        help="activation rows M (default: 37710)",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        nargs="+",
        default=[5_376, 7_168, 14_336],
        help="linear/weight widths K; raw SwiGLU input width is 2K",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--input-activation",
        type=_parse_input_activation,
        metavar="{none,swiglu}",
        default=None,
        help="raw-input activation; SwiGLU expects [up | gate] with width 2K",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-comfy-kitchen",
        action="store_true",
        help="include comfy-kitchen's optional preallocated CUDA preparation kernel",
    )
    add_output_arguments(parser, record_name="preparation phase")
    add_compiler_inspection_arguments(parser)
    return parser.parse_args(argv)


def _load_comfy_kitchen_cuda() -> ModuleType:
    """Load the version-pinned development adapter only when explicitly requested."""
    try:
        module = importlib.import_module("comfy_kitchen.backends.cuda")
    except (ImportError, OSError) as error:
        raise SystemExit(
            "--compare-comfy-kitchen requires an optional comfy-kitchen installation "
            "with its CUDA backend"
        ) from error
    if not hasattr(module, "_C") or not callable(getattr(module, "_wrap_for_dlpack", None)):
        raise SystemExit(
            "the installed comfy-kitchen CUDA backend is incompatible with the "
            "preparation benchmark; use comfy-kitchen==0.2.28"
        )
    return module


def _comfy_preparation_launcher(
    comfy_kitchen: ModuleType,
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    input_activation_code: int,
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
            input_activation_code,
            0,
            stream_ptr,
        )

    return launch


def _assert_fused_quality(
    split_qdata: torch.Tensor,
    split_scale: torch.Tensor,
    fused_qdata: torch.Tensor,
    fused_scale: torch.Tensor,
    dtype: torch.dtype,
    input_activation: str | None,
) -> None:
    if input_activation is None:
        torch.testing.assert_close(fused_qdata, split_qdata, rtol=0, atol=0)
        torch.testing.assert_close(fused_scale, split_scale, rtol=0, atol=0)
        return

    qdata_error = (split_qdata.to(torch.int16) - fused_qdata.to(torch.int16)).abs().max().item()
    if qdata_error > 1:
        raise AssertionError(f"fused SwiGLU qdata differs from split path by {qdata_error}")
    torch.testing.assert_close(
        fused_scale,
        split_scale,
        rtol=2 * torch.finfo(dtype).eps,
        atol=0,
    )


def _benchmark_width(
    rows: int,
    in_features: int,
    dtype: torch.dtype,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
    comfy_kitchen: ModuleType | None,
    input_activation: str | None,
) -> tuple[PreparationPhaseResult, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    raw_activation = torch.randn(
        rows,
        _raw_input_features(in_features, input_activation),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    activation = _apply_input_activation(raw_activation, input_activation)
    rotated = torch.empty_like(activation)
    split_qdata = torch.empty_like(activation, dtype=torch.int8)
    split_scale = torch.empty(rows, device="cuda", dtype=torch.float32)
    fused_qdata = torch.empty_like(split_qdata)
    fused_scale = torch.empty_like(split_scale)
    dtype_code = triton_backend._input_dtype_code(dtype)

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
            input_activation_code=1 if input_activation == "swiglu" else 0,
        )

    split()
    fused()
    sample_rows = min(rows, 64)
    sample = slice(0, sample_rows)
    _assert_fused_quality(
        split_qdata[sample],
        split_scale[sample],
        fused_qdata[sample],
        fused_scale[sample],
        dtype,
        input_activation,
    )

    phases: list[tuple[str, Callable[[], object]]] = [("fused", fused)]
    baseline_phase = _baseline_phase(input_activation)
    if input_activation is None:
        phases[:0] = [
            ("rotate", rotate),
            ("quantize", quantize),
            ("split", split),
        ]
    if comfy_kitchen is not None:
        comfy_activation = _comfy_input(raw_activation, input_activation)
        comfy_qdata = torch.empty_like(split_qdata)
        comfy_scale = torch.empty((rows, 1), device="cuda", dtype=torch.float32)
        comfy_launch = _comfy_preparation_launcher(
            comfy_kitchen,
            comfy_activation,
            comfy_qdata,
            comfy_scale,
            2 if input_activation == "swiglu" else 0,
        )
        comfy_launch()
        comfy_qdata_error = (
            (comfy_qdata[sample].to(torch.int16) - fused_qdata[sample].to(torch.int16))
            .abs()
            .max()
            .item()
        )
        if comfy_qdata_error > 2:
            raise AssertionError(f"comfy-kitchen qdata differs from Piper by {comfy_qdata_error}")
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
    results = []
    for phase, _launch in phases:
        timing = timings[phase]
        traffic = _minimum_global_bytes(
            phase,
            rows,
            in_features,
            dtype.itemsize,
            input_activation,
        )
        results.append(
            PreparationPhaseResult(
                phase=phase,
                provider="comfy-kitchen" if phase == "comfy-kitchen" else "piper-triton",
                operation_provenance=_PHASE_PROVENANCE[phase],
                timing=timing,
                minimum_global_bytes=traffic,
                effective_minimum_tbps=_effective_tbps(traffic, timing.median_ms),
                baseline_phase=baseline_phase,
                speedup_vs_baseline=baseline_median / timing.median_ms,
            )
        )
    return tuple(results)


def _print_phase_result(
    in_features: int,
    input_activation: str | None,
    result: PreparationPhaseResult,
) -> None:
    print(
        f"| {in_features} | {_raw_input_features(in_features, input_activation)} "
        f"| {result.phase} | {result.timing.display()} "
        f"| {result.minimum_global_bytes / 1e9:.3f} "
        f"| {result.effective_minimum_tbps:.3f} | {result.speedup_vs_baseline:.2f}x |"
    )


def _preparation_records(
    rows: int,
    in_features: int,
    dtype_name: str,
    input_activation: str | None,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
    results: Sequence[PreparationPhaseResult],
    environment: EnvironmentInfo,
) -> list[BenchmarkRecord]:
    """Convert phase measurements to the common machine-readable schema."""
    shape = {
        "rows": rows,
        "in_features": in_features,
        "raw_input_features": _raw_input_features(in_features, input_activation),
    }
    input_activation_name = input_activation or "none"
    input_layout = "up_gate" if input_activation == "swiglu" else "plain"
    records = []
    for result in results:
        records.append(
            BenchmarkRecord(
                benchmark="convrot-preparation",
                provider=result.provider,
                shape=shape,
                configuration={
                    "dtype": dtype_name,
                    "group_size": 256,
                    "input_activation": input_activation_name,
                    "input_layout": input_layout,
                    "phase": result.phase,
                    "baseline_provider": "piper-triton",
                    "baseline_phase": result.baseline_phase,
                    "operation_provenance": result.operation_provenance,
                    "prepared_execution_scope": "fixed_source_preallocated_outputs",
                    "seed": seed,
                },
                timings=PhaseTimings(
                    warmup_ms=warmup_ms,
                    measurement_time_ms=measurement_time_ms,
                    first_call_ms=None,
                    preparation=None,
                    prepared_execution=result.timing,
                    operator_end_to_end=None,
                ),
                quality=None,
                environment=environment,
                extra={
                    "minimum_global_bytes": result.minimum_global_bytes,
                    "effective_minimum_tbps": result.effective_minimum_tbps,
                    "speedup_vs_baseline": result.speedup_vs_baseline,
                },
            )
        )
    return records


def _inspection_provider(args: argparse.Namespace) -> BenchmarkProvider[None, None]:
    jit_functions = {"fused": triton_backend._rotate_quantize_rows_kernel}
    if args.input_activation is None:
        jit_functions = {
            "rotate": triton_backend._rotate_groups_kernel,
            "quantize": triton_backend._quantize_rows_kernel,
            **jit_functions,
        }
    return BenchmarkProvider(
        name="triton-convrot-preparation",
        prepare=lambda: None,
        run=lambda _prepared: None,
        configuration={
            "rows": args.rows,
            "in_features": args.in_features[0],
            "dtype": args.dtype,
            "group_size": 256,
            "input_activation": args.input_activation or "none",
            "input_layout": "up_gate" if args.input_activation == "swiglu" else "plain",
        },
        triton_jit_functions=jit_functions,
    )


def _compiler_requested(args: argparse.Namespace) -> bool:
    return args.compiler_report or output_target(args, option_prefix="compiler") is not None


def _validate_args(args: argparse.Namespace) -> None:
    if args.rows <= 0 or any(width <= 0 for width in args.in_features):
        raise SystemExit("--rows and every --in-features value must be positive")
    if any(width % 256 for width in args.in_features):
        raise SystemExit("every --in-features value must be divisible by 256")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if _compiler_requested(args) and len(args.in_features) != 1:
        raise SystemExit("compiler inspection requires exactly one --in-features value")
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot preparation benchmarking requires a CUDA GPU")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """Run allocation-free phase timings for each requested activation width."""
    args = _parse_args(argv)
    _validate_args(args)
    dtype = _dtype(args.dtype)
    comfy_kitchen = _load_comfy_kitchen_cuda() if args.compare_comfy_kitchen else None
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name} ({environment.gpu_architecture}); "
        f"Torch: {environment.torch_version}; Triton: {environment.triton_version}"
    )
    print(
        f"rows={args.rows} dtype={args.dtype} group_size=256 "
        f"input_activation={args.input_activation or 'none'} "
        f"baseline=piper-{_baseline_phase(args.input_activation)}\n"
    )
    print(
        "| linear K | raw input features | phase | device p50 [p20, p80] (ms) "
        "| minimum global traffic (GB) | effective minimum traffic (TB/s) "
        f"| speedup vs Piper {_baseline_phase(args.input_activation)} |"
    )
    print("|---:|---:|:---|---:|---:|---:|---:|")
    records: list[BenchmarkRecord] = []
    for in_features in args.in_features:
        results = _benchmark_width(
            args.rows,
            in_features,
            dtype,
            args.seed,
            args.warmup_ms,
            args.measurement_time_ms,
            comfy_kitchen,
            args.input_activation,
        )
        for result in results:
            _print_phase_result(in_features, args.input_activation, result)
        records.extend(
            _preparation_records(
                args.rows,
                in_features,
                args.dtype,
                args.input_activation,
                args.seed,
                args.warmup_ms,
                args.measurement_time_ms,
                results,
                environment,
            )
        )
    write_records(records, output_target(args))

    compiler_output = output_target(args, option_prefix="compiler")
    if _compiler_requested(args):
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
