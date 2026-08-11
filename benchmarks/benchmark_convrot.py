"""Benchmark Piper ConvRot entrypoints against portable and optional providers."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import torch
from lib.convrot import (
    CONVROT_DTYPE_NAMES,
    ConvRotConfig,
    ConvRotShape,
    comfy_convrot_input,
    convrot_dtype,
    raw_input_features,
)
from lib.convrot_providers import (
    make_convrot_workload,
    make_public_convrot_provider,
    make_reference_convrot_provider,
    sample_convrot_measurement,
)
from lib.environment import EnvironmentInfo, capture_environment
from lib.providers import BenchmarkProvider, ProviderMeasurement, measure_provider
from lib.quality import QualityMetrics, measure_quality
from lib.reporting import (
    BenchmarkRecord,
    add_output_arguments,
    output_target,
    write_records,
)

from piper_kernels.convrot._rotation import SUPPORTED_GROUP_SIZES


def _minimax_h3_shapes(rows: int) -> tuple[ConvRotShape, ...]:
    """Return the principal bias-free MiniMax H3 transformer projections."""
    return (
        ConvRotShape("qkv", rows, 21_504, 5_376, has_bias=False),
        ConvRotShape("attention-out", rows, 5_376, 7_168, has_bias=False),
        ConvRotShape("mlp-fc1", rows, 28_672, 5_376, has_bias=False),
        ConvRotShape(
            "mlp-fc2",
            rows,
            5_376,
            14_336,
            input_activation="swiglu",
            has_bias=False,
        ),
    )


_MINIMAX_H3_5S_SHAPES = _minimax_h3_shapes(37_710)
_MINIMAX_H3_128K_SHAPES = _minimax_h3_shapes(131_072)
_MAX_SWIGLU_RELATIVE_L2_ERROR = 0.01
_MAX_COMFY_RELATIVE_L2_ERROR = 0.02
_CUSTOM_SHAPE_DEFAULTS = {
    "rows": [1, 16, 64, 256],
    "out_features": 4096,
    "in_features": 4096,
    "input_activation": None,
    "no_bias": False,
}
_CUSTOM_SHAPE_OPTIONS = {
    "rows": "--rows",
    "out_features": "--out-features",
    "in_features": "--in-features",
    "input_activation": "--input-activation",
    "no_bias": "--no-bias",
}


@dataclass(slots=True, frozen=True)
class Result:
    """Timing and sampled-quality result for one activation and weight shape."""

    quality_row_indices: tuple[int, ...]
    input_preparation: str | None
    piper: ProviderMeasurement[torch.Tensor]
    reference: ProviderMeasurement[torch.Tensor] | None
    quality: QualityMetrics
    comfy_kitchen: ProviderMeasurement[torch.Tensor] | None = None
    comfy_kitchen_quality: QualityMetrics | None = None

    @property
    def speedup(self) -> float | None:
        """Return the warmed reference-to-Piper execution-time ratio when timed."""
        if self.reference is None:
            return None
        return (
            self.reference.timings.prepared_execution.median_ms
            / self.piper.timings.prepared_execution.median_ms
        )

    @property
    def comfy_kitchen_speedup(self) -> float | None:
        """Return the Comfy Kitchen-to-Piper execution-time ratio when requested."""
        if self.comfy_kitchen is None:
            return None
        return (
            self.comfy_kitchen.timings.prepared_execution.median_ms
            / self.piper.timings.prepared_execution.median_ms
        )


def _validated_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    input_activation: str | None,
) -> QualityMetrics:
    quality = measure_quality(actual, expected)
    if input_activation == "swiglu":
        if (
            quality.nonfinite_mismatch_count
            or quality.relative_l2_error > _MAX_SWIGLU_RELATIVE_L2_ERROR
        ):
            raise AssertionError(
                "SwiGLU quality exceeded the declared limit: "
                f"relative L2 {quality.relative_l2_error:.6f}, "
                f"non-finite mismatches {quality.nonfinite_mismatch_count}"
            )
    else:
        torch.testing.assert_close(actual, expected)
    return quality


def _package_version(distribution: str, module: ModuleType) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        value = getattr(module, "__version__", None)
        return str(value) if value is not None else "unknown"


def _load_comfy_kitchen() -> ModuleType:
    try:
        module = importlib.import_module("comfy_kitchen")
    except (ImportError, OSError) as error:
        raise SystemExit(
            "--compare-comfy-kitchen requires an optional comfy-kitchen installation "
            "with its CUDA backend"
        ) from error
    if not callable(getattr(module, "int8_linear", None)):
        raise SystemExit("the installed comfy-kitchen does not expose int8_linear")
    return module


def _comfy_backend_context(
    comfy_kitchen: ModuleType,
) -> AbstractContextManager[object]:
    """Prefer the public CUDA-backend selector when the installed version exposes it."""
    use_backend = getattr(comfy_kitchen, "use_backend", None)
    if not callable(use_backend):
        return nullcontext()
    return cast(AbstractContextManager[object], use_backend("cuda"))


def _comfy_provider_configuration(
    common_configuration: dict[str, object],
    shape: ConvRotShape,
    installed_version: str,
) -> dict[str, object]:
    """Describe Comfy Kitchen's public operator and input-layout adaptation."""
    return {
        **common_configuration,
        "installed_version": installed_version,
        "operation_entrypoint": "comfy_kitchen.int8_linear",
        "input_preparation": "provider-managed" if shape.input_activation else "none",
        "provider_input_layout": "gate_up" if shape.input_activation == "swiglu" else "plain",
        "input_layout_adapter": shape.input_activation == "swiglu",
    }


@torch.inference_mode()
def _run_shape(
    shape: ConvRotShape,
    config: ConvRotConfig,
    warmup_ms: int,
    measurement_time_ms: int,
    comfy_kitchen: ModuleType | None,
    skip_reference_timing: bool,
) -> Result:
    workload = make_convrot_workload(
        shape,
        config,
        device=torch.device("cuda"),
    )
    piper_measurement = measure_provider(
        make_public_convrot_provider(workload),
        warmup_ms=warmup_ms,
        measurement_time_ms=measurement_time_ms,
        measure_preparation=False,
    )
    piper_measurement = sample_convrot_measurement(workload, piper_measurement)

    if skip_reference_timing:
        reference_measurement = None
        reference_quality_output = workload.sampled_reference()
    else:
        reference_measurement = sample_convrot_measurement(
            workload,
            measure_provider(
                make_reference_convrot_provider(workload),
                warmup_ms=warmup_ms,
                measurement_time_ms=measurement_time_ms,
                measure_first_call=False,
                measure_preparation=False,
            ),
        )
        reference_quality_output = reference_measurement.output
    piper_quality = _validated_quality(
        piper_measurement.output,
        reference_quality_output,
        shape.input_activation,
    )

    comfy_measurement = None
    comfy_quality = None
    if comfy_kitchen is not None:
        activation, qdata, scale, bias = workload.inputs
        comfy_activation = comfy_convrot_input(activation, shape.input_activation)

        def comfy_optimized() -> torch.Tensor:
            return comfy_kitchen.int8_linear(
                comfy_activation,
                qdata,
                scale,
                bias,
                config.dtype,
                convrot=True,
                convrot_groupsize=config.group_size,
                input_act=shape.input_activation,
            )

        comfy_config = _comfy_provider_configuration(
            workload.common_configuration(),
            shape,
            _package_version("comfy-kitchen", comfy_kitchen),
        )
        with _comfy_backend_context(comfy_kitchen):
            comfy_measurement = sample_convrot_measurement(
                workload,
                measure_provider(
                    BenchmarkProvider(
                        name="comfy-kitchen",
                        prepare=lambda: workload.inputs,
                        run=lambda _prepared: comfy_optimized(),
                        synchronize=torch.cuda.synchronize,
                        configuration=comfy_config,
                    ),
                    warmup_ms=warmup_ms,
                    measurement_time_ms=measurement_time_ms,
                    measure_preparation=False,
                ),
            )
        comfy_quality = measure_quality(comfy_measurement.output, reference_quality_output)
        if (
            comfy_quality.nonfinite_mismatch_count
            or comfy_quality.relative_l2_error > _MAX_COMFY_RELATIVE_L2_ERROR
        ):
            raise AssertionError(
                "comfy-kitchen quality exceeded the declared limit: "
                f"relative L2 {comfy_quality.relative_l2_error:.6f}, "
                f"non-finite mismatches {comfy_quality.nonfinite_mismatch_count}"
            )

    return Result(
        quality_row_indices=workload.sampled_row_indices,
        input_preparation=workload.input_preparation,
        piper=piper_measurement,
        reference=reference_measurement,
        quality=piper_quality,
        comfy_kitchen=comfy_measurement,
        comfy_kitchen_quality=comfy_quality,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["custom", "minimax-h3-5s", "minimax-h3-128k"],
        default="custom",
        help="benchmark custom dimensions or a principal MiniMax H3 shape matrix",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="custom-shape activation rows M (default: 1 16 64 256)",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=argparse.SUPPRESS,
        help="custom-shape linear output width N (default: 4096)",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=argparse.SUPPRESS,
        help=("custom-shape linear/weight width K; raw SwiGLU input width is 2K (default: 4096)"),
    )
    parser.add_argument("--group-size", type=int, choices=SUPPORTED_GROUP_SIZES, default=256)
    parser.add_argument(
        "--input-activation",
        choices=("swiglu",),
        default=argparse.SUPPRESS,
        help="custom-shape raw-input activation; SwiGLU expects [up | gate] with width 2K",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        default=argparse.SUPPRESS,
        help="omit bias from the custom shape",
    )
    parser.add_argument(
        "--dtype",
        choices=CONVROT_DTYPE_NAMES,
        default="bfloat16",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-comfy-kitchen",
        action="store_true",
        help="benchmark the optional comfy-kitchen CUDA ConvRot provider",
    )
    parser.add_argument(
        "--skip-reference-timing",
        action="store_true",
        help=(
            "validate a stratified row sample but skip the full portable-reference timing; "
            "this is automatic for the 128K preset"
        ),
    )
    add_output_arguments(parser)
    arguments = parser.parse_args(argv)
    arguments.custom_shape_options = tuple(
        option for name, option in _CUSTOM_SHAPE_OPTIONS.items() if hasattr(arguments, name)
    )
    for name, default in _CUSTOM_SHAPE_DEFAULTS.items():
        if not hasattr(arguments, name):
            setattr(arguments, name, default)
    return arguments


def _benchmark_shapes(args: argparse.Namespace) -> tuple[ConvRotShape, ...]:
    if args.preset == "minimax-h3-5s":
        return _MINIMAX_H3_5S_SHAPES
    if args.preset == "minimax-h3-128k":
        return _MINIMAX_H3_128K_SHAPES
    return tuple(
        ConvRotShape(
            "custom",
            rows,
            args.out_features,
            args.in_features,
            input_activation=args.input_activation,
            has_bias=not args.no_bias,
        )
        for rows in args.rows
    )


def _skip_reference_timing(args: argparse.Namespace) -> bool:
    return args.skip_reference_timing or args.preset == "minimax-h3-128k"


def _validate_args(args: argparse.Namespace) -> None:
    if args.preset != "custom" and args.custom_shape_options:
        options = ", ".join(args.custom_shape_options)
        raise SystemExit(f"--preset {args.preset} cannot be combined with {options}")
    if args.preset == "custom" and (
        any(rows <= 0 for rows in args.rows) or args.out_features <= 0 or args.in_features <= 0
    ):
        raise SystemExit("rows, out_features, and in_features must all be positive")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if args.preset == "custom" and args.in_features % args.group_size:
        raise SystemExit("every in_features value must be divisible by --group-size")
    if args.compare_comfy_kitchen and args.dtype == "float32":
        raise SystemExit("comfy-kitchen comparison supports float16 and bfloat16")


def _print_header(compare_comfy_kitchen: bool, skip_reference_timing: bool) -> None:
    columns = [
        "case",
        "input activation",
        "Piper input preparation",
        "bias",
        "M (rows)",
        "N (output features)",
        "linear K",
        "raw input features",
        "first Piper call, wall (ms)",
        "Piper fixed-source execution, device p50 [p20, p80] (ms)",
    ]
    if compare_comfy_kitchen:
        columns.extend(
            [
                "comfy-kitchen fixed-source execution, device p50 [p20, p80] (ms)",
                "comfy-kitchen / Piper",
            ]
        )
    if not skip_reference_timing:
        if not compare_comfy_kitchen:
            columns.append("reference fixed-source execution, device p50 [p20, p80] (ms)")
        columns.append("reference / Piper")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(":---" if index < 4 else "---:" for index in range(len(columns))) + "|")


def _print_result(shape: ConvRotShape, result: Result) -> None:
    first_call_ms = result.piper.timings.first_call_ms
    assert first_call_ms is not None
    cells = [
        shape.name,
        shape.input_activation or "none",
        result.input_preparation or "not applicable",
        str(shape.has_bias),
        str(shape.rows),
        str(shape.out_features),
        str(shape.in_features),
        str(raw_input_features(shape.in_features, shape.input_activation)),
        f"{first_call_ms:.3f}",
        result.piper.timings.prepared_execution.display(),
    ]
    if result.comfy_kitchen is not None:
        assert result.comfy_kitchen_speedup is not None
        cells.extend(
            [
                result.comfy_kitchen.timings.prepared_execution.display(),
                f"{result.comfy_kitchen_speedup:.2f}x",
            ]
        )
    if result.reference is not None:
        assert result.speedup is not None
        if result.comfy_kitchen is None:
            cells.append(result.reference.timings.prepared_execution.display())
        cells.append(f"{result.speedup:.2f}x")
    print("| " + " | ".join(cells) + " |")


def _records_for_result(
    shape: ConvRotShape,
    result: Result,
    environment: EnvironmentInfo,
) -> list[BenchmarkRecord]:
    shape_record = shape.as_dict()
    measurements = [
        (result.piper, result.quality),
        (result.comfy_kitchen, result.comfy_kitchen_quality),
    ]
    if result.reference is not None:
        measurements.append(
            (
                result.reference,
                measure_quality(result.reference.output, result.reference.output),
            )
        )
    records = []
    for measurement, quality in measurements:
        if measurement is None or quality is None:
            continue
        records.append(
            BenchmarkRecord(
                benchmark="convrot-linear",
                provider=measurement.provider,
                shape=shape_record,
                configuration=measurement.configuration,
                timings=measurement.timings,
                quality=quality,
                environment=environment,
                extra={
                    "quality_rows": len(result.quality_row_indices),
                    "quality_row_indices": list(result.quality_row_indices),
                },
            )
        )
    return records


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested benchmark matrix and print a Markdown table."""
    args = _parse_args(argv)
    _validate_args(args)
    shapes = _benchmark_shapes(args)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot benchmarking requires a Triton-supported GPU")

    config = ConvRotConfig(
        dtype=convrot_dtype(args.dtype),
        group_size=args.group_size,
        seed=args.seed,
    )
    comfy_kitchen = _load_comfy_kitchen() if args.compare_comfy_kitchen else None
    skip_reference_timing = _skip_reference_timing(args)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name}; backend: {environment.accelerator_backend}; "
        f"architecture: {environment.gpu_architecture}"
    )
    print(f"Torch: {torch.__version__}; dtype: {config.dtype}; group size: {config.group_size}")
    if args.preset == "minimax-h3-128k":
        print("Reference: sampled boundary rows (full reference timing disabled for 128K)")
    print()
    _print_header(args.compare_comfy_kitchen, skip_reference_timing)
    records: list[BenchmarkRecord] = []
    for shape in shapes:
        result = _run_shape(
            shape,
            config,
            args.warmup_ms,
            args.measurement_time_ms,
            comfy_kitchen,
            skip_reference_timing,
        )
        _print_result(shape, result)
        records.extend(_records_for_result(shape, result, environment))
        del result
    write_records(records, output_target(args))


if __name__ == "__main__":
    main()
