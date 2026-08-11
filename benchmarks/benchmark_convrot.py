"""Benchmark Piper ConvRot entrypoints with portable-reference quality."""

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

_MIN_PIPER_SQNR_DB = 20.0
_MAX_COMFY_RELATIVE_L2_ERROR = 0.02


@dataclass(slots=True, frozen=True)
class Result:
    """Provider timing and full-reference quality for one shape."""

    input_preparation: str | None
    piper: ProviderMeasurement[None]
    quality: QualityMetrics
    comfy_kitchen: ProviderMeasurement[None] | None = None
    comfy_kitchen_quality: QualityMetrics | None = None

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
) -> QualityMetrics:
    quality = measure_quality(actual, expected)
    if quality.nonfinite_mismatch_count or not quality.sqnr_db >= _MIN_PIPER_SQNR_DB:
        raise AssertionError(
            "ConvRot quality exceeded the declared limit: "
            f"SQNR {quality.sqnr_db:.2f} dB, "
            f"non-finite mismatches {quality.nonfinite_mismatch_count}"
        )
    return quality


def _without_output(
    measurement: ProviderMeasurement[torch.Tensor],
) -> ProviderMeasurement[None]:
    """Retain timing metadata after its potentially large output is validated."""
    return ProviderMeasurement(
        provider=measurement.provider,
        output=None,
        timings=measurement.timings,
        configuration=measurement.configuration,
    )


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
) -> Result:
    workload = make_convrot_workload(
        shape,
        config,
        device=torch.device("cuda"),
    )
    reference_output = workload.reference()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    piper_with_output = measure_provider(
        make_public_convrot_provider(workload),
        warmup_ms=warmup_ms,
        measurement_time_ms=measurement_time_ms,
        measure_preparation=False,
    )
    piper_quality = _validated_quality(
        piper_with_output.output,
        reference_output,
    )
    piper_measurement = _without_output(piper_with_output)
    del piper_with_output

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
            comfy_with_output = measure_provider(
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
            )
        comfy_quality = measure_quality(comfy_with_output.output, reference_output)
        if (
            comfy_quality.nonfinite_mismatch_count
            or comfy_quality.relative_l2_error > _MAX_COMFY_RELATIVE_L2_ERROR
        ):
            raise AssertionError(
                "comfy-kitchen quality exceeded the declared limit: "
                f"relative L2 {comfy_quality.relative_l2_error:.6f}, "
                f"non-finite mismatches {comfy_quality.nonfinite_mismatch_count}"
            )
        comfy_measurement = _without_output(comfy_with_output)
        del comfy_with_output, comfy_optimized, comfy_activation

    return Result(
        input_preparation=workload.input_preparation,
        piper=piper_measurement,
        quality=piper_quality,
        comfy_kitchen=comfy_measurement,
        comfy_kitchen_quality=comfy_quality,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256],
        help="activation rows M (default: 1 16 64 256)",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=4096,
        help="linear output width N (default: 4096)",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=4096,
        help="linear/weight width K; raw SwiGLU input width is 2K (default: 4096)",
    )
    parser.add_argument("--group-size", type=int, choices=SUPPORTED_GROUP_SIZES, default=256)
    parser.add_argument(
        "--input-activation",
        choices=("swiglu",),
        default=None,
        help="raw-input activation; SwiGLU expects [up | gate] with width 2K",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="omit bias",
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
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _benchmark_shapes(args: argparse.Namespace) -> tuple[ConvRotShape, ...]:
    return tuple(
        ConvRotShape(
            "linear",
            rows,
            args.out_features,
            args.in_features,
            input_activation=args.input_activation,
            has_bias=not args.no_bias,
        )
        for rows in args.rows
    )


def _validate_args(args: argparse.Namespace) -> None:
    if any(rows <= 0 for rows in args.rows) or args.out_features <= 0 or args.in_features <= 0:
        raise SystemExit("rows, out_features, and in_features must all be positive")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if args.in_features % args.group_size:
        raise SystemExit("every in_features value must be divisible by --group-size")
    if args.compare_comfy_kitchen and args.dtype == "float32":
        raise SystemExit("comfy-kitchen comparison supports float16 and bfloat16")


def _print_header(compare_comfy_kitchen: bool) -> None:
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
    records = []
    for measurement, quality in measurements:
        if measurement is None:
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
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name}; backend: {environment.accelerator_backend}; "
        f"architecture: {environment.gpu_architecture}"
    )
    print(f"Torch: {torch.__version__}; dtype: {config.dtype}; group size: {config.group_size}")
    print()
    _print_header(args.compare_comfy_kitchen)
    records: list[BenchmarkRecord] = []
    for shape in shapes:
        result = _run_shape(
            shape,
            config,
            args.warmup_ms,
            args.measurement_time_ms,
            comfy_kitchen,
        )
        _print_result(shape, result)
        records.extend(_records_for_result(shape, result, environment))
        del result
        torch.cuda.empty_cache()
    write_records(records, output_target(args))


if __name__ == "__main__":
    main()
