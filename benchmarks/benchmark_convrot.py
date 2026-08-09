"""Benchmark Piper ConvRot entrypoints against portable and optional providers."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import cast

import torch
from lib import (
    BenchmarkProvider,
    BenchmarkRecord,
    EnvironmentInfo,
    ProviderMeasurement,
    QualityMetrics,
    add_output_arguments,
    capture_environment,
    measure_provider,
    measure_quality,
    output_target,
    write_records,
)

from piper_kernels.convrot import ConvRotInt8Tensor, convrot_linear
from piper_kernels.convrot.int8 import dispatch as convrot_dispatch
from piper_kernels.convrot.int8.reference import reference_linear


@dataclass(slots=True, frozen=True)
class BenchmarkShape:
    """One named ConvRot linear shape, where ``in_features`` is linear K."""

    name: str
    rows: int
    out_features: int
    in_features: int
    input_activation: str | None = None
    has_bias: bool = True


def _minimax_h3_shapes(rows: int) -> tuple[BenchmarkShape, ...]:
    """Return the principal bias-free MiniMax H3 transformer projections."""
    return (
        BenchmarkShape("qkv", rows, 21_504, 5_376, has_bias=False),
        BenchmarkShape("attention-out", rows, 5_376, 7_168, has_bias=False),
        BenchmarkShape("mlp-fc1", rows, 28_672, 5_376, has_bias=False),
        BenchmarkShape(
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
_MAX_QUALITY_ROWS = 256
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


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _parse_input_activation(value: str) -> str | None:
    """Map the CLI compatibility spelling ``none`` to Python ``None``."""
    if value == "none":
        return None
    if value == "swiglu":
        return value
    raise argparse.ArgumentTypeError("input activation must be 'none' or 'swiglu'")


def _raw_input_features(shape: BenchmarkShape) -> int:
    """Return the source width before applying an optional input activation."""
    return shape.in_features * (2 if shape.input_activation == "swiglu" else 1)


def _quality_row_indices(shape: BenchmarkShape) -> tuple[int, ...]:
    """Stratify quality rows and retain every first signed-32-bit crossing."""
    target = min(shape.rows, _MAX_QUALITY_ROWS)
    if target == shape.rows:
        return tuple(range(shape.rows))

    sampled = {round(index * (shape.rows - 1) / (target - 1)) for index in range(target)}
    critical = {0, shape.rows - 1}
    raw_input_width = _raw_input_features(shape)
    for row_width in (raw_input_width, shape.in_features, shape.out_features):
        boundary = ((1 << 31) + row_width - 1) // row_width
        critical.update(
            row for row in (boundary - 1, boundary, boundary + 1) if 0 <= row < shape.rows
        )

    for row in sorted(critical):
        if row in sampled:
            continue
        victim = min(sampled - critical, key=lambda candidate: abs(candidate - row))
        sampled.remove(victim)
        sampled.add(row)
    return tuple(sorted(sampled))


def _apply_input_activation(activation: torch.Tensor, input_activation: str | None) -> torch.Tensor:
    """Apply the benchmark's public raw-input activation contract."""
    if input_activation is None:
        return activation
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    up, gate = activation.chunk(2, dim=-1)
    return up * torch.nn.functional.silu(gate)


def _comfy_input(activation: torch.Tensor, input_activation: str | None) -> torch.Tensor:
    """Adapt Piper's [up|gate] contract to Comfy Kitchen 0.2.x [gate|up]."""
    if input_activation is None:
        return activation
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    up, gate = activation.chunk(2, dim=-1)
    return torch.cat((gate, up), dim=-1)


def _assert_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    input_activation: str | None,
) -> None:
    if input_activation == "swiglu":
        quality = measure_quality(actual, expected)
        if (
            quality.nonfinite_mismatch_count
            or quality.relative_l2_error > _MAX_SWIGLU_RELATIVE_L2_ERROR
        ):
            raise AssertionError(
                "SwiGLU quality exceeded the declared limit: "
                f"relative L2 {quality.relative_l2_error:.6f}, "
                f"non-finite mismatches {quality.nonfinite_mismatch_count}"
            )
        return
    torch.testing.assert_close(actual, expected)


def _sample_measurement(
    measurement: ProviderMeasurement[torch.Tensor],
    quality_index: torch.Tensor,
) -> ProviderMeasurement[torch.Tensor]:
    """Drop a potentially multi-gigabyte result after retaining sampled rows."""
    sampled = measurement.output.index_select(0, quality_index)
    return replace(measurement, output=sampled)


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


def _selected_input_preparation(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    group_size: int,
    input_activation: str | None,
) -> str | None:
    """Report the input-activation preparation selected by public dispatch."""
    if input_activation is None:
        return None
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    if convrot_dispatch._can_use_triton_swiglu(activation, qdata, group_size):
        return "fused"
    return "materialized"


@torch.inference_mode()
def _run_shape(
    shape: BenchmarkShape,
    group_size: int,
    dtype: torch.dtype,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
    comfy_kitchen: ModuleType | None,
    skip_reference_timing: bool,
) -> Result:
    rows = shape.rows
    out_features = shape.out_features
    in_features = shape.in_features
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    scale = (
        torch.rand(
            out_features,
            1,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    activation = torch.randn(
        rows,
        _raw_input_features(shape),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    bias = (
        torch.randn(out_features, device="cuda", dtype=dtype, generator=generator)
        if shape.has_bias
        else None
    )
    weight = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=group_size,
        logical_dtype=dtype,
    )
    input_preparation = _selected_input_preparation(
        activation,
        qdata,
        group_size,
        shape.input_activation,
    )

    quality_row_indices = _quality_row_indices(shape)
    quality_index = torch.tensor(quality_row_indices, device="cuda")

    def reference(value: torch.Tensor = activation) -> torch.Tensor:
        return reference_linear(
            _apply_input_activation(value, shape.input_activation),
            qdata,
            scale,
            group_size,
            bias,
        )

    def piper_operator() -> torch.Tensor:
        if shape.input_activation == "swiglu":
            return convrot_linear(
                activation,
                weight,
                bias,
                input_activation="swiglu",
            )
        return torch.nn.functional.linear(activation, weight, bias)

    common_config = {
        "dtype": str(dtype).removeprefix("torch."),
        "group_size": group_size,
        "input_activation": shape.input_activation or "none",
        "input_layout": "up_gate" if shape.input_activation == "swiglu" else "plain",
        "has_bias": shape.has_bias,
        "seed": seed,
        "prepared_execution_scope": "complete_operator_on_fixed_source_tensors",
    }
    piper_config = {
        **common_config,
        "operation_entrypoint": (
            "piper_kernels.convrot.convrot_linear"
            if shape.input_activation == "swiglu"
            else "torch.nn.functional.linear"
        ),
        "input_preparation": input_preparation or "none",
    }
    piper_measurement = measure_provider(
        BenchmarkProvider(
            name="piper-convrot",
            prepare=lambda: None,
            run=lambda _prepared: piper_operator(),
            synchronize=torch.cuda.synchronize,
            configuration=piper_config,
        ),
        warmup_ms=warmup_ms,
        measurement_time_ms=measurement_time_ms,
        measure_preparation=False,
    )
    piper_measurement = _sample_measurement(piper_measurement, quality_index)

    if skip_reference_timing:
        reference_measurement = None
        sampled_activation = activation.index_select(0, quality_index)
        reference_quality_output = reference(sampled_activation)
    else:
        reference_config = {
            **common_config,
            "operation_entrypoint": "piper_kernels.convrot.int8.reference.reference_linear",
            "input_preparation": "materialized" if shape.input_activation else "none",
        }
        full_reference_measurement = measure_provider(
            BenchmarkProvider(
                name="torch-reference",
                prepare=lambda: None,
                run=lambda _prepared: reference(),
                synchronize=torch.cuda.synchronize,
                configuration=reference_config,
            ),
            warmup_ms=warmup_ms,
            measurement_time_ms=measurement_time_ms,
            measure_first_call=False,
            measure_preparation=False,
        )
        reference_measurement = _sample_measurement(full_reference_measurement, quality_index)
        del full_reference_measurement
        reference_quality_output = reference_measurement.output
    _assert_quality(piper_measurement.output, reference_quality_output, shape.input_activation)

    comfy_measurement = None
    comfy_quality = None
    if comfy_kitchen is not None:
        comfy_activation = _comfy_input(activation, shape.input_activation)

        def comfy_optimized() -> torch.Tensor:
            return comfy_kitchen.int8_linear(
                comfy_activation,
                qdata,
                scale,
                bias,
                dtype,
                convrot=True,
                convrot_groupsize=group_size,
                input_act=shape.input_activation,
            )

        comfy_config = {
            **common_config,
            "version": _package_version("comfy-kitchen", comfy_kitchen),
            "operation_entrypoint": "comfy_kitchen.int8_linear",
            "input_preparation": "provider-managed" if shape.input_activation else "none",
            "input_layout": "gate_up" if shape.input_activation == "swiglu" else "plain",
            "input_layout_adapter": shape.input_activation == "swiglu",
        }
        with _comfy_backend_context(comfy_kitchen):
            full_comfy_measurement = measure_provider(
                BenchmarkProvider(
                    name="comfy-kitchen",
                    prepare=lambda: None,
                    run=lambda _prepared: comfy_optimized(),
                    synchronize=torch.cuda.synchronize,
                    configuration=comfy_config,
                ),
                warmup_ms=warmup_ms,
                measurement_time_ms=measurement_time_ms,
                measure_preparation=False,
            )
        comfy_measurement = _sample_measurement(full_comfy_measurement, quality_index)
        del full_comfy_measurement
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
        quality_row_indices=quality_row_indices,
        input_preparation=input_preparation,
        piper=piper_measurement,
        reference=reference_measurement,
        quality=measure_quality(piper_measurement.output, reference_quality_output),
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
    parser.add_argument("--group-size", type=int, choices=[16, 64, 256], default=256)
    parser.add_argument(
        "--input-activation",
        type=_parse_input_activation,
        metavar="{none,swiglu}",
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
        choices=["bfloat16", "float16", "float32"],
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


def _benchmark_shapes(args: argparse.Namespace) -> tuple[BenchmarkShape, ...]:
    if args.preset == "minimax-h3-5s":
        return _MINIMAX_H3_5S_SHAPES
    if args.preset == "minimax-h3-128k":
        return _MINIMAX_H3_128K_SHAPES
    return tuple(
        BenchmarkShape(
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


def _validate_args(args: argparse.Namespace, shapes: Sequence[BenchmarkShape]) -> None:
    if args.preset != "custom" and args.custom_shape_options:
        options = ", ".join(args.custom_shape_options)
        raise SystemExit(f"--preset {args.preset} cannot be combined with {options}")
    if any(
        dimension <= 0
        for shape in shapes
        for dimension in (shape.rows, shape.out_features, shape.in_features)
    ):
        raise SystemExit("rows, out_features, and in_features must all be positive")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if any(shape.in_features % args.group_size for shape in shapes):
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


def _print_result(shape: BenchmarkShape, result: Result) -> None:
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
        str(_raw_input_features(shape)),
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
    shape: BenchmarkShape,
    result: Result,
    environment: EnvironmentInfo,
) -> list[BenchmarkRecord]:
    shape_record = {
        "case": shape.name,
        "rows": shape.rows,
        "out_features": shape.out_features,
        "in_features": shape.in_features,
        "raw_input_features": _raw_input_features(shape),
        "input_activation": shape.input_activation or "none",
        "input_layout": "up_gate" if shape.input_activation == "swiglu" else "plain",
        "has_bias": shape.has_bias,
    }
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
    shapes = _benchmark_shapes(args)
    _validate_args(args, shapes)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot benchmarking requires a Triton-supported GPU")

    dtype = _dtype(args.dtype)
    comfy_kitchen = _load_comfy_kitchen() if args.compare_comfy_kitchen else None
    skip_reference_timing = _skip_reference_timing(args)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name}; backend: {environment.accelerator_backend}; "
        f"architecture: {environment.gpu_architecture}"
    )
    print(f"Torch: {torch.__version__}; dtype: {dtype}; group size: {args.group_size}")
    if args.preset == "minimax-h3-128k":
        print("Reference: sampled boundary rows (full reference timing disabled for 128K)")
    print()
    _print_header(args.compare_comfy_kitchen, skip_reference_timing)
    records: list[BenchmarkRecord] = []
    for shape in shapes:
        result = _run_shape(
            shape,
            args.group_size,
            dtype,
            args.seed,
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
