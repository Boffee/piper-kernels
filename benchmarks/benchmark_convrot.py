"""Benchmark ConvRot's Triton backend against the portable PyTorch reference."""

import argparse
import importlib
import importlib.metadata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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

from piper_kernels.convrot import ConvRotInt8Tensor, linear_input_act
from piper_kernels.convrot.int8.reference import reference_linear


@dataclass(slots=True, frozen=True)
class BenchmarkShape:
    """One named ConvRot linear shape."""

    name: str
    rows: int
    out_features: int
    in_features: int
    input_act: str | None = None
    has_bias: bool = True


_MINIMAX_H3_5S_SHAPES = (
    BenchmarkShape("qkv", 37_710, 21_504, 5_376, has_bias=False),
    BenchmarkShape("attention-out", 37_710, 5_376, 7_168, has_bias=False),
    BenchmarkShape("mlp-fc1", 37_710, 28_672, 5_376, has_bias=False),
    BenchmarkShape("mlp-fc2", 37_710, 5_376, 14_336, input_act="swiglu", has_bias=False),
)
_MAX_QUALITY_ROWS = 256
_MAX_SWIGLU_RELATIVE_L2_ERROR = 0.01
_MAX_COMFY_RELATIVE_L2_ERROR = 0.02


@dataclass(slots=True, frozen=True)
class Result:
    """Timing result for one activation and weight shape."""

    rows: int
    out_features: int
    in_features: int
    quality_rows: int
    quality_row_indices: tuple[int, ...]
    triton: ProviderMeasurement[torch.Tensor]
    reference: ProviderMeasurement[torch.Tensor] | None
    quality: QualityMetrics
    comfy_kitchen: ProviderMeasurement[torch.Tensor] | None = None
    comfy_kitchen_quality: QualityMetrics | None = None

    @property
    def speedup(self) -> float | None:
        """Return the warmed reference-to-Triton speed ratio."""
        if self.reference is None:
            return None
        return (
            self.reference.timings.prepared_execution.median_ms
            / self.triton.timings.prepared_execution.median_ms
        )

    @property
    def comfy_kitchen_speedup(self) -> float | None:
        """Return the Comfy Kitchen-to-Piper execution-time ratio when requested."""
        if self.comfy_kitchen is None:
            return None
        return (
            self.comfy_kitchen.timings.prepared_execution.median_ms
            / self.triton.timings.prepared_execution.median_ms
        )


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _quality_row_indices(shape: BenchmarkShape) -> tuple[int, ...]:
    """Sample rows across M and preserve every signed-32-bit boundary."""
    target = min(shape.rows, _MAX_QUALITY_ROWS)
    if target == shape.rows:
        return tuple(range(shape.rows))

    sampled = {round(index * (shape.rows - 1) / (target - 1)) for index in range(target)}
    critical = {0, shape.rows - 1}
    raw_input_width = shape.in_features * (2 if shape.input_act == "swiglu" else 1)
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


def _apply_input_act(activation: torch.Tensor, input_act: str | None) -> torch.Tensor:
    if input_act != "swiglu":
        return activation
    gate, up = activation.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _assert_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    input_act: str | None,
) -> None:
    if input_act == "swiglu":
        quality = measure_quality(actual, expected)
        if (
            quality.nonfinite_mismatch_count
            or quality.relative_l2_error > _MAX_SWIGLU_RELATIVE_L2_ERROR
        ):
            raise AssertionError(
                "fused SwiGLU quality exceeded the declared limit: "
                f"relative L2 {quality.relative_l2_error:.6f}, "
                f"non-finite mismatches {quality.nonfinite_mismatch_count}"
            )
    else:
        torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def _run_shape(
    shape: BenchmarkShape,
    group_size: int,
    dtype: torch.dtype,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
    compare_comfy_kitchen: bool,
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
        in_features * (2 if shape.input_act == "swiglu" else 1),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    bias = (
        torch.randn(out_features, device="cuda", dtype=dtype, generator=generator)
        if shape.has_bias
        else None
    )
    weight = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=group_size,
        dtype=dtype,
    )

    quality_row_indices = _quality_row_indices(shape)
    quality_index = torch.tensor(quality_row_indices, device="cuda")

    def reference(value: torch.Tensor = activation) -> torch.Tensor:
        return reference_linear(
            _apply_input_act(value, shape.input_act), qdata, scale, group_size, bias
        )

    def optimized() -> torch.Tensor:
        if shape.input_act == "swiglu":
            return linear_input_act(activation, weight, "swiglu", bias)
        return torch.nn.functional.linear(activation, weight, bias)

    provider_config = {
        "dtype": str(dtype).removeprefix("torch."),
        "group_size": group_size,
        "input_act": shape.input_act or "none",
        "has_bias": shape.has_bias,
        "seed": seed,
    }
    optimized_measurement = measure_provider(
        BenchmarkProvider(
            name="triton",
            prepare=lambda: None,
            run=lambda _prepared: optimized(),
            synchronize=torch.cuda.synchronize,
            configuration=provider_config,
        ),
        warmup_ms=warmup_ms,
        measurement_time_ms=measurement_time_ms,
        measure_preparation=False,
    )
    if skip_reference_timing:
        reference_measurement = None
        reference_quality_output = reference(activation.index_select(0, quality_index))
    else:
        reference_measurement = measure_provider(
            BenchmarkProvider(
                name="torch-reference",
                prepare=lambda: None,
                run=lambda _prepared: reference(),
                synchronize=torch.cuda.synchronize,
                configuration=provider_config,
            ),
            warmup_ms=warmup_ms,
            measurement_time_ms=measurement_time_ms,
            measure_first_call=False,
            measure_preparation=False,
        )
        reference_quality_output = reference_measurement.output.index_select(0, quality_index)
    optimized_quality_output = optimized_measurement.output.index_select(0, quality_index)
    _assert_quality(optimized_quality_output, reference_quality_output, shape.input_act)

    comfy_measurement = None
    comfy_quality = None
    if compare_comfy_kitchen:
        output_dtype_code = 1 if dtype is torch.float16 else 2

        def comfy_optimized() -> torch.Tensor:
            return torch.ops.comfy_kitchen.int8_linear(
                activation,
                qdata,
                scale,
                bias,
                output_dtype_code,
                True,
                group_size,
                shape.input_act,
            )

        comfy_measurement = measure_provider(
            BenchmarkProvider(
                name="comfy-kitchen",
                prepare=lambda: None,
                run=lambda _prepared: comfy_optimized(),
                synchronize=torch.cuda.synchronize,
                configuration={
                    **provider_config,
                    "version": importlib.metadata.version("comfy-kitchen"),
                },
            ),
            warmup_ms=warmup_ms,
            measurement_time_ms=measurement_time_ms,
            measure_preparation=False,
        )
        comfy_quality = measure_quality(
            comfy_measurement.output.index_select(0, quality_index),
            reference_quality_output,
        )
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
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        quality_rows=len(quality_row_indices),
        quality_row_indices=quality_row_indices,
        triton=optimized_measurement,
        reference=reference_measurement,
        quality=measure_quality(
            optimized_quality_output,
            reference_quality_output,
        ),
        comfy_kitchen=comfy_measurement,
        comfy_kitchen_quality=comfy_quality,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["custom", "minimax-h3-5s"],
        default="custom",
        help=(
            "benchmark custom dimensions or the principal MiniMax H3 5-second "
            "transformer linear shapes"
        ),
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--group-size", type=int, choices=[16, 64, 256], default=256)
    parser.add_argument("--input-act", choices=["none", "swiglu"], default="none")
    parser.add_argument("--no-bias", action="store_true")
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
        help="benchmark ComfyUI's comfy-kitchen ConvRot operator when installed",
    )
    parser.add_argument(
        "--skip-reference-timing",
        action="store_true",
        help=(
            "validate a stratified row sample but skip the full portable-reference timing; "
            "use this for memory-intensive shapes"
        ),
    )
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _benchmark_shapes(args: argparse.Namespace) -> tuple[BenchmarkShape, ...]:
    if args.preset == "minimax-h3-5s":
        return _MINIMAX_H3_5S_SHAPES
    return tuple(
        BenchmarkShape(
            "custom",
            rows,
            args.out_features,
            args.in_features,
            input_act=None if args.input_act == "none" else args.input_act,
            has_bias=not args.no_bias,
        )
        for rows in args.rows
    )


def _print_header(compare_comfy_kitchen: bool, skip_reference_timing: bool) -> None:
    columns = [
        "case",
        "act",
        "bias",
        "M",
        "N",
        "K",
        "first Triton call, wall (ms)",
        "Triton prepared execution, device p50 [p20, p80] (ms)",
    ]
    if compare_comfy_kitchen:
        columns.extend(
            [
                "comfy-kitchen prepared execution, device p50 [p20, p80] (ms)",
                "comfy-kitchen / Triton",
            ]
        )
    if not skip_reference_timing:
        if not compare_comfy_kitchen:
            columns.append("reference prepared execution, device p50 [p20, p80] (ms)")
        columns.append("reference / Triton")
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(":---" if index < 3 else "---:" for index in range(len(columns))) + "|")


def _print_result(shape: BenchmarkShape, result: Result) -> None:
    first_call_ms = result.triton.timings.first_call_ms
    assert first_call_ms is not None
    cells = [
        shape.name,
        shape.input_act or "none",
        str(shape.has_bias),
        str(result.rows),
        str(result.out_features),
        str(result.in_features),
        f"{first_call_ms:.3f}",
        result.triton.timings.prepared_execution.display(),
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
        "rows": result.rows,
        "out_features": result.out_features,
        "in_features": result.in_features,
        "input_act": shape.input_act or "none",
        "has_bias": shape.has_bias,
    }
    measurements = [
        (result.triton, result.quality),
        (result.comfy_kitchen, result.comfy_kitchen_quality),
    ]
    if result.reference is not None:
        reference_quality_output = result.reference.output.index_select(
            0,
            torch.tensor(result.quality_row_indices, device="cuda"),
        )
        measurements.append(
            (
                result.reference,
                measure_quality(reference_quality_output, reference_quality_output),
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
                    "quality_rows": result.quality_rows,
                    "quality_row_indices": list(result.quality_row_indices),
                },
            )
        )
    return records


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested benchmark matrix and print a Markdown table."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot benchmarking requires a Triton-supported GPU")
    shapes = _benchmark_shapes(args)
    if any(
        dimension <= 0
        for shape in shapes
        for dimension in (shape.rows, shape.out_features, shape.in_features)
    ):
        raise SystemExit("rows, out_features, and in_features must all be positive")
    if any(shape.in_features % args.group_size for shape in shapes):
        raise SystemExit("every in_features value must be divisible by --group-size")
    if args.compare_comfy_kitchen:
        if args.dtype == "float32":
            raise SystemExit("comfy-kitchen comparison supports float16 and bfloat16")
        try:
            importlib.import_module("comfy_kitchen")
        except ImportError as error:
            raise SystemExit(
                "--compare-comfy-kitchen requires the optional comfy-kitchen package"
            ) from error

    dtype = _dtype(args.dtype)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name}; backend: {environment.accelerator_backend}; "
        f"architecture: {environment.gpu_architecture}"
    )
    print(f"Torch: {torch.__version__}; dtype: {dtype}; group size: {args.group_size}")
    print()
    _print_header(args.compare_comfy_kitchen, args.skip_reference_timing)
    records: list[BenchmarkRecord] = []
    for shape in shapes:
        result = _run_shape(
            shape,
            args.group_size,
            dtype,
            args.seed,
            args.warmup_ms,
            args.measurement_time_ms,
            args.compare_comfy_kitchen,
            args.skip_reference_timing,
        )
        _print_result(shape, result)
        records.extend(_records_for_result(shape, result, environment))
        del result
    write_records(records, output_target(args))


if __name__ == "__main__":
    main()
