"""Benchmark ConvRot's Triton backend against the portable PyTorch reference."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from _lib import (
    BenchmarkProvider,
    BenchmarkRecord,
    ProviderMeasurement,
    QualityMetrics,
    add_output_arguments,
    capture_environment,
    measure_provider,
    measure_quality,
    output_target,
    write_records,
)

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._int8.reference import reference_linear


@dataclass(slots=True, frozen=True)
class Result:
    """Timing result for one activation and weight shape."""

    rows: int
    out_features: int
    in_features: int
    triton: ProviderMeasurement[torch.Tensor]
    reference: ProviderMeasurement[torch.Tensor]
    quality: QualityMetrics

    @property
    def speedup(self) -> float:
        """Return the warmed reference-to-Triton speed ratio."""
        return (
            self.reference.timings.prepared_execution.median_ms
            / self.triton.timings.prepared_execution.median_ms
        )


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@torch.inference_mode()
def _run_shape(
    rows: int,
    out_features: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    seed: int,
    warmup_ms: int,
    measurement_time_ms: int,
) -> Result:
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
        in_features,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    bias = torch.randn(out_features, device="cuda", dtype=dtype, generator=generator)
    weight = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=group_size,
        dtype=dtype,
    )

    def reference() -> torch.Tensor:
        return reference_linear(activation, qdata, scale, group_size, bias)

    def optimized() -> torch.Tensor:
        return torch.nn.functional.linear(activation, weight, bias)

    provider_config = {
        "dtype": str(dtype).removeprefix("torch."),
        "group_size": group_size,
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
    torch.testing.assert_close(
        optimized_measurement.output,
        reference_measurement.output,
        rtol=0,
        atol=0,
    )

    return Result(
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        triton=optimized_measurement,
        reference=reference_measurement,
        quality=measure_quality(optimized_measurement.output, reference_measurement.output),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--group-size", type=int, choices=[16, 64, 256], default=256)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested benchmark matrix and print a Markdown table."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot benchmarking requires a Triton-supported GPU")
    if args.in_features % args.group_size:
        raise SystemExit("--in-features must be divisible by --group-size")

    dtype = _dtype(args.dtype)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name}; backend: {environment.accelerator_backend}; "
        f"architecture: {environment.gpu_architecture}"
    )
    print(f"Torch: {torch.__version__}; dtype: {dtype}; group size: {args.group_size}")
    print()
    print(
        "| M | N | K | first Triton call (ms) "
        "| Triton prepared execution p50 [p20, p80] (ms) "
        "| reference prepared execution p50 [p20, p80] (ms) | reference / Triton |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    records: list[BenchmarkRecord] = []
    for rows in args.rows:
        result = _run_shape(
            rows,
            args.out_features,
            args.in_features,
            args.group_size,
            dtype,
            args.seed,
            args.warmup_ms,
            args.measurement_time_ms,
        )
        first_call_ms = result.triton.timings.first_call_ms
        assert first_call_ms is not None
        print(
            f"| {result.rows} | {result.out_features} | {result.in_features} "
            f"| {first_call_ms:.3f} "
            f"| {result.triton.timings.prepared_execution.display()} "
            f"| {result.reference.timings.prepared_execution.display()} "
            f"| {result.speedup:.2f}x |"
        )
        shape = {
            "rows": result.rows,
            "out_features": result.out_features,
            "in_features": result.in_features,
        }
        for measurement, quality in (
            (result.triton, result.quality),
            (result.reference, measure_quality(result.reference.output, result.reference.output)),
        ):
            records.append(
                BenchmarkRecord(
                    benchmark="convrot-linear",
                    provider=measurement.provider,
                    shape=shape,
                    configuration=measurement.configuration,
                    timings=measurement.timings,
                    quality=quality,
                    environment=environment,
                )
            )
    write_records(records, output_target(args))


if __name__ == "__main__":
    main()
