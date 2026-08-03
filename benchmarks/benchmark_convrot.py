"""Benchmark ConvRot's Triton backend against the portable PyTorch reference."""

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import triton.testing

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._int8.reference import reference_linear


@dataclass(slots=True, frozen=True)
class Result:
    """Timing result for one activation and weight shape."""

    rows: int
    out_features: int
    in_features: int
    cold_ms: float
    triton_ms: float
    reference_ms: float

    @property
    def speedup(self) -> float:
        """Return the warmed reference-to-Triton speed ratio."""
        return self.reference_ms / self.triton_ms


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _do_bench(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


@torch.inference_mode()
def _run_shape(
    rows: int,
    out_features: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    warmup_ms: int,
    repeat_ms: int,
) -> Result:
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        device="cuda",
        dtype=torch.int8,
    )
    scale = torch.rand(out_features, 1, device="cuda", dtype=torch.float32) * 0.01
    activation = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    bias = torch.randn(out_features, device="cuda", dtype=dtype)
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

    expected = reference()
    torch.cuda.synchronize()
    started = time.perf_counter()
    actual = optimized()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - started) * 1_000
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    return Result(
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        cold_ms=cold_ms,
        triton_ms=_do_bench(optimized, warmup_ms, repeat_ms),
        reference_ms=_do_bench(reference, warmup_ms, repeat_ms),
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
    parser.add_argument("--repeat-ms", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested benchmark matrix and print a Markdown table."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot benchmarking requires a CUDA-capable GPU")
    if args.in_features % args.group_size:
        raise SystemExit("--in-features must be divisible by --group-size")

    dtype = _dtype(args.dtype)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Torch: {torch.__version__}; dtype: {dtype}; group size: {args.group_size}")
    print()
    print("| M | N | K | cold Triton (ms) | Triton (ms) | reference (ms) | speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for rows in args.rows:
        result = _run_shape(
            rows,
            args.out_features,
            args.in_features,
            args.group_size,
            dtype,
            args.warmup_ms,
            args.repeat_ms,
        )
        print(
            f"| {result.rows} | {result.out_features} | {result.in_features} "
            f"| {result.cold_ms:.3f} | {result.triton_ms:.3f} "
            f"| {result.reference_ms:.3f} | {result.speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
