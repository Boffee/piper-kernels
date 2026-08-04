"""Benchmark the unpacked INT4-range ConvRot QK Sage2++ experiment."""

import argparse
from collections.abc import Callable, Sequence

import torch
import triton.testing

from piper_kernels.attention import sage_attention
from piper_kernels.attention._sage2pp.experiments import (
    triton_sage_attention_int4_convrot,
)


def _do_bench(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    error = actual.float() - expected.float()
    mean_error = error.abs().mean().item()
    max_error = error.abs().max().item()
    signal = expected.float().square().mean()
    noise = error.square().mean().clamp_min(1e-30)
    sqnr = (10 * torch.log10(signal / noise)).item()
    return mean_error, max_error, sqnr


@torch.inference_mode()
def _run_shape(  # noqa: PLR0913, PLR0917
    query_length: int,
    key_length: int,
    batch: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    rotation_group: int,
    grouped_qk: bool,
    warmup_ms: int,
    repeat_ms: int,
) -> None:
    query = torch.randn(batch, heads, query_length, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(batch, heads, key_length, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)

    def int8_sage() -> torch.Tensor:
        return sage_attention(query, key, value, is_causal=is_causal)

    def int4_convrot() -> torch.Tensor:
        return triton_sage_attention_int4_convrot(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            rotation_group=rotation_group,
            grouped_qk=grouped_qk,
        )

    def sdpa() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )

    expected = sdpa()
    int8_output = int8_sage()
    int4_output = int4_convrot()
    int8_quality = _quality(int8_output, expected)
    int4_quality = _quality(int4_output, expected)
    int8_ms = _do_bench(int8_sage, warmup_ms, repeat_ms)
    int4_ms = _do_bench(int4_convrot, warmup_ms, repeat_ms)
    sdpa_ms = _do_bench(sdpa, warmup_ms, repeat_ms)

    print(
        f"| {query_length} | {key_length} | {int8_ms:.4f} | {int4_ms:.4f} "
        f"| {int8_ms / int4_ms:.2f}x | {sdpa_ms:.4f} "
        f"| {int8_quality[0]:.6f}/{int8_quality[1]:.6f}/{int8_quality[2]:.2f} "
        f"| {int4_quality[0]:.6f}/{int4_quality[1]:.6f}/{int4_quality[2]:.2f} |"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--kv-sequence", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--rotation-group", type=int, choices=[16, 64], default=64)
    parser.add_argument(
        "--grouped-qk",
        action="store_true",
        help="use coarse per-warp/per-block QK scales instead of per-thread scales",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the INT4-range experiment and print latency plus error metrics."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("INT4 ConvRot benchmarking requires a CUDA-capable GPU")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9) and capability[0] != 12:
        raise SystemExit("The experiment requires consumer Ada SM89 or Blackwell SM12x")
    if (
        args.causal
        and args.kv_sequence is not None
        and any(sequence != args.kv_sequence for sequence in args.sequence)
    ):
        raise SystemExit("causal attention requires equal query and key/value lengths")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"GPU: {torch.cuda.get_device_name()}; capability: SM{capability[0]}{capability[1]}")
    print(
        "INT4 values are unpacked in INT8 storage: timings do not represent native INT4 MMA. "
        f"rotation group: {args.rotation_group}; grouped QK: {args.grouped_qk}"
    )
    print()
    print(
        "| query | key/value | 8+8 (ms) | unpacked 4+8 ConvRot (ms) | 8+8 / 4+8 "
        "| SDPA (ms) | 8+8 mean/max/SQNR | 4+8 mean/max/SQNR |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for query_length in args.sequence:
        key_length = args.kv_sequence or query_length
        _run_shape(
            query_length,
            key_length,
            args.batch,
            args.heads,
            args.head_dim,
            dtype,
            args.causal,
            args.rotation_group,
            args.grouped_qk,
            args.warmup_ms,
            args.repeat_ms,
        )


if __name__ == "__main__":
    main()
