"""Benchmark unpacked UINT4-P and ConvRot-INT4-V Sage2++ experiments."""

import argparse
from collections.abc import Callable, Sequence

import torch
import triton.testing

from piper_kernels.attention._sage2pp.backends.triton import _run_sage_attention
from piper_kernels.attention._sage2pp.experiments import (
    triton_sage_attention_uint4_pv_convrot,
    triton_sage_attention_uint4_pv_paired_convrot,
)


def _do_bench(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = actual.float() - expected.float()
    signal = expected.float().square().mean()
    noise = error.square().mean().clamp_min(1e-30)
    sqnr = float(10 * torch.log10(signal / noise))
    relative_l1 = float(error.abs().sum() / expected.float().abs().sum().clamp_min(1e-30))
    return sqnr, relative_l1


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
    scale = head_dim**-0.5

    def canonical() -> torch.Tensor:
        return _run_sage_attention(
            query,
            key,
            value,
            scale,
            is_causal,
            qk_quantization_range=127,
            grouped_qk=grouped_qk,
        )

    def uint4_pv() -> torch.Tensor:
        return triton_sage_attention_uint4_pv_convrot(
            query,
            key,
            value,
            scale,
            is_causal,
            rotation_group=0,
            grouped_qk=grouped_qk,
        )

    def feature_convrot() -> torch.Tensor:
        return triton_sage_attention_uint4_pv_convrot(
            query,
            key,
            value,
            scale,
            is_causal,
            rotation_group=rotation_group,
            grouped_qk=grouped_qk,
        )

    def paired_convrot() -> torch.Tensor:
        return triton_sage_attention_uint4_pv_paired_convrot(
            query,
            key,
            value,
            scale,
            is_causal,
            rotation_group=rotation_group,
            grouped_qk=grouped_qk,
        )

    def sdpa() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal
        )

    expected = sdpa()
    providers = (canonical, uint4_pv, feature_convrot, paired_convrot)
    outputs = [provider() for provider in providers]
    timings = [_do_bench(provider, warmup_ms, repeat_ms) for provider in providers]
    qualities = [_quality(output, expected) for output in outputs]
    print(
        f"| {query_length} | {key_length} | {timings[0]:.4f} | {timings[1]:.4f} "
        f"| {timings[2]:.4f} | {timings[3]:.4f} | "
        f"{qualities[0][0]:.2f}/{qualities[0][1]:.4f} | "
        f"{qualities[1][0]:.2f}/{qualities[1][1]:.4f} | "
        f"{qualities[2][0]:.2f}/{qualities[2][1]:.4f} | "
        f"{qualities[3][0]:.2f}/{qualities[3][1]:.4f} |"
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
    parser.add_argument("--grouped-qk", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run synthetic latency and quality comparisons for the PV experiments."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("UINT4 PV benchmarking requires a CUDA-capable GPU")
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
        "UINT4/INT4 codes use unpacked INT8 storage and INT8 tl.dot; timings are prototype "
        f"orchestration costs, not native four-bit MMA. rotation group: {args.rotation_group}"
    )
    print()
    print(
        "| query | key/value | 8+8 ms | UINT4-P/INT4-V ms | feature-rot ms | paired-rot ms "
        "| 8+8 SQNR/L1 | direct SQNR/L1 | feature SQNR/L1 | paired SQNR/L1 |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for query_length in args.sequence:
        _run_shape(
            query_length,
            args.kv_sequence or query_length,
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
