"""Benchmark paired-Hadamard INT8 PV with grouped V RMS scales."""

import argparse
from collections.abc import Callable, Sequence

import torch
import triton.testing

from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.int8_pv_convrot_rms import (
    _launch_int8_pv_convrot_rms_attention,
    _prepare_int8_pv_convrot_rms_inputs,
    triton_sage_attention_int8_pv_convrot_rms,
)


def _bench(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = actual.float() - expected.float()
    sqnr = 10 * torch.log10(
        expected.float().square().mean() / error.square().mean().clamp_min(1e-30)
    )
    relative_l1 = error.abs().sum() / expected.float().abs().sum().clamp_min(1e-30)
    return float(sqnr), float(relative_l1)


@torch.inference_mode()
def _run_shape(
    sequence: int,
    batch: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    grouped_qk: bool,
    warmup_ms: int,
    repeat_ms: int,
) -> None:
    query = torch.randn(batch, heads, sequence, head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention_scale = head_dim**-0.5
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    block_m = _sage_backend._select_query_block(query, batch, heads, sequence)

    prepared = {
        group_tiles: _prepare_int8_pv_convrot_rms_inputs(
            query,
            key,
            value,
            attention_scale,
            grouped_qk=grouped_qk,
            value_rms_group_tiles=group_tiles,
        )
        for group_tiles in (1, 2)
    }
    outputs = {group_tiles: torch.empty_like(query) for group_tiles in (1, 2)}

    def hot(group_tiles: int) -> torch.Tensor:
        return _launch_int8_pv_convrot_rms_attention(
            prepared[group_tiles],
            outputs[group_tiles],
            sequence,
            sequence,
            False,
            grouped_qk=grouped_qk,
            value_rms_group_tiles=group_tiles,
            local_probability=False,
            block_m=block_m,
            num_warps=4,
            num_stages=3,
        )

    def end_to_end(group_tiles: int) -> torch.Tensor:
        return triton_sage_attention_int8_pv_convrot_rms(
            query,
            key,
            value,
            attention_scale,
            False,
            grouped_qk=grouped_qk,
            value_rms_group_tiles=group_tiles,
            local_probability=False,
        )

    for group_tiles in (1, 2):
        hot(group_tiles)
        end_to_end(group_tiles)
    hot_timings = {
        group_tiles: _bench(lambda group_tiles=group_tiles: hot(group_tiles), warmup_ms, repeat_ms)
        for group_tiles in (1, 2)
    }
    end_to_end_timings = {
        group_tiles: _bench(
            lambda group_tiles=group_tiles: end_to_end(group_tiles), warmup_ms, repeat_ms
        )
        for group_tiles in (1, 2)
    }
    qualities = {group_tiles: _quality(hot(group_tiles), expected) for group_tiles in (1, 2)}
    print(
        f"| {sequence} | {hot_timings[1]:.5f} | {hot_timings[2]:.5f} "
        f"| {hot_timings[1] / hot_timings[2]:.3f}x "
        f"| {end_to_end_timings[1]:.5f} | {end_to_end_timings[2]:.5f} "
        f"| {end_to_end_timings[1] / end_to_end_timings[2]:.3f}x "
        f"| {qualities[1][0]:.2f}/{qualities[1][1]:.4f} "
        f"| {qualities[2][0]:.2f}/{qualities[2][1]:.4f} | M{block_m} |"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--grouped-qk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Measure one- and two-tile V RMS scale groups."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("The grouped RMS benchmark requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9) and capability[0] != 12:
        raise SystemExit("The benchmark requires consumer Ada SM89 or Blackwell SM12x")
    grouped_qk = capability[0] == 12 if args.grouped_qk is None else args.grouped_qk
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"GPU: {torch.cuda.get_device_name()}; capability: SM{capability[0]}{capability[1]}")
    print("Both variants use K=64 IMMA/Hadamard tiles; only V RMS scale grouping differs.")
    print()
    print(
        "| N | G1 hot ms | G2 hot ms | G1/G2 hot | G1 e2e ms | G2 e2e ms "
        "| G1/G2 e2e | G1 SQNR/L1 | G2 SQNR/L1 | config |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    for sequence in args.sequence:
        _run_shape(
            sequence,
            args.batch,
            args.heads,
            args.head_dim,
            dtype,
            grouped_qk,
            args.warmup_ms,
            args.repeat_ms,
        )


if __name__ == "__main__":
    main()
