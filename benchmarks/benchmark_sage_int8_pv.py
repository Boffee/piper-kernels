"""Benchmark direct signed-INT8 PV against Triton Sage2++ FP8 PV."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.testing

from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.int8_pv import (
    _launch_int8_pv_attention,
    _prepare_block_int8_pv_inputs,
    _prepare_int8_pv_inputs,
)


@dataclass(slots=True, frozen=True)
class Config:
    """Attention launch configuration."""

    block_m: int
    block_n: int
    num_stages: int

    def display(self) -> str:
        return f"M{self.block_m}/N{self.block_n}/S{self.num_stages}"


def _bench(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = actual.float() - expected.float()
    sqnr = 10 * torch.log10(
        expected.float().square().mean() / error.square().mean().clamp_min(1e-30)
    )
    relative_l1 = error.abs().sum() / expected.float().abs().sum().clamp_min(1e-30)
    return float(sqnr), float(relative_l1)


def _select_fastest(
    configs: Sequence[Config],
    make_function: Callable[[Config], Callable[[], torch.Tensor]],
    tune_ms: int,
) -> tuple[Config, Callable[[], torch.Tensor]]:
    candidates: list[tuple[float, Config, Callable[[], torch.Tensor]]] = []
    for config in configs:
        function = make_function(config)
        try:
            function()
            latency = _bench(function, tune_ms, tune_ms)
        except triton.runtime.errors.OutOfResources:
            continue
        candidates.append((latency, config, function))
    _, config, function = min(candidates, key=lambda candidate: candidate[0])
    return config, function


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
    tune_ms: int,
) -> None:
    query = torch.randn(batch, heads, sequence, head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = head_dim**-0.5
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    fixed_prepared = _prepare_int8_pv_inputs(query, key, value, scale, grouped_qk=grouped_qk)
    block_prepared = _prepare_block_int8_pv_inputs(query, key, value, scale, grouped_qk=grouped_qk)
    query_int8, key_int8, _, query_scale, key_scale, folded_fp8_scale = fixed_prepared
    value_fp8 = torch.empty(
        (batch, heads, head_dim, sequence),
        device=value.device,
        dtype=torch.float8_e4m3fn,
    )
    _sage_backend._quantize_value_kernel[(triton.cdiv(sequence, 64), heads, batch)](
        value,
        folded_fp8_scale,
        value_fp8,
        sequence,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_fp8.stride(0),
        value_fp8.stride(1),
        value_fp8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=64,
        output_transposed=True,
        num_warps=4,
    )
    fp8_output = torch.empty_like(query)
    fp8_fp32_output = torch.empty_like(query)
    fixed_output = torch.empty_like(query)
    block_output = torch.empty_like(query)

    block_ms = (32, 64) if sequence <= 512 else (32, 64, 128)
    common_configs = tuple(Config(block_m, 64, stages) for block_m in block_ms for stages in (2, 3))
    fixed_configs = common_configs + tuple(
        Config(block_m, 128, stages) for block_m in block_ms for stages in (2, 3)
    )

    def make_fp8(
        output: torch.Tensor, *, accumulator_fp32: bool
    ) -> Callable[[Config], Callable[[], torch.Tensor]]:
        def make(config: Config) -> Callable[[], torch.Tensor]:
            use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
                query,
                config.block_m,
                head_dim,
                sequence,
                True,
            )
            key_argument, value_argument = _sage_backend._make_attention_arguments(
                key_int8,
                value_fp8,
                batch,
                heads,
                sequence,
                head_dim,
                True,
                use_tensor_descriptors,
            )

            def run() -> torch.Tensor:
                _sage_backend._sage_attention_kernel[
                    (triton.cdiv(sequence, config.block_m), heads, batch)
                ](
                    query_int8,
                    key_argument,
                    value_argument,
                    query_scale,
                    key_scale,
                    folded_fp8_scale,
                    output,
                    sequence,
                    sequence,
                    is_causal=False,
                    grouped_qk=grouped_qk,
                    pv_accumulator_fp32=accumulator_fp32,
                    heads=heads,
                    head_dim=head_dim,
                    block_m=config.block_m,
                    block_n=64,
                    value_transposed=True,
                    use_tensor_descriptors=use_tensor_descriptors,
                    num_warps=4,
                    num_stages=config.num_stages,
                )
                return output

            return run

        return make

    def make_int8(
        prepared: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        *,
        block_scaled: bool,
    ) -> Callable[[Config], Callable[[], torch.Tensor]]:
        def make(config: Config) -> Callable[[], torch.Tensor]:
            use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
                query,
                config.block_m,
                head_dim,
                sequence,
                True,
            )

            def run() -> torch.Tensor:
                return _launch_int8_pv_attention(
                    prepared,
                    output,
                    sequence,
                    sequence,
                    False,
                    grouped_qk=grouped_qk,
                    block_m=config.block_m,
                    block_n=config.block_n,
                    num_warps=4,
                    num_stages=config.num_stages,
                    block_scaled_pv=block_scaled,
                    use_tensor_descriptors=use_tensor_descriptors,
                )

            return run

        return make

    fp8_config, fp8_hot = _select_fastest(
        common_configs,
        make_fp8(fp8_output, accumulator_fp32=False),
        tune_ms,
    )
    fp8_fp32_config, fp8_fp32_hot = _select_fastest(
        common_configs,
        make_fp8(fp8_fp32_output, accumulator_fp32=True),
        tune_ms,
    )
    fixed_config, fixed_hot = _select_fastest(
        fixed_configs,
        make_int8(fixed_prepared, fixed_output, block_scaled=False),
        tune_ms,
    )
    block_config, block_hot = _select_fastest(
        common_configs,
        make_int8(block_prepared, block_output, block_scaled=True),
        tune_ms,
    )

    fp8_hot_ms = _bench(fp8_hot, warmup_ms, repeat_ms)
    fp8_fp32_hot_ms = _bench(fp8_fp32_hot, warmup_ms, repeat_ms)
    fixed_hot_ms = _bench(fixed_hot, warmup_ms, repeat_ms)
    block_hot_ms = _bench(block_hot, warmup_ms, repeat_ms)
    fp8_quality = _quality(fp8_hot(), expected)
    fp8_fp32_quality = _quality(fp8_fp32_hot(), expected)
    fixed_quality = _quality(fixed_hot(), expected)
    block_quality = _quality(block_hot(), expected)

    print(
        f"| {sequence} | {fp8_hot_ms:.5f} | {fp8_fp32_hot_ms:.5f} "
        f"| {fixed_hot_ms:.5f} | {block_hot_ms:.5f} "
        f"| {fp8_fp32_hot_ms / fp8_hot_ms:.2f}x | {fixed_hot_ms / fp8_hot_ms:.2f}x "
        f"| {block_hot_ms / fp8_hot_ms:.2f}x | {fp8_quality[0]:.2f}/{fp8_quality[1]:.4f} "
        f"| {fp8_fp32_quality[0]:.2f}/{fp8_fp32_quality[1]:.4f} "
        f"| {fixed_quality[0]:.2f}/{fixed_quality[1]:.4f} "
        f"| {block_quality[0]:.2f}/{block_quality[1]:.4f} "
        f"| {fp8_config.display()} | {fp8_fp32_config.display()} "
        f"| {fixed_config.display()} | {block_config.display()} |"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--grouped-qk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tune-ms", type=int, default=50)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run prequantized FP8 and direct signed-INT8 PV comparisons."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("INT8 PV benchmarking requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9) and capability[0] != 12:
        raise SystemExit("The benchmark requires consumer Ada SM89 or Blackwell SM12x")
    grouped_qk = capability[0] == 12 if args.grouped_qk is None else args.grouped_qk
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name()}; capability: SM{capability[0]}{capability[1]}")
    print("Hot-loop timings use prequantized Q/K/V and select the fastest listed configuration.")
    print()
    print(
        "| N | FP8-FP16 ms | FP8-FP32 ms | fixed INT8 ms | block INT8 ms "
        "| FP32/FP16 | fixed/FP16 | block/FP16 | FP8-FP16 SQNR/L1 "
        "| FP8-FP32 SQNR/L1 | fixed SQNR/L1 | block SQNR/L1 "
        "| FP16 config | FP32 config | fixed config | block config |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---|:---|")
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
            args.tune_ms,
        )


if __name__ == "__main__":
    main()
