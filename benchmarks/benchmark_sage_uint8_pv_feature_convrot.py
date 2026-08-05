"""Benchmark feature-only ConvRot V with UINT8-equivalent P."""

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
from piper_kernels.attention._sage2pp.experiments.uint8_pv_feature_convrot import (
    _launch_uint8_pv_feature_convrot_attention,
    _prepare_uint8_pv_feature_convrot_inputs,
    triton_sage_attention_uint8_pv_feature_convrot,
)


@dataclass(slots=True, frozen=True)
class Config:
    """Attention launch configuration."""

    block_m: int
    num_stages: int

    def display(self) -> str:
        return f"M{self.block_m}/S{self.num_stages}"


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
# Benchmark drivers intentionally expose every shape/timing dimension.
# ruff: noqa: PLR0913, PLR0915, PLR0917
def _run_shape(
    sequence: int,
    batch: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    grouped_qk: bool,
    rotations: Sequence[int],
    scale_axes: Sequence[str],
    probability_scale_modes: Sequence[str],
    value_scale_floors: Sequence[float],
    affine_probability: bool,
    native_uint8_mma: bool,
    integer_output_recurrence: bool,
    integer_tile_exponent_recurrence: bool,
    paired_int32_tiles: bool,
    split_pv_head_dim: bool,
    tile_common_log_denominator: bool,
    narrow_int8_log_denominator: bool,
    running_max_probability_recurrence: bool,
    scale_forward_log_recurrence: bool,
    optimize_pv_scaling: bool,
    fp32_pv_scale_metadata: bool | None,
    scaled_fp16_numerator: bool,
    scaled_fp16_denominator: bool,
    warmup_ms: int,
    repeat_ms: int,
    tune_ms: int,
) -> None:
    torch.manual_seed(9100 + sequence)
    query = torch.randn(batch, heads, sequence, head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scale = head_dim**-0.5
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    fixed_prepared = _prepare_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
    )
    block_prepared = _prepare_block_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
    )
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
    variants = tuple(
        (scale_axis, rotation, probability_scale_mode, value_scale_floor)
        for scale_axis in scale_axes
        for rotation in rotations
        for probability_scale_mode in (
            probability_scale_modes if scale_axis == "key" else ("dynamic",)
        )
        for value_scale_floor in (
            value_scale_floors
            if scale_axis == "key" and probability_scale_mode == "tile"
            else (0.0,)
        )
    )
    feature_prepared = {
        variant: _prepare_uint8_pv_feature_convrot_inputs(
            query,
            key,
            value,
            scale,
            grouped_qk=grouped_qk,
            rotation_group=variant[1],
            value_scale_axis=variant[0],
            value_scale_floor=variant[3],
            probability_scale_mode=variant[2],
            affine_probability=affine_probability,
            native_uint8_mma=native_uint8_mma,
            tile_common_log_denominator=tile_common_log_denominator,
            narrow_int8_log_denominator=narrow_int8_log_denominator,
            scale_forward_log_recurrence=scale_forward_log_recurrence,
            fp32_scale_forward_metadata=(
                optimize_pv_scaling
                and (
                    fp32_pv_scale_metadata
                    if fp32_pv_scale_metadata is not None
                    else not scaled_fp16_numerator
                )
            ),
            precompute_pv_multiplier=optimize_pv_scaling,
        )
        for variant in variants
    }

    fp8_output = torch.empty_like(query)
    fixed_output = torch.empty_like(query)
    block_output = torch.empty_like(query)
    rotated_outputs = {
        variant: torch.empty(query.shape, device=query.device, dtype=torch.float32)
        for variant in variants
    }
    feature_outputs = {variant: torch.empty_like(query) for variant in variants}
    block_ms = (32, 64) if sequence <= 512 else (32, 64, 128)
    configs = tuple(Config(block_m, stages) for block_m in block_ms for stages in (2, 3))

    def make_fp8(config: Config) -> Callable[[], torch.Tensor]:
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
                fp8_output,
                sequence,
                sequence,
                is_causal=False,
                grouped_qk=grouped_qk,
                pv_accumulator_fp32=False,
                heads=heads,
                head_dim=head_dim,
                block_m=config.block_m,
                block_n=64,
                value_transposed=True,
                use_tensor_descriptors=use_tensor_descriptors,
                num_warps=4,
                num_stages=config.num_stages,
            )
            return fp8_output

        return run

    def make_block(config: Config) -> Callable[[], torch.Tensor]:
        use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
            query,
            config.block_m,
            head_dim,
            sequence,
            True,
        )

        def run() -> torch.Tensor:
            return _launch_int8_pv_attention(
                block_prepared,
                block_output,
                sequence,
                sequence,
                False,
                grouped_qk=grouped_qk,
                block_m=config.block_m,
                num_warps=4,
                num_stages=config.num_stages,
                block_scaled_pv=True,
                use_tensor_descriptors=use_tensor_descriptors,
            )

        return run

    def make_fixed(config: Config) -> Callable[[], torch.Tensor]:
        use_tensor_descriptors = (
            _sage_backend._should_use_split_pv_tensor_descriptors(
                query,
                config.block_m,
                head_dim,
                sequence,
                True,
            )
            if split_pv_head_dim
            else _sage_backend._should_use_attention_tensor_descriptors(
                query,
                config.block_m,
                head_dim,
                sequence,
                True,
            )
        )

        def run() -> torch.Tensor:
            return _launch_int8_pv_attention(
                fixed_prepared,
                fixed_output,
                sequence,
                sequence,
                False,
                grouped_qk=grouped_qk,
                block_m=config.block_m,
                block_n=64,
                num_warps=4,
                num_stages=config.num_stages,
                split_pv_head_dim=split_pv_head_dim,
                use_tensor_descriptors=use_tensor_descriptors,
            )

        return run

    def make_feature(
        variant: tuple[str, int, str, float],
    ) -> Callable[[Config], Callable[[], torch.Tensor]]:
        def make(config: Config) -> Callable[[], torch.Tensor]:
            if split_pv_head_dim:
                use_tensor_descriptors = (
                    torch.cuda.get_device_capability(query.device)[0] == 12
                    and scaled_fp16_numerator
                    and config.block_m == 128
                    and head_dim == 128
                    and sequence % 16 == 0
                ) or _sage_backend._should_use_split_pv_tensor_descriptors(
                    query,
                    config.block_m,
                    head_dim,
                    sequence,
                    True,
                )
            else:
                use_tensor_descriptors = (
                    (not affine_probability or native_uint8_mma)
                    and variant[1] == 0
                    and _sage_backend._should_use_attention_tensor_descriptors(
                        query,
                        config.block_m,
                        head_dim,
                        sequence,
                        True,
                    )
                )

            def run() -> torch.Tensor:
                return _launch_uint8_pv_feature_convrot_attention(
                    feature_prepared[variant],
                    rotated_outputs[variant],
                    feature_outputs[variant],
                    sequence,
                    sequence,
                    False,
                    grouped_qk=grouped_qk,
                    rotation_group=variant[1],
                    value_scale_axis=variant[0],
                    probability_scale_mode=variant[2],
                    fuse_output_rotation=True,
                    block_m=config.block_m,
                    num_warps=4,
                    num_stages=config.num_stages,
                    affine_probability=affine_probability,
                    native_uint8_mma=native_uint8_mma,
                    integer_output_recurrence=integer_output_recurrence,
                    integer_tile_exponent_recurrence=integer_tile_exponent_recurrence,
                    paired_int32_tiles=paired_int32_tiles,
                    split_pv_head_dim=split_pv_head_dim,
                    tile_common_log_denominator=tile_common_log_denominator,
                    narrow_int8_log_denominator=narrow_int8_log_denominator,
                    running_max_probability_recurrence=running_max_probability_recurrence,
                    scale_forward_log_recurrence=scale_forward_log_recurrence,
                    factored_pv_scaling=optimize_pv_scaling,
                    precomputed_pv_multiplier=optimize_pv_scaling,
                    scaled_fp16_numerator=scaled_fp16_numerator,
                    scaled_fp16_denominator=scaled_fp16_denominator,
                    unmasked_self_attention=(
                        sequence % config.block_m == 0
                        and sequence % 64 == 0
                        and scale_forward_log_recurrence
                        and native_uint8_mma
                        and split_pv_head_dim
                        and optimize_pv_scaling
                    ),
                    use_tensor_descriptors=use_tensor_descriptors,
                    maxnreg=(
                        168
                        if split_pv_head_dim
                        and scale_forward_log_recurrence
                        and torch.cuda.get_device_capability()[0] == 12
                        and sequence >= 1024
                        and not scaled_fp16_numerator
                        else None
                    ),
                )

            return run

        return make

    fp8_config, fp8_hot = _select_fastest(configs, make_fp8, tune_ms)
    fixed_config, fixed_hot = _select_fastest(configs, make_fixed, tune_ms)
    block_config, block_hot = _select_fastest(configs, make_block, tune_ms)
    feature_selected = {
        variant: _select_fastest(configs, make_feature(variant), tune_ms) for variant in variants
    }
    fp8_ms = _bench(fp8_hot, warmup_ms, repeat_ms)
    fixed_ms = _bench(fixed_hot, warmup_ms, repeat_ms)
    block_ms_result = _bench(block_hot, warmup_ms, repeat_ms)
    feature_ms = {
        variant: _bench(function, warmup_ms, repeat_ms)
        for variant, (_, function) in feature_selected.items()
    }
    e2e_ms = {
        variant: _bench(
            lambda variant=variant: triton_sage_attention_uint8_pv_feature_convrot(
                query,
                key,
                value,
                scale,
                False,
                rotation_group=variant[1],
                value_scale_axis=variant[0],
                probability_scale_mode=variant[2],
                value_scale_floor=variant[3],
                grouped_qk=grouped_qk,
                affine_probability=affine_probability,
                native_uint8_mma=native_uint8_mma,
                integer_output_recurrence=integer_output_recurrence,
                integer_tile_exponent_recurrence=integer_tile_exponent_recurrence,
                paired_int32_tiles=paired_int32_tiles,
                split_pv_head_dim=split_pv_head_dim,
                tile_common_log_denominator=tile_common_log_denominator,
                narrow_int8_log_denominator=narrow_int8_log_denominator,
                running_max_probability_recurrence=running_max_probability_recurrence,
                scale_forward_log_recurrence=scale_forward_log_recurrence,
                optimize_pv_scaling=optimize_pv_scaling,
                fp32_pv_scale_metadata=fp32_pv_scale_metadata,
                scaled_fp16_numerator=scaled_fp16_numerator,
                scaled_fp16_denominator=scaled_fp16_denominator,
            ),
            warmup_ms,
            repeat_ms,
        )
        for variant in variants
    }
    fp8_quality = _quality(fp8_hot(), expected)
    fixed_quality = _quality(fixed_hot(), expected)
    block_quality = _quality(block_hot(), expected)
    feature_quality = {
        variant: _quality(function(), expected)
        for variant, (_, function) in feature_selected.items()
    }

    print(
        f"| {sequence} | FP8 | {fp8_ms:.5f} | - | {fp8_quality[0]:.2f}/"
        f"{fp8_quality[1]:.4f} | {fp8_config.display()} |"
    )
    print(
        f"| {sequence} | signed INT8 fixed"
        f"{' split-D64' if split_pv_head_dim else ''} | {fixed_ms:.5f} | - | "
        f"{fixed_quality[0]:.2f}/{fixed_quality[1]:.4f} | {fixed_config.display()} |"
    )
    print(
        f"| {sequence} | signed INT8 block | {block_ms_result:.5f} | - | "
        f"{block_quality[0]:.2f}/{block_quality[1]:.4f} | {block_config.display()} |"
    )
    for variant in variants:
        scale_axis, rotation, probability_scale_mode, value_scale_floor = variant
        config, _ = feature_selected[variant]
        probability_encoding = (
            "native UINT8" if native_uint8_mma else "UINT8" if affine_probability else "signed INT8"
        )
        recurrence = (
            " INT32-tile-exponent"
            if integer_tile_exponent_recurrence
            else " INT32-recurrence"
            if integer_output_recurrence
            else ""
        )
        recurrence += " paired-Q10" if paired_int32_tiles else ""
        pv_schedule = " split-D64" if split_pv_head_dim else ""
        denominator = " tile-common-denom" if tile_common_log_denominator else ""
        denominator += " narrow-INT8-denom" if narrow_int8_log_denominator else ""
        recurrence += " running-max-P" if running_max_probability_recurrence else ""
        recurrence += " scale-forward" if scale_forward_log_recurrence else ""
        recurrence += " optimized-PV-scale" if optimize_pv_scaling else ""
        recurrence += " scaled-FP16-numerator" if scaled_fp16_numerator else ""
        recurrence += " scaled-FP16-denominator" if scaled_fp16_denominator else ""
        recurrence += (
            " R168"
            if split_pv_head_dim
            and scale_forward_log_recurrence
            and torch.cuda.get_device_capability()[0] == 12
            and sequence >= 1024
            and not scaled_fp16_numerator
            else ""
        )
        print(
            f"| {sequence} | {probability_encoding}{recurrence}{pv_schedule}{denominator} "
            f"{scale_axis}-scale H{rotation} "
            f"P-{probability_scale_mode} V-floor-{value_scale_floor:g} | "
            f"{feature_ms[variant]:.5f} | {e2e_ms[variant]:.5f} "
            f"| {feature_quality[variant][0]:.2f}/"
            f"{feature_quality[variant][1]:.4f} | {config.display()} |"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--grouped-qk", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--affine-probability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use affine UINT8-equivalent P; disable for nonnegative signed INT8 P",
    )
    parser.add_argument(
        "--native-uint8-mma",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use native UINT8 x INT8 MMA; requires a mixed-sign Triton compiler",
    )
    parser.add_argument(
        "--integer-output-recurrence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep the PV numerator in INT32 and convert only in the epilogue",
    )
    parser.add_argument(
        "--integer-tile-exponent-recurrence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use tile-local UINT8 P and align INT32 partials with block exponents",
    )
    parser.add_argument(
        "--paired-int32-tiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="merge pairs of K64 INT32 partials with a Q10 row scale",
    )
    parser.add_argument(
        "--split-pv-head-dim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compute D128 PV as two sequential D64 dots",
    )
    parser.add_argument(
        "--tile-common-log-denominator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="approximate per-key inverse scales with one geometric-mean scale per K64 tile",
    )
    parser.add_argument(
        "--narrow-int8-log-denominator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compute the probability-weighted denominator with a narrow INT8 MMA",
    )
    parser.add_argument(
        "--running-max-probability-recurrence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="quantize P in running-max coordinates and delay its constant scale to the epilogue",
    )
    parser.add_argument(
        "--scale-forward-log-recurrence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="move exact per-key V scaling from the denominator into P construction",
    )
    parser.add_argument(
        "--optimize-pv-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="precompute the FP32 UINT8 probability multiplier during V quantization",
    )
    parser.add_argument(
        "--scaled-fp16-numerator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep the unnormalized key-scaled PV numerator in a fixed 2^-16 FP16 coordinate",
    )
    parser.add_argument(
        "--fp32-pv-scale-metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override FP16/FP32 storage for the precomputed PV multiplier",
    )
    parser.add_argument(
        "--scaled-fp16-denominator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep the softmax denominator in a fixed 2^-4 FP16 coordinate",
    )
    parser.add_argument(
        "--rotations", type=int, nargs="+", choices=[0, 16, 64], default=[0, 16, 64]
    )
    parser.add_argument(
        "--scale-axes",
        nargs="+",
        choices=["feature", "key"],
        default=["feature", "key"],
    )
    parser.add_argument(
        "--probability-scale-modes",
        nargs="+",
        choices=["dynamic", "tile", "log"],
        default=["dynamic", "tile", "log"],
        help="per-key V only; feature-scale controls always use dynamic",
    )
    parser.add_argument(
        "--value-scale-floors",
        type=float,
        nargs="+",
        default=[0.0, 0.0625, 0.125, 0.25],
        help="fractions of the K-tile maximum; applied to tile-scaled per-key V",
    )
    parser.add_argument("--tune-ms", type=int, default=50)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0912
    """Measure full-range P and feature-only V rotations."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("The feature ConvRot benchmark requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9) and capability[0] != 12:
        raise SystemExit("The benchmark requires consumer Ada SM89 or Blackwell SM12x")
    grouped_qk = capability[0] == 12 if args.grouped_qk is None else args.grouped_qk
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if args.integer_output_recurrence and args.integer_tile_exponent_recurrence:
        raise SystemExit("select only one integer recurrence")
    if args.paired_int32_tiles and (
        not args.affine_probability
        or args.scale_axes != ["key"]
        or args.probability_scale_modes != ["log"]
        or args.integer_output_recurrence
        or args.integer_tile_exponent_recurrence
    ):
        raise SystemExit(
            "--paired-int32-tiles requires UINT8, per-key log scaling, and FP32 recurrence"
        )
    if args.split_pv_head_dim and (
        args.head_dim != 128
        or args.rotations != [0]
        or args.scale_axes != ["key"]
        or args.probability_scale_modes != ["log"]
        or args.integer_output_recurrence
        or args.integer_tile_exponent_recurrence
    ):
        raise SystemExit(
            "--split-pv-head-dim requires D128, H0, per-key log scaling, and FP32 recurrence"
        )
    if args.tile_common_log_denominator and (
        args.scale_axes != ["key"] or args.probability_scale_modes != ["log"]
    ):
        raise SystemExit(
            "--tile-common-log-denominator requires --scale-axes key --probability-scale-modes log"
        )
    if args.narrow_int8_log_denominator and (
        args.scale_axes != ["key"]
        or args.probability_scale_modes != ["log"]
        or args.integer_output_recurrence
        or args.integer_tile_exponent_recurrence
        or args.paired_int32_tiles
    ):
        raise SystemExit(
            "--narrow-int8-log-denominator requires per-key log scaling and FP32 recurrence"
        )
    if args.tile_common_log_denominator and args.narrow_int8_log_denominator:
        raise SystemExit("select only one approximate log denominator")
    if args.running_max_probability_recurrence and (
        args.scale_axes != ["key"]
        or args.probability_scale_modes != ["log"]
        or args.integer_output_recurrence
        or args.integer_tile_exponent_recurrence
        or args.paired_int32_tiles
        or args.tile_common_log_denominator
        or args.narrow_int8_log_denominator
    ):
        raise SystemExit(
            "--running-max-probability-recurrence requires exact per-key log FP32 recurrence"
        )
    if args.scale_forward_log_recurrence and (
        args.scale_axes != ["key"]
        or args.probability_scale_modes != ["log"]
        or args.running_max_probability_recurrence
        or args.integer_output_recurrence
        or args.integer_tile_exponent_recurrence
        or args.tile_common_log_denominator
        or args.narrow_int8_log_denominator
    ):
        raise SystemExit(
            "--scale-forward-log-recurrence requires exact per-key log FP32 recurrence"
        )
    if args.optimize_pv_scaling and (
        not args.scale_forward_log_recurrence
        or not args.native_uint8_mma
        or not args.split_pv_head_dim
    ):
        raise SystemExit(
            "--optimize-pv-scaling requires native UINT8 split-D128 scale-forward recurrence"
        )
    if args.fp32_pv_scale_metadata is not None and not args.optimize_pv_scaling:
        raise SystemExit("PV scale metadata override requires --optimize-pv-scaling")
    if args.scaled_fp16_numerator and (
        not args.scale_forward_log_recurrence
        or not args.native_uint8_mma
        or not args.split_pv_head_dim
        or not args.affine_probability
        or max(args.sequence) > 131072
    ):
        raise SystemExit(
            "--scaled-fp16-numerator requires affine native UINT8 split-D128 "
            "scale-forward recurrence with N <= 131072"
        )
    if args.scaled_fp16_denominator and not args.scaled_fp16_numerator:
        raise SystemExit("--scaled-fp16-denominator requires --scaled-fp16-numerator")
    if (args.integer_output_recurrence or args.integer_tile_exponent_recurrence) and (
        args.scale_axes != ["key"] or args.probability_scale_modes != ["log"]
    ):
        raise SystemExit(
            "--integer-output-recurrence requires --scale-axes key --probability-scale-modes log"
        )
    print(f"GPU: {torch.cuda.get_device_name()}; capability: SM{capability[0]}{capability[1]}")
    probability_encoding = (
        "native UINT8"
        if args.native_uint8_mma
        else "UINT8"
        if args.affine_probability
        else "signed INT8"
    )
    print(
        f"Hot {probability_encoding} timings include the inverse when enabled but exclude "
        "Q/K/V preparation."
    )
    print("Feature inverse rotations are fused into the attention epilogue.")
    print(
        "End-to-end timings include Q/K preparation, V rotation/quantization, and output inverse."
    )
    print()
    print("| N | variant | hot ms | e2e ms | SQNR/L1 | config |")
    print("|---:|:---|---:|---:|---:|:---|")
    for sequence in args.sequence:
        _run_shape(
            sequence,
            args.batch,
            args.heads,
            args.head_dim,
            dtype,
            grouped_qk,
            args.rotations,
            args.scale_axes,
            args.probability_scale_modes,
            args.value_scale_floors,
            args.affine_probability,
            args.native_uint8_mma,
            args.integer_output_recurrence,
            args.integer_tile_exponent_recurrence,
            args.paired_int32_tiles,
            args.split_pv_head_dim,
            args.tile_common_log_denominator,
            args.narrow_int8_log_denominator,
            args.running_max_probability_recurrence,
            args.scale_forward_log_recurrence,
            args.optimize_pv_scaling,
            args.fp32_pv_scale_metadata,
            args.scaled_fp16_numerator,
            args.scaled_fp16_denominator,
            args.warmup_ms,
            args.repeat_ms,
            args.tune_ms,
        )


if __name__ == "__main__":
    main()
