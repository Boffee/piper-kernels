"""Measure Sage INT4 ConvRot quality on real FLUX.2 Klein activations."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    import diffusers.models.transformers.transformer_flux2 as flux2_module
    from diffusers import Flux2KleinPipeline
except ModuleNotFoundError:
    flux2_module = None
    Flux2KleinPipeline = None

from piper_kernels.attention._convrot_reference import rotate_attention_groups
from piper_kernels.attention._sage2pp.backends.triton import _run_sage_attention
from piper_kernels.attention._sage2pp.experiments import (
    triton_sage_attention_int8_pv,
    triton_sage_attention_int8_pv_block_scaled,
    triton_sage_attention_int8_pv_per_key_log,
    triton_sage_attention_uint4_pv_convrot,
    triton_sage_attention_uint4_pv_paired_convrot,
    triton_sage_attention_uint8_pv_feature_convrot,
)
from piper_kernels.attention._sage2pp.reference import (
    _quantize_key_per_thread,
    _quantize_query_per_thread,
)


@dataclass(slots=True, frozen=True)
class Measurement:
    """Quality metrics for one quantization variant at one captured attention."""

    prompt: int
    step: int
    layer: int
    sequence: int
    variant: str
    score_sqnr: float
    output_sqnr: float
    output_relative_l1: float


def _sqnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    noise = (actual.float() - expected.float()).square().mean().clamp_min(1e-30)
    signal = expected.float().square().mean().clamp_min(1e-30)
    return float(10 * torch.log10(signal / noise))


def _relative_l1(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).abs().sum()
    signal = expected.float().abs().sum().clamp_min(1e-30)
    return float(error / signal)


def _quantized_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    quantization_range: int,
    rotation_group: int | None,
) -> torch.Tensor:
    if rotation_group is not None:
        query = rotate_attention_groups(query.float(), rotation_group)
        key = rotate_attention_groups(key.float(), rotation_group)
    key = key.float() - key.float().mean(dim=2, keepdim=True)
    query_int, query_scale = _quantize_query_per_thread(query, quantization_range)
    key_int, key_scale = _quantize_key_per_thread(key, quantization_range)
    integer_scores = torch.matmul(query_int.float(), key_int.transpose(-1, -2).float())
    return integer_scores * query_scale[..., None] * key_scale[:, :, None, :]


def _pv_operand_ablation(
    probabilities: torch.Tensor,
    value: torch.Tensor,
    *,
    quantize_probability: bool,
    quantize_value: bool,
) -> torch.Tensor:
    output = torch.zeros(
        (*probabilities.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    for start in range(0, value.shape[2], 64):
        stop = min(start + 64, value.shape[2])
        probability_block = probabilities[..., start:stop]
        value_block = value[:, :, start:stop].float()
        if quantize_probability:
            probability_max = probability_block.amax(dim=-1)
            probability_scale = torch.where(
                probability_max > 0,
                probability_max / 15,
                torch.ones_like(probability_max),
            )
            probability_block = (probability_block / probability_scale[..., None]).round().clamp(
                0, 15
            ) * probability_scale[..., None]
        if quantize_value:
            value_scale = value_block.abs().amax(dim=2) / 7 + 1e-7
            value_block = (value_block / value_scale[:, :, None, :]).round().clamp(
                -7, 7
            ) * value_scale[:, :, None, :]
        output += torch.matmul(probability_block, value_block)
    return output


def _layer_name(layer: int) -> str:
    return f"joint.{layer}" if layer < 5 else f"single.{layer - 5}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 5, 14, 24])
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[0, 3])
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="repeat to test multiple prompts",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--pv-diagnostics",
        action="store_true",
        help="materialize softmax to isolate P-only and V-only four-bit error",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "int8",
            "int8_pv_fixed",
            "int8_pv_block",
            "int8_pv_key_log_signed",
            "int4",
            "int4_rot16",
            "int4_rot64",
            "uint4_pv",
            "uint4_pv_rot16",
            "uint4_pv_rot64",
            "uint4_pv_pairrot16",
            "uint4_pv_pairrot64",
            "uint8_pv",
            "uint8_pv_rot16",
            "uint8_pv_rot64",
            "uint8_pv_key",
            "uint8_pv_key_rot16",
            "uint8_pv_key_rot64",
            "uint8_pv_key_log",
            "uint8_pv_key_rot64_log",
            "uint8_pv_key_tile",
            "uint8_pv_key_rot16_tile",
            "uint8_pv_key_rot64_tile",
            "uint8_pv_key_tile_floor16",
            "uint8_pv_key_tile_floor8",
            "uint8_pv_key_tile_floor4",
            "uint8_pv_key_rot64_tile_floor16",
            "uint8_pv_key_rot64_tile_floor8",
            "uint8_pv_key_rot64_tile_floor4",
        ],
        help="defaults to every quantization variant",
    )
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0915
    """Run exact inference while shadow-evaluating selected attention calls."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("The real-model SQNR benchmark requires a CUDA GPU")
    if min(args.capture_steps) < 0 or max(args.capture_steps) >= args.steps:
        raise SystemExit("capture steps must be within the denoising step range")
    if min(args.layers) < 0 or max(args.layers) >= 25:
        raise SystemExit("FLUX.2 Klein attention layers are numbered 0 through 24")

    if Flux2KleinPipeline is None or flux2_module is None:
        raise SystemExit(
            "Run with the optional model dependencies through uv; see benchmarks/README.md"
        )

    prompts = args.prompt or [
        "A red fox standing in a snowy pine forest, cinematic lighting, detailed photograph",
        "A futuristic glass city beside the ocean at sunset, wide-angle architectural photograph",
    ]
    qk_variants = (
        ("int8", 127, None),
        ("int4", 7, None),
        ("int4_rot16", 7, 16),
        ("int4_rot64", 7, 64),
    )
    int8_pv_variants = (
        ("int8_pv_fixed", triton_sage_attention_int8_pv),
        ("int8_pv_block", triton_sage_attention_int8_pv_block_scaled),
        ("int8_pv_key_log_signed", triton_sage_attention_int8_pv_per_key_log),
    )
    pv_variants = (
        ("uint4_pv", 0, False),
        ("uint4_pv_rot16", 16, False),
        ("uint4_pv_rot64", 64, False),
        ("uint4_pv_pairrot16", 16, True),
        ("uint4_pv_pairrot64", 64, True),
    )
    uint8_pv_variants = (
        ("uint8_pv", 0, "feature", "dynamic", 0.0),
        ("uint8_pv_rot16", 16, "feature", "dynamic", 0.0),
        ("uint8_pv_rot64", 64, "feature", "dynamic", 0.0),
        ("uint8_pv_key", 0, "key", "dynamic", 0.0),
        ("uint8_pv_key_rot16", 16, "key", "dynamic", 0.0),
        ("uint8_pv_key_rot64", 64, "key", "dynamic", 0.0),
        ("uint8_pv_key_log", 0, "key", "log", 0.0),
        ("uint8_pv_key_rot64_log", 64, "key", "log", 0.0),
        ("uint8_pv_key_tile", 0, "key", "tile", 0.0),
        ("uint8_pv_key_rot16_tile", 16, "key", "tile", 0.0),
        ("uint8_pv_key_rot64_tile", 64, "key", "tile", 0.0),
        ("uint8_pv_key_tile_floor16", 0, "key", "tile", 1 / 16),
        ("uint8_pv_key_tile_floor8", 0, "key", "tile", 1 / 8),
        ("uint8_pv_key_tile_floor4", 0, "key", "tile", 1 / 4),
        ("uint8_pv_key_rot64_tile_floor16", 64, "key", "tile", 1 / 16),
        ("uint8_pv_key_rot64_tile_floor8", 64, "key", "tile", 1 / 8),
        ("uint8_pv_key_rot64_tile_floor4", 64, "key", "tile", 1 / 4),
    )
    all_variant_names = tuple(
        [name for name, _, _ in qk_variants]
        + [name for name, _ in int8_pv_variants]
        + [name for name, _, _ in pv_variants]
        + [name for name, _, _, _, _ in uint8_pv_variants]
    )
    diagnostic_variant_names = (
        "int8_qk_exact_pv",
        "uint4_p_exact_v",
        "exact_p_int4_v",
        "uint4_p_int4_v",
    )
    selected_variant_names = set(args.variants or all_variant_names)

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cuda")
    original_dispatch = flux2_module.dispatch_attention_fn
    measurements: list[Measurement] = []
    calls_per_step = 25
    call_index = 0
    prompt_index = 0

    def measured_dispatch(  # noqa: PLR0912
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *dispatch_args: object,
        **dispatch_kwargs: object,
    ) -> torch.Tensor:
        nonlocal call_index
        exact_output = original_dispatch(
            query,
            key,
            value,
            *dispatch_args,
            **dispatch_kwargs,
        )
        step = call_index // calls_per_step
        layer = call_index % calls_per_step
        call_index += 1
        if step not in args.capture_steps or layer not in args.layers:
            return exact_output

        query_h = query.permute(0, 2, 1, 3).contiguous()
        key_h = key.permute(0, 2, 1, 3).contiguous()
        value_h = value.permute(0, 2, 1, 3).contiguous()
        key_centered = key_h.float() - key_h.float().mean(dim=2, keepdim=True)
        exact_scores = torch.matmul(query_h.float(), key_centered.transpose(-1, -2))

        expected_output = exact_output.permute(0, 2, 1, 3)
        for variant, quantization_range, rotation_group in qk_variants:
            if variant not in selected_variant_names:
                continue
            quantized_scores = _quantized_scores(
                query_h,
                key_h,
                quantization_range,
                rotation_group,
            )
            quantized_output = _run_sage_attention(
                query_h,
                key_h,
                value_h,
                query.shape[-1] ** -0.5,
                False,
                qk_quantization_range=quantization_range,
                grouped_qk=False,
                rotation_group=rotation_group,
            )
            measurements.append(
                Measurement(
                    prompt=prompt_index,
                    step=step,
                    layer=layer,
                    sequence=query.shape[1],
                    variant=variant,
                    score_sqnr=_sqnr(quantized_scores, exact_scores),
                    output_sqnr=_sqnr(quantized_output, expected_output),
                    output_relative_l1=_relative_l1(quantized_output, expected_output),
                )
            )
        if selected_variant_names.intersection(name for name, _ in int8_pv_variants):
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            for variant, implementation in int8_pv_variants:
                if variant not in selected_variant_names:
                    continue
                quantized_output = implementation(
                    query_h,
                    key_h,
                    value_h,
                    query.shape[-1] ** -0.5,
                    False,
                    grouped_qk=False,
                )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        if selected_variant_names.intersection(name for name, _, _ in pv_variants):
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            for variant, rotation_group, paired_rotation in pv_variants:
                if variant not in selected_variant_names:
                    continue
                implementation = (
                    triton_sage_attention_uint4_pv_paired_convrot
                    if paired_rotation
                    else triton_sage_attention_uint4_pv_convrot
                )
                quantized_output = implementation(
                    query_h,
                    key_h,
                    value_h,
                    query.shape[-1] ** -0.5,
                    False,
                    rotation_group=rotation_group,
                    grouped_qk=False,
                )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        if selected_variant_names.intersection(name for name, _, _, _, _ in uint8_pv_variants):
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            for (
                variant,
                rotation_group,
                value_scale_axis,
                probability_scale_mode,
                value_scale_floor,
            ) in uint8_pv_variants:
                if variant not in selected_variant_names:
                    continue
                quantized_output = triton_sage_attention_uint8_pv_feature_convrot(
                    query_h,
                    key_h,
                    value_h,
                    query.shape[-1] ** -0.5,
                    False,
                    rotation_group=rotation_group,
                    value_scale_axis=value_scale_axis,
                    probability_scale_mode=probability_scale_mode,
                    value_scale_floor=value_scale_floor,
                    grouped_qk=False,
                )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        if args.pv_diagnostics:
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            probabilities = torch.softmax(
                int8_scores * query.shape[-1] ** -0.5,
                dim=-1,
            )
            diagnostic_outputs = (
                (
                    "int8_qk_exact_pv",
                    _pv_operand_ablation(
                        probabilities,
                        value_h,
                        quantize_probability=False,
                        quantize_value=False,
                    ),
                ),
                (
                    "uint4_p_exact_v",
                    _pv_operand_ablation(
                        probabilities,
                        value_h,
                        quantize_probability=True,
                        quantize_value=False,
                    ),
                ),
                (
                    "exact_p_int4_v",
                    _pv_operand_ablation(
                        probabilities,
                        value_h,
                        quantize_probability=False,
                        quantize_value=True,
                    ),
                ),
                (
                    "uint4_p_int4_v",
                    _pv_operand_ablation(
                        probabilities,
                        value_h,
                        quantize_probability=True,
                        quantize_value=True,
                    ),
                ),
            )
            for variant, diagnostic_output in diagnostic_outputs:
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(diagnostic_output, expected_output),
                        output_relative_l1=_relative_l1(diagnostic_output, expected_output),
                    )
                )
        return exact_output

    flux2_module.dispatch_attention_fn = measured_dispatch
    try:
        for current_prompt, prompt in enumerate(prompts):
            prompt_index = current_prompt
            call_index = 0
            generator = torch.Generator("cuda").manual_seed(args.seed + current_prompt)
            pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                max_sequence_length=args.max_sequence_length,
                generator=generator,
                output_type="latent",
            )
            expected_calls = args.steps * calls_per_step
            if call_index != expected_calls:
                raise RuntimeError(
                    f"expected {expected_calls} attention calls, observed {call_index}"
                )
    finally:
        flux2_module.dispatch_attention_fn = original_dispatch

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Model: {args.model}; prompts: {len(prompts)}; resolution: "
        f"{args.width}x{args.height}; denoising steps: {args.steps}"
    )
    print()
    print(
        "| prompt | step | layer | sequence | variant | score SQNR | output SQNR | output rel-L1 |"
    )
    print("|---:|---:|:---|---:|:---|---:|---:|---:|")
    for item in measurements:
        print(
            f"| {item.prompt} | {item.step} | {_layer_name(item.layer)} | {item.sequence} "
            f"| {item.variant} | {item.score_sqnr:.2f} dB | {item.output_sqnr:.2f} dB "
            f"| {item.output_relative_l1:.4f} |"
        )

    print()
    print(
        "| variant | mean score SQNR | worst score SQNR | mean output SQNR "
        "| worst output SQNR | mean rel-L1 |"
    )
    print("|:---|---:|---:|---:|---:|---:|")
    aggregate_variant_names = all_variant_names + (
        diagnostic_variant_names if args.pv_diagnostics else ()
    )
    for variant in aggregate_variant_names:
        if variant not in selected_variant_names and variant not in diagnostic_variant_names:
            continue
        selected = [item for item in measurements if item.variant == variant]
        mean_score = sum(item.score_sqnr for item in selected) / len(selected)
        mean_output = sum(item.output_sqnr for item in selected) / len(selected)
        mean_l1 = sum(item.output_relative_l1 for item in selected) / len(selected)
        print(
            f"| {variant} | {mean_score:.2f} dB | "
            f"{min(item.score_sqnr for item in selected):.2f} dB | {mean_output:.2f} dB "
            f"| {min(item.output_sqnr for item in selected):.2f} dB | {mean_l1:.4f} |"
        )


if __name__ == "__main__":
    main()
