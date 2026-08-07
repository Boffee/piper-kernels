"""Measure Sage INT4 ConvRot quality on real FLUX.2 Klein activations."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

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
    triton_sage_attention_uint8_pv_bucketed_grouped,
    triton_sage_attention_uint8_pv_feature_convrot,
    triton_sage_attention_uint8_pv_int32_recurrence,
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
    attention_pass: int
    layer: int
    sequence: int
    variant: str
    score_sqnr: float
    output_sqnr: float
    output_relative_l1: float


@dataclass(slots=True, frozen=True)
class AlignmentDiagnostics:
    """Row and probability-code statistics for power-of-two coordinate alignment."""

    gap_counts: tuple[int, int, int, int, int, int]
    exponent_advances: int
    merged_rows: int
    local_nonzero_codes: int
    nearest_nonzero_codes: int
    dithered_nonzero_codes: int


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


def _int8_pv_key_log_reference(
    scores: torch.Tensor,
    value: torch.Tensor,
    block_n: int,
    *,
    center_value: bool = False,
    probability_dtype: torch.dtype | None = None,
    accumulator_dtype: torch.dtype | None = None,
    normalized_accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate per-key INT8 V with tile-local UINT8 P in FP32."""
    value_float = value.float()
    value_mean = value_float.mean(dim=2, keepdim=True) if center_value else None
    if value_mean is not None:
        value_float = value_float - value_mean
    value_scale = value_float.abs().amax(dim=-1) / 127 + 1e-7
    value_int = (value_float / value_scale[..., None]).round().clamp(-127, 127)
    value_log_scale = torch.log(value_scale)
    value_inverse_scale = value_scale.reciprocal()

    accumulator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    running_max = torch.full_like(denominator, -torch.inf)
    for start in range(0, value.shape[2], block_n):
        stop = min(start + block_n, value.shape[2])
        score_block = scores[..., start:stop] + value_log_scale[:, :, None, start:stop]
        block_max = score_block.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        current_weight = torch.exp(block_max - next_max)
        probabilities = torch.exp(score_block - block_max[..., None])
        if probability_dtype is not None:
            probabilities = probabilities.to(probability_dtype).float()
        probability_codes = (probabilities * 255 + 0.5).floor().clamp(0, 255)
        integer_partial = torch.matmul(
            probability_codes,
            value_int[:, :, start:stop],
        )
        next_denominator = (
            denominator * old_weight
            + (probabilities * value_inverse_scale[:, :, None, start:stop]).sum(dim=-1)
            * current_weight
        )
        if normalized_accumulator_dtype is None:
            accumulator = (
                accumulator * old_weight[..., None]
                + integer_partial * (current_weight / 255)[..., None]
            )
            if accumulator_dtype is not None:
                accumulator = accumulator.to(accumulator_dtype).float()
        else:
            inverse_next_denominator = next_denominator.clamp_min(1e-30).reciprocal()
            old_output_weight = denominator * old_weight * inverse_next_denominator
            tile_output_weight = current_weight * inverse_next_denominator
            accumulator = (
                (
                    accumulator * old_output_weight[..., None]
                    + integer_partial * (tile_output_weight / 255)[..., None]
                )
                .to(normalized_accumulator_dtype)
                .float()
            )
        denominator = next_denominator
        running_max = next_max
    output = (
        accumulator
        if normalized_accumulator_dtype is not None
        else accumulator / denominator.clamp_min(1e-30)[..., None]
    )
    return output if value_mean is None else output + value_mean


def _permute_key_value_by_v_scale(
    key: torch.Tensor,
    value: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group noncausal K/V rows with similar V ranges and keep them contiguous."""
    if mode == "none":
        return key, value
    value_range = value.float().abs().amax(dim=-1).clamp_min(1e-30)
    if mode == "sort":
        order_key = value_range
    elif mode.startswith("log2x"):
        hash_within_bucket = mode.endswith("hash")
        bucket_mode = mode.removesuffix("hash")
        buckets_per_octave = int(bucket_mode.removeprefix("log2x"))
        if buckets_per_octave not in (1, 2, 4):
            raise ValueError("log2 V-scale buckets must use 1, 2, or 4 bins per octave")
        # Small integer radix keys retain original order within each scale bin.
        bucket_key = torch.floor(
            torch.log2(value_range) * buckets_per_octave
        ).to(torch.int32)
        if hash_within_bucket:
            positions = torch.arange(
                value.shape[2],
                device=value.device,
                dtype=torch.int64,
            )
            tie_break = (positions * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
            order_key = bucket_key.to(torch.int64) * (1 << 31) + tie_break
        else:
            order_key = bucket_key
    else:
        raise ValueError(f"unknown V-scale permutation mode: {mode!r}")
    order = torch.argsort(order_key, dim=-1, stable=True)
    key_order = order[..., None].expand_as(key)
    value_order = order[..., None].expand_as(value)
    return torch.gather(key, 2, key_order), torch.gather(value, 2, value_order)


def _centered_key_scaled_uint8_sort_low_to_high(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
) -> torch.Tensor:
    """Run the quality-first exact centered-V ordering through direct INT8 preparation."""
    if is_causal:
        raise ValueError("V sorting is valid only for noncausal attention")
    # Direct ordered preparation targets the grouped SM12x execution path.
    grouped_qk = True
    return triton_sage_attention_uint8_pv_feature_convrot(
        query,
        key,
        value,
        scale,
        is_causal,
        rotation_group=0,
        value_scale_axis="key",
        probability_scale_mode="log",
        grouped_qk=grouped_qk,
        affine_probability=True,
        native_uint8_mma=True,
        split_pv_head_dim=True,
        scale_forward_log_recurrence=True,
        optimize_pv_scaling=True,
        scaled_fp16_numerator=True,
        center_value=True,
        sort_value_rows=True,
    )


def _int8_pv_output_scaled_reference(  # noqa: PLR0912, PLR0913, PLR0915, PLR0917
    scores: torch.Tensor,
    value: torch.Tensor,
    block_n: int,
    feature_group: int,
    clip_rank: int = 0,
    value_block_n: int | None = None,
    power_of_two_value_scale: bool = False,
    integer_pair_alignment: bool = False,
    global_value_scale: bool = False,
    scale_run_n: int | None = None,
    global_probability_codes: bool = False,
    dither_probability_codes: bool = False,
    power_of_two_probability_weight: bool = False,
    scaled_fp16_numerator: bool = False,
) -> torch.Tensor:
    """Evaluate UINT8 P with V scales that can be applied after each INT8 dot.

    ``clip_rank`` selects a robust scale: zero uses the maximum, one ignores the
    largest magnitude in each Kxfeature scaling group, and so on. Values beyond
    the selected range are still clipped to the signed INT8 domain.
    """
    if value.shape[-1] % feature_group:
        raise ValueError("feature scale group must divide the head dimension")
    value_block_n = value_block_n or block_n
    if block_n % value_block_n:
        raise ValueError("value block must divide the probability block")
    if integer_pair_alignment and block_n != 2 * value_block_n:
        raise ValueError("integer alignment currently requires two V scale blocks")
    if scale_run_n is not None and (
        scale_run_n < block_n or scale_run_n % block_n
    ):
        raise ValueError("scale run must be a multiple of the probability block")
    accumulator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    running_max = torch.full_like(denominator, -torch.inf)
    feature_groups = value.shape[-1] // feature_group
    global_value_int: torch.Tensor | None = None
    global_scale_vector: torch.Tensor | None = None
    run_value_int: torch.Tensor | None = None
    run_scale_vectors: list[torch.Tensor] = []
    previous_accumulator_scale: torch.Tensor | None = None
    if global_value_scale:
        grouped_value = value.float().reshape(
            *value.shape[:-1],
            feature_groups,
            feature_group,
        )
        value_scale = grouped_value.abs().amax(dim=(2, 4)) / 127 + 1e-7
        global_scale_vector = value_scale.repeat_interleave(feature_group, dim=-1)
        global_value_int = (
            (grouped_value / value_scale[:, :, None, :, None])
            .round()
            .clamp(-127, 127)
            .reshape_as(value)
        )
    elif scale_run_n is not None:
        run_value_int = torch.empty_like(value, dtype=torch.float32)
        for run_start in range(0, value.shape[2], scale_run_n):
            run_stop = min(run_start + scale_run_n, value.shape[2])
            value_run = value[:, :, run_start:run_stop].float()
            grouped_value = value_run.reshape(
                *value_run.shape[:-1],
                feature_groups,
                feature_group,
            )
            value_scale = grouped_value.abs().amax(dim=(2, 4)) / 127 + 1e-7
            run_scale_vectors.append(
                value_scale.repeat_interleave(feature_group, dim=-1)
            )
            run_value_int[:, :, run_start:run_stop] = (
                (grouped_value / value_scale[:, :, None, :, None])
                .round()
                .clamp(-127, 127)
                .reshape_as(value_run)
            )

    for start in range(0, value.shape[2], block_n):
        stop = min(start + block_n, value.shape[2])
        score_block = scores[..., start:stop]
        block_max = score_block.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        if global_probability_codes:
            current_weight = torch.ones_like(block_max)
            probabilities = torch.exp(score_block - next_max[..., None])
        else:
            current_weight = torch.exp(block_max - next_max)
            probabilities = torch.exp(score_block - block_max[..., None])
        if dither_probability_codes:
            key_indices = torch.arange(start, stop, device=value.device, dtype=torch.int64)
            key_hash = (key_indices * 1103515245 + 12345) & 0xFFFF
            rounding_offset = key_hash.to(torch.float32) * (1.0 / 65536.0)
        else:
            rounding_offset = 0.5
        probability_codes = (
            probabilities * 255 + rounding_offset
        ).floor().clamp(0, 255)
        integer_partials: list[torch.Tensor] = []
        scale_vectors: list[torch.Tensor] = []
        if run_value_int is not None and scale_run_n is not None:
            integer_partials.append(
                torch.matmul(probability_codes, run_value_int[:, :, start:stop])
            )
            scale_vectors.append(run_scale_vectors[start // scale_run_n])
        elif global_value_int is not None and global_scale_vector is not None:
            integer_partials.append(
                torch.matmul(probability_codes, global_value_int[:, :, start:stop])
            )
            scale_vectors.append(global_scale_vector)
        for value_start in (
            range(start, stop, value_block_n)
            if global_value_int is None and run_value_int is None
            else ()
        ):
            value_stop = min(value_start + value_block_n, stop)
            value_block = value[:, :, value_start:value_stop].float()
            grouped_value = value_block.reshape(
                *value_block.shape[:-1],
                feature_groups,
                feature_group,
            )
            if clip_rank:
                scale_samples = (
                    grouped_value.abs()
                    .permute(0, 1, 3, 2, 4)
                    .flatten(start_dim=3)
                )
                if clip_rank >= scale_samples.shape[-1]:
                    raise ValueError("clip rank must be smaller than the scale group")
                value_scale = torch.topk(
                    scale_samples,
                    k=clip_rank + 1,
                    dim=-1,
                ).values[..., clip_rank] / 127 + 1e-7
            else:
                value_scale = grouped_value.abs().amax(dim=(2, 4)) / 127 + 1e-7
            if power_of_two_value_scale:
                value_scale = torch.exp2(torch.ceil(torch.log2(value_scale)))
            scale_vector = value_scale.repeat_interleave(feature_group, dim=-1)
            value_int = (
                (grouped_value / value_scale[:, :, None, :, None])
                .round()
                .clamp(-127, 127)
                .reshape_as(value_block)
            )
            probability_start = value_start - start
            probability_stop = value_stop - start
            integer_partials.append(
                torch.matmul(
                    probability_codes[..., probability_start:probability_stop],
                    value_int,
                )
            )
            scale_vectors.append(scale_vector)

        if integer_pair_alignment:
            common_scale = torch.maximum(scale_vectors[0], scale_vectors[1])
            aligned_partials = []
            for integer_partial, scale_vector in zip(
                integer_partials,
                scale_vectors,
                strict=True,
            ):
                scaled_magnitude = (
                    integer_partial.abs()
                    * (scale_vector / common_scale)[:, :, None, :]
                )
                aligned_partials.append(
                    integer_partial.sign() * torch.floor(scaled_magnitude + 0.5)
                )
            tile_numerator = (aligned_partials[0] + aligned_partials[1]) * common_scale[
                :, :, None, :
            ]
        else:
            tile_numerator = sum(
                integer_partial * scale_vector[:, :, None, :]
                for integer_partial, scale_vector in zip(
                    integer_partials,
                    scale_vectors,
                    strict=True,
                )
            )

        if scaled_fp16_numerator:
            if scale_run_n is None or len(integer_partials) != 1:
                raise ValueError("FP16 recurrence requires one run-scaled V partial")
            current_accumulator_scale = scale_vectors[0]
            if previous_accumulator_scale is not None and start % scale_run_n == 0:
                accumulator = (
                    accumulator.to(torch.float16)
                    * (previous_accumulator_scale / current_accumulator_scale)[
                        :, :, None, :
                    ].to(torch.float16)
                ).to(torch.float16).float()
            next_denominator = denominator * old_weight + probabilities.sum(
                dim=-1
            ) * current_weight
            scaled_partial = (
                integer_partials[0].float() * (1.0 / 65536.0)
            ).to(torch.float16)
            accumulator = (
                accumulator.to(torch.float16)
                * old_weight[..., None].to(torch.float16)
                + scaled_partial
                * current_weight[..., None].to(torch.float16)
            ).to(torch.float16).float()
            denominator = next_denominator
            previous_accumulator_scale = current_accumulator_scale
            running_max = next_max
            continue

        numerator_weight = (
            torch.exp2(torch.round(torch.log2(current_weight.clamp_min(1e-30))))
            if power_of_two_probability_weight
            else current_weight
        )
        accumulator = (
            accumulator * old_weight[..., None]
            + tile_numerator * (numerator_weight / 255)[..., None]
        )
        denominator = (
            denominator * old_weight
            + probabilities.sum(dim=-1) * current_weight
        )
        running_max = next_max
    if scaled_fp16_numerator:
        if previous_accumulator_scale is None:
            raise ValueError("scaled FP16 recurrence requires at least one V scale run")
        return (
            accumulator
            * 65536.0
            * previous_accumulator_scale[:, :, None, :]
            / 255
            / denominator.clamp_min(1e-30)[..., None]
        )
    return accumulator / denominator.clamp_min(1e-30)[..., None]


def _rounded_power_of_two_shift(values: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Match the integer recurrence's rounded arithmetic right shift in FP32."""
    scales = torch.exp2(shifts.float())
    return torch.floor((values + scales[..., None] * 0.5) / scales[..., None])


def _rounded_group_shift_int64(
    values: torch.Tensor,
    group_shifts: torch.Tensor,
    feature_group: int,
) -> torch.Tensor:
    """Apply the prospective Triton recurrence's grouped rounded right shift."""
    shifts = group_shifts.repeat_interleave(feature_group, dim=-1).clamp(0, 62)
    safe_shifts = shifts.clamp_min(1)
    rounding = torch.ones_like(safe_shifts, dtype=torch.int64) << (safe_shifts - 1)
    shifted = (values + rounding) >> shifts
    return torch.where(shifts == 0, values, shifted)


def _int8_pv_output_scaled_int32_reference(  # noqa: PLR0912, PLR0915
    scores: torch.Tensor,
    value: torch.Tensor,
    block_n: int,
    feature_group: int,
    *,
    integer_alignment: bool = True,
    q8_coefficient_mantissa: bool = False,
    common_feature_exponent: bool = False,
) -> torch.Tensor:
    """Evaluate a grouped block-floating INT32 numerator after V-range sorting.

    P retains a tile-local 8-bit range. V scales are rounded up to powers of two,
    letting each completed INT32 dot be aligned by shifts instead of converted and
    rescaled in FP32 on every K tile.
    """
    if value.shape[-1] % feature_group:
        raise ValueError("feature scale group must divide the head dimension")
    log2_e = 1.4426950408889634
    scores_log2 = scores.float() * log2_e
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    denominator_max = torch.full_like(denominator, -torch.inf)
    accumulator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.int64 if integer_alignment else torch.float32,
    )
    running_exponent: torch.Tensor | None = None
    feature_groups = value.shape[-1] // feature_group

    for start in range(0, value.shape[2], block_n):
        stop = min(start + block_n, value.shape[2])
        score_block = scores_log2[..., start:stop]
        block_score_max = score_block.amax(dim=-1)
        if q8_coefficient_mantissa:
            probability_codes = (
                torch.exp2(score_block - block_score_max[..., None]) * 255 + 0.5
            ).floor().clamp(0, 255)
        else:
            block_score_exponent = torch.ceil(block_score_max).to(torch.int64)
            probability_codes = (
                torch.exp2(score_block - block_score_exponent[..., None].float()) * 255 + 0.5
            ).floor().clamp(0, 255)

        value_block = value[:, :, start:stop].float()
        grouped_value = value_block.reshape(
            *value_block.shape[:-1],
            feature_groups,
            feature_group,
        )
        raw_value_scale = grouped_value.abs().amax(dim=(2, 4)) / 127 + 1e-7
        if q8_coefficient_mantissa:
            value_scale = raw_value_scale
        else:
            value_exponent = torch.ceil(torch.log2(raw_value_scale)).to(torch.int64)
            value_scale = torch.exp2(value_exponent.float())
        value_int = (
            (grouped_value / value_scale[:, :, None, :, None])
            .round()
            .clamp(-127, 127)
            .reshape_as(value_block)
        )
        partial = torch.matmul(probability_codes, value_int).round().to(torch.int64)
        if q8_coefficient_mantissa:
            coefficient_log2 = (
                block_score_max[..., None]
                + torch.log2(value_scale)[:, :, None, :]
            )
            tile_exponent = torch.ceil(
                coefficient_log2.amax(dim=-1, keepdim=True)
                if common_feature_exponent
                else coefficient_log2
            ).to(torch.int64)
            coefficient_mantissa_q8 = (
                torch.exp2(coefficient_log2 - tile_exponent.float()) * 256 + 0.5
            ).floor().to(torch.int64)
            partial = (partial * coefficient_mantissa_q8.repeat_interleave(
                feature_group,
                dim=-1,
            ) + 128) >> 8
        else:
            tile_exponent = block_score_exponent[..., None] + value_exponent[:, :, None, :]
        next_denominator_max = torch.maximum(
            denominator_max,
            score_block.amax(dim=-1),
        )

        if not integer_alignment:
            old_weight = torch.exp2(denominator_max - next_denominator_max)
            tile_scale = torch.exp2(
                tile_exponent.float() - next_denominator_max[..., None]
            ).repeat_interleave(feature_group, dim=-1)
            accumulator = (
                accumulator * old_weight[..., None]
                + partial.float() * tile_scale
            )
        elif running_exponent is None:
            running_exponent = tile_exponent
            accumulator = partial
        else:
            next_exponent = torch.maximum(running_exponent, tile_exponent)
            accumulator = _rounded_group_shift_int64(
                accumulator,
                next_exponent - running_exponent,
                value.shape[-1] if common_feature_exponent else feature_group,
            ) + _rounded_group_shift_int64(
                partial,
                next_exponent - tile_exponent,
                value.shape[-1] if common_feature_exponent else feature_group,
            )
            running_exponent = next_exponent

        denominator = denominator * torch.exp2(
            denominator_max - next_denominator_max
        ) + torch.exp2(score_block - next_denominator_max[..., None]).sum(dim=-1)
        denominator_max = next_denominator_max

    if not integer_alignment:
        return accumulator / (denominator[..., None] * 255.0)
    if running_exponent is None:
        raise ValueError("INT32 reference requires a nonempty key sequence")
    output_exponent = running_exponent.repeat_interleave(
        value.shape[-1] if common_feature_exponent else feature_group,
        dim=-1,
    )
    output_scale = torch.exp2(
        output_exponent.float() - denominator_max[..., None]
    )
    return accumulator.float() * output_scale / (denominator[..., None] * 255.0)


def _int8_pv_key_log_alignment_references(  # noqa: PLR0915
    scores: torch.Tensor,
    value: torch.Tensor,
    block_n: int = 64,
) -> tuple[dict[str, torch.Tensor], AlignmentDiagnostics]:
    """Compare post-dot alignment with cheaper pre-dot UINT8-code alignment."""
    log2_e = 1.4426950408889634
    scores_log2 = scores.float() * log2_e
    value_float = value.float()
    value_scale = value_float.abs().amax(dim=-1) / 127 + 1e-7
    value_int = (value_float / value_scale[..., None]).round().clamp(-127, 127)
    value_log2_scale = torch.log2(value_scale)

    output_shape = (*scores.shape[:-1], value.shape[-1])
    postdot = torch.zeros(output_shape, device=value.device, dtype=torch.float32)
    predot_nearest = torch.zeros_like(postdot)
    dither_bits = (1, 2, 4, 10)
    predot_dithered = {bits: torch.zeros_like(postdot) for bits in dither_bits}
    predot_key_dithered = torch.zeros_like(postdot)
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    denominator_max = torch.full_like(denominator, -torch.inf)
    running_exponent: torch.Tensor | None = None

    gap_counts = [0, 0, 0, 0, 0, 0]
    exponent_advances = 0
    merged_rows = 0
    local_nonzero_codes = 0
    nearest_nonzero_codes = 0
    dithered_nonzero_codes = 0
    batch_indices = torch.arange(value.shape[0], device=value.device)[:, None, None, None]
    head_indices = torch.arange(value.shape[1], device=value.device)[None, :, None, None]
    query_indices = torch.arange(scores.shape[2], device=value.device)[None, None, :, None]

    for start in range(0, value.shape[2], block_n):
        stop = min(start + block_n, value.shape[2])
        score_block = scores_log2[..., start:stop]
        transformed_scores = score_block + value_log2_scale[:, :, None, start:stop]
        block_exponent = torch.ceil(transformed_scores.amax(dim=-1)).to(torch.int32)
        probability_code_values = (
            torch.exp2(transformed_scores - block_exponent[..., None].float()) * 255
        )
        probability_codes = (probability_code_values + 0.5).floor().clamp(0, 255)
        value_block = value_int[:, :, start:stop]
        partial = torch.matmul(probability_codes, value_block)

        block_denominator_max = score_block.amax(dim=-1)
        next_denominator_max = torch.maximum(denominator_max, block_denominator_max)
        denominator = denominator * torch.exp2(denominator_max - next_denominator_max) + torch.exp2(
            score_block - next_denominator_max[..., None]
        ).sum(dim=-1)
        denominator_max = next_denominator_max

        if running_exponent is None:
            running_exponent = block_exponent
            postdot = partial
            predot_nearest = partial
            predot_dithered = {bits: partial.clone() for bits in dither_bits}
            predot_key_dithered = partial
            local_nonzero_codes += int((probability_codes != 0).sum())
            nearest_nonzero_codes += int((probability_codes != 0).sum())
            dithered_nonzero_codes += int((probability_codes != 0).sum())
            continue

        next_exponent = torch.maximum(running_exponent, block_exponent)
        old_shift = next_exponent - running_exponent
        block_shift = next_exponent - block_exponent
        gap = torch.abs(running_exponent - block_exponent)
        for bucket in range(5):
            gap_counts[bucket] += int((gap == bucket).sum())
        gap_counts[5] += int((gap >= 5).sum())
        exponent_advances += int((block_exponent > running_exponent).sum())
        merged_rows += gap.numel()

        postdot = _rounded_power_of_two_shift(postdot, old_shift) + _rounded_power_of_two_shift(
            partial,
            block_shift,
        )
        shift_scale = torch.exp2(block_shift.float())
        nearest_codes = torch.floor(probability_codes / shift_scale[..., None] + 0.5)
        predot_nearest = _rounded_power_of_two_shift(
            predot_nearest,
            old_shift,
        ) + torch.matmul(nearest_codes, value_block)

        key_indices = torch.arange(start, stop, device=value.device)[None, None, None, :]
        phase = (
            batch_indices * 251
            + head_indices * 199
            + query_indices * 73
            + key_indices * 151
        )
        for bits in dither_bits:
            levels = 1 << bits
            dither = ((phase & (levels - 1)).float() + 0.5) / levels
            dithered_codes = torch.floor(
                probability_codes / shift_scale[..., None] + dither
            )
            predot_dithered[bits] = _rounded_power_of_two_shift(
                predot_dithered[bits],
                old_shift,
            ) + torch.matmul(dithered_codes, value_block)
        key_phase = head_indices * 199 + key_indices * 151
        key_dither = ((key_phase & 1023).float() + 0.5) * (1.0 / 1024.0)
        key_dithered_codes = torch.floor(
            probability_codes / shift_scale[..., None] + key_dither
        )
        predot_key_dithered = _rounded_power_of_two_shift(
            predot_key_dithered,
            old_shift,
        ) + torch.matmul(key_dithered_codes, value_block)

        local_nonzero_codes += int((probability_codes != 0).sum())
        nearest_nonzero_codes += int((nearest_codes != 0).sum())
        dithered_nonzero_codes += int((dithered_codes != 0).sum())
        running_exponent = next_exponent

    if running_exponent is None:
        raise ValueError("alignment reference requires at least one key")
    output_scale = (
        torch.exp2(running_exponent.float() - denominator_max)
        / 255
        / denominator.clamp_min(1e-30)
    )
    outputs = {
        "int8_pv_key_log_pow2_postdot": postdot * output_scale[..., None],
        "int8_pv_key_log_pow2_predot_nearest": predot_nearest * output_scale[..., None],
    }
    outputs.update(
        {
            f"int8_pv_key_log_pow2_predot_dithered_b{bits}": accumulator
            * output_scale[..., None]
            for bits, accumulator in predot_dithered.items()
        }
    )
    outputs["int8_pv_key_log_pow2_predot_dithered_key_b10"] = (
        predot_key_dithered * output_scale[..., None]
    )
    diagnostics = AlignmentDiagnostics(
        gap_counts=tuple(gap_counts),  # type: ignore[arg-type]
        exponent_advances=exponent_advances,
        merged_rows=merged_rows,
        local_nonzero_codes=local_nonzero_codes,
        nearest_nonzero_codes=nearest_nonzero_codes,
        dithered_nonzero_codes=dithered_nonzero_codes,
    )
    return outputs, diagnostics


def _int8_pv_key_log_paired_reference(
    scores: torch.Tensor,
    value: torch.Tensor,
    scale_bits: int,
) -> torch.Tensor:
    """Merge pairs of independently normalized K64 INT32 partials."""
    value_float = value.float()
    value_scale = value_float.abs().amax(dim=-1) / 127 + 1e-7
    value_int = (value_float / value_scale[..., None]).round().clamp(-127, 127)
    value_log_scale = torch.log(value_scale)
    value_inverse_scale = value_scale.reciprocal()

    accumulator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    running_max = torch.full_like(denominator, -torch.inf)
    integer_scale = 1 << scale_bits
    integer_rounding = integer_scale >> 1
    for pair_start in range(0, value.shape[2], 128):
        subtiles: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for start in range(pair_start, min(pair_start + 128, value.shape[2]), 64):
            stop = min(start + 64, value.shape[2])
            score_block = scores[..., start:stop] + value_log_scale[:, :, None, start:stop]
            block_max = score_block.amax(dim=-1)
            probabilities = torch.exp(score_block - block_max[..., None])
            probability_codes = (probabilities * 255 + 0.5).floor().clamp(0, 255)
            integer_partial = torch.matmul(
                probability_codes,
                value_int[:, :, start:stop],
            ).to(torch.int64)
            denominator_partial = (probabilities * value_inverse_scale[:, :, None, start:stop]).sum(
                dim=-1
            )
            subtiles.append((block_max, integer_partial, denominator_partial))

        pair_max = torch.stack([item[0] for item in subtiles]).amax(dim=0)
        pair_integer = torch.zeros_like(subtiles[0][1])
        pair_denominator = torch.zeros_like(subtiles[0][2])
        for block_max, integer_partial, denominator_partial in subtiles:
            weight = torch.exp(block_max - pair_max)
            weight_integer = (weight * integer_scale + 0.5).floor().to(torch.int64)
            product = integer_partial * weight_integer[..., None]
            rounded = torch.where(
                product >= 0,
                (product + integer_rounding) >> scale_bits,
                -((-product + integer_rounding) >> scale_bits),
            )
            pair_integer += rounded
            pair_denominator += denominator_partial * weight

        next_max = torch.maximum(running_max, pair_max)
        old_weight = torch.exp(running_max - next_max)
        current_weight = torch.exp(pair_max - next_max)
        accumulator = (
            accumulator * old_weight[..., None]
            + pair_integer.float() * (current_weight / 255)[..., None]
        )
        denominator = denominator * old_weight + pair_denominator * current_weight
        running_max = next_max
    return accumulator / denominator.clamp_min(1e-30)[..., None]


def _int8_pv_key_log_sampled_pair_reference(
    scores: torch.Tensor,
    value: torch.Tensor,
    headroom_log2: float,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Use interleaved K64 samples to choose an approximate K128 UINT8 coordinate."""
    if scores.shape[-1] % 128:
        raise ValueError("sampled K128 reference requires a sequence divisible by 128")
    value_float = value.float()
    value_scale = value_float.abs().amax(dim=-1) / 127 + 1e-7
    value_int = (value_float / value_scale[..., None]).round().clamp(-127, 127)
    transformed_scores = scores + torch.log(value_scale)[:, :, None, :]
    headroom = headroom_log2 * torch.log(torch.tensor(2.0, device=scores.device))

    accumulator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    running_max = torch.full_like(denominator, -torch.inf)
    clipped_rows = 0
    sampled_rows = 0
    maximum_gap_log2 = 0.0
    maximum_score_gap_log2 = 0.0
    maximum_scale_gap_log2 = 0.0
    for pair_start in range(0, value.shape[2], 128):
        sample_indices = torch.arange(pair_start, pair_start + 128, 2, device=value.device)
        heldout_indices = sample_indices + 1
        sample_scores = transformed_scores.index_select(-1, sample_indices)
        heldout_scores = transformed_scores.index_select(-1, heldout_indices)
        sample_max = sample_scores.amax(dim=-1)
        heldout_max = heldout_scores.amax(dim=-1)
        estimated_max = sample_max + headroom
        gap_log2 = (heldout_max - estimated_max) * 1.4426950408889634
        clipped_rows += int((gap_log2 > 0).sum())
        sampled_rows += gap_log2.numel()
        maximum_gap_log2 = max(maximum_gap_log2, float(gap_log2.amax()))
        sample_score_max = scores.index_select(-1, sample_indices).amax(dim=-1)
        heldout_score_max = scores.index_select(-1, heldout_indices).amax(dim=-1)
        maximum_score_gap_log2 = max(
            maximum_score_gap_log2,
            float(((heldout_score_max - sample_score_max) * 1.4426950408889634).amax()),
        )
        sample_scale_max = torch.log2(value_scale.index_select(2, sample_indices)).amax(dim=-1)
        heldout_scale_max = torch.log2(value_scale.index_select(2, heldout_indices)).amax(dim=-1)
        maximum_scale_gap_log2 = max(
            maximum_scale_gap_log2,
            float((heldout_scale_max - sample_scale_max).amax()),
        )

        sample_codes = (
            torch.exp(sample_scores - estimated_max[..., None]) * 255 + 0.5
        ).floor().clamp(0, 255)
        heldout_codes = (
            torch.exp(heldout_scores - estimated_max[..., None]) * 255 + 0.5
        ).floor().clamp(0, 255)
        integer_partial = torch.matmul(
            sample_codes,
            value_int.index_select(2, sample_indices),
        ) + torch.matmul(
            heldout_codes,
            value_int.index_select(2, heldout_indices),
        )

        pair_max = torch.maximum(sample_max, heldout_max)
        next_max = torch.maximum(running_max, pair_max)
        old_weight = torch.exp(running_max - next_max)
        sampled_weight = torch.exp(estimated_max - next_max)
        pair_weight = torch.exp(pair_max - next_max)
        pair_denominator = torch.exp(
            scores.index_select(-1, sample_indices) - pair_max[..., None]
        ).sum(dim=-1) + torch.exp(
            scores.index_select(-1, heldout_indices) - pair_max[..., None]
        ).sum(dim=-1)
        accumulator = (
            accumulator * old_weight[..., None]
            + integer_partial * (sampled_weight / 255)[..., None]
        )
        denominator = denominator * old_weight + pair_denominator * pair_weight
        running_max = next_max

    return (
        accumulator / denominator.clamp_min(1e-30)[..., None],
        clipped_rows / sampled_rows,
        maximum_gap_log2,
        maximum_score_gap_log2,
        maximum_scale_gap_log2,
    )


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
    parser.add_argument("--guidance-scale", type=float, default=1.0)
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
            "int8_pv_bucketed_group4_k128",
            "int8_pv_bucketed_group4_k128_run512",
            "int8_pv_bucketed_group4_k64_run512_scaled_fp16_numerator",
            "int8_pv_bucketed_group4_k128_run512_quarter_octave",
            "int8_pv_bucketed_group4_k128_one_octave",
            "int8_pv_bucketed_group4_k128_quarter_octave",
            "int8_pv_bucketed_group4_k128_global_p",
            "int8_pv_bucketed_group8_k128",
            "int8_pv_block",
            "int8_pv_key_log_signed",
            "int8_pv_key_log_int32",
            "int8_pv_key_log_int32_tile",
            "int8_pv_key_log_int32_tile_predot_dithered",
            "int8_pv_key_log_ref64",
            "int8_pv_key_log_ref64_v_centered",
            "int8_pv_key_log_ref128",
            "int8_pv_key_log_pair_q10",
            "int8_pv_key_log_ref64_fp16_p",
            "int8_pv_key_log_ref64_fp16_acc",
            "int8_pv_key_log_ref64_fp16_p_acc",
            "int8_pv_key_log_ref64_fp16_norm_acc",
            "int8_pv_output_k64_feature",
            "int8_pv_output_k32_feature",
            "int8_pv_output_k64_group16",
            "int8_pv_output_k64_feature_run512_global_p",
            "int8_pv_output_k64_feature_run512_global_p_dither",
            "int8_pv_output_k64_feature_run512_pow2weight",
            "int8_pv_output_k64_group4_run512_global_p",
            "int8_pv_output_k64_group4_run512_global_p_dither",
            "int8_pv_output_k64_group4_run1024_global_p",
            "int8_pv_output_k32_group16",
            "int8_pv_output_k64_scalar",
            "int8_pv_output_k64_feature_sort",
            "int8_pv_output_k64_group16_sort",
            "int8_pv_output_k64_scalar_sort",
            "int8_pv_output_k64_feature_log2x2",
            "int8_pv_output_k64_group16_log2x2",
            "int8_pv_output_k64_scalar_log2x2",
            "int8_pv_output_k64_group2_log2x1_h64",
            "int8_pv_output_k64_group4_log2x1_h64",
            "int8_pv_output_k64_group8_log2x1_h64",
            "int8_pv_output_k64_group16_log2x1_h64",
            "int8_pv_output_k64_group32_log2x1_h64",
            "int8_pv_output_k64_group64_log2x1_h64",
            "int8_pv_output_k128_group16_log2x1_h64",
            "int8_pv_output_k128_group32_log2x1_h64",
            "int8_pv_output_k128_group4_log2x1_h64",
            "int8_pv_output_k128_group4_log2x2_h64_run256",
            "int8_pv_output_k128_group4_log2x2_h64_run512",
            "int8_pv_output_k64_group4_log2x2_h64_run512_scaled_fp16_numerator",
            "int8_pv_output_k128_group4_log2x2_h64_run512_pow2weight",
            "int8_pv_output_k128_group4_log2x2_h64_run1024",
            "int8_pv_output_k128_group4_log2x2_h64_run2048",
            "int8_pv_output_k128_group8_log2x1_h64",
            "int8_pv_output_k128_group4_log2x1_h64_int32",
            "int8_pv_output_k128_group4_log2x1_h64_int32_q8mantissa",
            "int8_pv_output_k128_group4_log2x1_h64_int32_q8mantissa_commonexp",
            "int8_pv_output_k128_group4_log2x1_h64_pow2_fp32",
            "int8_pv_output_k128_group8_log2x1_h64_int32",
            "int8_pv_output_k128_group8_log2x1hash_h64",
            "int8_pv_output_k128_group4_log2x1hash_h64",
            "int8_pv_output_k256_feature_log2x1_h64",
            "int8_pv_output_k256_group2_log2x1_h64",
            "int8_pv_output_k256_group4_log2x1_h64",
            "int8_pv_output_k64_scalar_sort_h64",
            "int8_pv_output_k64_scalar_log2x1_h64",
            "int8_pv_output_k64_scalar_log2x2_h64",
            "int8_pv_output_k64_feature_log2x1",
            "int8_pv_output_k64_feature_log2x4",
            "int8_pv_output_k64_feature_sort_clip1",
            "int8_pv_output_k64_feature_sort_clip2",
            "int8_pv_output_k64_feature_log2x1_clip1",
            "int8_pv_output_k64_feature_log2x1_clip2",
            "int8_pv_output_k64_feature_log2x1_clip4",
            "int8_pv_output_k32_feature_sort",
            "int8_pv_output_k32_feature_log2x1",
            "int8_pv_output_k32_feature_log2x2",
            "int8_pv_output_k32_feature_log2x4",
            "int8_pv_output_k64p_k32v_feature_sort",
            "int8_pv_output_k64p_k32v_feature_log2x1",
            "int8_pv_output_k64p_k32v_feature_log2x1_pow2",
            "int8_pv_output_k64p_k32v_feature_log2x1_pow2_align",
            "int8_pv_output_k64p_globalv_feature",
            "int8_pv_output_k64p_globalv_feature_sort",
            "int8_pv_output_k64p_globalv_feature_log2x1",
            "int8_pv_output_k64p_globalv_feature_log2x2",
            "int8_pv_output_k64_feature_sort_h64",
            "int8_pv_output_k64_feature_log2x1_h64",
            "int8_pv_output_k64_feature_log2x2_h64",
            "int8_pv_key_log_pow2_postdot",
            "int8_pv_key_log_pow2_predot_nearest",
            "int8_pv_key_log_pow2_predot_dithered_b1",
            "int8_pv_key_log_pow2_predot_dithered_b2",
            "int8_pv_key_log_pow2_predot_dithered_b4",
            "int8_pv_key_log_pow2_predot_dithered_b10",
            "int8_pv_key_log_pow2_predot_dithered_key_b10",
            "int8_pv_key_log_sampled_pair_h0",
            "int8_pv_key_log_sampled_pair_h025",
            "int8_pv_key_log_sampled_pair_h05",
            "int8_pv_key_log_sampled_pair_h1",
            "int8_pv_key_log_sampled_pair_h15",
            "int8_pv_key_log_sampled_pair_h2",
            "int8_pv_key_log_sampled_pair_h25",
            "int8_pv_key_log_sampled_pair_h3",
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
            "uint8_pv_key_log_narrow_denominator",
            "uint8_pv_key_log_running_max",
            "uint8_pv_key_log_scale_forward",
            "uint8_pv_key_log_scale_forward_optimized_pv",
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator",
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator_sort_low_to_high",
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator_fp32_metadata",
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator_scaled_fp16_denominator",
            "uint8_pv_key_log_tile_common",
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
def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0912, PLR0915
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
        (
            "int8_pv_bucketed_group4_k128",
            triton_sage_attention_uint8_pv_bucketed_grouped,
        ),
        (
            "int8_pv_bucketed_group4_k128_run512",
            partial(
                triton_sage_attention_uint8_pv_bucketed_grouped,
                scale_run_n=512,
            ),
        ),
        (
            "int8_pv_bucketed_group4_k64_run512_scaled_fp16_numerator",
            partial(
                triton_sage_attention_uint8_pv_bucketed_grouped,
                block_n=64,
                scale_run_n=512,
                scaled_fp16_numerator=True,
            ),
        ),
        (
            "int8_pv_bucketed_group4_k128_run512_quarter_octave",
            partial(
                triton_sage_attention_uint8_pv_bucketed_grouped,
                scale_run_n=512,
                range_bucket_log2_scale=4,
            ),
        ),
        (
            "int8_pv_bucketed_group4_k128_one_octave",
            partial(
                triton_sage_attention_uint8_pv_bucketed_grouped,
                range_bucket_log2_scale=1,
            ),
        ),
        (
            "int8_pv_bucketed_group4_k128_quarter_octave",
            partial(
                triton_sage_attention_uint8_pv_bucketed_grouped,
                range_bucket_log2_scale=4,
            ),
        ),
        (
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator_sort_low_to_high",
            _centered_key_scaled_uint8_sort_low_to_high,
        ),
        (
            "int8_pv_bucketed_group4_k128_global_p",
            partial(
                triton_sage_attention_uint8_pv_bucketed_grouped,
                local_probability_codes=False,
            ),
        ),
        (
            "int8_pv_bucketed_group8_k128",
            partial(triton_sage_attention_uint8_pv_bucketed_grouped, feature_group=8),
        ),
        ("int8_pv_fixed", triton_sage_attention_int8_pv),
        ("int8_pv_block", triton_sage_attention_int8_pv_block_scaled),
        ("int8_pv_key_log_signed", triton_sage_attention_int8_pv_per_key_log),
        ("int8_pv_key_log_int32", triton_sage_attention_uint8_pv_int32_recurrence),
        (
            "int8_pv_key_log_int32_tile",
            partial(triton_sage_attention_uint8_pv_int32_recurrence, tile_exponent=True),
        ),
        (
            "int8_pv_key_log_int32_tile_predot_dithered",
            partial(
                triton_sage_attention_uint8_pv_feature_convrot,
                rotation_group=0,
                value_scale_axis="key",
                probability_scale_mode="log",
                affine_probability=True,
                native_uint8_mma=True,
                integer_tile_exponent_recurrence=True,
                predot_exponent_alignment=True,
                dithered_predot_alignment=True,
                split_pv_head_dim=True,
                scale_forward_log_recurrence=True,
                optimize_pv_scaling=True,
            ),
        ),
    )
    int8_pv_reference_variants = (
        ("int8_pv_key_log_ref64", 64, None, None, None, None, False),
        ("int8_pv_key_log_ref64_v_centered", 64, None, None, None, None, True),
        ("int8_pv_key_log_ref128", 128, None, None, None, None, False),
        ("int8_pv_key_log_pair_q10", 64, 10, None, None, None, False),
        ("int8_pv_key_log_ref64_fp16_p", 64, None, torch.float16, None, None, False),
        ("int8_pv_key_log_ref64_fp16_acc", 64, None, None, torch.float16, None, False),
        (
            "int8_pv_key_log_ref64_fp16_p_acc",
            64,
            None,
            torch.float16,
            torch.float16,
            None,
            False,
        ),
        (
            "int8_pv_key_log_ref64_fp16_norm_acc",
            64,
            None,
            None,
            None,
            torch.float16,
            False,
        ),
    )
    output_scaled_reference_variants = (
        ("int8_pv_output_k64_feature", 64, 1, "none", 0),
        ("int8_pv_output_k32_feature", 32, 1, "none", 0),
        ("int8_pv_output_k64_group16", 64, 16, "none", 0),
        ("int8_pv_output_k64_feature_run512_global_p", 64, 1, "none", 0),
        ("int8_pv_output_k64_feature_run512_global_p_dither", 64, 1, "none", 0),
        ("int8_pv_output_k64_feature_run512_pow2weight", 64, 1, "none", 0),
        ("int8_pv_output_k64_group4_run512_global_p", 64, 4, "none", 0),
        ("int8_pv_output_k64_group4_run512_global_p_dither", 64, 4, "none", 0),
        ("int8_pv_output_k64_group4_run1024_global_p", 64, 4, "none", 0),
        ("int8_pv_output_k32_group16", 32, 16, "none", 0),
        ("int8_pv_output_k64_scalar", 64, 128, "none", 0),
        ("int8_pv_output_k64_feature_sort", 64, 1, "sort", 0),
        ("int8_pv_output_k64_group16_sort", 64, 16, "sort", 0),
        ("int8_pv_output_k64_scalar_sort", 64, 128, "sort", 0),
        ("int8_pv_output_k64_feature_log2x2", 64, 1, "log2x2", 0),
        ("int8_pv_output_k64_group16_log2x2", 64, 16, "log2x2", 0),
        ("int8_pv_output_k64_scalar_log2x2", 64, 128, "log2x2", 0),
        ("int8_pv_output_k64_group2_log2x1_h64", 64, 2, "log2x1", 64),
        ("int8_pv_output_k64_group4_log2x1_h64", 64, 4, "log2x1", 64),
        (
            "int8_pv_output_k64_group4_log2x2_h64_run512_scaled_fp16_numerator",
            64,
            4,
            "log2x2",
            64,
        ),
        ("int8_pv_output_k64_group8_log2x1_h64", 64, 8, "log2x1", 64),
        ("int8_pv_output_k64_group16_log2x1_h64", 64, 16, "log2x1", 64),
        ("int8_pv_output_k64_group32_log2x1_h64", 64, 32, "log2x1", 64),
        ("int8_pv_output_k64_group64_log2x1_h64", 64, 64, "log2x1", 64),
        ("int8_pv_output_k128_group16_log2x1_h64", 128, 16, "log2x1", 64),
        ("int8_pv_output_k128_group32_log2x1_h64", 128, 32, "log2x1", 64),
        ("int8_pv_output_k128_group4_log2x1_h64", 128, 4, "log2x1", 64),
        ("int8_pv_output_k128_group4_log2x2_h64_run256", 128, 4, "log2x2", 64),
        ("int8_pv_output_k128_group4_log2x2_h64_run512", 128, 4, "log2x2", 64),
        (
            "int8_pv_output_k128_group4_log2x2_h64_run512_pow2weight",
            128,
            4,
            "log2x2",
            64,
        ),
        ("int8_pv_output_k128_group4_log2x2_h64_run1024", 128, 4, "log2x2", 64),
        ("int8_pv_output_k128_group4_log2x2_h64_run2048", 128, 4, "log2x2", 64),
        ("int8_pv_output_k128_group8_log2x1_h64", 128, 8, "log2x1", 64),
        ("int8_pv_output_k128_group4_log2x1_h64_int32", 128, 4, "log2x1", 64),
        (
            "int8_pv_output_k128_group4_log2x1_h64_int32_q8mantissa",
            128,
            4,
            "log2x1",
            64,
        ),
        (
            "int8_pv_output_k128_group4_log2x1_h64_int32_q8mantissa_commonexp",
            128,
            4,
            "log2x1",
            64,
        ),
        ("int8_pv_output_k128_group4_log2x1_h64_pow2_fp32", 128, 4, "log2x1", 64),
        ("int8_pv_output_k128_group8_log2x1_h64_int32", 128, 8, "log2x1", 64),
        (
            "int8_pv_output_k128_group8_log2x1hash_h64",
            128,
            8,
            "log2x1hash",
            64,
        ),
        (
            "int8_pv_output_k128_group4_log2x1hash_h64",
            128,
            4,
            "log2x1hash",
            64,
        ),
        ("int8_pv_output_k256_feature_log2x1_h64", 256, 1, "log2x1", 64),
        ("int8_pv_output_k256_group2_log2x1_h64", 256, 2, "log2x1", 64),
        ("int8_pv_output_k256_group4_log2x1_h64", 256, 4, "log2x1", 64),
        ("int8_pv_output_k64_scalar_sort_h64", 64, 128, "sort", 64),
        ("int8_pv_output_k64_scalar_log2x1_h64", 64, 128, "log2x1", 64),
        ("int8_pv_output_k64_scalar_log2x2_h64", 64, 128, "log2x2", 64),
        ("int8_pv_output_k64_feature_log2x1", 64, 1, "log2x1", 0),
        ("int8_pv_output_k64_feature_log2x4", 64, 1, "log2x4", 0),
        ("int8_pv_output_k64_feature_sort_clip1", 64, 1, "sort", 0),
        ("int8_pv_output_k64_feature_sort_clip2", 64, 1, "sort", 0),
        ("int8_pv_output_k64_feature_log2x1_clip1", 64, 1, "log2x1", 0),
        ("int8_pv_output_k64_feature_log2x1_clip2", 64, 1, "log2x1", 0),
        ("int8_pv_output_k64_feature_log2x1_clip4", 64, 1, "log2x1", 0),
        ("int8_pv_output_k32_feature_sort", 32, 1, "sort", 0),
        ("int8_pv_output_k32_feature_log2x1", 32, 1, "log2x1", 0),
        ("int8_pv_output_k32_feature_log2x2", 32, 1, "log2x2", 0),
        ("int8_pv_output_k32_feature_log2x4", 32, 1, "log2x4", 0),
        ("int8_pv_output_k64p_k32v_feature_sort", 64, 1, "sort", 0),
        ("int8_pv_output_k64p_k32v_feature_log2x1", 64, 1, "log2x1", 0),
        ("int8_pv_output_k64p_k32v_feature_log2x1_pow2", 64, 1, "log2x1", 0),
        (
            "int8_pv_output_k64p_k32v_feature_log2x1_pow2_align",
            64,
            1,
            "log2x1",
            0,
        ),
        ("int8_pv_output_k64p_globalv_feature", 64, 1, "none", 0),
        ("int8_pv_output_k64p_globalv_feature_sort", 64, 1, "sort", 0),
        ("int8_pv_output_k64p_globalv_feature_log2x1", 64, 1, "log2x1", 0),
        ("int8_pv_output_k64p_globalv_feature_log2x2", 64, 1, "log2x2", 0),
        ("int8_pv_output_k64_feature_sort_h64", 64, 1, "sort", 64),
        ("int8_pv_output_k64_feature_log2x1_h64", 64, 1, "log2x1", 64),
        ("int8_pv_output_k64_feature_log2x2_h64", 64, 1, "log2x2", 64),
    )
    alignment_reference_variant_names = (
        "int8_pv_key_log_pow2_postdot",
        "int8_pv_key_log_pow2_predot_nearest",
        "int8_pv_key_log_pow2_predot_dithered_b1",
        "int8_pv_key_log_pow2_predot_dithered_b2",
        "int8_pv_key_log_pow2_predot_dithered_b4",
        "int8_pv_key_log_pow2_predot_dithered_b10",
        "int8_pv_key_log_pow2_predot_dithered_key_b10",
    )
    sampled_pair_reference_variants = (
        ("int8_pv_key_log_sampled_pair_h0", 0.0),
        ("int8_pv_key_log_sampled_pair_h025", 0.25),
        ("int8_pv_key_log_sampled_pair_h05", 0.5),
        ("int8_pv_key_log_sampled_pair_h1", 1.0),
        ("int8_pv_key_log_sampled_pair_h15", 1.5),
        ("int8_pv_key_log_sampled_pair_h2", 2.0),
        ("int8_pv_key_log_sampled_pair_h25", 2.5),
        ("int8_pv_key_log_sampled_pair_h3", 3.0),
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
        ("uint8_pv_key_log_narrow_denominator", 0, "key", "log", 0.0),
        ("uint8_pv_key_log_running_max", 0, "key", "log", 0.0),
        ("uint8_pv_key_log_scale_forward", 0, "key", "log", 0.0),
        ("uint8_pv_key_log_scale_forward_optimized_pv", 0, "key", "log", 0.0),
        (
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator",
            0,
            "key",
            "log",
            0.0,
        ),
        (
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator_fp32_metadata",
            0,
            "key",
            "log",
            0.0,
        ),
        (
            "uint8_pv_key_log_scale_forward_optimized_pv_scaled_fp16_numerator_scaled_fp16_denominator",
            0,
            "key",
            "log",
            0.0,
        ),
        ("uint8_pv_key_log_tile_common", 0, "key", "log", 0.0),
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
        + [name for name, _, _, _, _, _, _ in int8_pv_reference_variants]
        + [name for name, _, _, _, _ in output_scaled_reference_variants]
        + list(alignment_reference_variant_names)
        + [name for name, _ in sampled_pair_reference_variants]
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
    alignment_diagnostics: list[AlignmentDiagnostics] = []
    sampled_pair_diagnostics: list[tuple[str, float, float, float, float]] = []
    attention_passes = 2 if args.guidance_scale > 1.0 else 1
    layers_per_pass = 25
    calls_per_step = attention_passes * layers_per_pass
    call_index = 0
    prompt_index = 0

    def measured_dispatch(  # noqa: PLR0912, PLR0915
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
        call_within_step = call_index % calls_per_step
        attention_pass = call_within_step // layers_per_pass
        layer = call_within_step % layers_per_pass
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
                    attention_pass=attention_pass,
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
                        attention_pass=attention_pass,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        if selected_variant_names.intersection(
            name for name, _, _, _, _, _, _ in int8_pv_reference_variants
        ):
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            scaled_int8_scores = int8_scores * query.shape[-1] ** -0.5
            for (
                variant,
                block_n,
                pair_scale_bits,
                probability_dtype,
                accumulator_dtype,
                normalized_accumulator_dtype,
                center_value,
            ) in int8_pv_reference_variants:
                if variant not in selected_variant_names:
                    continue
                quantized_output = (
                    _int8_pv_key_log_reference(
                        scaled_int8_scores,
                        value_h,
                        block_n,
                        center_value=center_value,
                        probability_dtype=probability_dtype,
                        accumulator_dtype=accumulator_dtype,
                        normalized_accumulator_dtype=normalized_accumulator_dtype,
                    )
                    if pair_scale_bits is None
                    else _int8_pv_key_log_paired_reference(
                        scaled_int8_scores,
                        value_h,
                        pair_scale_bits,
                    )
                )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        attention_pass=attention_pass,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        selected_output_scaled = [
            item
            for item in output_scaled_reference_variants
            if item[0] in selected_variant_names
        ]
        if selected_output_scaled:
            permuted_inputs: dict[
                tuple[str, int],
                tuple[torch.Tensor, torch.Tensor, float],
            ] = {}
            input_modes = {(item[3], item[4]) for item in selected_output_scaled}
            for permutation_mode, rotation_group in input_modes:
                value_for_quantization = (
                    rotate_attention_groups(value_h.float(), rotation_group)
                    if rotation_group
                    else value_h
                )
                permuted_key, permuted_value = _permute_key_value_by_v_scale(
                    key_h,
                    value_for_quantization,
                    permutation_mode,
                )
                permuted_key_centered = (
                    permuted_key.float()
                    - permuted_key.float().mean(dim=2, keepdim=True)
                )
                permuted_exact_scores = torch.matmul(
                    query_h.float(),
                    permuted_key_centered.transpose(-1, -2),
                )
                permuted_int8_scores = _quantized_scores(
                    query_h,
                    permuted_key,
                    127,
                    None,
                )
                permuted_inputs[(permutation_mode, rotation_group)] = (
                    permuted_int8_scores * query.shape[-1] ** -0.5,
                    permuted_value,
                    _sqnr(permuted_int8_scores, permuted_exact_scores),
                )
            for (
                variant,
                block_n,
                feature_group,
                permutation_mode,
                rotation_group,
            ) in selected_output_scaled:
                scaled_scores, permuted_value, score_sqnr = permuted_inputs[
                    (permutation_mode, rotation_group)
                ]
                if "_int32" in variant or variant.endswith("_pow2_fp32"):
                    quantized_output = _int8_pv_output_scaled_int32_reference(
                        scaled_scores,
                        permuted_value,
                        block_n,
                        feature_group,
                        integer_alignment="_int32" in variant,
                        q8_coefficient_mantissa="_int32_q8mantissa" in variant,
                        common_feature_exponent=variant.endswith("_commonexp"),
                    )
                else:
                    quantized_output = _int8_pv_output_scaled_reference(
                        scaled_scores,
                        permuted_value,
                        block_n,
                        feature_group,
                        clip_rank=(
                            4
                            if variant.endswith("clip4")
                            else 2
                            if variant.endswith("clip2")
                            else 1
                            if variant.endswith("clip1")
                            else 0
                        ),
                        value_block_n=(32 if "_k64p_k32v_" in variant else None),
                        power_of_two_value_scale=("_pow2" in variant),
                        integer_pair_alignment=variant.endswith("_pow2_align"),
                        global_value_scale=("_globalv_" in variant),
                        scale_run_n=(
                            int(
                                variant.rsplit("_run", maxsplit=1)[1].split(
                                    "_", maxsplit=1
                                )[0]
                            )
                            if "_run" in variant
                            else None
                        ),
                        global_probability_codes="_global_p" in variant,
                        dither_probability_codes=variant.endswith("_dither"),
                        power_of_two_probability_weight=variant.endswith(
                            "_pow2weight"
                        ),
                        scaled_fp16_numerator=variant.endswith(
                            "_scaled_fp16_numerator"
                        ),
                    )
                if rotation_group:
                    quantized_output = rotate_attention_groups(
                        quantized_output,
                        rotation_group,
                    )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        attention_pass=attention_pass,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        if selected_variant_names.intersection(alignment_reference_variant_names):
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            scaled_int8_scores = int8_scores * query.shape[-1] ** -0.5
            alignment_outputs, diagnostics = _int8_pv_key_log_alignment_references(
                scaled_int8_scores,
                value_h,
            )
            alignment_diagnostics.append(diagnostics)
            for variant in alignment_reference_variant_names:
                if variant not in selected_variant_names:
                    continue
                quantized_output = alignment_outputs[variant]
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        attention_pass=attention_pass,
                        layer=layer,
                        sequence=query.shape[1],
                        variant=variant,
                        score_sqnr=int8_score_sqnr,
                        output_sqnr=_sqnr(quantized_output, expected_output),
                        output_relative_l1=_relative_l1(quantized_output, expected_output),
                    )
                )
        if selected_variant_names.intersection(name for name, _ in sampled_pair_reference_variants):
            int8_scores = _quantized_scores(query_h, key_h, 127, None)
            int8_score_sqnr = _sqnr(int8_scores, exact_scores)
            scaled_int8_scores = int8_scores * query.shape[-1] ** -0.5
            for variant, headroom_log2 in sampled_pair_reference_variants:
                if variant not in selected_variant_names:
                    continue
                (
                    quantized_output,
                    clipped_row_fraction,
                    maximum_gap_log2,
                    maximum_score_gap_log2,
                    maximum_scale_gap_log2,
                ) = _int8_pv_key_log_sampled_pair_reference(
                    scaled_int8_scores,
                    value_h,
                    headroom_log2,
                )
                sampled_pair_diagnostics.append(
                    (
                        variant,
                        clipped_row_fraction,
                        maximum_gap_log2,
                        maximum_score_gap_log2,
                        maximum_scale_gap_log2,
                    )
                )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        attention_pass=attention_pass,
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
                        attention_pass=attention_pass,
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
                optimized_pv_scaling = "_optimized_pv" in variant
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
                    native_uint8_mma=optimized_pv_scaling,
                    split_pv_head_dim=optimized_pv_scaling,
                    tile_common_log_denominator=variant.endswith("_tile_common"),
                    narrow_int8_log_denominator=variant.endswith("_narrow_denominator"),
                    running_max_probability_recurrence=variant.endswith("_running_max"),
                    scale_forward_log_recurrence="_scale_forward" in variant,
                    optimize_pv_scaling=optimized_pv_scaling,
                    fp32_pv_scale_metadata=(
                        True if variant.endswith("_fp32_metadata") else None
                    ),
                    scaled_fp16_numerator="_scaled_fp16_numerator" in variant,
                    scaled_fp16_denominator=variant.endswith(
                        "_scaled_fp16_denominator"
                    ),
                )
                measurements.append(
                    Measurement(
                        prompt=prompt_index,
                        step=step,
                        attention_pass=attention_pass,
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
                        attention_pass=attention_pass,
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
                guidance_scale=args.guidance_scale,
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
        f"{args.width}x{args.height}; denoising steps: {args.steps}; "
        f"guidance: {args.guidance_scale:g}"
    )
    print()
    print(
        "| prompt | step | pass | layer | sequence | variant | score SQNR | output SQNR "
        "| output rel-L1 |"
    )
    print("|---:|---:|---:|:---|---:|:---|---:|---:|---:|")
    for item in measurements:
        print(
            f"| {item.prompt} | {item.step} | {item.attention_pass} "
            f"| {_layer_name(item.layer)} | {item.sequence} "
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

    if sampled_pair_diagnostics:
        print()
        print(
            "| variant | mean clipped rows | worst transformed gap "
            "| worst score-only gap | worst V-scale-only gap |"
        )
        print("|:---|---:|---:|---:|---:|")
        for variant, _ in sampled_pair_reference_variants:
            selected = [item for item in sampled_pair_diagnostics if item[0] == variant]
            if not selected:
                continue
            mean_clipped = sum(item[1] for item in selected) / len(selected)
            maximum_gap = max(item[2] for item in selected)
            maximum_score_gap = max(item[3] for item in selected)
            maximum_scale_gap = max(item[4] for item in selected)
            print(
                f"| {variant} | {mean_clipped:.2%} | {maximum_gap:.2f} log2 "
                f"| {maximum_score_gap:.2f} log2 | {maximum_scale_gap:.2f} log2 |"
            )

    if alignment_diagnostics:
        gap_counts = tuple(
            sum(item.gap_counts[bucket] for item in alignment_diagnostics) for bucket in range(6)
        )
        merged_rows = sum(item.merged_rows for item in alignment_diagnostics)
        exponent_advances = sum(item.exponent_advances for item in alignment_diagnostics)
        local_nonzero = sum(item.local_nonzero_codes for item in alignment_diagnostics)
        nearest_nonzero = sum(item.nearest_nonzero_codes for item in alignment_diagnostics)
        dithered_nonzero = sum(item.dithered_nonzero_codes for item in alignment_diagnostics)
        print()
        print("Power-of-two alignment diagnostics:")
        labels = ("0", "1", "2", "3", "4", "5+")
        print(
            "  row exponent gaps: "
            + ", ".join(
                f"{label}={count / merged_rows:.2%}"
                for label, count in zip(labels, gap_counts, strict=True)
            )
        )
        print(f"  running exponent advances: {exponent_advances / merged_rows:.2%}")
        print(
            "  nonzero P-code retention: "
            f"nearest={nearest_nonzero / local_nonzero:.2%}, "
            f"dithered={dithered_nonzero / local_nonzero:.2%}"
        )


if __name__ == "__main__":
    main()
