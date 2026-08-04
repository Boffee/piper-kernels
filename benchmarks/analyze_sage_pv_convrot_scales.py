"""Analyze fixed-scale paired-ConvRot INT8 PV on real FLUX.2 activations."""

# ruff: noqa: PLR0912, PLR0915

import argparse
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

try:
    import diffusers.models.transformers.transformer_flux2 as flux2_module
    from diffusers import Flux2KleinPipeline
except ModuleNotFoundError:
    flux2_module = None
    Flux2KleinPipeline = None

from piper_kernels.attention._convrot_reference import rotate_attention_groups

_BLOCK_K = 64
_INT8_RANGE = 127


@dataclass(slots=True)
class CalibrationSamples:
    """Rotated operand maxima captured from one calibration prompt."""

    probability: dict[int, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    value: dict[int, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    probability_max_over_rms: dict[int, list[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )
    value_max_over_rms: dict[int, list[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )
    value_max_over_sequence_rms: dict[int, list[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )
    value_max_over_group_rms: dict[int, list[torch.Tensor]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass(slots=True, frozen=True)
class StaticScale:
    """Fixed scales for rotated P and V."""

    probability: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True, frozen=True)
class Quality:
    """One attention-output quality measurement."""

    step: int
    layer: int
    variant: str
    sqnr: float
    relative_l1: float
    probability_clip_fraction: float
    value_clip_fraction: float


@dataclass(slots=True)
class ClipCounter:
    """Count elements outside a fixed quantizer's representable range."""

    clipped: int = 0
    total: int = 0

    def add(self, values: torch.Tensor, scale: torch.Tensor) -> None:
        self.clipped += int((values.abs() > scale * _INT8_RANGE).sum())
        self.total += values.numel()

    @property
    def fraction(self) -> float:
        return self.clipped / self.total


def _sqnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().clamp_min(1e-30)
    signal = expected.float().square().mean().clamp_min(1e-30)
    return float(10 * torch.log10(signal / error))


def _relative_l1(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).abs().sum()
    signal = expected.float().abs().sum().clamp_min(1e-30)
    return float(error / signal)


def _padded_rotated_value(value: torch.Tensor, start: int) -> torch.Tensor:
    block = value[:, :, start : start + _BLOCK_K].float()
    if block.shape[2] != _BLOCK_K:
        padded = block.new_zeros((*block.shape[:2], _BLOCK_K, block.shape[-1]))
        padded[:, :, : block.shape[2]] = block
        block = padded
    return rotate_attention_groups(block.transpose(-1, -2), _BLOCK_K).transpose(-1, -2)


def _rotated_probability(probability: torch.Tensor) -> torch.Tensor:
    if probability.shape[-1] != _BLOCK_K:
        padded = probability.new_zeros((*probability.shape[:-1], _BLOCK_K))
        padded[..., : probability.shape[-1]] = probability
        probability = padded
    return rotate_attention_groups(probability, _BLOCK_K)


def _online_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    local_probability: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    key = key.float() - key.float().mean(dim=2, keepdim=True)
    query = query.float()
    running_max = torch.full(
        query.shape[:3], -float("inf"), device=query.device, dtype=torch.float32
    )
    denominator = torch.zeros_like(running_max)
    attention_scale = query.shape[-1] ** -0.5
    for start in range(0, key.shape[2], _BLOCK_K):
        stop = min(start + _BLOCK_K, key.shape[2])
        scores = torch.matmul(query, key[:, :, start:stop].transpose(-1, -2)) * attention_scale
        block_max = scores.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        if local_probability:
            current_weight = torch.exp(block_max - next_max)
            probability = torch.exp(scores - block_max[..., None])
        else:
            current_weight = torch.ones_like(old_weight)
            probability = torch.exp(scores - next_max[..., None])
        denominator = denominator * old_weight + probability.sum(dim=-1) * current_weight
        yield (
            old_weight,
            current_weight,
            denominator,
            _rotated_probability(probability),
            _padded_rotated_value(value, start),
        )
        running_max = next_max


def _collect_calibration(
    samples: CalibrationSamples,
    layer: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    local_probability: bool,
    value_rms_group_tiles: Sequence[int],
) -> None:
    sequence_value_rms = value.float().square().mean(dim=2).sqrt().clamp_min(1e-30)
    for block_index, (_, _, _, probability, rotated_value) in enumerate(
        _online_blocks(query, key, value, local_probability=local_probability)
    ):
        probability_maximum = probability.abs().amax(dim=-1)
        value_maximum = rotated_value.abs().amax(dim=2)
        probability_rms = probability.square().mean(dim=-1).sqrt().clamp_min(1e-30)
        value_rms = rotated_value.square().mean(dim=2).sqrt().clamp_min(1e-30)
        samples.probability[layer].append(probability_maximum.to(device="cpu"))
        samples.value[layer].append(value_maximum.to(device="cpu"))
        samples.probability_max_over_rms[layer].append(
            (probability_maximum / probability_rms).to(device="cpu")
        )
        samples.value_max_over_rms[layer].append((value_maximum / value_rms).to(device="cpu"))
        samples.value_max_over_sequence_rms[layer].append(
            (value_maximum / sequence_value_rms).to(device="cpu")
        )
        for group_tiles in value_rms_group_tiles:
            group_index = block_index // group_tiles
            group_start = group_index * group_tiles * _BLOCK_K
            group_stop = min(group_start + group_tiles * _BLOCK_K, value.shape[2])
            group_rms = (
                value[:, :, group_start:group_stop]
                .float()
                .square()
                .mean(dim=2)
                .sqrt()
                .clamp_min(1e-30)
            )
            samples.value_max_over_group_rms[group_tiles].append(
                (value_maximum / group_rms).to(device="cpu")
            )


def _quantile(values: torch.Tensor, percentile: float, dim: int | None = None) -> torch.Tensor:
    return torch.quantile(values.float(), percentile / 100.0, dim=dim)


def _calibrate(
    samples: CalibrationSamples,
    layers: Sequence[int],
    percentile: float,
    value_rms_group_tiles: Sequence[int],
) -> dict[str, dict[int, StaticScale]]:
    probability_global = torch.cat(
        [item.reshape(-1) for layer in layers for item in samples.probability[layer]]
    )
    value_global = torch.cat(
        [item.reshape(-1) for layer in layers for item in samples.value[layer]]
    )
    global_scale = StaticScale(
        probability=_quantile(probability_global, percentile) / _INT8_RANGE,
        value=_quantile(value_global, percentile) / _INT8_RANGE,
    )
    result: dict[str, dict[int, StaticScale]] = {
        "global": dict.fromkeys(layers, global_scale),
        "layer": {},
        "head_channel": {},
        "layer_p_dynamic_v": {},
        "head_p_dynamic_v": {},
    }
    probability_ratio = torch.cat(
        [item.reshape(-1) for layer in layers for item in samples.probability_max_over_rms[layer]]
    )
    value_ratio = torch.cat(
        [item.reshape(-1) for layer in layers for item in samples.value_max_over_rms[layer]]
    )
    rms_factor = StaticScale(
        probability=_quantile(probability_ratio, percentile) / _INT8_RANGE,
        value=_quantile(value_ratio, percentile) / _INT8_RANGE,
    )
    result["rms_factor"] = dict.fromkeys(layers, rms_factor)
    value_sequence_ratio = torch.cat(
        [
            item.reshape(-1)
            for layer in layers
            for item in samples.value_max_over_sequence_rms[layer]
        ]
    )
    sequence_rms_factor = _quantile(value_sequence_ratio, percentile) / _INT8_RANGE
    result["dynamic_p_sequence_rms_v"] = dict.fromkeys(
        layers,
        StaticScale(
            probability=rms_factor.probability,
            value=sequence_rms_factor,
        ),
    )
    for group_tiles in value_rms_group_tiles:
        group_ratio = torch.cat(samples.value_max_over_group_rms[group_tiles])
        group_factor = _quantile(group_ratio.reshape(-1), percentile) / _INT8_RANGE
        result[f"dynamic_p_vrms_g{group_tiles}"] = dict.fromkeys(
            layers,
            StaticScale(
                probability=rms_factor.probability,
                value=group_factor,
            ),
        )
        result[f"rms_p_vrms_g{group_tiles}"] = dict.fromkeys(
            layers,
            StaticScale(
                probability=rms_factor.probability,
                value=group_factor,
            ),
        )
    result["fixed_p_dynamic_v"] = dict.fromkeys(
        layers,
        StaticScale(probability=global_scale.probability, value=rms_factor.value),
    )
    result["dynamic_p_rms_v"] = dict.fromkeys(layers, rms_factor)
    result["rms_p_dynamic_v"] = dict.fromkeys(layers, rms_factor)
    result["fixed_p_rms_v"] = dict.fromkeys(
        layers,
        StaticScale(probability=global_scale.probability, value=rms_factor.value),
    )
    for layer in layers:
        probability = torch.stack(samples.probability[layer])
        value = torch.stack(samples.value[layer])
        result["layer"][layer] = StaticScale(
            probability=_quantile(probability.reshape(-1), percentile) / _INT8_RANGE,
            value=_quantile(value.reshape(-1), percentile) / _INT8_RANGE,
        )
        # Calibration tensors are [samples, batch, heads, queries] for P and
        # [samples, batch, heads, channels] for V. Static P uses one scale per
        # head; static V retains one scale per head/output channel.
        probability_by_head = probability.permute(2, 0, 1, 3).reshape(probability.shape[2], -1)
        value_by_head_channel = value.permute(2, 3, 0, 1).reshape(
            value.shape[2], value.shape[3], -1
        )
        result["head_channel"][layer] = StaticScale(
            probability=(_quantile(probability_by_head, percentile, dim=1) / _INT8_RANGE).reshape(
                1, -1, 1, 1
            ),
            value=(_quantile(value_by_head_channel, percentile, dim=2) / _INT8_RANGE).reshape(
                1, value.shape[2], 1, value.shape[3]
            ),
        )
        result["layer_p_dynamic_v"][layer] = StaticScale(
            probability=result["layer"][layer].probability,
            value=rms_factor.value,
        )
        result["head_p_dynamic_v"][layer] = StaticScale(
            probability=result["head_channel"][layer].probability,
            value=rms_factor.value,
        )
    return result


def _quantize(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (values / scale).round().clamp(-_INT8_RANGE, _INT8_RANGE)


def _evaluate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    expected: torch.Tensor,
    scales: dict[str, StaticScale],
    step: int,
    layer: int,
    *,
    local_probability: bool,
) -> list[Quality]:
    accumulators = {
        name: torch.zeros_like(query, dtype=torch.float32) for name in ("dynamic", *scales)
    }
    probability_clips = {name: ClipCounter() for name in scales}
    value_clips = {name: ClipCounter() for name in scales}
    sequence_value_rms = value.float().square().mean(dim=2, keepdim=True).sqrt()
    group_rms_cache: dict[tuple[int, int], torch.Tensor] = {}
    latest_denominator = None
    for block_index, (
        old_weight,
        current_weight,
        block_denominator,
        probability,
        rotated_value,
    ) in enumerate(_online_blocks(query, key, value, local_probability=local_probability)):
        latest_denominator = block_denominator
        for accumulator in accumulators.values():
            accumulator.mul_(old_weight[..., None])

        probability_scale = probability.abs().amax(dim=-1, keepdim=True) / _INT8_RANGE + 1e-30
        value_scale = rotated_value.abs().amax(dim=2, keepdim=True) / _INT8_RANGE + 1e-30
        dynamic_probability = _quantize(probability, probability_scale)
        dynamic_value = _quantize(rotated_value, value_scale)
        accumulators["dynamic"] += (
            torch.matmul(dynamic_probability, dynamic_value)
            * probability_scale
            * value_scale
            * current_weight[..., None]
        )

        for name, fixed in scales.items():
            if name in ("rms_factor", "rms_p_dynamic_v") or name.startswith("rms_p_vrms_g"):
                probability_fixed = (
                    probability.square().mean(dim=-1, keepdim=True).sqrt()
                    * fixed.probability.to(probability.device)
                ).clamp_min(1e-30)
            elif name.startswith("dynamic_p_"):
                probability_fixed = probability_scale
            else:
                probability_fixed = fixed.probability.to(probability.device)
            if name == "dynamic_p_sequence_rms_v":
                value_fixed = (sequence_value_rms * fixed.value.to(rotated_value.device)).clamp_min(
                    1e-30
                )
            elif "_vrms_g" in name:
                group_tiles = int(name.rsplit("_vrms_g", maxsplit=1)[1])
                group_index = block_index // group_tiles
                cache_key = (group_tiles, group_index)
                if cache_key not in group_rms_cache:
                    group_start = group_index * group_tiles * _BLOCK_K
                    group_stop = min(
                        group_start + group_tiles * _BLOCK_K,
                        value.shape[2],
                    )
                    group_rms_cache[cache_key] = (
                        value[:, :, group_start:group_stop]
                        .float()
                        .square()
                        .mean(dim=2, keepdim=True)
                        .sqrt()
                    )
                value_fixed = (
                    group_rms_cache[cache_key] * fixed.value.to(rotated_value.device)
                ).clamp_min(1e-30)
            elif name in ("rms_factor", "dynamic_p_rms_v", "fixed_p_rms_v"):
                value_fixed = (
                    rotated_value.square().mean(dim=2, keepdim=True).sqrt()
                    * fixed.value.to(rotated_value.device)
                ).clamp_min(1e-30)
            elif name.endswith("_dynamic_v"):
                value_fixed = value_scale
            else:
                value_fixed = fixed.value.to(rotated_value.device)
            probability_clips[name].add(probability, probability_fixed)
            value_clips[name].add(rotated_value, value_fixed)
            accumulators[name] += (
                torch.matmul(
                    _quantize(probability, probability_fixed),
                    _quantize(rotated_value, value_fixed),
                )
                * probability_fixed
                * value_fixed
                * current_weight[..., None]
            )

    assert latest_denominator is not None
    results = []
    for name, accumulator in accumulators.items():
        output = accumulator / latest_denominator[..., None]
        results.append(
            Quality(
                step=step,
                layer=layer,
                variant=name,
                sqnr=_sqnr(output, expected),
                relative_l1=_relative_l1(output, expected),
                probability_clip_fraction=(
                    0.0 if name == "dynamic" else probability_clips[name].fraction
                ),
                value_clip_fraction=0.0 if name == "dynamic" else value_clips[name].fraction,
            )
        )
    return results


def _print_statistics(
    samples: CalibrationSamples,
    layers: Sequence[int],
    value_rms_group_tiles: Sequence[int],
) -> None:
    print("| operand | scope | CV | p50 max | p90 | p99 | p99.9 | maximum | max/p50 |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
    for operand, mapping in (
        ("P", samples.probability),
        ("V", samples.value),
    ):
        all_values = torch.cat(
            [item.reshape(-1) for layer in layers for item in mapping[layer]]
        ).float()
        for scope, values in (
            ("global", all_values),
            *(
                (f"layer {layer}", torch.cat([item.reshape(-1) for item in mapping[layer]]))
                for layer in layers
            ),
        ):
            mean = values.mean()
            p50, p90, p99, p999 = torch.quantile(
                values.float(), torch.tensor([0.5, 0.9, 0.99, 0.999])
            )
            maximum = values.max()
            print(
                f"| {operand} | {scope} | {float(values.std() / mean):.3f} "
                f"| {float(p50):.5f} | {float(p90):.5f} | {float(p99):.5f} "
                f"| {float(p999):.5f} | {float(maximum):.5f} "
                f"| {float(maximum / p50):.1f}x |"
            )
    print()
    print("| operand | max/RMS CV | p50 | p90 | p99 | p99.9 | maximum |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    for operand, mapping in (
        ("P", samples.probability_max_over_rms),
        ("V", samples.value_max_over_rms),
    ):
        values = torch.cat(
            [item.reshape(-1) for layer in layers for item in mapping[layer]]
        ).float()
        mean = values.mean()
        p50, p90, p99, p999 = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 0.999]))
        print(
            f"| {operand} | {float(values.std() / mean):.3f} | {float(p50):.3f} "
            f"| {float(p90):.3f} | {float(p99):.3f} | {float(p999):.3f} "
            f"| {float(values.max()):.3f} |"
        )
    sequence_values = torch.cat(
        [
            item.reshape(-1)
            for layer in layers
            for item in samples.value_max_over_sequence_rms[layer]
        ]
    ).float()
    p50, p90, p99, p999 = torch.quantile(sequence_values, torch.tensor([0.5, 0.9, 0.99, 0.999]))
    print(
        f"| V/sequence | {float(sequence_values.std() / sequence_values.mean()):.3f} "
        f"| {float(p50):.3f} | {float(p90):.3f} | {float(p99):.3f} "
        f"| {float(p999):.3f} | {float(sequence_values.max()):.3f} |"
    )
    for group_tiles in value_rms_group_tiles:
        values = torch.cat(samples.value_max_over_group_rms[group_tiles]).float().reshape(-1)
        p50, p90, p99, p999 = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 0.999]))
        print(
            f"| V/{group_tiles} tiles | {float(values.std() / values.mean()):.3f} "
            f"| {float(p50):.3f} | {float(p90):.3f} | {float(p99):.3f} "
            f"| {float(p999):.3f} | {float(values.max()):.3f} |"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 5, 14, 24])
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--local-probability", action="store_true")
    parser.add_argument(
        "--v-rms-group-tiles",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """Calibrate on one prompt, then evaluate fixed scales on another prompt."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("The ConvRot scale analysis requires a CUDA GPU")
    if Flux2KleinPipeline is None or flux2_module is None:
        raise SystemExit("Run with the optional model dependencies; see benchmarks/README.md")
    if any(group_tiles < 1 for group_tiles in args.v_rms_group_tiles):
        raise SystemExit("V RMS group tile counts must be positive")
    calibration_prompt = (
        "A red fox standing in a snowy pine forest, cinematic lighting, detailed photograph"
    )
    validation_prompt = (
        "A futuristic glass city beside the ocean at sunset, wide-angle architectural photograph"
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cuda")
    original_dispatch = flux2_module.dispatch_attention_fn
    samples = CalibrationSamples()
    calibrated: dict[str, dict[int, StaticScale]] | None = None
    qualities: list[Quality] = []
    prompt_index = 0
    call_index = 0
    calls_per_step = 25

    def measured_dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *dispatch_args: object,
        **dispatch_kwargs: object,
    ) -> torch.Tensor:
        nonlocal call_index
        exact = original_dispatch(query, key, value, *dispatch_args, **dispatch_kwargs)
        step = call_index // calls_per_step
        layer = call_index % calls_per_step
        call_index += 1
        if step not in args.capture_steps or layer not in args.layers:
            return exact
        query_h = query.permute(0, 2, 1, 3).contiguous()
        key_h = key.permute(0, 2, 1, 3).contiguous()
        value_h = value.permute(0, 2, 1, 3).contiguous()
        if prompt_index == 0:
            _collect_calibration(
                samples,
                layer,
                query_h,
                key_h,
                value_h,
                local_probability=args.local_probability,
                value_rms_group_tiles=args.v_rms_group_tiles,
            )
        else:
            assert calibrated is not None
            fixed = {name: by_layer[layer] for name, by_layer in calibrated.items()}
            qualities.extend(
                _evaluate(
                    query_h,
                    key_h,
                    value_h,
                    exact.permute(0, 2, 1, 3),
                    fixed,
                    step,
                    layer,
                    local_probability=args.local_probability,
                )
            )
        return exact

    flux2_module.dispatch_attention_fn = measured_dispatch
    try:
        for current_prompt, prompt in enumerate((calibration_prompt, validation_prompt)):
            prompt_index = current_prompt
            call_index = 0
            if current_prompt == 1:
                calibrated = _calibrate(
                    samples,
                    args.layers,
                    args.percentile,
                    args.v_rms_group_tiles,
                )
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
    finally:
        flux2_module.dispatch_attention_fn = original_dispatch

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Model: {args.model}; resolution: {args.width}x{args.height}; "
        f"fixed-scale calibration percentile: {args.percentile}; "
        f"local probability normalization: {args.local_probability}"
    )
    print()
    _print_statistics(samples, args.layers, args.v_rms_group_tiles)
    print()
    print("| variant | mean SQNR | worst SQNR | mean rel-L1 | P clipped | V clipped |")
    print("|:---|---:|---:|---:|---:|---:|")
    assert calibrated is not None
    for variant in ("dynamic", *calibrated):
        selected = [quality for quality in qualities if quality.variant == variant]
        print(
            f"| {variant} | {sum(item.sqnr for item in selected) / len(selected):.2f} dB "
            f"| {min(item.sqnr for item in selected):.2f} dB "
            f"| {sum(item.relative_l1 for item in selected) / len(selected):.4f} "
            f"| {sum(item.probability_clip_fraction for item in selected) / len(selected):.6f} "
            f"| {sum(item.value_clip_fraction for item in selected) / len(selected):.6f} |"
        )


if __name__ == "__main__":
    main()
