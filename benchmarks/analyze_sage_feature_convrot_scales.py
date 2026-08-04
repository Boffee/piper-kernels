"""Analyze per-key V scale ratios after feature-axis Hadamard rotations."""

import argparse
from collections import defaultdict
from collections.abc import Sequence

import torch

try:
    import diffusers.models.transformers.transformer_flux2 as flux2_module
    from diffusers import Flux2KleinPipeline
except ModuleNotFoundError:
    flux2_module = None
    Flux2KleinPipeline = None

from piper_kernels.attention._convrot_reference import rotate_attention_groups

_TILE_K = 64


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 5, 14, 24])
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def _signed_features(heads: int, width: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(0x5A6E2025)
    bits = torch.randint(
        0,
        2,
        (1, heads, 1, width),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    return bits.float().mul_(2).sub_(1)


def _rotate(
    value: torch.Tensor,
    group: int,
    *,
    signed: bool,
) -> torch.Tensor:
    value = value.float()
    if signed:
        value = value * _signed_features(value.shape[1], value.shape[-1], value.device)
    return value if group == 0 else rotate_attention_groups(value, group)


def _tile_ratios(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key_length = scale.shape[-1]
    padded_length = ((key_length + _TILE_K - 1) // _TILE_K) * _TILE_K
    padded = torch.zeros((*scale.shape[:-1], padded_length), device=scale.device)
    padded[..., :key_length] = scale
    grouped = padded.reshape(*scale.shape[:-1], -1, _TILE_K)
    valid = torch.arange(padded_length, device=scale.device).reshape(-1, _TILE_K) < key_length
    maximum = grouped.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    ratio = (grouped / maximum).masked_select(valid)
    median = grouped.masked_fill(~valid, torch.nan).nanmedian(dim=-1).values
    maximum_over_median = maximum.squeeze(-1) / median.clamp_min(1e-30)
    return ratio, maximum_over_median


def _quantiles(values: torch.Tensor, probabilities: Sequence[float]) -> list[float]:
    points = torch.tensor(probabilities, dtype=torch.float32)
    return [float(item) for item in torch.quantile(values.float(), points)]


def _query_conditioned_scale_usage(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return max(P*r) and r at the score winner for every query/K tile."""
    query_h = query.permute(0, 2, 1, 3).float()
    key_h = key.permute(0, 2, 1, 3).float()
    value_h = value.permute(0, 2, 1, 3).float()
    key_length = key_h.shape[2]
    alphas: list[torch.Tensor] = []
    winner_ratios: list[torch.Tensor] = []
    for start in range(0, key_length, _TILE_K):
        stop = min(start + _TILE_K, key_length)
        value_scale = value_h[:, :, start:stop].abs().amax(dim=-1)
        ratio = value_scale / value_scale.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        scores = (
            torch.matmul(
                query_h,
                key_h[:, :, start:stop].transpose(-1, -2),
            )
            * query_h.shape[-1] ** -0.5
        )
        winner = scores.argmax(dim=-1)
        probability = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        alphas.append((probability * ratio[:, :, None, :]).amax(dim=-1).reshape(-1).cpu())
        winner_ratios.append(
            torch.gather(
                ratio[:, :, None, :].expand_as(scores),
                -1,
                winner[..., None],
            )
            .reshape(-1)
            .cpu()
        )
    return torch.cat(alphas), torch.cat(winner_ratios)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0915
    """Capture real V tensors and report per-key scale concentration."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("The feature-scale analyzer requires a CUDA GPU")
    if Flux2KleinPipeline is None or flux2_module is None:
        raise SystemExit("Run with the optional model dependencies; see benchmarks/README.md")

    prompts = args.prompt or [
        "A red fox standing in a snowy pine forest, cinematic lighting, detailed photograph",
        "A futuristic glass city beside the ocean at sunset, wide-angle architectural photograph",
    ]
    variants = (
        ("none", 0, False),
        ("H16", 16, False),
        ("H64", 64, False),
        ("signed H16", 16, True),
        ("signed H64", 64, True),
    )
    ratios: dict[str, list[torch.Tensor]] = defaultdict(list)
    tile_spreads: dict[str, list[torch.Tensor]] = defaultdict(list)
    max_over_rms: dict[str, list[torch.Tensor]] = defaultdict(list)
    value_sqnr: dict[str, list[float]] = defaultdict(list)
    alpha_by_layer: dict[str, list[torch.Tensor]] = defaultdict(list)
    winner_ratio_by_layer: dict[str, list[torch.Tensor]] = defaultdict(list)

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cuda")
    original_dispatch = flux2_module.dispatch_attention_fn
    calls_per_step = 25
    call_index = 0

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

        value_h = value.permute(0, 2, 1, 3).contiguous().float()
        alpha, winner_ratio = _query_conditioned_scale_usage(query, key, value)
        alpha_by_layer["all"].append(alpha)
        alpha_by_layer[f"layer {layer}"].append(alpha)
        winner_ratio_by_layer["all"].append(winner_ratio)
        winner_ratio_by_layer[f"layer {layer}"].append(winner_ratio)
        row_rms = value_h.square().mean(dim=-1).sqrt().clamp_min(1e-30)
        rms_ratio, rms_spread = _tile_ratios(row_rms)
        ratios["RMS floor"].append(rms_ratio.cpu())
        tile_spreads["RMS floor"].append(rms_spread.cpu())

        signal = value_h.square().mean().clamp_min(1e-30)
        for name, group, signed in variants:
            rotated = _rotate(value_h, group, signed=signed)
            maximum = rotated.abs().amax(dim=-1).clamp_min(1e-30)
            ratio, spread = _tile_ratios(maximum)
            ratios[name].append(ratio.cpu())
            tile_spreads[name].append(spread.cpu())
            max_over_rms[name].append((maximum / row_rms).reshape(-1).cpu())

            scale = maximum[..., None] / 127
            quantized = (rotated / scale).round().clamp(-127, 127)
            reconstructed = quantized * scale
            if group:
                reconstructed = rotate_attention_groups(reconstructed, group)
            if signed:
                reconstructed *= _signed_features(
                    value_h.shape[1],
                    value_h.shape[-1],
                    value_h.device,
                )
            noise = (reconstructed - value_h).square().mean().clamp_min(1e-30)
            value_sqnr[name].append(float(10 * torch.log10(signal / noise)))
        return exact

    flux2_module.dispatch_attention_fn = measured_dispatch
    try:
        for prompt_index, prompt in enumerate(prompts):
            call_index = 0
            generator = torch.Generator("cuda").manual_seed(args.seed + prompt_index)
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
    print(f"Model: {args.model}; captures per variant: {len(value_sqnr['none'])}")
    print()
    print(
        "| variant | r min | r p1 | r p10 | r p50 | tile max/median p50 | p90 "
        "| max/RMS p50 | p99 | V SQNR |"
    )
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("RMS floor", *(item[0] for item in variants)):
        ratio = torch.cat(ratios[name])
        spread = torch.cat(tile_spreads[name])
        r_min, r_p1, r_p10, r_p50 = _quantiles(ratio, [0.0, 0.01, 0.1, 0.5])
        spread_p50, spread_p90 = _quantiles(spread, [0.5, 0.9])
        if name == "RMS floor":
            max_rms_p50 = max_rms_p99 = float("nan")
            sqnr = float("nan")
        else:
            max_rms_p50, max_rms_p99 = _quantiles(
                torch.cat(max_over_rms[name]),
                [0.5, 0.99],
            )
            sqnr = sum(value_sqnr[name]) / len(value_sqnr[name])
        print(
            f"| {name} | {r_min:.4f} | {r_p1:.4f} | {r_p10:.4f} | {r_p50:.4f} "
            f"| {spread_p50:.2f}x | {spread_p90:.2f}x | {max_rms_p50:.2f} "
            f"| {max_rms_p99:.2f} | {sqnr:.2f} dB |"
        )

    print()
    print("Query-conditioned use of a tile-wide probability scale (unrotated V):")
    print("| scope | alpha p1 | p10 | p50 | winner-r p1 | p10 | p50 | alpha<1/8 | alpha<1/4 |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    scopes = ("all", *(f"layer {layer}" for layer in args.layers))
    for scope in scopes:
        alpha = torch.cat(alpha_by_layer[scope])
        winner_ratio = torch.cat(winner_ratio_by_layer[scope])
        alpha_p1, alpha_p10, alpha_p50 = _quantiles(alpha, [0.01, 0.1, 0.5])
        winner_p1, winner_p10, winner_p50 = _quantiles(winner_ratio, [0.01, 0.1, 0.5])
        below_eighth = float((alpha < 0.125).float().mean())
        below_quarter = float((alpha < 0.25).float().mean())
        print(
            f"| {scope} | {alpha_p1:.4f} | {alpha_p10:.4f} | {alpha_p50:.4f} "
            f"| {winner_p1:.4f} | {winner_p10:.4f} | {winner_p50:.4f} "
            f"| {below_eighth:.2%} | {below_quarter:.2%} |"
        )


if __name__ == "__main__":
    main()
