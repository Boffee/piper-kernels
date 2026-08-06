"""Measure hardware-granular PV skipping on real FLUX.2 Klein activations."""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    import diffusers.models.transformers.transformer_flux2 as flux2_module
    from diffusers import Flux2KleinPipeline
except ModuleNotFoundError:
    flux2_module = None
    Flux2KleinPipeline = None

from analyze_sage_per_feature_transforms import (
    Capture,
    Transform,
    _apply_transform,
    _build_transforms,
    _quantized_scores,
    _sqnr,
)

from piper_kernels.attention._sage2pp.backends.triton import _run_sage_attention

_P_RANGE = 255
_V_RANGE = 127
_RANGE_BUCKET_LOG2_SCALE = 2
_GATES = ("v_only", "score", "value", "online_mass", "mass", "contribution")


@dataclass(slots=True)
class TileAnalysis:
    """Quantized output contributions and cheap/oracle risks for every PV tile."""

    outputs: torch.Tensor
    risks: dict[str, torch.Tensor]


@dataclass(slots=True)
class Measurement:
    """One held-out sparse-output measurement."""

    skip_fraction: float
    sqnr: float
    relative_l1: float


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
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=("original", "local", "global"),
        default=["original", "local", "global"],
    )
    parser.add_argument("--local-sort-n", type=int, default=512)
    parser.add_argument("--k-tiles", type=int, nargs="+", default=[128])
    parser.add_argument("--query-groups", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--scale-run-n", type=int, default=512)
    parser.add_argument("--feature-group", type=int, default=16)
    parser.add_argument("--basis", default="ZCA cond8")
    parser.add_argument("--gates", nargs="+", choices=_GATES, default=list(_GATES))
    parser.add_argument("--skip-targets", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.feature_group <= 0:
        raise ValueError("--feature-group must be positive")
    if args.scale_run_n <= 0 or args.local_sort_n <= 0:
        raise ValueError("scale and local-sort runs must be positive")
    if any(tile <= 0 or tile % 32 for tile in args.k_tiles):
        raise ValueError("every K tile must be a positive multiple of 32")
    if any(group <= 0 for group in args.query_groups):
        raise ValueError("every query group must be positive")
    if any(target <= 0 or target >= 1 for target in args.skip_targets):
        raise ValueError("skip targets must be strictly between zero and one")


def _capture_flux_activations(args: argparse.Namespace) -> list[Capture]:
    if Flux2KleinPipeline is None or flux2_module is None:
        raise RuntimeError("install the optional diffusers benchmark dependencies")
    prompts = args.prompt or [
        "A red fox standing in a snowy pine forest, cinematic lighting, detailed photograph",
        "A futuristic glass city beside the ocean at sunset, wide-angle architectural photograph",
        "An astronaut repairing a satellite above Earth, realistic space photography",
        "A watercolor illustration of an old bookshop on a rainy evening, warm window light",
    ]
    if len(prompts) < 2:
        raise ValueError("provide at least two prompts for calibration and held-out evaluation")

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    if args.sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    original_dispatch = flux2_module.dispatch_attention_fn
    captures: list[Capture] = []
    calls_per_step = 25
    call_index = 0
    prompt_index = 0

    def capturing_dispatch(
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
        if step in args.capture_steps and layer in args.layers:
            captures.append(
                Capture(
                    prompt=prompt_index,
                    step=step,
                    layer=layer,
                    query=query.permute(0, 2, 1, 3).contiguous().cpu(),
                    key=key.permute(0, 2, 1, 3).contiguous().cpu(),
                    value=value.permute(0, 2, 1, 3).contiguous().cpu(),
                    expected=exact.permute(0, 2, 1, 3).contiguous().cpu(),
                )
            )
        return exact

    flux2_module.dispatch_attention_fn = capturing_dispatch
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
    return captures


def _bucket_order(value: torch.Tensor, mode: str, local_sort_n: int) -> torch.Tensor:
    sequence = value.shape[2]
    row_range = value.abs().amax(dim=-1).clamp_min(1e-30)
    bucket = torch.floor(torch.log2(row_range) * _RANGE_BUCKET_LOG2_SCALE)
    if mode == "original":
        return torch.arange(sequence, device=value.device).expand(*value.shape[:2], sequence)
    if mode == "global":
        return torch.argsort(bucket, dim=-1, stable=True)

    order = torch.empty_like(bucket, dtype=torch.int64)
    for start in range(0, sequence, local_sort_n):
        stop = min(start + local_sort_n, sequence)
        local_order = torch.argsort(bucket[..., start:stop], dim=-1, stable=True)
        order[..., start:stop] = local_order + start
    return order


def _permute_keys(
    scores: torch.Tensor,
    value: torch.Tensor,
    order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_scores = torch.gather(
        scores,
        -1,
        order[:, :, None, :].expand_as(scores),
    )
    sorted_value = torch.gather(
        value,
        2,
        order[..., None].expand_as(value),
    )
    return sorted_scores, sorted_value


def _quantize_grouped_value(
    value: torch.Tensor,
    feature_group: int,
    scale_run_n: int,
) -> torch.Tensor:
    if value.shape[-1] % feature_group:
        raise ValueError("the feature width must be divisible by --feature-group")
    reconstructed = torch.empty_like(value)
    feature_groups = value.shape[-1] // feature_group
    for start in range(0, value.shape[2], scale_run_n):
        stop = min(start + scale_run_n, value.shape[2])
        run = value[:, :, start:stop]
        grouped = run.reshape(*run.shape[:-1], feature_groups, feature_group)
        scale = grouped.abs().amax(dim=(2, 4), keepdim=True) / _V_RANGE + 1e-7
        reconstructed[:, :, start:stop] = (
            (grouped / scale)
            .round()
            .clamp(-_V_RANGE, _V_RANGE)
            .mul(scale)
            .reshape_as(run)
        )
    return reconstructed


def _analyze_tiles(
    scores: torch.Tensor,
    transformed_value: torch.Tensor,
    transform: Transform,
    k_tile: int,
    feature_group: int,
    scale_run_n: int,
) -> TileAnalysis:
    reconstructed = _quantize_grouped_value(
        transformed_value,
        feature_group,
        scale_run_n,
    )
    physical_value = _apply_transform(reconstructed, transform.inverse)
    value_range = transformed_value.abs().amax(dim=-1).clamp_min(1e-30)
    global_max = scores.amax(dim=-1)
    denominator = torch.exp(scores - global_max[..., None]).sum(dim=-1).clamp_min(1e-30)
    running_max = torch.full_like(global_max, -float("inf"))
    running_denominator = torch.zeros_like(global_max)
    output_tiles = []
    v_only_risks = []
    score_risks = []
    value_risks = []
    online_mass_risks = []
    mass_risks = []
    contribution_risks = []

    for start in range(0, scores.shape[-1], k_tile):
        stop = min(start + k_tile, scores.shape[-1])
        block_scores = scores[..., start:stop]
        block_max = block_scores.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        local_probability = torch.exp(block_scores - block_max[..., None])
        old_weight = torch.exp(running_max - next_max)
        current_weight = torch.exp(block_max - next_max)
        current_mass = local_probability.sum(dim=-1) * current_weight
        next_denominator = running_denominator * old_weight + current_mass
        global_coefficient = torch.exp(block_max - global_max) / denominator
        probability_codes = (local_probability * _P_RANGE).round().clamp(0, _P_RANGE)
        effective_probability = (
            probability_codes
            * (global_coefficient / _P_RANGE)[..., None]
        )
        tile_output = torch.matmul(effective_probability, physical_value[:, :, start:stop])

        output_tiles.append(tile_output)
        v_only_risks.append(
            value_range[:, :, start:stop]
            .amax(dim=-1)[:, :, None]
            .expand_as(block_max)
        )
        score_risks.append(block_max - next_max)
        value_risks.append(
            (
                block_scores
                - next_max[..., None]
                + torch.log(value_range[:, :, None, start:stop])
            ).amax(dim=-1)
        )
        online_mass_risks.append(current_mass / next_denominator.clamp_min(1e-30))
        mass_risks.append(local_probability.sum(dim=-1) * global_coefficient)
        contribution_risks.append(tile_output.square().sum(dim=-1).sqrt())
        running_max = next_max
        running_denominator = next_denominator

    return TileAnalysis(
        outputs=torch.stack(output_tiles),
        risks={
            "v_only": torch.stack(v_only_risks),
            "score": torch.stack(score_risks),
            "value": torch.stack(value_risks),
            "online_mass": torch.stack(online_mass_risks),
            "mass": torch.stack(mass_risks),
            "contribution": torch.stack(contribution_risks),
        },
    )


def _group_risk(row_risk: torch.Tensor, query_group: int) -> torch.Tensor:
    sequence = row_risk.shape[-1]
    padded_sequence = ((sequence + query_group - 1) // query_group) * query_group
    if padded_sequence != sequence:
        row_risk = torch.nn.functional.pad(
            row_risk,
            (0, padded_sequence - sequence),
            value=float("inf"),
        )
    return row_risk.reshape(*row_risk.shape[:-1], -1, query_group).amax(dim=-1)


def _calibrated_threshold(risks: list[torch.Tensor], target: float) -> torch.Tensor:
    # Each input is [K tiles, batch, heads, query groups]. Select independently per head.
    flattened = [risk.permute(1, 2, 0, 3).flatten(start_dim=2) for risk in risks]
    values = torch.cat(flattened, dim=-1).sort(dim=-1).values
    index = min(max(round(target * values.shape[-1]) - 1, 0), values.shape[-1] - 1)
    return values[..., index]


def _sparse_output(
    analysis: TileAnalysis,
    group_risk: torch.Tensor,
    threshold: torch.Tensor,
    query_group: int,
) -> tuple[torch.Tensor, float]:
    skip = group_risk <= threshold[None, :, :, None]
    row_skip = skip.repeat_interleave(query_group, dim=-1)[..., : analysis.outputs.shape[-2]]
    output = (analysis.outputs * (~row_skip)[..., None]).sum(dim=0)
    return output, float(skip.float().mean())


def _relative_l1(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().abs().sum().clamp_min(1e-30)
    return float((actual.float() - expected.float()).abs().sum() / denominator)


def _prepare_capture(
    capture: Capture,
    transform: Transform,
    order_name: str,
    local_sort_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = capture.query.to("cuda")
    key = capture.key.to("cuda")
    value = capture.value.to(device="cuda", dtype=torch.float32)
    scores = _quantized_scores(query, key)
    transformed = _apply_transform(value, transform.forward)
    order = _bucket_order(transformed, order_name, local_sort_n)
    return _permute_keys(scores, transformed, order)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0912, PLR0915
    """Calibrate skip gates on one prompt and evaluate them on held-out prompts."""
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("the PV skipping analyzer requires a CUDA GPU")

    captures = _capture_flux_activations(args)
    calibration = [capture for capture in captures if capture.prompt == 0]
    evaluation = [capture for capture in captures if capture.prompt > 0]
    transforms = _build_transforms(calibration, args.layers, torch.device("cuda"))
    if args.basis not in transforms:
        raise ValueError(f"unknown basis {args.basis!r}; choose from {tuple(transforms)}")

    calibration_risks: dict[tuple[str, int, int, str, int], list[torch.Tensor]] = defaultdict(list)
    for capture in calibration:
        transform = transforms[args.basis][capture.layer]
        for order_name in args.orders:
            scores, value = _prepare_capture(
                capture,
                transform,
                order_name,
                args.local_sort_n,
            )
            for k_tile in args.k_tiles:
                analysis = _analyze_tiles(
                    scores,
                    value,
                    transform,
                    k_tile,
                    args.feature_group,
                    args.scale_run_n,
                )
                for query_group in args.query_groups:
                    for gate in args.gates:
                        calibration_risks[
                            (order_name, k_tile, query_group, gate, capture.layer)
                        ].append(_group_risk(analysis.risks[gate], query_group).cpu())

    thresholds: dict[tuple[str, int, int, str, float, int], torch.Tensor] = {}
    for key, risks in calibration_risks.items():
        order_name, k_tile, query_group, gate, layer = key
        for target in args.skip_targets:
            thresholds[(order_name, k_tile, query_group, gate, target, layer)] = (
                _calibrated_threshold(risks, target).to("cuda")
            )

    canonical_measurements: list[tuple[float, float]] = []
    dense_measurements: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    sparse_measurements: dict[
        tuple[str, int, int, str, float], list[Measurement]
    ] = defaultdict(list)

    for capture in evaluation:
        expected = capture.expected.to("cuda")
        query = capture.query.to("cuda")
        key = capture.key.to("cuda")
        value = capture.value.to("cuda")
        canonical = _run_sage_attention(
            query,
            key,
            value,
            query.shape[-1] ** -0.5,
            False,
            qk_quantization_range=127,
            grouped_qk=False,
            rotation_group=None,
        )
        canonical_measurements.append(
            (_sqnr(canonical, expected), _relative_l1(canonical, expected))
        )

        transform = transforms[args.basis][capture.layer]
        for order_name in args.orders:
            scores, transformed_value = _prepare_capture(
                capture,
                transform,
                order_name,
                args.local_sort_n,
            )
            for k_tile in args.k_tiles:
                analysis = _analyze_tiles(
                    scores,
                    transformed_value,
                    transform,
                    k_tile,
                    args.feature_group,
                    args.scale_run_n,
                )
                dense = analysis.outputs.sum(dim=0)
                dense_measurements[(order_name, k_tile)].append(
                    (_sqnr(dense, expected), _relative_l1(dense, expected))
                )
                for query_group in args.query_groups:
                    grouped_risks = {
                        gate: _group_risk(analysis.risks[gate], query_group)
                        for gate in args.gates
                    }
                    for gate, group_risk in grouped_risks.items():
                        for target in args.skip_targets:
                            threshold = thresholds[
                                (
                                    order_name,
                                    k_tile,
                                    query_group,
                                    gate,
                                    target,
                                    capture.layer,
                                )
                            ]
                            sparse, skip_fraction = _sparse_output(
                                analysis,
                                group_risk,
                                threshold,
                                query_group,
                            )
                            sparse_measurements[
                                (order_name, k_tile, query_group, gate, target)
                            ].append(
                                Measurement(
                                    skip_fraction=skip_fraction,
                                    sqnr=_sqnr(sparse, expected),
                                    relative_l1=_relative_l1(sparse, expected),
                                )
                            )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: {args.model}")
    print(
        f"Basis: {args.basis}; feature group: {args.feature_group}; "
        f"scale run: K{args.scale_run_n}"
    )
    print(
        f"Calibration captures: {len(calibration)} from prompt 0; "
        f"held-out captures: {len(evaluation)}"
    )
    print(f"Sequence lengths: {sorted({capture.query.shape[2] for capture in captures})}")

    canonical_sqnr = [measurement[0] for measurement in canonical_measurements]
    canonical_l1 = [measurement[1] for measurement in canonical_measurements]
    print()
    print("Dense controls:")
    print("| method | K tile | mean SQNR | worst SQNR | mean relative L1 |")
    print("|:---|---:|---:|---:|---:|")
    print(
        f"| canonical Sage2++ | - | {sum(canonical_sqnr) / len(canonical_sqnr):.2f} dB "
        f"| {min(canonical_sqnr):.2f} dB | {sum(canonical_l1) / len(canonical_l1):.4f} |"
    )
    for (order_name, k_tile), values in dense_measurements.items():
        sqnr = [value[0] for value in values]
        relative_l1 = [value[1] for value in values]
        print(
            f"| {order_name} dense | {k_tile} | {sum(sqnr) / len(sqnr):.2f} dB "
            f"| {min(sqnr):.2f} dB | {sum(relative_l1) / len(relative_l1):.4f} |"
        )

    print()
    print("Calibrated PV skipping on held-out prompts:")
    print(
        "| order | K tile | Q group | gate | calibration target | actual skip "
        "| mean SQNR | worst SQNR | mean relative L1 |"
    )
    print("|:---|---:|---:|:---|---:|---:|---:|---:|---:|")
    for key, values in sparse_measurements.items():
        order_name, k_tile, query_group, gate, target = key
        print(
            f"| {order_name} | {k_tile} | {query_group} | {gate} | {target:.0%} "
            f"| {sum(value.skip_fraction for value in values) / len(values):.1%} "
            f"| {sum(value.sqnr for value in values) / len(values):.2f} dB "
            f"| {min(value.sqnr for value in values):.2f} dB "
            f"| {sum(value.relative_l1 for value in values) / len(values):.4f} |"
        )


if __name__ == "__main__":
    main()
