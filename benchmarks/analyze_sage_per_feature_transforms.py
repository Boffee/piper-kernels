"""Evaluate calibrated feature bases for per-feature INT8 V on real FLUX activations."""

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
    triton_sage_attention_uint8_pv_bucketed_grouped,
    triton_sage_attention_uint8_pv_feature_convrot,
)
from piper_kernels.attention._sage2pp.reference import (
    _quantize_key_per_thread,
    _quantize_query_per_thread,
)

_TILE_K = 64
_GROUPED_TILE_K = 128
_GROUPED_SCALE_RUN_K = 512
_RANGE_BUCKET_LOG2_SCALE = 2
_P_RANGE = 255
_V_RANGE = 127


@dataclass(slots=True)
class Capture:
    """One exact attention invocation stored on CPU."""

    prompt: int
    step: int
    layer: int
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    expected: torch.Tensor


@dataclass(slots=True, frozen=True)
class Transform:
    """A per-head feature transform and its inverse."""

    forward: torch.Tensor
    inverse: torch.Tensor


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


def _apply_transform(value: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhnd,hde->bhne", value, transform)


def _hadamard_matrix(
    heads: int,
    width: int,
    group: int,
    device: torch.device,
) -> torch.Tensor:
    identity = torch.eye(width, device=device).reshape(1, 1, width, width)
    matrix = rotate_attention_groups(identity, group).reshape(width, width)
    return matrix.expand(heads, -1, -1).contiguous()


def _random_orthogonal(
    heads: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(0xFEA705E)
    matrices = torch.randn(
        heads,
        width,
        width,
        generator=generator,
        device=device,
    )
    orthogonal, triangular = torch.linalg.qr(matrices)
    signs = torch.where(
        triangular.diagonal(dim1=-2, dim2=-1) < 0,
        -1.0,
        1.0,
    )
    return orthogonal * signs[:, None, :]


def _build_transforms(
    calibration: list[Capture],
    layers: Sequence[int],
    device: torch.device,
) -> dict[str, dict[int, Transform]]:
    example = calibration[0].value
    heads = example.shape[1]
    width = example.shape[-1]
    identity = torch.eye(width, device=device).expand(heads, -1, -1).contiguous()
    hadamard = _hadamard_matrix(heads, width, 64, device)
    random = _random_orthogonal(heads, width, device)
    transforms: dict[str, dict[int, Transform]] = {
        "identity": {},
        "H64": {},
        "random orthogonal": {},
        "PCA": {},
        "PCA whiten": {},
        "diagonal RMS": {},
        "ZCA cond2": {},
        "ZCA cond4": {},
        "ZCA cond8": {},
        "ZCA cond16": {},
        "Cholesky cond4": {},
    }
    for layer in layers:
        values = torch.cat(
            [
                item.value.to(device=device, dtype=torch.float32)
                for item in calibration
                if item.layer == layer
            ],
            dim=2,
        )
        second_moment = torch.einsum("bhnd,bhne->hde", values, values)
        second_moment /= values.shape[0] * values.shape[2]
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        minimum = eigenvalues.amax(dim=-1, keepdim=True) * 1e-6
        feature_gain = eigenvalues.clamp_min(minimum).rsqrt()
        whiten = eigenvectors * feature_gain[:, None, :]
        whiten_inverse = feature_gain.reciprocal()[:, :, None] * eigenvectors.transpose(-1, -2)
        diagonal_gain = second_moment.diagonal(dim1=-2, dim2=-1).clamp_min(minimum).rsqrt()
        diagonal = torch.diag_embed(diagonal_gain)
        diagonal_inverse = torch.diag_embed(diagonal_gain.reciprocal())
        bounded_eigenvalues_by_condition = {
            condition: eigenvalues.clamp_min(eigenvalues.amax(dim=-1, keepdim=True) / condition**2)
            for condition in (2, 4, 8, 16)
        }
        bounded_covariance = torch.matmul(
            eigenvectors * bounded_eigenvalues_by_condition[4][:, None, :],
            eigenvectors.transpose(-1, -2),
        )
        cholesky = torch.linalg.cholesky(bounded_covariance)
        cholesky_whiten = torch.linalg.solve_triangular(
            cholesky.transpose(-1, -2),
            identity,
            upper=True,
        )
        cholesky_inverse = cholesky.transpose(-1, -2)
        transforms["identity"][layer] = Transform(identity, identity)
        transforms["H64"][layer] = Transform(hadamard, hadamard.transpose(-1, -2))
        transforms["random orthogonal"][layer] = Transform(random, random.transpose(-1, -2))
        transforms["PCA"][layer] = Transform(eigenvectors, eigenvectors.transpose(-1, -2))
        transforms["PCA whiten"][layer] = Transform(whiten, whiten_inverse)
        transforms["diagonal RMS"][layer] = Transform(diagonal, diagonal_inverse)
        for condition, bounded_eigenvalues in bounded_eigenvalues_by_condition.items():
            bounded_gain = bounded_eigenvalues.rsqrt()
            zca = torch.matmul(
                eigenvectors * bounded_gain[:, None, :],
                eigenvectors.transpose(-1, -2),
            )
            zca_inverse = torch.matmul(
                eigenvectors * bounded_gain.reciprocal()[:, None, :],
                eigenvectors.transpose(-1, -2),
            )
            transforms[f"ZCA cond{condition}"][layer] = Transform(zca, zca_inverse)
        transforms["Cholesky cond4"][layer] = Transform(
            cholesky_whiten,
            cholesky_inverse,
        )
    return transforms


def _quantized_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    key_centered = key.float() - key.float().mean(dim=2, keepdim=True)
    query_int, query_scale = _quantize_query_per_thread(query, _V_RANGE)
    key_int, key_scale = _quantize_key_per_thread(key_centered, _V_RANGE)
    integer_scores = torch.matmul(query_int.float(), key_int.transpose(-1, -2).float())
    return (
        integer_scores * query_scale[..., None] * key_scale[:, :, None, :] * query.shape[-1] ** -0.5
    )


def _quantize_value_per_feature(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructed = torch.empty_like(value)
    noise = torch.zeros((), device=value.device)
    for start in range(0, value.shape[2], _TILE_K):
        block = value[:, :, start : start + _TILE_K]
        scale = block.abs().amax(dim=2, keepdim=True) / _V_RANGE + 1e-7
        block_reconstructed = (block / scale).round().clamp(-_V_RANGE, _V_RANGE) * scale
        reconstructed[:, :, start : start + _TILE_K] = block_reconstructed
        noise += (block_reconstructed - block).square().sum()
    return reconstructed, noise


def _quantized_attention(
    scores: torch.Tensor,
    value: torch.Tensor,
    transform: Transform,
) -> tuple[torch.Tensor, float]:
    rotated = _apply_transform(value.float(), transform.forward)
    reconstructed, _ = _quantize_value_per_feature(rotated)
    accumulator = torch.zeros_like(value, dtype=torch.float32)
    denominator = torch.zeros(value.shape[:3], device=value.device, dtype=torch.float32)
    running_max = torch.full_like(denominator, -float("inf"))
    for start in range(0, value.shape[2], _TILE_K):
        stop = min(start + _TILE_K, value.shape[2])
        block_scores = scores[..., start:stop]
        block_max = block_scores.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        current_weight = torch.exp(block_max - next_max)
        probability = torch.exp(block_scores - block_max[..., None])
        probability_quantized = (probability * _P_RANGE).round() / _P_RANGE
        accumulator *= old_weight[..., None]
        accumulator += (
            torch.matmul(
                probability_quantized,
                reconstructed[:, :, start:stop],
            )
            * current_weight[..., None]
        )
        denominator = denominator * old_weight + probability.sum(dim=-1) * current_weight
        running_max = next_max
    rotated_output = accumulator / denominator.clamp_min(1e-30)[..., None]
    output = _apply_transform(rotated_output, transform.inverse)
    reconstructed_value = _apply_transform(reconstructed, transform.inverse)
    signal = value.float().square().sum().clamp_min(1e-30)
    noise = (reconstructed_value - value.float()).square().sum().clamp_min(1e-30)
    value_sqnr = float(10 * torch.log10(signal / noise))
    return output, value_sqnr


def _sort_transformed_value(
    scores: torch.Tensor,
    value: torch.Tensor,
    transform: Transform,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a feature basis and stably sort paired score/V rows by range."""
    transformed = _apply_transform(value.float(), transform.forward)
    row_range = transformed.abs().amax(dim=-1).clamp_min(1e-30)
    bucket = torch.floor(torch.log2(row_range) * _RANGE_BUCKET_LOG2_SCALE)
    order = torch.argsort(bucket, dim=-1, stable=True)
    sorted_value = torch.gather(
        transformed,
        2,
        order[..., None].expand_as(transformed),
    )
    sorted_scores = torch.gather(
        scores,
        -1,
        order[:, :, None, :].expand_as(scores),
    )
    return sorted_scores, sorted_value


def _grouped_scale_run_attention(
    scores: torch.Tensor,
    value: torch.Tensor,
    transform: Transform,
    feature_group: int,
) -> torch.Tensor:
    """Reference K128 UINT8-P attention with K512 grouped INT8-V scales."""
    feature_groups = value.shape[-1] // feature_group
    value_int = torch.empty_like(value)
    scale_vectors: list[torch.Tensor] = []
    for run_start in range(0, value.shape[2], _GROUPED_SCALE_RUN_K):
        run_stop = min(run_start + _GROUPED_SCALE_RUN_K, value.shape[2])
        value_run = value[:, :, run_start:run_stop]
        grouped = value_run.reshape(
            *value_run.shape[:-1],
            feature_groups,
            feature_group,
        )
        value_scale = grouped.abs().amax(dim=(2, 4)) / _V_RANGE + 1e-7
        scale_vectors.append(value_scale.repeat_interleave(feature_group, dim=-1))
        value_int[:, :, run_start:run_stop] = (
            (grouped / value_scale[:, :, None, :, None])
            .round()
            .clamp(-_V_RANGE, _V_RANGE)
            .reshape_as(value_run)
        )

    accumulator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=value.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros(scores.shape[:-1], device=value.device, dtype=torch.float32)
    running_max = torch.full_like(denominator, -float("inf"))
    for start in range(0, value.shape[2], _GROUPED_TILE_K):
        stop = min(start + _GROUPED_TILE_K, value.shape[2])
        block_scores = scores[..., start:stop]
        block_max = block_scores.amax(dim=-1)
        next_max = torch.maximum(running_max, block_max)
        old_weight = torch.exp(running_max - next_max)
        current_weight = torch.exp(block_max - next_max)
        probability = torch.exp(block_scores - block_max[..., None])
        probability_codes = (probability * _P_RANGE).round().clamp(0, _P_RANGE)
        scale_vector = scale_vectors[start // _GROUPED_SCALE_RUN_K]
        partial = torch.matmul(probability_codes, value_int[:, :, start:stop])
        accumulator = (
            accumulator * old_weight[..., None]
            + partial
            * scale_vector[:, :, None, :]
            * (current_weight / _P_RANGE)[..., None]
        )
        denominator = denominator * old_weight + probability.sum(dim=-1) * current_weight
        running_max = next_max
    transformed_output = accumulator / denominator.clamp_min(1e-30)[..., None]
    return _apply_transform(transformed_output, transform.inverse)


def _sqnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    signal = expected.float().square().mean().clamp_min(1e-30)
    noise = (actual.float() - expected.float()).square().mean().clamp_min(1e-30)
    return float(10 * torch.log10(signal / noise))


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0912, PLR0915
    """Calibrate on the first prompt and evaluate transforms on held-out prompts."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("The per-feature transform analyzer requires a CUDA GPU")
    if Flux2KleinPipeline is None or flux2_module is None:
        raise SystemExit("Run with the optional model dependencies; see benchmarks/README.md")
    prompts = args.prompt or [
        "A red fox standing in a snowy pine forest, cinematic lighting, detailed photograph",
        "A futuristic glass city beside the ocean at sunset, wide-angle architectural photograph",
        "An astronaut repairing a satellite above Earth, realistic space photography",
        "A watercolor illustration of an old bookshop on a rainy evening, warm window light",
    ]
    if len(prompts) < 2:
        raise SystemExit("provide at least two prompts for calibration and held-out evaluation")

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cuda")
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

    calibration = [item for item in captures if item.prompt == 0]
    evaluation = [item for item in captures if item.prompt > 0]
    transforms = _build_transforms(calibration, args.layers, torch.device("cuda"))
    value_measurements: dict[str, list[float]] = {name: [] for name in transforms}
    output_measurements: dict[str, list[float]] = {name: [] for name in transforms}
    layer_measurements: dict[tuple[str, int], list[float]] = {}
    per_key_measurements: list[float] = []
    fp8_measurements: list[float] = []
    grouped_transform_names = ("identity", "H64", "ZCA cond4", "ZCA cond8")
    grouped_feature_groups = (4, 8, 16, 32, 64)
    grouped_measurements: dict[tuple[str, int], list[float]] = {
        (name, feature_group): []
        for name in grouped_transform_names
        for feature_group in grouped_feature_groups
    }
    grouped_zca_triton_groups = (16, 32, 64)
    grouped_zca_triton_measurements: dict[int, list[float]] = {
        feature_group: [] for feature_group in grouped_zca_triton_groups
    }

    for capture in evaluation:
        query = capture.query.to("cuda")
        key = capture.key.to("cuda")
        value = capture.value.to("cuda")
        expected = capture.expected.to("cuda")
        scores = _quantized_scores(query, key)
        fp8_output = _run_sage_attention(
            query,
            key,
            value,
            query.shape[-1] ** -0.5,
            False,
            qk_quantization_range=127,
            grouped_qk=False,
            rotation_group=None,
        )
        fp8_measurements.append(_sqnr(fp8_output, expected))
        per_key_output = triton_sage_attention_uint8_pv_feature_convrot(
            query,
            key,
            value,
            query.shape[-1] ** -0.5,
            False,
            rotation_group=0,
            value_scale_axis="key",
            probability_scale_mode="dynamic",
            grouped_qk=False,
        )
        per_key_measurements.append(_sqnr(per_key_output, expected))
        for name, by_layer in transforms.items():
            transform = by_layer[capture.layer]
            output, value_sqnr = _quantized_attention(
                scores,
                value,
                transform,
            )
            output_sqnr = _sqnr(output, expected)
            value_measurements[name].append(value_sqnr)
            output_measurements[name].append(output_sqnr)
            layer_measurements.setdefault((name, capture.layer), []).append(output_sqnr)
            if name in grouped_transform_names:
                sorted_scores, sorted_value = _sort_transformed_value(
                    scores,
                    value,
                    transform,
                )
                for feature_group in grouped_feature_groups:
                    grouped_output = _grouped_scale_run_attention(
                        sorted_scores,
                        sorted_value,
                        transform,
                        feature_group,
                    )
                    grouped_measurements[(name, feature_group)].append(
                        _sqnr(grouped_output, expected)
                    )
        zca_transform = transforms["ZCA cond8"][capture.layer]
        transformed_value = _apply_transform(
            value.float(),
            zca_transform.forward,
        ).to(value.dtype)
        for feature_group in grouped_zca_triton_groups:
            transformed_output = triton_sage_attention_uint8_pv_bucketed_grouped(
                query,
                key,
                transformed_value,
                query.shape[-1] ** -0.5,
                False,
                feature_group=feature_group,
                scale_run_n=_GROUPED_SCALE_RUN_K,
                maxnreg=224,
            )
            grouped_zca_triton_measurements[feature_group].append(
                _sqnr(
                    _apply_transform(transformed_output.float(), zca_transform.inverse),
                    expected,
                )
            )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: {args.model}")
    print(
        f"Calibration captures: {len(calibration)} from prompt 0; "
        f"held-out captures: {len(evaluation)} from {len(prompts) - 1} prompt(s)"
    )
    print()
    print("| transform | mean V SQNR | mean output SQNR | worst output SQNR |")
    print("|:---|---:|---:|---:|")
    for name in transforms:
        values = value_measurements[name]
        outputs = output_measurements[name]
        print(
            f"| {name} | {sum(values) / len(values):.2f} dB "
            f"| {sum(outputs) / len(outputs):.2f} dB | {min(outputs):.2f} dB |"
        )
    print(
        f"| FP8 PV Triton | - | "
        f"{sum(fp8_measurements) / len(fp8_measurements):.2f} dB "
        f"| {min(fp8_measurements):.2f} dB |"
    )
    print(
        f"| per-key dynamic Triton | - | "
        f"{sum(per_key_measurements) / len(per_key_measurements):.2f} dB "
        f"| {min(per_key_measurements):.2f} dB |"
    )

    print()
    print("| layer | " + " | ".join(transforms) + " |")
    print("|---:|" + "---:|" * len(transforms))
    for layer in args.layers:
        cells = []
        for name in transforms:
            values = layer_measurements[(name, layer)]
            cells.append(f"{sum(values) / len(values):.2f}")
        print(f"| {layer} | " + " | ".join(cells) + " |")

    print()
    print("Sorted K512 scale runs with K128 UINT8-P tiles:")
    print("| transform | V group | mean output SQNR | worst output SQNR |")
    print("|:---|---:|---:|---:|")
    for name in grouped_transform_names:
        for feature_group in grouped_feature_groups:
            outputs = grouped_measurements[(name, feature_group)]
            print(
                f"| {name} | {feature_group} | {sum(outputs) / len(outputs):.2f} dB "
                f"| {min(outputs):.2f} dB |"
            )
    for feature_group in grouped_zca_triton_groups:
        outputs = grouped_zca_triton_measurements[feature_group]
        print(
            f"| ZCA cond8 actual Triton | {feature_group} "
            f"| {sum(outputs) / len(outputs):.2f} dB | {min(outputs):.2f} dB |"
        )


if __name__ == "__main__":
    main()
