"""Isolate the selected key-scaled INT8 PV recurrence instruction chain."""

# Triton JIT pointer arguments intentionally have no Python runtime types.
# The benchmark callback is consumed before the next sequence-loop iteration.
# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915, PLR0917

import argparse
import json
from collections.abc import Sequence

import torch
import triton
import triton.language as tl
import triton.testing
from profile_sage_pv_variant import _compiled_kernel, _compiler_report

_VARIANTS = (
    "persistent-int32",
    "fp32",
    "fp16",
    "fp16-old-weight",
    "fp16-current-weight",
    "fp16-weighted",
    "fp16-weighted-exp2",
    "fp16-preweighted-exp2",
    "fp16-qk",
    "fp16-weighted-qk",
    "attention-fixed",
    "attention-local",
    "attention-local-shared-weight",
    "attention-local-select-fma",
)


@triton.jit
def _pv_recurrence_kernel(
    probability_ptr,
    value_ptr,
    query_ptr,
    key_ptr,
    old_weight_ptr,
    current_weight_ptr,
    output_ptr,
    max_output_ptr,
    tiles,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    head_dim: tl.constexpr,
    fp32_recurrence: tl.constexpr,
    fp16_recurrence: tl.constexpr,
    apply_old_weight: tl.constexpr,
    apply_current_weight: tl.constexpr,
    compute_weights: tl.constexpr,
    preweight_before_downcast: tl.constexpr,
    derive_weights_from_qk: tl.constexpr,
    form_probability_from_scores: tl.constexpr,
    local_probability_coordinate: tl.constexpr,
    shared_weight_recurrence: tl.constexpr,
    select_fma_recurrence: tl.constexpr,
):
    """Run two D64 integer PV dots with one selected recurrence boundary."""
    program = tl.program_id(0)
    half_head_dim: tl.constexpr = head_dim // 2
    offsets_m = tl.arange(0, block_m)
    offsets_k = tl.arange(0, block_k)
    offsets_d = tl.arange(0, half_head_dim)
    offsets_qk_d = tl.arange(0, head_dim)

    if derive_weights_from_qk:
        query = tl.load(query_ptr + offsets_m[:, None] * head_dim + offsets_qk_d[None, :])
        running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
        if form_probability_from_scores:
            denominator = tl.zeros((block_m,), dtype=tl.float32)

    if fp16_recurrence:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
    elif fp32_recurrence:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    else:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.int32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.int32)

    for tile in tl.range(0, tiles, disable_licm=True):
        if derive_weights_from_qk:
            key = tl.load(
                key_ptr
                + tile * head_dim * block_k
                + offsets_qk_d[:, None] * block_k
                + offsets_k[None, :]
            )
            scores = tl.dot(query, key, out_dtype=tl.int32).to(tl.float32)
            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(running_max, block_max)
            if shared_weight_recurrence or select_fma_recurrence:
                block_is_new_max = block_max > running_max
                shared_weight = tl.exp2(-tl.abs(block_max - running_max))
                old_weight = tl.where(block_is_new_max, shared_weight, 1.0)
                current_weight = tl.where(block_is_new_max, 1.0, shared_weight)
            else:
                old_weight = tl.exp2(running_max - next_max)
                current_weight = tl.exp2(block_max - next_max)
        if form_probability_from_scores:
            if local_probability_coordinate:
                probability_float = tl.exp2(scores - block_max[:, None])
                tile_weight = current_weight
            else:
                probability_float = tl.exp2(scores - next_max[:, None])
                tile_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
            probability = tl.minimum(
                127.0,
                probability_float * 127.0 + 0.5,
            ).to(tl.int8)
            denominator = denominator * old_weight + tl.sum(probability_float, axis=1) * tile_weight
        else:
            probability = tl.load(
                probability_ptr
                + tile * block_m * block_k
                + offsets_m[:, None] * block_k
                + offsets_k[None, :]
            )
        value_base = tile * block_k * head_dim + offsets_k[:, None] * head_dim
        value_low = tl.load(value_ptr + value_base + offsets_d[None, :])
        value_high = tl.load(value_ptr + value_base + half_head_dim + offsets_d[None, :])

        if fp16_recurrence or fp32_recurrence:
            partial_low = tl.dot(probability, value_low, out_dtype=tl.int32)
            partial_high = tl.dot(probability, value_high, out_dtype=tl.int32)
        else:
            accumulator_low = tl.dot(
                probability,
                value_low,
                accumulator_low,
                out_dtype=tl.int32,
            )
            accumulator_high = tl.dot(
                probability,
                value_high,
                accumulator_high,
                out_dtype=tl.int32,
            )

        if fp32_recurrence:
            accumulator_low += partial_low.to(tl.float32)
            accumulator_high += partial_high.to(tl.float32)
        elif fp16_recurrence:
            if apply_old_weight and not derive_weights_from_qk:
                old_weight = tl.load(old_weight_ptr + tile * block_m + offsets_m)
                if compute_weights:
                    old_weight = tl.exp2(old_weight)
            if apply_current_weight and not derive_weights_from_qk:
                current_weight = tl.load(current_weight_ptr + tile * block_m + offsets_m)
                if compute_weights:
                    current_weight = tl.exp2(current_weight)
            if preweight_before_downcast:
                partial_low_scaled = (
                    partial_low.to(tl.float32) * (current_weight * (1.0 / 65536.0))[:, None]
                ).to(tl.float16)
                partial_high_scaled = (
                    partial_high.to(tl.float32) * (current_weight * (1.0 / 65536.0))[:, None]
                ).to(tl.float16)
            else:
                partial_low_scaled = (partial_low.to(tl.float32) * (1.0 / 65536.0)).to(tl.float16)
                partial_high_scaled = (partial_high.to(tl.float32) * (1.0 / 65536.0)).to(tl.float16)
            if select_fma_recurrence:
                weighted_low = tl.where(
                    block_is_new_max[:, None],
                    accumulator_low,
                    partial_low_scaled,
                )
                unweighted_low = tl.where(
                    block_is_new_max[:, None],
                    partial_low_scaled,
                    accumulator_low,
                )
                weighted_high = tl.where(
                    block_is_new_max[:, None],
                    accumulator_high,
                    partial_high_scaled,
                )
                unweighted_high = tl.where(
                    block_is_new_max[:, None],
                    partial_high_scaled,
                    accumulator_high,
                )
                accumulator_low = (
                    weighted_low * shared_weight[:, None].to(tl.float16) + unweighted_low
                )
                accumulator_high = (
                    weighted_high * shared_weight[:, None].to(tl.float16) + unweighted_high
                )
            elif apply_old_weight and apply_current_weight:
                if preweight_before_downcast:
                    accumulator_low = (
                        accumulator_low * old_weight[:, None].to(tl.float16) + partial_low_scaled
                    )
                    accumulator_high = (
                        accumulator_high * old_weight[:, None].to(tl.float16) + partial_high_scaled
                    )
                else:
                    accumulator_low = accumulator_low * old_weight[:, None].to(
                        tl.float16
                    ) + partial_low_scaled * current_weight[:, None].to(tl.float16)
                    accumulator_high = accumulator_high * old_weight[:, None].to(
                        tl.float16
                    ) + partial_high_scaled * current_weight[:, None].to(tl.float16)
            elif apply_old_weight:
                accumulator_low = (
                    accumulator_low * old_weight[:, None].to(tl.float16) + partial_low_scaled
                )
                accumulator_high = (
                    accumulator_high * old_weight[:, None].to(tl.float16) + partial_high_scaled
                )
            elif apply_current_weight:
                accumulator_low += partial_low_scaled * current_weight[:, None].to(tl.float16)
                accumulator_high += partial_high_scaled * current_weight[:, None].to(tl.float16)
            else:
                accumulator_low += partial_low_scaled
                accumulator_high += partial_high_scaled
        if derive_weights_from_qk:
            running_max = next_max

    output_base = output_ptr + program * block_m * head_dim
    if form_probability_from_scores:
        output_low = accumulator_low.to(tl.float32) / denominator[:, None]
        output_high = accumulator_high.to(tl.float32) / denominator[:, None]
    else:
        output_low = accumulator_low.to(tl.float32)
        output_high = accumulator_high.to(tl.float32)
    tl.store(
        output_base + offsets_m[:, None] * head_dim + offsets_d[None, :],
        output_low,
    )
    tl.store(
        output_base + offsets_m[:, None] * head_dim + half_head_dim + offsets_d[None, :],
        output_high,
    )
    if derive_weights_from_qk:
        tl.store(max_output_ptr + program * block_m + offsets_m, running_max)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=_VARIANTS)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--block-m", type=int, choices=[64, 128], default=128)
    parser.add_argument("--probability-dtype", choices=["int8", "uint8"], default="uint8")
    parser.add_argument("--num-warps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--num-stages", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=1500)
    return parser.parse_args(argv)


def _ttgir_report() -> dict[str, object]:
    """Count layout and arithmetic operations in the sole compiled specialization."""
    ttgir = str(_compiled_kernel(_pv_recurrence_kernel).asm["ttgir"])
    operation_counts = {
        operation: ttgir.count(operation)
        for operation in (
            "ttg.convert_layout",
            "tt.broadcast",
            "tt.expand_dims",
            "ttg.local_alloc",
            "ttg.local_load",
            "tt.dot",
            "arith.mulf",
            "arith.addf",
            "arith.extf",
            "arith.truncf",
        )
    }
    return {
        "operation_counts": operation_counts,
        "convert_layout_lines": [
            line.strip() for line in ttgir.splitlines() if "ttg.convert_layout" in line
        ],
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """Compile and measure one recurrence specialization."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.sequence % 64:
        raise SystemExit("sequence must be divisible by K64")
    if args.probability_dtype == "uint8" and torch.cuda.get_device_capability()[0] != 12:
        raise SystemExit("the native UINT8 control currently targets consumer SM120")

    block_k = 64
    head_dim = 128
    tiles = args.sequence // block_k
    programs = args.heads * triton.cdiv(args.sequence, args.block_m)
    probability_dtype = torch.uint8 if args.probability_dtype == "uint8" else torch.int8
    torch.manual_seed(7100 + args.sequence)
    probability = torch.randint(
        0,
        16,
        (tiles, args.block_m, block_k),
        device="cuda",
        dtype=probability_dtype,
    )
    value = torch.randint(
        -8,
        8,
        (tiles, block_k, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    query = torch.randint(
        -8,
        8,
        (args.block_m, head_dim),
        device="cuda",
        dtype=torch.int8,
    )
    key = torch.randint(
        -8,
        8,
        (tiles, head_dim, block_k),
        device="cuda",
        dtype=torch.int8,
    )
    old_log2_weight = torch.empty(
        (tiles, args.block_m),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-0.25, 0.0)
    current_log2_weight = torch.empty_like(old_log2_weight).uniform_(-1.0, 0.0)
    output = torch.empty(
        (programs, args.block_m, head_dim),
        device="cuda",
        dtype=torch.float32,
    )
    max_output = torch.empty(
        (programs, args.block_m),
        device="cuda",
        dtype=torch.float32,
    )

    fp32_recurrence = args.variant == "fp32"
    fp16_recurrence = args.variant.startswith(("fp16", "attention"))
    apply_old_weight = args.variant in (
        "fp16-old-weight",
        "fp16-weighted",
        "fp16-weighted-exp2",
        "fp16-preweighted-exp2",
        "fp16-weighted-qk",
        "attention-fixed",
        "attention-local",
        "attention-local-shared-weight",
        "attention-local-select-fma",
    )
    apply_current_weight = args.variant in (
        "fp16-current-weight",
        "fp16-weighted",
        "fp16-weighted-exp2",
        "fp16-preweighted-exp2",
        "fp16-weighted-qk",
        "attention-local",
        "attention-local-shared-weight",
        "attention-local-select-fma",
    )
    compute_weights = args.variant.endswith("exp2")
    preweight_before_downcast = args.variant == "fp16-preweighted-exp2"
    derive_weights_from_qk = args.variant.endswith("qk") or args.variant.startswith("attention")
    form_probability_from_scores = args.variant.startswith("attention")
    local_probability_coordinate = args.variant.startswith("attention-local")
    shared_weight_recurrence = args.variant == "attention-local-shared-weight"
    select_fma_recurrence = args.variant == "attention-local-select-fma"

    def launch() -> None:
        _pv_recurrence_kernel[(programs,)](
            probability,
            value,
            query,
            key,
            old_log2_weight,
            current_log2_weight,
            output,
            max_output,
            tiles,
            block_m=args.block_m,
            block_k=block_k,
            head_dim=head_dim,
            fp32_recurrence=fp32_recurrence,
            fp16_recurrence=fp16_recurrence,
            apply_old_weight=apply_old_weight,
            apply_current_weight=apply_current_weight,
            compute_weights=compute_weights,
            preweight_before_downcast=preweight_before_downcast,
            derive_weights_from_qk=derive_weights_from_qk,
            form_probability_from_scores=form_probability_from_scores,
            local_probability_coordinate=local_probability_coordinate,
            shared_weight_recurrence=shared_weight_recurrence,
            select_fma_recurrence=select_fma_recurrence,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    launch()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("recurrence produced non-finite output")
    latency_ms = float(
        triton.testing.do_bench(
            launch,
            warmup=args.warmup_ms,
            rep=args.repeat_ms,
        )
    )
    integer_ops = programs * tiles * 2 * args.block_m * block_k * (head_dim // 2) * 2
    print(
        json.dumps(
            {
                "variant": args.variant,
                "probability_dtype": args.probability_dtype,
                "sequence": args.sequence,
                "tiles": tiles,
                "programs": programs,
                "block_m": args.block_m,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
                "effective_integer_tops": integer_ops / (latency_ms * 1e9),
                "ttgir_operations": _ttgir_report(),
                **_compiler_report(_pv_recurrence_kernel, latency_ms),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
