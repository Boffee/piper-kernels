"""Tune a small Piper Attention schedule search offline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from lib import (
    AttentionShape,
    BenchmarkProvider,
    TuningCandidate,
    TuningPhase,
    TuningRecord,
    UnsupportedTuningCandidateError,
    add_output_arguments,
    capture_environment,
    make_attention_inputs,
    measure_quality,
    output_target,
    tune_candidates,
    write_records,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import triton as piper_attention_backend
from piper_kernels.attention.piper_attention.dispatch import _default_center_value

_SCHEDULES = ("pointer", "tensor-descriptor")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--kv-sequence", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--center-value",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the production centering policy",
    )
    parser.add_argument(
        "--schedules",
        choices=_SCHEDULES,
        nargs="+",
        default=list(_SCHEDULES),
    )
    parser.add_argument(
        "--phase",
        type=TuningPhase,
        choices=tuple(TuningPhase),
        default=TuningPhase.PREPARED_EXECUTION,
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measurement-time-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    add_output_arguments(parser, record_name="tuning candidate")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    lengths = (args.sequence, args.kv_sequence or args.sequence)
    if any(length <= 0 for length in lengths):
        raise SystemExit("attention sequence lengths must be positive")
    if args.batch_size <= 0 or args.heads <= 0:
        raise SystemExit("batch size and heads must be positive")
    if args.causal and lengths[0] != lengths[1]:
        raise SystemExit("causal attention requires equal query and key/value lengths")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")


def _make_candidate(
    schedule: str,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    scale: float,
    is_causal: bool,
    center_value: bool,
    sort_value_rows: bool,
    target: AcceleratorTarget,
    common_configuration: Mapping[str, object],
) -> TuningCandidate[object, torch.Tensor]:
    use_tensor_descriptors = schedule == "tensor-descriptor"

    def make_provider() -> BenchmarkProvider[object, torch.Tensor]:
        if use_tensor_descriptors and not target.is_cuda_capability(12):
            raise UnsupportedTuningCandidateError(
                "the tensor-descriptor candidate currently targets SM12x"
            )
        query, key, value = inputs

        def prepare() -> object:
            return piper_attention_backend._prepare_piper_attention(
                query,
                key,
                value,
                scale,
                is_causal,
                center_value,
                native_uint8=True,
                sort_value_rows=sort_value_rows,
                use_tensor_descriptors=use_tensor_descriptors,
            )

        def run(prepared: object) -> torch.Tensor:
            return piper_attention_backend._launch_piper_attention(
                cast(piper_attention_backend._PreparedPiperAttention, prepared)
            )

        return BenchmarkProvider(
            name=f"piper_attention_{schedule.replace('-', '_')}",
            prepare=prepare,
            run=run,
            synchronize=torch.cuda.synchronize,
        )

    return TuningCandidate(
        name=schedule,
        configuration={
            **common_configuration,
            "algorithm": "piper_attention",
            "load_path": schedule,
            "center_value": center_value,
            "value_row_order": ("centered_range_ascending" if sort_value_rows else "original"),
        },
        make_provider=make_provider,
    )


def _print_results(records: Sequence[TuningRecord]) -> None:
    print("| candidate | status | selected | p50 (ms) | SQNR (dB) | reason |")
    print("|:---|:---|:---:|---:|---:|:---|")
    for record in records:
        timing = "-" if record.timing is None else f"{record.timing.median_ms:.3f}"
        quality = "-" if record.quality is None else f"{record.quality.sqnr_db:.2f}"
        print(
            f"| {record.candidate} | {record.status.value} | {record.selected} "
            f"| {timing} | {quality} | {record.reason or ''} |"
        )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("Piper Attention tuning requires an available NVIDIA GPU")
    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    if not target.supports_uint8_int8_mma:
        raise SystemExit("Piper Attention tuning requires NVIDIA SM8x or SM12x")

    key_value_length = args.kv_sequence or args.sequence
    shape = AttentionShape(
        args.batch_size,
        args.heads,
        args.sequence,
        key_value_length,
        args.head_dim,
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    inputs = make_attention_inputs(shape, dtype=dtype, device=device, generator=generator)
    query, key, value = inputs
    scale = args.head_dim**-0.5
    center_value = (
        _default_center_value(query, key, args.causal, target)
        if args.center_value is None
        else args.center_value
    )
    sort_value_rows = piper_attention_backend._should_sort_value_rows(
        center_value=center_value,
        target=target,
        is_causal=args.causal,
        head_dim=args.head_dim,
        key_length=key_value_length,
    )
    common_configuration = {
        "dtype": args.dtype,
        "is_causal": args.causal,
        "scale": scale,
        "seed": args.seed,
    }
    candidates = tuple(
        _make_candidate(
            schedule,
            inputs,
            scale=scale,
            is_causal=args.causal,
            center_value=center_value,
            sort_value_rows=sort_value_rows,
            target=target,
            common_configuration=common_configuration,
        )
        for schedule in args.schedules
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=args.causal,
        scale=scale,
    )
    run = tune_candidates(
        candidates,
        tuning="piper_attention_load_path",
        shape=shape.as_dict(),
        environment=capture_environment(Path(__file__).resolve().parents[1]),
        phase=args.phase,
        warmup_ms=args.warmup_ms,
        measurement_time_ms=args.measurement_time_ms,
        measure_candidate_quality=lambda output: measure_quality(output, expected),
    )
    _print_results(run.records)
    write_records(run.records, output_target(args))
    if run.winner is None:
        raise SystemExit("no tuning candidate passed")
    print(f"selected: {run.winner.candidate}")


def main() -> None:
    """Run the Piper Attention schedule search."""
    _main()


if __name__ == "__main__":
    main()
