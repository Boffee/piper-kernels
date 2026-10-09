"""Run fixed, hardware-independent operator workloads with one measurement protocol."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from lib.cases import diagnostic_cases, select_cases
from lib.environment import capture_environment
from lib.reporting import add_output_arguments, output_target, write_records
from lib.suite import Measurement, SuiteRecord, implementations, run_implementation


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--list", action="store_true", help="list cases without initializing a GPU")
    parser.add_argument(
        "--case", action="append", default=[], help="case identity or shell-style pattern"
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        choices=(
            "attention",
            "sparse_attention",
            "linear",
            "ffn",
            "conv3d",
            "pipeline",
        ),
    )
    parser.add_argument(
        "--provider", action="append", default=[], help="implementation name; repeat to compare"
    )
    parser.add_argument("--device", default="cuda", help="device shared by all cases in this run")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-ms", type=int, default=500)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if args.warmup_ms < 0 or args.measurement_ms <= 0:
        parser.error("warmup must be non-negative and measurement time must be positive")
    return args


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "No CUDA/ROCm device available. Use --list to inspect the common cases."
            )
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise SystemExit("The suite supports CUDA/ROCm, or explicit CPU reference runs.")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cases = select_cases(tuple(args.case), families=tuple(args.family))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.list:
        for case in cases:
            print(f"{case.id}: {case.as_dict()}")
        if not args.case:
            print("Small diagnostics (select explicitly with --case):")
            for case in diagnostic_cases():
                if not args.family or case.family in args.family:
                    print(f"  {case.id}")
        return 0
    device = _device(args.device)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    measurement = Measurement(args.warmup_ms, args.measurement_ms)
    selected = [(case, implementations(case, device)) for case in cases]
    available = {item.name for _, providers in selected for item in providers}
    unknown = set(args.provider) - available
    if unknown:
        raise SystemExit(f"Unknown provider selection: {', '.join(sorted(unknown))}")
    records: list[SuiteRecord] = []
    destination = output_target(args)
    for case, providers in selected:
        for provider in providers:
            if args.provider and provider.name not in args.provider:
                continue
            record = run_implementation(
                case,
                provider,
                device=device,
                environment=environment,
                measurement=measurement,
            )
            records.append(record)
            write_records(records, destination)
            timing = record.timings
            detail = f"{timing.median_ms:.3f} ms" if timing else record.reason
            print(f"{case.id} / {provider.name}: {record.status} {detail}", flush=True)
    return int(any(record.status == "failed" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
