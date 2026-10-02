"""Diagnose ConvRot INT8 convolution layouts and prepared convolution schedules.

Use benchmark.py for stable cross-accelerator comparisons. This diagnostic can
consume the same named case and add alternate layouts or an offline tile search.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import torch
from lib.cases import Conv3DCase, named_case
from lib.environment import EnvironmentInfo, capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.suite_types import normal_tensor
from lib.timing import DeviceTimings, measure_device
from torch.nn import functional

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import conv3d, group_norm_silu_conv3d, reference
from piper_kernels.conv3d.convrot.int8 import triton as shared
from piper_kernels.conv3d.convrot.int8._amd import policy as amd_policy
from piper_kernels.conv3d.convrot.int8._dispatch import default_execution_plan
from piper_kernels.conv3d.convrot.int8._interfaces import ConvolutionPolicy
from piper_kernels.conv3d.convrot.int8._nvidia import policy as nvidia_policy
from piper_kernels.conv3d.convrot.int8._plan import ConvolutionExecutionPlan, ConvolutionSchedule
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor


class ConvolutionFlags(TypedDict):
    symmetric_spatial_padding: bool
    right_spatial_padding: bool
    residual: torch.Tensor | None


def _shape(value: str) -> tuple[int, ...]:
    try:
        dimensions = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use N,C,T,H,W,O") from error
    if len(dimensions) != 6 or min(dimensions) < 1:
        raise argparse.ArgumentTypeError("use six positive N,C,T,H,W,O dimensions")
    channels = dimensions[1]
    if channels < 64 or channels > 4096 or channels & (channels - 1):
        raise argparse.ArgumentTypeError("C must be a power of two in [64, 4096]")
    if min(dimensions[3:5]) < 2:
        raise argparse.ArgumentTypeError("reflection padding requires H,W >= 2")
    return dimensions


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="use an unchanged convolution case from the shared catalog")
    parser.add_argument("--shape", type=_shape, action="append", help="N,C,T,H,W,O; repeatable")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--measurement-time-ms", "--rep-ms", dest="measurement_time_ms", type=int, default=100
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=871)
    parser.add_argument("--tune", action="store_true", help="sweep prepared convolution tiles")
    parser.add_argument(
        "--vendor-benchmark", action="store_true", help="enable vendor convolution algorithm search"
    )
    parser.add_argument(
        "--skip-reference-timing",
        action="store_true",
        help="still check the INT8 reference, but time only native and FP16 implementations",
    )
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    args.group_norm_silu = None
    if args.case is not None:
        try:
            case = named_case(args.case)
        except ValueError as error:
            parser.error(str(error))
        if not isinstance(case, Conv3DCase):
            parser.error("--case requires a convolution case")
        supplied = {part.split("=", 1)[0] for part in (sys.argv[1:] if argv is None else argv)}
        if supplied & {"--shape", "--dtype", "--seed"}:
            parser.error("--case cannot be combined with workload overrides")
        args.shape = [
            (case.batch, case.channels, case.frames, case.height, case.width, case.out_channels)
        ]
        args.dtype, args.seed = case.dtype, case.seed
        args.group_norm_silu = case.group_norm_silu
    if args.measurement_time_ms <= 0 or args.samples < 1 or args.warmup_ms < 0 or args.device < 0:
        parser.error("requires positive duration/samples and non-negative warmup/device")
    return args


def _convolution_policy(target: AcceleratorTarget) -> ConvolutionPolicy:
    if nvidia_policy.supports_target(target):
        return cast(ConvolutionPolicy, nvidia_policy)
    if amd_policy.supports_target(target):
        return cast(ConvolutionPolicy, amd_policy)
    raise ValueError(f"ConvRot INT8 convolution benchmarking has no optimized backend for {target}")


def _standard_conv3d(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Include the same spatial reflection and leading two zero frames as ConvRot."""
    padded = functional.pad(activation, (1, 1, 1, 1, 0, 0), mode="reflect")
    padded = functional.pad(padded, (0, 0, 0, 0, 2, 0))
    return functional.conv3d(padded, weight)


def _standard_group_norm_silu_conv3d(
    activation: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    """Standard eager FP16 intermediates, with normalization isolated per frame."""
    batch, channels, frames, height, width = activation.shape
    memory_format = (
        torch.channels_last_3d
        if activation.is_contiguous(memory_format=torch.channels_last_3d)
        else torch.contiguous_format
    )
    framewise = activation.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    normalized = functional.group_norm(framewise, 32, norm_weight, norm_bias, 1e-6)
    normalized = normalized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    normalized = normalized.contiguous(memory_format=memory_format)
    return _standard_conv3d(functional.silu(normalized), weight)


def _measure_implementations(
    implementations: Mapping[str, Callable[[], torch.Tensor]],
    *,
    warmup_ms: int,
    measurement_time_ms: int,
    samples: int,
    skip_reference_timing: bool,
) -> dict[str, DeviceTimings]:
    # Warm/check every implementation before timing, including a skipped reference.
    # FP16 is a separate numerical baseline, not an oracle for quantized outputs.
    torch.testing.assert_close(
        implementations["native"](), implementations["reference"](), atol=4e-3, rtol=4e-3
    )
    torch.testing.assert_close(
        implementations["fp16"](), implementations["fp16_channels_last"](), atol=2e-3, rtol=2e-3
    )
    timings = {}
    for name, operation in implementations.items():
        if name == "reference" and skip_reference_timing:
            continue
        timings[name] = measure_device(
            operation,
            warmup_ms=warmup_ms,
            measurement_time_ms=measurement_time_ms,
            samples=samples,
        )
    return timings


def _benchmark_shape(
    shape: tuple[int, ...],
    args: argparse.Namespace,
    environment: EnvironmentInfo,
    target: AcceleratorTarget,
) -> list[BenchmarkRecord[DeviceTimings]]:
    batch, channels, frames, height, width, outputs = shape
    policy = _convolution_policy(target)
    activation = normal_tensor(
        (batch, channels, frames, height, width),
        device=torch.device("cuda"),
        dtype=getattr(torch, args.dtype),
        seed=args.seed,
    )
    group_size = 64 if channels <= 128 else 256
    input_scale = torch.tensor(8 / 127)
    dense = normal_tensor(
        (outputs, channels, 3, 3, 3),
        device=torch.device("cpu"),
        dtype=getattr(torch, args.dtype),
        seed=args.seed + 1,
        scale=(27 * channels) ** -0.5,
    )
    weight = ConvRotInt8Tensor.from_hp(
        dense,
        group_size=group_size,
        act_per_tensor_scale=input_scale,
    ).to(device="cuda")
    input_scale = input_scale.to(device="cuda")
    norm_weight, norm_bias = (
        torch.ones(channels, device="cuda"),
        torch.zeros(channels, device="cuda"),
    )
    operands = (
        weight.qdata,
        weight.scale,
        None,
        group_size,
        input_scale,
        (1, 1, 1),
    )
    flags: ConvolutionFlags = {
        "symmetric_spatial_padding": True,
        "right_spatial_padding": False,
        "residual": None,
    }
    # Reconstruct the logical (inverse-rotated) filter once, not inside timing.
    # FP16 intentionally omits activation rotation/quantization and retains FP16
    # intermediate rounding, so it is a performance baseline, not an INT8 oracle.
    fp16_activation = activation.to(torch.float16)
    fp16_weight = weight.dequantize(output_dtype=torch.float16)
    channels_last_activation = fp16_activation.contiguous(memory_format=torch.channels_last_3d)
    channels_last_weight = fp16_weight.contiguous(memory_format=torch.channels_last_3d)
    fp16_norm_weight, fp16_norm_bias = norm_weight.half(), norm_bias.half()
    operations = {
        "conv3d": {
            "native": lambda: conv3d(activation, weight, padding="reflect"),
            "reference": lambda: reference.conv3d(activation, *operands, **flags),
            "fp16": lambda: _standard_conv3d(fp16_activation, fp16_weight),
            "fp16_channels_last": lambda: _standard_conv3d(
                channels_last_activation, channels_last_weight
            ),
        },
        "group_norm_silu_conv3d": {
            "native": lambda: group_norm_silu_conv3d(
                activation, norm_weight, norm_bias, 32, 1e-6, weight, padding="reflect"
            ),
            "reference": lambda: reference.group_norm_silu_conv3d(
                activation, norm_weight, norm_bias, 32, 1e-6, *operands, **flags
            ),
            "fp16": lambda: _standard_group_norm_silu_conv3d(
                fp16_activation, fp16_weight, fp16_norm_weight, fp16_norm_bias
            ),
            "fp16_channels_last": lambda: _standard_group_norm_silu_conv3d(
                channels_last_activation, channels_last_weight, fp16_norm_weight, fp16_norm_bias
            ),
        },
    }
    if args.group_norm_silu is not None:
        name = "group_norm_silu_conv3d" if args.group_norm_silu else "conv3d"
        operations = {name: operations[name]}
    records = []
    shape_record: dict[str, int] = dict(
        zip(("batch", "channels", "frames", "height", "width", "out_channels"), shape, strict=True)
    )
    configuration = {
        "dtype": args.dtype,
        "group_size": group_size,
        "seed": args.seed,
        "fp16_baseline_dtype": "float16",
        "vendor_benchmark": args.vendor_benchmark,
        "vendor_convolution_enabled": torch.backends.cudnn.enabled,
        "vendor_convolution_version": torch.backends.cudnn.version(),
        "vendor_convolution_deterministic": torch.backends.cudnn.deterministic,
        "miopen_suggest_nhwc": os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC")
        if target.is_amd_hip
        else None,
    }
    for name, implementations in operations.items():
        timings = _measure_implementations(
            implementations,
            warmup_ms=args.warmup_ms,
            measurement_time_ms=args.measurement_time_ms,
            samples=args.samples,
            skip_reference_timing=args.skip_reference_timing,
        )
        for provider, timing in timings.items():
            records.append(
                BenchmarkRecord(
                    benchmark="convrot-conv3d",
                    provider=provider,
                    shape=shape_record,
                    configuration={
                        **configuration,
                        "operation": name,
                        "phase": "operator_end_to_end",
                    },
                    timings=timing,
                    environment=environment,
                )
            )
            print(
                f"{shape} {name}/{provider}: cache-flushed {timing.cache_flushed.display()} ms; "
                f"graph {timing.graph.display()} ms",
                flush=True,
            )
    if not args.tune:
        return records
    execution_plan_for = partial(
        default_execution_plan,
        activation,
        weight.qdata,
        (1, 1, 1),
        policy=policy,
        target=target,
        group_norm=bool(args.group_norm_silu),
        symmetric_spatial_padding=True,
        right_spatial_padding=False,
    )
    production_plan = execution_plan_for()
    if args.group_norm_silu:
        prepared = shared._prepare_group_norm_silu_input(
            activation,
            norm_weight,
            norm_bias,
            32,
            1e-6,
            group_size,
            input_scale,
            schedule=production_plan.preparation,
            accelerator_backend=target.backend,
        )
    else:
        prepared = shared._prepare_input(
            activation,
            group_size,
            input_scale,
            schedule=production_plan.preparation,
            accelerator_backend=target.backend,
        )
    candidates = (
        ConvolutionSchedule(m, n, k, warps, stages)
        for m, n, k, warps in (
            (32, 64, 64, 4),
            (32, 64, 128, 4),
            (32, 64, 256, 4),
            (64, 64, 64, 4),
            (64, 64, 128, 4),
            (64, 128, 64, 4),
            (64, 128, 128, 4),
            (64, 128, 256, 4),
            (128, 128, 64, 4),
            (128, 128, 128, 8),
            (128, 256, 64, 8),
        )
        for stages in (1, 2)
    )
    expected = shared._conv3d_prepared(
        prepared,
        weight.qdata,
        weight.scale,
        None,
        weight.act_per_tensor_scale,
        (1, 1, 1),
        execution_plan=production_plan,
        **flags,
    )
    for schedule in candidates:
        candidate = execution_plan_for(convolution_schedule=schedule)

        def run(candidate: ConvolutionExecutionPlan = candidate) -> torch.Tensor:
            return shared._conv3d_prepared(
                prepared,
                weight.qdata,
                weight.scale,
                None,
                weight.act_per_tensor_scale,
                (1, 1, 1),
                execution_plan=candidate,
                **flags,
            )

        torch.testing.assert_close(run(), expected, atol=0, rtol=0)
        timing = measure_device(
            run,
            warmup_ms=args.warmup_ms,
            measurement_time_ms=args.measurement_time_ms,
            samples=args.samples,
        )
        records.append(
            BenchmarkRecord(
                benchmark="convrot-conv3d",
                provider="candidate",
                shape=shape_record,
                configuration={
                    **configuration,
                    "operation": "group_norm_silu_conv3d" if args.group_norm_silu else "conv3d",
                    "phase": "prepared_execution",
                    "schedule": schedule._asdict(),
                },
                timings=timing,
                environment=environment,
            )
        )
        print(f"{shape} {schedule}: graph {timing.graph.display()} ms", flush=True)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot INT8 convolution benchmarking requires a CUDA or ROCm GPU")
    torch.cuda.set_device(args.device)
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    _convolution_policy(target)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    print(
        f"GPU: {environment.gpu_name}; backend: {target.backend}; "
        f"architecture: {target.architecture}"
    )
    records = []
    with torch.inference_mode():
        for shape in args.shape or [
            (1, 128, 5, 64, 64, 128),
            (1, 256, 3, 32, 32, 256),
            (1, 512, 3, 16, 16, 512),
        ]:
            with torch.backends.cudnn.flags(benchmark=args.vendor_benchmark):
                records.extend(_benchmark_shape(shape, args, environment, target))
    write_records(records, output_target(args))


if __name__ == "__main__":
    main()
