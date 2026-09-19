"""Compare RDNA4 INT8 Conv3D with its reference and standard ROCm FP16 Conv3D.

Synthetic N,C,T,H,W,O shapes exercise the H3-style encoder channel sizes. These
are operator measurements, not checkpoint quality or end-to-end encoder results.
No GPU clock or power settings are changed. JSON lines go to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import torch
import triton
from lib.environment import capture_environment
from torch.nn import functional
from triton.testing import do_bench, do_bench_cudagraph

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import conv3d, group_norm_silu_conv3d, reference
from piper_kernels.conv3d.convrot.int8 import triton as shared
from piper_kernels.conv3d.convrot.int8._amd import policy
from piper_kernels.conv3d.convrot.int8._interfaces import ConvolutionPolicy
from piper_kernels.conv3d.convrot.int8._plan import ConvolutionPlan
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
    parser.add_argument("--shape", type=_shape, action="append", help="N,C,T,H,W,O; repeatable")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--tune", action="store_true", help="sweep prepared convolution tiles")
    parser.add_argument(
        "--miopen-benchmark", action="store_true", help="enable vendor convolution algorithm search"
    )
    parser.add_argument(
        "--skip-reference-timing",
        action="store_true",
        help="still check the INT8 reference, but time only native and FP16 implementations",
    )
    args = parser.parse_args(argv)
    if args.rep_ms <= 0:
        parser.error("--rep-ms must be positive")
    return args


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
    rep_ms: int,
    skip_reference_timing: bool,
) -> dict[str, float]:
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
        timings[f"{name}_ms"] = cast(
            float, do_bench(operation, warmup=25, rep=rep_ms, return_mode="median")
        )
        timings[f"{name}_graph_ms"] = cast(
            float, do_bench_cudagraph(operation, rep=rep_ms, return_mode="median")
        )
    return timings


def _benchmark_shape(shape: tuple[int, ...], args: argparse.Namespace) -> None:
    batch, channels, frames, height, width, outputs = shape
    torch.manual_seed(871)
    activation = torch.randn(
        batch, channels, frames, height, width, device="cuda", dtype=getattr(torch, args.dtype)
    )
    group_size = 64 if channels <= 128 else 256
    input_scale = torch.tensor(0.02, device="cuda")
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(-16, 16, (outputs, 3, 3, 3, channels), device="cuda", dtype=torch.int8),
        torch.full((outputs,), 0.001, device="cuda"),
        group_size=group_size,
        logical_dtype=torch.float16,
        act_per_tensor_scale=input_scale,
    )
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
    for name, implementations in operations.items():
        timings = _measure_implementations(
            implementations, rep_ms=args.rep_ms, skip_reference_timing=args.skip_reference_timing
        )
        print(json.dumps({"shape_ncthwo": shape, "operation": name, **timings}), flush=True)
    if not args.tune:
        return
    prepared = shared._prepare_input(
        activation,
        group_size,
        weight.act_per_tensor_scale,
        policy=policy,
        accelerator_backend="hip",
    )
    candidates = (
        ConvolutionPlan(m, n, k, warps, stages)
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
        policy=policy,
        **flags,
    )
    for plan in candidates:
        candidate = cast(
            ConvolutionPolicy,
            SimpleNamespace(
                convolution_plan=lambda *_, plan=plan: plan,
                preparation_plan=policy.preparation_plan,
                use_weight_descriptor=policy.use_weight_descriptor,
            ),
        )

        def run(candidate: ConvolutionPolicy = candidate) -> torch.Tensor:
            return shared._conv3d_prepared(
                prepared,
                weight.qdata,
                weight.scale,
                None,
                weight.act_per_tensor_scale,
                (1, 1, 1),
                policy=candidate,
                **flags,
            )

        torch.testing.assert_close(run(), expected, atol=0, rtol=0)
        elapsed = cast(float, do_bench_cudagraph(run, rep=args.rep_ms, return_mode="median"))
        print(
            json.dumps(
                {
                    "shape_ncthwo": shape,
                    "phase": "prepared",
                    "plan": plan._asdict(),
                    "graph_ms": elapsed,
                }
            ),
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("requires ROCm PyTorch and RDNA4")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not policy.supports_target(target):
        raise SystemExit(f"unsupported target: {target}")
    torch.backends.cudnn.benchmark = args.miopen_benchmark
    print(
        json.dumps(
            {
                "environment": capture_environment(Path(__file__).resolve().parents[1]).as_dict(),
                "gpu": torch.cuda.get_device_name(),
                "target": str(target),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "dtype": args.dtype,
                "rep_ms": args.rep_ms,
                "skip_reference_timing": args.skip_reference_timing,
                "fp16_baseline_dtype": "float16",
                "vendor_convolution_enabled": torch.backends.cudnn.enabled,
                "vendor_convolution_version": torch.backends.cudnn.version(),
                "miopen_benchmark": torch.backends.cudnn.benchmark,
                "miopen_deterministic": torch.backends.cudnn.deterministic,
                "miopen_suggest_nhwc": os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC", "0"),
            }
        ),
        flush=True,
    )
    with torch.inference_mode():
        for shape in args.shape or [
            (1, 128, 5, 64, 64, 128),
            (1, 256, 3, 32, 32, 256),
            (1, 512, 3, 16, 16, 512),
        ]:
            _benchmark_shape(shape, args)


if __name__ == "__main__":
    main()
