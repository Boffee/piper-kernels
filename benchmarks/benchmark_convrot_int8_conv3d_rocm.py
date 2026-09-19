"""Compare native RDNA4 Conv3D with its portable reference; optionally sweep GEMM tiles.

Synthetic N,C,T,H,W,O shapes exercise the H3-style encoder channel sizes. These
are operator measurements, not checkpoint quality or end-to-end encoder results.
No GPU clock or power settings are changed. JSON lines go to stdout.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import torch
import triton
from lib.environment import capture_environment
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
    args = parser.parse_args(argv)
    if args.rep_ms <= 0:
        parser.error("--rep-ms must be positive")
    return args


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
    operations = (
        (
            "conv3d",
            lambda: conv3d(activation, weight, padding="reflect"),
            lambda: reference.conv3d(activation, *operands, **flags),
        ),
        (
            "group_norm_silu_conv3d",
            lambda: group_norm_silu_conv3d(
                activation, norm_weight, norm_bias, 32, 1e-6, weight, padding="reflect"
            ),
            lambda: reference.group_norm_silu_conv3d(
                activation, norm_weight, norm_bias, 32, 1e-6, *operands, **flags
            ),
        ),
    )
    for name, native, portable in operations:
        torch.testing.assert_close(native(), portable(), atol=4e-3, rtol=4e-3)
        timings = {}
        for implementation, operation in (("native", native), ("reference", portable)):
            timings[f"{implementation}_ms"] = do_bench(
                operation, warmup=25, rep=args.rep_ms, return_mode="median"
            )
            timings[f"{implementation}_graph_ms"] = do_bench_cudagraph(
                operation, rep=args.rep_ms, return_mode="median"
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
    print(
        json.dumps(
            {
                "environment": capture_environment(Path(__file__).resolve().parents[1]).as_dict(),
                "gpu": torch.cuda.get_device_name(),
                "target": str(target),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "dtype": args.dtype,
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
