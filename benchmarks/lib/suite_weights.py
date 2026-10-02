"""Shared linear, FFN, and convolution workloads across accelerator backends."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels._triton.targets import AcceleratorTarget

from .cases import Conv3DCase, FFNCase, LinearCase
from .quality import QualityMetrics, measure_quality
from .suite_types import Implementation, Operation, QualityCheck, normal_tensor, sample_indices

# Optional format and kernel imports stay inside the selected implementation.
# ruff: noqa: PLC0415

type WeightFormat = Literal["torch", "convrot_int8", "nvfp4", "convrot_nvfp4"]
_FORMATS: tuple[WeightFormat, ...] = ("torch", "convrot_int8", "nvfp4", "convrot_nvfp4")
_CPU = torch.device("cpu")


def _group_size(width: int) -> int:
    return next(group for group in (256, 64, 16) if width % group == 0)


def _support(
    format_name: WeightFormat, device: torch.device, *, convolution: bool = False
) -> str | None:
    if format_name == "torch":
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return "native implementation requires a supported CUDA or ROCm accelerator"
    if importlib.util.find_spec("triton") is None:
        return "native implementation requires Triton"
    target = AcceleratorTarget.from_device(device)
    if format_name == "convrot_int8":
        if convolution:
            from piper_kernels.conv3d.convrot.int8._amd import policy as amd
            from piper_kernels.conv3d.convrot.int8._nvidia import policy as nvidia
        else:
            from piper_kernels.linear.convrot.int8._amd import policy as amd
            from piper_kernels.linear.convrot.int8._nvidia import policy as nvidia

        if not (amd.supports_target(target) or nvidia.supports_target(target)):
            return "no native ConvRot INT8 backend for this accelerator"
    elif not target.is_cuda_capability(12, 0):
        return "native NVFP4 execution requires exact NVIDIA SM120"
    return (
        "quantized weight construction requires TorchAO"
        if importlib.util.find_spec("torchao") is None
        else None
    )


def _dense_weight(rows: int, columns: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    return normal_tensor((rows, columns), dtype=dtype, device=_CPU, seed=seed, scale=columns**-0.5)


def _pack_weight(
    weight: torch.Tensor, format_name: WeightFormat, device: torch.device
) -> torch.Tensor:
    if format_name == "torch":
        return weight.to(device)
    if format_name == "convrot_int8":
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        return ConvRotInt8Tensor.from_hp(weight, group_size=_group_size(weight.shape[1])).to(device)
    from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

    from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
    from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

    activation = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=True,
    )

    if format_name == "convrot_nvfp4":
        return ConvRotNVFP4Tensor.from_hp(
            weight,
            group_size=_group_size(weight.shape[1]),
            compute_per_tensor_scale=True,
            is_swizzled_scales=True,
            act_quant_kwargs=activation,
        ).to(device)
    return PiperNVFP4Tensor.from_hp(
        weight,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant_kwargs=activation,
    ).to(device)


def _quantized_comparison(actual: torch.Tensor, expected: torch.Tensor) -> QualityMetrics:
    """Check implementation agreement separately from quantization's FP error."""
    metrics = measure_quality(actual, expected)
    if (
        metrics.actual_nonfinite_count
        or metrics.reference_nonfinite_count
        or metrics.relative_l2_error > 0.02
    ):
        raise ValueError("native output differs from the portable quantized reference")
    return metrics


def _linear(case: LinearCase, device: torch.device, format_name: WeightFormat) -> Operation:
    dtype = getattr(torch, case.dtype)
    source = normal_tensor((case.rows, case.in_features), dtype=dtype, device=_CPU, seed=case.seed)
    dense = _dense_weight(case.out_features, case.in_features, dtype, case.seed + 1)
    bias = (
        normal_tensor((case.out_features,), dtype=dtype, device=_CPU, seed=case.seed + 2, scale=0.1)
        if case.bias
        else None
    )
    selected = sample_indices(case.rows, device=_CPU)
    expected = F.linear(
        source[selected].double(), dense.double(), None if bias is None else bias.double()
    )
    weight = _pack_weight(dense, format_name, device)
    quantized_expected = None
    if format_name == "convrot_int8":
        from piper_kernels.linear.convrot.int8 import reference
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        packed = cast(ConvRotInt8Tensor, weight)
        quantized_expected = reference.linear(
            source[selected], packed.qdata.cpu(), packed.scale.cpu(), packed.group_size, bias
        )
    activation = source.to(device)
    device_bias = None if bias is None else bias.to(device)

    def check(actual: torch.Tensor) -> QualityCheck:
        sampled = actual[selected.to(device)].cpu().double()
        comparisons = {}
        if quantized_expected is not None:
            comparisons["portable_quantized_linear"] = _quantized_comparison(
                sampled, quantized_expected.double()
            )
        return QualityCheck(
            measure_quality(sampled, expected),
            "fp64_linear_from_original_weights",
            sampled.numel(),
            actual.numel(),
            0.25 if "nvfp4" in format_name else 0.05,
            comparisons,
        )

    return Operation(
        lambda: F.linear(activation, weight, device_bias),
        check,
        {
            "format": format_name,
            "activation_scaling": None if format_name == "torch" else "dynamic",
            "group_size": _group_size(case.in_features)
            if format_name.startswith("convrot")
            else None,
            "bias": case.bias,
        },
    )


def _ffn_reference(
    source: torch.Tensor, weights: list[torch.Tensor], activation: str
) -> torch.Tensor:
    projected = F.linear(source.double(), weights[0].double())
    if activation == "gelu":
        activated = F.gelu(projected, approximate="tanh")
    else:
        activated = projected * F.silu(F.linear(source.double(), weights[1].double()))
    return F.linear(activated, weights[-1].double())


def _ffn_launch(
    source: torch.Tensor,
    weights: list[torch.Tensor],
    case: FFNCase,
    format_name: WeightFormat,
) -> tuple[Callable[[], torch.Tensor], int | None]:
    if format_name == "torch":

        def run() -> torch.Tensor:
            up = F.linear(source, weights[0])
            activated = (
                F.gelu(up, approximate="tanh")
                if case.activation == "gelu"
                else up * F.silu(F.linear(source, weights[1]))
            )
            return F.linear(activated, weights[-1])

        return run, None
    if format_name == "convrot_int8":
        from piper_kernels.fusions.convrot_int8_ffn import _core
        from piper_kernels.fusions.convrot_int8_gelu_ffn import triton as gelu
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        packed = [cast(ConvRotInt8Tensor, weight) for weight in weights]
        linears = tuple(
            _core.LinearOperands(weight.qdata, weight.scale, None, weight.group_size, None)
            for weight in packed
        )
        chunk_rows = (
            gelu._default_chunk_rows(source, packed[0].qdata, packed[-1].qdata, gated_updates=False)
            if case.activation == "gelu"
            else _core.DEFAULT_CHUNK_ROWS
        )
        return partial(
            _core.run_chunked_ffn,
            source,
            linears[:-1],
            linears[-1],
            "gelu_tanh" if case.activation == "gelu" else "swiglu",
            chunk_rows,
        ), chunk_rows

    from piper_kernels.fusions.convrot_nvfp4_gelu_ffn import triton as rotated_gelu
    from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import triton as rotated_swiglu
    from piper_kernels.fusions.nvfp4_ffn import _core as nv_core
    from piper_kernels.fusions.nvfp4_gelu_ffn import triton as nv_gelu
    from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
    from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

    nv_weights = [cast(PiperNVFP4Tensor, weight) for weight in weights]
    groups = [
        cast(ConvRotNVFP4Tensor, weight).group_size if format_name == "convrot_nvfp4" else None
        for weight in nv_weights
    ]
    nv_linears = tuple(
        nv_core.LinearOperands(
            weight.qdata, weight.scale, weight.per_tensor_scale, None, None, True, weight.high_first
        )
        for weight in nv_weights
    )
    if case.activation == "gelu":
        preparation = rotated_gelu._preparation_backends(groups[0], groups[-1], False, False)
        chunk_rows = nv_gelu._default_chunk_rows(
            source, nv_weights[0].qdata, nv_weights[-1].qdata, gated_updates=False
        )
    else:
        preparation = rotated_swiglu._preparation_backends(
            groups[1], groups[0], groups[-1], False, False, False
        )
        chunk_rows = nv_core.DEFAULT_CHUNK_ROWS
    return partial(
        nv_core.run_chunked_ffn, source, nv_linears[:-1], nv_linears[-1], chunk_rows, *preparation
    ), chunk_rows


def _ffn(case: FFNCase, device: torch.device, format_name: WeightFormat) -> Operation:
    dtype = getattr(torch, case.dtype)
    source = normal_tensor((case.rows, case.width), dtype=dtype, device=_CPU, seed=case.seed)
    shapes = [(case.intermediate, case.width)] * (1 if case.activation == "gelu" else 2)
    shapes.append((case.width, case.intermediate))
    dense = [
        _dense_weight(rows, columns, dtype, case.seed + index + 1)
        for index, (rows, columns) in enumerate(shapes)
    ]
    selected = sample_indices(case.rows, device=_CPU)
    expected = _ffn_reference(source[selected], dense, case.activation)
    weights = [_pack_weight(weight, format_name, device) for weight in dense]
    run, chunk_rows = _ffn_launch(source.to(device), weights, case, format_name)

    def check(actual: torch.Tensor) -> QualityCheck:
        sampled = actual[selected.to(device)].cpu().double()
        return QualityCheck(
            measure_quality(sampled, expected),
            "fp64_ffn_from_original_weights",
            sampled.numel(),
            actual.numel(),
            0.35 if "nvfp4" in format_name else 0.1,
        )

    return Operation(
        run,
        check,
        {
            "format": format_name,
            "activation": case.activation,
            "activation_scaling": None if format_name == "torch" else "dynamic",
            "source_group_size": _group_size(case.width)
            if format_name.startswith("convrot")
            else None,
            "down_group_size": _group_size(case.intermediate)
            if format_name.startswith("convrot")
            else None,
            "chunk_rows": chunk_rows,
            "bias": False,
        },
    )


def _normalize_frames(source: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = source.shape
    framewise = source.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    normalized = F.group_norm(framewise, 32, eps=1e-6)
    return F.silu(normalized).reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _convolution_patches(source: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather only sampled neighborhoods, with causal time and reflected space."""
    _, _, frames, height, width = source.shape
    batch = positions // (frames * height * width)
    frame = positions // (height * width) % frames
    row, column = positions // width % height, positions % width
    patches = []
    for dt in range(3):
        time = frame + dt - 2
        for dh in range(-1, 2):
            y = (row + dh).abs()
            y = torch.where(y >= height, 2 * height - 2 - y, y)
            for dw in range(-1, 2):
                x = (column + dw).abs()
                x = torch.where(x >= width, 2 * width - 2 - x, x)
                values = source[batch, :, time.clamp_min(0), y, x]
                patches.append(values * (time >= 0)[:, None])
    return torch.stack(patches, dim=1)


def _conv3d(case: Conv3DCase, device: torch.device, format_name: WeightFormat) -> Operation:
    dtype = getattr(torch, case.dtype)
    shape = (case.batch, case.channels, case.frames, case.height, case.width)
    source = normal_tensor(shape, dtype=dtype, device=_CPU, seed=case.seed)
    dense = normal_tensor(
        (case.out_channels, case.channels, 3, 3, 3),
        dtype=dtype,
        device=_CPU,
        seed=case.seed + 1,
        scale=(27 * case.channels) ** -0.5,
    )
    positions = sample_indices(case.batch * case.frames * case.height * case.width, device=_CPU)
    normalized = _normalize_frames(source.float()) if case.group_norm_silu else source.float()
    patches = _convolution_patches(normalized, positions)
    expected = patches.flatten(1).double() @ dense.permute(0, 2, 3, 4, 1).flatten(1).double().T
    activation = source.to(device)
    quantized_expected = None
    if format_name == "torch":
        weight = dense.to(device)

        def run() -> torch.Tensor:
            value = _normalize_frames(activation) if case.group_norm_silu else activation
            value = F.pad(value, (1, 1, 1, 1, 0, 0), mode="reflect")
            return F.conv3d(F.pad(value, (0, 0, 0, 0, 2, 0)), weight)
    else:
        from piper_kernels.conv3d.convrot.int8 import conv3d, group_norm_silu_conv3d
        from piper_kernels.weights.convrot._rotation import rotate_groups
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        group = _group_size(case.channels)
        input_scale = torch.tensor(8 / 127, dtype=torch.float32)
        packed = ConvRotInt8Tensor.from_hp(
            dense, group_size=group, act_per_tensor_scale=input_scale
        )
        quantized_patches = (rotate_groups(patches, group) / input_scale).round().clamp(-128, 127)
        quantized_expected = (
            quantized_patches.flatten(1).double() @ packed.qdata.flatten(1).double().T
        )
        quantized_expected *= input_scale.double() * packed.scale.flatten().double()
        weight = packed.to(device)
        norm_weight, norm_bias = (
            torch.ones(case.channels, device=device),
            torch.zeros(case.channels, device=device),
        )

        def run() -> torch.Tensor:
            if case.group_norm_silu:
                return group_norm_silu_conv3d(
                    activation, norm_weight, norm_bias, 32, 1e-6, weight, padding="reflect"
                )
            return conv3d(activation, weight, padding="reflect")

    def check(actual: torch.Tensor) -> QualityCheck:
        b = positions // (case.frames * case.height * case.width)
        t = positions // (case.height * case.width) % case.frames
        h, w = positions // case.width % case.height, positions % case.width
        sampled = actual[b.to(device), :, t.to(device), h.to(device), w.to(device)].cpu().double()
        comparisons = (
            {}
            if quantized_expected is None
            else {
                "portable_quantized_convolution": _quantized_comparison(sampled, quantized_expected)
            }
        )
        return QualityCheck(
            measure_quality(sampled, expected),
            "fp64_causal_reflect_convolution_from_original_weights",
            sampled.numel(),
            actual.numel(),
            0.05,
            comparisons,
        )

    return Operation(
        run,
        check,
        {
            "format": format_name,
            "padding": "causal_temporal_reflect_spatial",
            "group_norm_silu": case.group_norm_silu,
            "group_norm_groups": 32 if case.group_norm_silu else None,
            "activation_scale": None if format_name == "torch" else 8 / 127,
            "activation_scale_source": None if format_name == "torch" else "synthetic_fixed",
            "group_size": _group_size(case.channels) if format_name == "convrot_int8" else None,
        },
    )


def implementations(
    case: LinearCase | FFNCase | Conv3DCase, device: torch.device
) -> list[Implementation]:
    """Enumerate stable implementations without constructing tensors or compiling."""
    if isinstance(case, Conv3DCase):
        return [
            Implementation(
                name, partial(_conv3d, case, device, name), _support(name, device, convolution=True)
            )
            for name in ("torch", "convrot_int8")
        ]
    if isinstance(case, FFNCase):
        return [
            Implementation(name, partial(_ffn, case, device, name), _support(name, device))
            for name in _FORMATS
        ]
    return [
        Implementation(name, partial(_linear, case, device, name), _support(name, device))
        for name in _FORMATS
    ]
