"""Portable INT8 ConvRot reference implementation."""

import torch

from piper_kernels._stochastic_quantization import _stochastic_round_to_int

from .._rotation import rotate_groups, validate_group_size

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def quantize_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and quantize a dense two-dimensional weight per output row."""
    validate_group_size(group_size)
    if weight.ndim != 2:
        raise ValueError(
            f"ConvRot INT8 high-precision weight must be 2-D, got shape {tuple(weight.shape)}"
        )
    if weight.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            "ConvRot INT8 high-precision weight must use float16, bfloat16, or float32, "
            f"got {weight.dtype}"
        )
    if weight.device.type == "meta":
        raise ValueError("ConvRot INT8 cannot quantize a meta tensor without values")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"ConvRot in_features {weight.shape[1]} is not divisible by group size {group_size}"
        )
    return dynamic_quantize_rows(rotate_groups(weight, group_size))


def validate_storage(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    """Validate INT8 ConvRot storage and its logical floating-point dtype."""
    validate_group_size(group_size)
    if qdata.dtype is not torch.int8 or qdata.ndim != 2:
        raise ValueError(
            f"ConvRot INT8 qdata must be a 2-D int8 tensor, got {qdata.dtype} {qdata.shape}"
        )
    if qdata.shape[1] % group_size:
        raise ValueError(
            f"ConvRot in_features {qdata.shape[1]} is not divisible by group size {group_size}"
        )
    expected_scale_shape = (qdata.shape[0], 1)
    if scale.dtype is not torch.float32 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"ConvRot INT8 scale must be float32 with shape {expected_scale_shape}, "
            f"got {scale.dtype} {tuple(scale.shape)} for qdata {tuple(qdata.shape)}"
        )
    if scale.device != qdata.device:
        raise ValueError(
            f"ConvRot INT8 qdata and scale must share a device, got {qdata.device}/{scale.device}"
        )
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            "ConvRot INT8 qdata and scale must be contiguous; "
            "use from_quantized to canonicalize storage"
        )
    if dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"ConvRot logical dtype must be float16, bfloat16, or float32, got {dtype}"
        )


def dynamic_quantize_rows(
    value: torch.Tensor,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize each row to signed INT8 with a float32 scale."""
    if value.ndim > 0 and value.shape[-1] == 0:
        scale = torch.full(
            (*value.shape[:-1], 1),
            1e-30,
            dtype=torch.float32,
            device=value.device,
        )
        return value.to(torch.int8), scale
    scale = (value.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    logical_scale = scale.to(value.dtype)
    if value.dtype is torch.float16:
        scale_underflowed = logical_scale == 0
        safe_logical_scale = torch.where(
            scale_underflowed,
            torch.ones_like(logical_scale),
            logical_scale,
        )
        scaled = value / safe_logical_scale
        scaled = torch.where(scale_underflowed, value.float() / scale, scaled.float())
    else:
        # The minimum scale remains representable in bfloat16 and float32.
        scaled = value / logical_scale
    qdata = scaled.round().clamp(-128, 127).to(torch.int8)
    if rounding_seed is not None:
        stochastic_scaled = value.to(torch.float32) / scale
        qdata = _stochastic_round_to_int(
            stochastic_scaled,
            seed=rounding_seed,
            quant_min=-128,
            quant_max=127,
            deterministic=qdata,
        ).to(torch.int8)
    return qdata, scale


def reference_addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Add a matrix product to a logical ConvRot weight and requantize it in place."""
    if beta == 0:
        rotated_weight = torch.zeros(qdata.shape, device=qdata.device, dtype=mat1.dtype)
    else:
        rotated_weight = qdata.to(mat1.dtype) * scale.to(mat1.dtype)
    rotated_mat2 = rotate_groups(mat2, group_size)
    merged = torch.addmm(rotated_weight, mat1, rotated_mat2, beta=beta, alpha=alpha)
    merged_qdata, merged_scale = dynamic_quantize_rows(
        merged,
        rounding_seed=rounding_seed,
    )
    qdata.copy_(merged_qdata)
    scale.copy_(merged_scale)


def _empty_inner_linear(
    activation: torch.Tensor,
    out_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Return the dense-linear result when the reduction dimension is empty."""
    result = activation.new_zeros((*activation.shape[:-1], out_features))
    return result if bias is None else result + bias


def reference_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the portable PyTorch ConvRot W8A8 linear implementation."""
    original_shape = activation.shape
    if original_shape[-1] == 0:
        return _empty_inner_linear(activation, qdata.shape[0], bias)
    activation_2d = activation.reshape(-1, original_shape[-1])
    rotated = rotate_groups(activation_2d, group_size)
    activation_qdata, activation_scale = dynamic_quantize_rows(rotated)
    if activation.device.type == "cpu":
        accumulated = activation_qdata.to(torch.int32) @ qdata.T.to(torch.int32)
    else:
        # Float32 represents each INT8 product exactly. Only very long reductions
        # can round the integer sum, which is preferable to rejecting the shape.
        accumulated = activation_qdata.float() @ qdata.T.float()
    del rotated, activation_qdata

    # Reuse the FP32 accumulator for the epilogue. Out-of-place broadcasts retain
    # multiple full [rows, out_features] temporaries, which is prohibitive for
    # large reference workloads even though the final BF16 output itself fits.
    result = accumulated.to(torch.float32)
    result.mul_(activation_scale.to(torch.float32))
    result.mul_(scale.reshape(1, -1).to(torch.float32))
    if bias is not None:
        result.add_(bias.to(torch.float32))
    return result.to(activation.dtype).reshape(*original_shape[:-1], qdata.shape[0])


def reference_swiglu_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize raw ``[up | gate]`` SwiGLU before a portable ConvRot linear."""
    in_features = qdata.shape[1]
    up, gate = torch.split(activation, [in_features, in_features], dim=-1)
    return reference_linear(
        up * torch.nn.functional.silu(gate),
        qdata,
        scale,
        group_size,
        bias,
    )
