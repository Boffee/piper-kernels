"""Validated storage-level entrypoints for ConvRot INT8."""

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from . import reference

try:
    from . import triton as triton_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton_backend = None


def _supports_triton(input: torch.Tensor) -> bool:  # noqa: A002
    return triton_backend is not None and AcceleratorTarget.from_device(
        input.device
    ).cuda_capability_at_least(7, 5)


def _validate_scalar(value: int | float | complex, name: str) -> float:
    if isinstance(value, complex):
        raise TypeError(f"ConvRot INT8 addmm_ {name} must be a real number, got {value}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"ConvRot INT8 addmm_ {name} must be finite, got {value}")
    return converted


def _validate_addmm(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    reference.validate_storage(qdata, scale, group_size, dtype)
    if qdata.device.type == "meta":
        raise ValueError("ConvRot INT8 addmm_ cannot update a meta tensor without values")
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(
            "ConvRot INT8 addmm_ matrices must be 2-D, "
            f"got shapes {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    expected_mat1 = (qdata.shape[0], mat2.shape[0])
    expected_mat2 = (mat1.shape[1], qdata.shape[1])
    if tuple(mat1.shape) != expected_mat1 or tuple(mat2.shape) != expected_mat2:
        raise ValueError(
            "ConvRot INT8 addmm_ shape mismatch: expected "
            f"mat1 {expected_mat1} and mat2 {expected_mat2} for weight {tuple(qdata.shape)}, "
            f"got {tuple(mat1.shape)} and {tuple(mat2.shape)}"
        )
    if mat1.device != qdata.device or mat2.device != qdata.device:
        raise ValueError(
            "ConvRot INT8 addmm_ weight and matrices must share a device, "
            f"got {qdata.device}/{mat1.device}/{mat2.device}"
        )
    if mat1.dtype is not dtype or mat2.dtype is not dtype:
        raise ValueError(
            "ConvRot INT8 addmm_ matrices must match the weight's logical dtype, "
            f"got {dtype}/{mat1.dtype}/{mat2.dtype}"
        )
    if mat1.layout is not torch.strided or mat2.layout is not torch.strided:
        raise ValueError("ConvRot INT8 addmm_ matrices must use strided layout")
    if torch.is_grad_enabled() and (
        scale.requires_grad or mat1.requires_grad or mat2.requires_grad
    ):
        raise RuntimeError(
            "ConvRot INT8 addmm_ does not support autograd; detach its inputs or use no_grad"
        )


def convrot_int8_addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
    rounding_seed: int | None = None,
) -> None:
    """Apply the logical ``beta * weight + alpha * mat1 @ mat2`` update in place."""
    _validate_addmm(qdata, scale, dtype, group_size, mat1, mat2)
    beta_float = _validate_scalar(beta, "beta")
    alpha_float = _validate_scalar(alpha, "alpha")
    if rounding_seed is not None:
        if isinstance(rounding_seed, bool) or not isinstance(rounding_seed, int):
            raise TypeError("ConvRot INT8 addmm_ rounding_seed must be an unsigned 64-bit integer")
        if not 0 <= rounding_seed < (1 << 64):
            raise ValueError("ConvRot INT8 addmm_ rounding_seed must be an unsigned 64-bit integer")
    if beta_float == 1 and alpha_float == 0:
        return
    seed_argument = (
        rounding_seed
        if rounding_seed is None or rounding_seed < (1 << 63)
        else rounding_seed - (1 << 64)
    )
    if _supports_triton(qdata):
        assert triton_backend is not None
        triton_backend.convrot_int8_addmm_(
            qdata,
            scale,
            mat1,
            mat2,
            group_size,
            beta_float,
            alpha_float,
            seed_argument,
        )
    else:
        reference.convrot_int8_addmm_(
            qdata,
            scale,
            mat1,
            mat2,
            group_size,
            beta_float,
            alpha_float,
            seed_argument,
        )


def _validate_linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None,
    expected_features: int,
) -> torch.Tensor | None:
    """Validate one linear and canonicalize its optional bias."""
    reference.validate_storage(qdata, scale, group_size, dtype)
    if input.ndim == 0 or input.shape[-1] != expected_features:
        actual = 0 if input.ndim == 0 else input.shape[-1]
        raise ValueError(
            f"ConvRot linear input has {actual} features, expected {expected_features}"
        )
    if input.device != qdata.device:
        raise ValueError(
            f"ConvRot input and weight must share a device, got {input.device}/{qdata.device}"
        )
    if input.dtype is not dtype:
        raise ValueError(
            f"ConvRot input must match the weight's logical dtype, got {input.dtype}/{dtype}"
        )
    if input.layout is not torch.strided:
        raise ValueError("ConvRot input must use strided layout")

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError(
                f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}"
            )
        expected_bias_shape = (qdata.shape[0],)
        if tuple(bias.shape) != expected_bias_shape:
            raise ValueError(
                f"ConvRot linear bias must have shape {expected_bias_shape}, "
                f"got {tuple(bias.shape)}"
            )
        if bias.device != input.device:
            raise ValueError(
                "ConvRot input, weight, and bias must share a device, "
                f"got {input.device}/{qdata.device}/{bias.device}"
            )
        if bias.dtype is not dtype:
            raise ValueError(
                f"ConvRot bias must match the weight's logical dtype, got {bias.dtype}/{dtype}"
            )
        if bias.layout is not torch.strided:
            raise ValueError("ConvRot linear bias must use strided layout")

    if torch.is_grad_enabled() and (
        input.requires_grad or scale.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "ConvRot INT8 linear does not support autograd; detach its inputs or use no_grad"
        )
    return None if bias is None else bias.contiguous()


def _run_linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run a validated ordinary ConvRot linear through the selected backend."""
    if _supports_triton(input):
        assert triton_backend is not None
        return triton_backend.convrot_int8_linear(input, qdata, scale, bias, group_size)
    return reference.convrot_int8_linear(input, qdata, scale, group_size, bias)


def convrot_int8_linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply raw ConvRot INT8 storage to a floating-point activation.

    ``qdata`` stores the two-dimensional INT8 weight in its rotated basis,
    and ``scale`` contains one float32 value per output channel. This is the
    internal storage-level ABI. Consumers should call
    :func:`torch.nn.functional.linear` with a ``ConvRotInt8Tensor`` weight.
    """
    bias = _validate_linear(
        input,
        qdata,
        scale,
        dtype,
        group_size,
        bias,
        qdata.shape[1],
    )
    if input.device.type == "meta":
        return input.new_empty((*input.shape[:-1], qdata.shape[0]))
    return _run_linear(input, qdata, scale, group_size, bias)


def convrot_int8_swiglu_linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply packed ``[up | gate]`` SwiGLU immediately before a ConvRot linear."""
    in_features = qdata.shape[1]
    bias = _validate_linear(
        input,
        qdata,
        scale,
        dtype,
        group_size,
        bias,
        2 * in_features,
    )

    output_shape = (*input.shape[:-1], qdata.shape[0])
    if input.device.type == "meta":
        return input.new_empty(output_shape)
    if _supports_triton(input):
        assert triton_backend is not None
        return triton_backend.convrot_int8_swiglu_linear(
            input,
            qdata,
            scale,
            bias,
            group_size,
        )

    up, gate = input.chunk(2, dim=-1)
    activated = up * torch.nn.functional.silu(gate)
    return _run_linear(activated, qdata, scale, group_size, bias)
