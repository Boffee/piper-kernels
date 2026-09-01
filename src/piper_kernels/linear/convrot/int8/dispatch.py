"""Validated storage-level entrypoints for ConvRot INT8."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._input_activations import input_activation_width
from piper_kernels.linear.convrot._update import (
    validate_real_scalar,
    validate_rounding_seed,
)

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


def addmm_(
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
    operation = "ConvRot INT8 addmm_"
    beta_float = validate_real_scalar(beta, "beta", operation=operation)
    alpha_float = validate_real_scalar(alpha, "alpha", operation=operation)
    validate_rounding_seed(rounding_seed, operation=operation)
    if beta_float == 1 and alpha_float == 0:
        return
    seed_argument = (
        rounding_seed
        if rounding_seed is None or rounding_seed < (1 << 63)
        else rounding_seed - (1 << 64)
    )
    if _supports_triton(qdata):
        assert triton_backend is not None
        triton_backend.addmm_(
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
        reference.addmm_(
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
    activation_fn: str | None,
) -> torch.Tensor:
    """Run a validated ConvRot linear through the selected backend."""
    if _supports_triton(input):
        assert triton_backend is not None
        return triton_backend.linear(
            input,
            qdata,
            scale,
            bias,
            group_size,
            activation_fn,
        )
    return reference.linear(
        input,
        qdata,
        scale,
        group_size,
        bias,
        activation_fn=activation_fn,
    )


def linear(
    input: torch.Tensor,  # noqa: A002
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None = None,
    *,
    activation_fn: str | None = None,
) -> torch.Tensor:
    """Apply raw ConvRot INT8 storage to a floating-point activation.

    ``qdata`` stores the two-dimensional INT8 weight in its rotated basis,
    and ``scale`` contains one float32 value per output channel. This is the
    internal storage-level ABI. Consumers should call
    :func:`torch.nn.functional.linear` with a ``ConvRotInt8Tensor`` weight.
    """
    in_features = qdata.shape[1]
    bias = _validate_linear(
        input,
        qdata,
        scale,
        dtype,
        group_size,
        bias,
        in_features * input_activation_width(activation_fn),
    )
    if input.device.type == "meta":
        return input.new_empty((*input.shape[:-1], qdata.shape[0]))
    return _run_linear(input, qdata, scale, group_size, bias, activation_fn)
