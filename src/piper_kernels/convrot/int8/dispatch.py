"""Backend selection for INT8 ConvRot operators."""

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .._torch_compat import is_fake_mode_active
from ._policy import can_fuse_rotation_quantization
from .reference import reference_addmm_, reference_linear, validate_storage

try:
    from .triton import (
        triton_convrot_int8_addmm_ as _triton_addmm_,
    )
    from .triton import (
        triton_convrot_int8_linear as _triton_linear,
    )
    from .triton import (
        triton_convrot_int8_swiglu_linear as _triton_swiglu_linear,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_addmm_ = None
    _triton_linear = None
    _triton_swiglu_linear = None


def _accelerator_target(device: torch.device) -> AcceleratorTarget | None:
    """Resolve a concrete target, failing closed for synthetic fake devices."""
    try:
        return AcceleratorTarget.from_device(device)
    except Exception:
        if torch.compiler.is_compiling():
            return None
        if not is_fake_mode_active():
            raise
        return None


def _needs_fake_cuda_kernel(tensor: torch.Tensor) -> bool:
    """Return whether a CUDA fake has no concrete target for decomposed factories."""
    if torch.compiler.is_compiling():
        return False
    return (
        tensor.device.type == "cuda"
        and is_fake_mode_active()
        and _accelerator_target(tensor.device) is None
    )


def _can_use_triton(activation: torch.Tensor, qdata: torch.Tensor) -> bool:
    if (
        _triton_linear is None
        or activation.device.type != "cuda"
        or activation.device != qdata.device
        or activation.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return False
    target = _accelerator_target(activation.device)
    return target is not None and target.cuda_capability_at_least(7, 5)


def _can_use_triton_swiglu(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    group_size: int,
) -> bool:
    if (
        _triton_swiglu_linear is None
        or activation.device != qdata.device
        or activation.ndim == 0
        or activation.shape[-1] == 0
    ):
        return False
    rows = activation.numel() // activation.shape[-1]
    if not can_fuse_rotation_quantization(
        rows,
        int(qdata.shape[1]),
        group_size,
        activation.dtype,
        activation.device,
        sm120=True,
    ):
        return False
    target = _accelerator_target(activation.device)
    return target is not None and target.is_cuda_capability(12, 0)


def _reference_swiglu_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Materialize ``[up | gate]`` SwiGLU before the portable ConvRot linear."""
    in_features = qdata.shape[1]
    up, gate = activation.split(in_features, dim=-1)
    return reference_linear(
        up * torch.nn.functional.silu(gate),
        qdata,
        scale,
        group_size,
        bias,
    )


@torch.library.custom_op("piper_kernels::convrot_int8_linear", mutates_args=())
def _convrot_int8_linear_op(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Run the portable implementation of the semantic INT8 ConvRot linear."""
    return reference_linear(activation, qdata, scale, group_size, bias)


@_convrot_int8_linear_op.register_kernel("cuda")
def _convrot_int8_linear_cuda(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Select the optimized CUDA implementation when the target supports it."""
    if _can_use_triton(activation, qdata):
        assert _triton_linear is not None
        return _triton_linear(activation, qdata, scale, bias, group_size)
    return reference_linear(activation, qdata, scale, group_size, bias)


@_convrot_int8_linear_op.register_fake
def _convrot_int8_linear_fake(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    _scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
) -> torch.Tensor:
    return activation.new_empty((*activation.shape[:-1], qdata.shape[0]))


@torch.library.custom_op("piper_kernels::convrot_int8_swiglu_linear", mutates_args=())
def _convrot_int8_swiglu_linear_op(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Run portable ``[up | gate]`` SwiGLU plus an INT8 ConvRot linear."""
    return _reference_swiglu_linear(activation, qdata, scale, group_size, bias)


@_convrot_int8_swiglu_linear_op.register_kernel("cuda")
def _convrot_int8_swiglu_linear_cuda(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Select fused CUDA SwiGLU only in its measured eligibility region."""
    if _can_use_triton_swiglu(activation, qdata, group_size):
        assert _triton_swiglu_linear is not None
        return _triton_swiglu_linear(activation, qdata, scale, bias, group_size)
    return _reference_swiglu_linear(activation, qdata, scale, group_size, bias)


@_convrot_int8_swiglu_linear_op.register_fake
def _convrot_int8_swiglu_linear_fake(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    _scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
) -> torch.Tensor:
    return activation.new_empty((*activation.shape[:-1], qdata.shape[0]))


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
    validate_storage(qdata, scale, group_size, dtype)
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


def _can_use_triton_addmm(qdata: torch.Tensor, mat1: torch.Tensor) -> bool:
    if _triton_addmm_ is None or qdata.device.type != "cuda" or mat1.device != qdata.device:
        return False
    target = _accelerator_target(qdata.device)
    return target is not None and target.cuda_capability_at_least(7, 5)


@torch.library.custom_op(
    "piper_kernels::convrot_int8_addmm_",
    mutates_args=("qdata", "scale"),
)
def _convrot_int8_addmm_op(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
) -> None:
    """Run the portable implementation of the semantic INT8 ConvRot update."""
    reference_addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha)


@_convrot_int8_addmm_op.register_kernel("cuda")
def _convrot_int8_addmm_cuda(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
) -> None:
    """Select the optimized CUDA implementation when the target supports it."""
    if _can_use_triton_addmm(qdata, mat1):
        assert _triton_addmm_ is not None
        _triton_addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha)
        return
    reference_addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha)


@_convrot_int8_addmm_op.register_fake
def _convrot_int8_addmm_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _mat1: torch.Tensor,
    _mat2: torch.Tensor,
    _group_size: int,
    _beta: float,
    _alpha: float,
) -> None:
    return None


def _addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
) -> None:
    """Apply the logical ``beta * weight + alpha * mat1 @ mat2`` update in place."""
    _validate_addmm(qdata, scale, dtype, group_size, mat1, mat2)
    beta_float = _validate_scalar(beta, "beta")
    alpha_float = _validate_scalar(alpha, "alpha")
    if beta_float == 1 and alpha_float == 0:
        return
    _convrot_int8_addmm_op(
        qdata,
        scale,
        mat1,
        mat2,
        group_size,
        beta_float,
        alpha_float,
    )


def _validate_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None,
    expected_features: int,
) -> torch.Tensor | None:
    """Validate the shared inference contract and canonicalize the optional bias."""
    validate_storage(qdata, scale, group_size, dtype)
    if activation.ndim == 0 or activation.shape[-1] != expected_features:
        actual = 0 if activation.ndim == 0 else activation.shape[-1]
        raise ValueError(
            f"ConvRot linear input has {actual} features, expected {expected_features}"
        )
    if activation.device != qdata.device:
        raise ValueError(
            "ConvRot activation and weight must share a device, "
            f"got {activation.device}/{qdata.device}"
        )
    if activation.dtype is not dtype:
        raise ValueError(
            "ConvRot activation must match the weight's logical dtype, "
            f"got {activation.dtype}/{dtype}"
        )
    if activation.layout is not torch.strided:
        raise ValueError("ConvRot activation must use strided layout")

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
        if bias.device != activation.device:
            raise ValueError(
                "ConvRot activation, weight, and bias must share a device, "
                f"got {activation.device}/{qdata.device}/{bias.device}"
            )
        if bias.dtype is not dtype:
            raise ValueError(
                f"ConvRot bias must match the weight's logical dtype, got {bias.dtype}/{dtype}"
            )
        if bias.layout is not torch.strided:
            raise ValueError("ConvRot linear bias must use strided layout")

    if torch.is_grad_enabled() and (
        activation.requires_grad or scale.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "ConvRot INT8 linear does not support autograd; detach its inputs or use no_grad"
        )
    return None if bias is None else bias.contiguous()


def _run_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run a validated ordinary ConvRot linear through the selected backend."""
    # Keep the portable path decomposable under torch.compile. The registered
    # CUDA kernel rechecks eligibility so direct semantic-op calls remain safe.
    if _needs_fake_cuda_kernel(activation) or _can_use_triton(activation, qdata):
        return _convrot_int8_linear_op(activation, qdata, scale, bias, group_size)
    return reference_linear(activation, qdata, scale, group_size, bias)


def _linear(
    activation: torch.Tensor,
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
        activation,
        qdata,
        scale,
        dtype,
        group_size,
        bias,
        qdata.shape[1],
    )
    if activation.device.type == "meta":
        return activation.new_empty((*activation.shape[:-1], qdata.shape[0]))
    return _run_linear(activation, qdata, scale, group_size, bias)


def _convrot_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    group_size: int,
    bias: torch.Tensor | None,
    input_activation: str,
) -> torch.Tensor:
    """Apply an explicit input activation immediately before a ConvRot linear."""
    if input_activation != "swiglu":
        raise ValueError(f"ConvRot input_activation must be 'swiglu', got {input_activation!r}")

    in_features = qdata.shape[1]
    bias = _validate_linear(
        activation,
        qdata,
        scale,
        dtype,
        group_size,
        bias,
        2 * in_features,
    )

    output_shape = (*activation.shape[:-1], qdata.shape[0])
    if activation.device.type == "meta":
        return activation.new_empty(output_shape)
    if in_features == 0:
        result = activation.new_zeros(output_shape, dtype=torch.float32)
        if bias is not None:
            result += bias.to(torch.float32)
        return result.to(activation.dtype)
    # As above, unsupported SwiGLU remains visible as portable PyTorch ops to
    # compilers instead of being hidden behind the semantic custom op.
    if _needs_fake_cuda_kernel(activation) or _can_use_triton_swiglu(
        activation,
        qdata,
        group_size,
    ):
        return _convrot_int8_swiglu_linear_op(
            activation,
            qdata,
            scale,
            bias,
            group_size,
        )

    up, gate = activation.split(in_features, dim=-1)
    activated = up * torch.nn.functional.silu(gate)
    return _run_linear(activated, qdata, scale, group_size, bias)
