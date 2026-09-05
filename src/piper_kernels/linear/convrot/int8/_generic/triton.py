"""Conservative shared Triton launchers without a GPU-model allowlist."""

# pyright: reportCallIssue=false

import torch
import triton

from piper_kernels._triton.stochastic_quantization import seed_argument
from piper_kernels.linear.convrot.triton import logical_dtype_code, rotate_input

from .._kernels.triton import _requantize_update_rows_kernel, quantize_rows_kernel


def supports_device(device: torch.device) -> bool:
    """Use the installed driver, not a tuned GEMM policy, to find a Triton device."""
    if device.type in ("cpu", "meta"):
        return False
    try:
        driver = triton.runtime.driver.active
        active_device = driver.get_active_torch_device()
        if getattr(active_device, "type", None) != device.type:
            return False
        if device.index is not None and getattr(active_device, "index", None) != device.index:
            # Do not infer capabilities for one GPU from another GPU's target.
            return False
        target = driver.get_current_target()
        # This is the compiler's CUDA baseline, not an INT8 MMA requirement.
        backend = getattr(target, "backend", None)
        architecture = getattr(target, "arch", None)
        return isinstance(backend, str) and (
            backend != "cuda" or (isinstance(architecture, int) and architecture >= 70)
        )
    except (RuntimeError, ImportError, AttributeError):
        # Probe before any kernel launch or mutation. Execution errors are not
        # swallowed: retrying a partially applied in-place update is unsafe.
        return False


def _reciprocal_scale(value: torch.Tensor) -> bool:
    return value.device.type == "cuda" and torch.version.hip is not None


def prepare_input(input, group_size, *, out):  # noqa: A002
    """Use separate rotation and row quantization to bound fused live storage."""
    width = input.shape[-1]
    value = input.reshape(-1, width)
    rotated = torch.empty_like(value)
    rotate_input(value, rotated, group_size, num_warps=4)
    quantize_rows_kernel[(value.shape[0],)](
        rotated,
        out[0],
        out[1],
        width,
        block_size=max(128, triton.next_power_of_2(width)),
        logical_dtype_code=logical_dtype_code(value.dtype),
        reciprocal_scale=_reciprocal_scale(value),
        num_warps=4,
    )
    return out


def addmm_(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed=None):
    """Rotate the right factor, multiply with PyTorch, and requantize in place."""
    has_update = alpha != 0 and mat1.shape[1] != 0
    if has_update:
        mat2_contiguous = mat2.contiguous()
        rotated_mat2 = torch.empty_like(mat2_contiguous)
        rotate_input(mat2_contiguous, rotated_mat2, group_size, num_warps=4)
        update = torch.mm(mat1, rotated_mat2)
    else:
        update = qdata
    _requantize_update_(
        qdata, scale, update, mat1.dtype, beta, alpha, rounding_seed, has_update=has_update
    )


def _requantize_update_(
    qdata, scale, update, logical_dtype, beta, alpha, rounding_seed, *, has_update
):
    """Refill existing rowwise storage with shared deterministic/stochastic math."""
    out_features, in_features = qdata.shape
    _requantize_update_rows_kernel[(out_features,)](
        qdata,
        scale,
        update,
        in_features,
        qdata.stride(0),
        qdata.stride(1),
        scale.stride(0),
        update.stride(0),
        update.stride(1),
        beta,
        alpha,
        seed_argument(rounding_seed),
        block_size=max(128, triton.next_power_of_2(in_features)),
        logical_dtype_code=logical_dtype_code(logical_dtype),
        has_base=beta != 0,
        has_update=has_update,
        stochastic=rounding_seed is not None,
        reciprocal_scale=_reciprocal_scale(qdata),
        num_warps=8,
    )


def add_(qdata, scale, update, group_size, alpha, rounding_seed=None):
    """Rotate a dense update and merge it into existing rowwise storage."""
    has_update = alpha != 0
    if has_update:
        update_contiguous = update.contiguous()
        rotated_update = torch.empty_like(update_contiguous)
        rotate_input(update_contiguous, rotated_update, group_size, num_warps=4)
    else:
        rotated_update = qdata
    _requantize_update_(
        qdata,
        scale,
        rotated_update,
        update.dtype,
        1.0,
        alpha,
        rounding_seed,
        has_update=has_update,
    )
