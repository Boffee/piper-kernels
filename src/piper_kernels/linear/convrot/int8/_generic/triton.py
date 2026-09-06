"""Conservative shared Triton launchers without a GPU-model allowlist."""

# pyright: reportCallIssue=false

import torch
import triton

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.stochastic_quantization import seed_argument
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.triton import logical_dtype_code, rotate_input

from .._gguf_policy import select_conversion_chunks
from .._kernels.triton import (
    convert_gguf_tiles_kernel,
    gguf_row_scales_kernel,
    quantize_rows_kernel,
    requantize_update_rows_kernel,
    rotate_quantize_rows_kernel,
)

_GGUF_TILE_SIZE = 1024
_GGUF_MAXIMA_BYTES = 1024 * 1024


def _reciprocal_scale(value: torch.Tensor) -> bool:
    return value.device.type == "cuda" and torch.version.hip is not None


def prepare_input(input, group_size, *, out):  # noqa: A002
    """Use separate rotation and row quantization to bound fused live storage."""
    width = input.shape[-1]
    value = input.reshape(-1, width)
    rotated = torch.empty_like(value)
    rotate_input(value, rotated, group_size, num_warps=4)
    with device_context(input.device):
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


def convert_gguf_out(data, quant_type, group_size, logical_dtype, qdata, scale):
    """Share fused conversion across accelerators, with a bounded wide-row fallback."""
    rows, width = qdata.shape
    if rows == 0 or width == 0:
        scale.fill_(1e-30)
        return
    target = AcceleratorTarget.from_device(data.device)
    chunks = select_conversion_chunks(target, width)
    with device_context(data.device):
        if chunks is not None:
            chunk_count, chunk_size = chunks
            rotate_quantize_rows_kernel[(rows,)](
                data,
                qdata,
                scale,
                width,
                chunk_size=chunk_size,
                chunk_count=chunk_count,
                group_size=group_size,
                inverse_sqrt_group=group_size**-0.5,
                logical_dtype_code=logical_dtype_code(logical_dtype),
                activation_fn=None,
                accelerator_backend=target.backend,
                gguf_quant_type=quant_type,
                num_warps=4,
            )
            return
        _convert_gguf_tiled_out(data, quant_type, group_size, logical_dtype, qdata, scale)


def _convert_gguf_tiled_out(data, quant_type, group_size, logical_dtype, qdata, scale):
    """Convert with at most 1 MiB of maxima, or one row's maxima."""
    rows, width = qdata.shape
    tiles = (width + _GGUF_TILE_SIZE - 1) // _GGUF_TILE_SIZE
    batch_rows = min(rows, max(1, _GGUF_MAXIMA_BYTES // (tiles * 4)))
    maxima = torch.empty((batch_rows, tiles), device=data.device, dtype=torch.float32)
    with device_context(data.device):
        for start in range(0, rows, batch_rows):
            stop = min(start + batch_rows, rows)
            packed, output, scales = data[start:stop], qdata[start:stop], scale[start:stop]
            for write_maxima in (True, False):
                convert_gguf_tiles_kernel[(stop - start, tiles)](
                    packed,
                    output,
                    scales,
                    maxima,
                    width,
                    tiles,
                    block_size=_GGUF_TILE_SIZE,
                    group_size=group_size,
                    logical_dtype_code=logical_dtype_code(logical_dtype),
                    quant_type=quant_type,
                    write_maxima=write_maxima,
                    num_warps=4,
                )
                if write_maxima:
                    gguf_row_scales_kernel[(stop - start,)](
                        maxima,
                        scales,
                        tiles,
                        block_size=triton.next_power_of_2(tiles),
                        reciprocal_scale=_reciprocal_scale(data),
                        num_warps=4,
                    )


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
    with device_context(qdata.device):
        requantize_update_rows_kernel[(out_features,)](
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
