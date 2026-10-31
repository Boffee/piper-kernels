"""Triton launchers for ConvRot INT8 weight conversion and updates."""

# pyright: reportCallIssue=false
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from piper_kernels._triton.convrot import rotate_input
from piper_kernels._triton.convrot_int8 import (
    convert_gguf_tiles_kernel,
    merge_lora_tiles_kernel,
    reduce_row_scales_kernel,
    requantize_update_rows_kernel,
    rotate_quantize_rows_kernel,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget

from ._gguf_policy import select_conversion_chunks
from ._plan import AddmmExecutionPlan
from ._policy import select_addmm_plan

_GGUF_TILE_SIZE = 1024
_GGUF_MAXIMA_BYTES = 1024 * 1024
# Amortize GEMM/launch overhead while bounding scratch independently of output rows.
_UPDATE_WORKSPACE_BYTES = 32 * 1024 * 1024
# PyTorch's metadata-only alias predicate is missing from its type stubs.
_shares_storage: Callable[[torch.Tensor, torch.Tensor], bool] = torch._C._is_alias_of  # pyright: ignore[reportAttributeAccessIssue]


def _addmm_fused_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
    *,
    execution_plan: AddmmExecutionPlan,
) -> None:
    """Merge contiguous INT8 rows with group-aligned tiles.

    The first pass computes maxima; the second recomputes and packs the
    product using complete-row scales. Keeping the old scales until all
    packing CTAs finish prevents a race with base-weight dequantization.
    One-column-tile rows can reduce and pack in one pass. Rotation and GEMM
    retain the batched path's dtype rounding boundaries, including autocast.
    """
    if alpha == 0 or mat1.shape[1] == 0:
        _requantize_update_(qdata, scale, qdata, beta, alpha, rounding_seed, has_update=False)
        return
    rows, width = qdata.shape
    plan = execution_plan
    tiles = triton.cdiv(width, plan.block_n)
    maxima = (
        torch.empty((rows, tiles), device=qdata.device, dtype=torch.float32) if tiles > 1 else None
    )
    new_scale = torch.empty_like(scale) if tiles > 1 else scale
    dtype = (
        torch.get_autocast_dtype(qdata.device.type)
        if torch.is_autocast_enabled(qdata.device.type)
        else mat1.dtype
    )
    matmul_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]
    with device_context(qdata.device):
        for write_maxima in (True, False) if tiles > 1 else (False,):
            merge_lora_tiles_kernel[(triton.cdiv(rows, plan.block_m), tiles)](
                qdata,
                scale,
                new_scale,
                mat1,
                mat2,
                maxima,
                rows,
                width,
                mat1.shape[1],
                *mat1.stride(),
                *mat2.stride(),
                beta,
                alpha,
                0 if rounding_seed is None else rounding_seed,
                group_size=group_size,
                matmul_dtype=matmul_dtype,
                has_base=beta != 0,
                stochastic=rounding_seed is not None,
                write_maxima=write_maxima,
                single_tile=tiles == 1,
                accelerator_backend="hip" if _reciprocal_scale(qdata) else qdata.device.type,
                block_m=plan.block_m,
                block_n=plan.block_n,
                block_k=plan.block_k,
                num_warps=plan.num_warps,
                num_stages=plan.num_stages,
            )
            if write_maxima:
                reduce_row_scales_kernel[(rows,)](
                    maxima,
                    new_scale,
                    tiles,
                    block_size=triton.next_power_of_2(tiles),
                    reciprocal_scale=_reciprocal_scale(qdata),
                    num_warps=4,
                )
        # Every packing CTA must finish reading the old scales before replacement.
        if tiles > 1:
            scale.copy_(new_scale)


def _reciprocal_scale(value: torch.Tensor) -> bool:
    return value.device.type == "cuda" and torch.version.hip is not None


def convert_gguf_out(data, quant_type, group_size, qdata, scale):
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
                activation_fn=None,
                accelerator_backend=target.backend,
                gguf_quant_type=quant_type,
                num_warps=4,
            )
            return
        _convert_gguf_tiled_out(data, quant_type, group_size, qdata, scale)


def _convert_gguf_tiled_out(data, quant_type, group_size, qdata, scale):
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
                    quant_type=quant_type,
                    write_maxima=write_maxima,
                    accelerator_backend="hip" if _reciprocal_scale(data) else data.device.type,
                    num_warps=4,
                )
                if write_maxima:
                    reduce_row_scales_kernel[(stop - start,)](
                        maxima,
                        scales,
                        tiles,
                        block_size=triton.next_power_of_2(tiles),
                        reciprocal_scale=_reciprocal_scale(data),
                        num_warps=4,
                    )


def addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Fuse supported matrix updates, with bounded GEMM batches on other targets."""
    if alpha != 0 and mat1.shape[1] != 0:
        # Tiles and row batches overwrite storage while other tiles/batches
        # may still read the factors. Snapshot only factors sharing that storage.
        if _shares_storage(mat1, qdata) or _shares_storage(mat1, scale):
            mat1 = mat1.clone()
        if _shares_storage(mat2, qdata) or _shares_storage(mat2, scale):
            mat2 = mat2.clone()
    plan = select_addmm_plan(
        AcceleratorTarget.from_device(qdata.device),
        group_size,
    )
    if plan is not None:
        _addmm_fused_(
            qdata,
            scale,
            mat1,
            mat2,
            group_size,
            beta,
            alpha,
            rounding_seed,
            execution_plan=plan,
        )
    else:
        _addmm_batched_(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed)


def _addmm_batched_(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed=None):
    """Rotate the right factor and merge row batches with a bounded product buffer."""
    has_update = alpha != 0 and mat1.shape[1] != 0
    if not has_update:
        _requantize_update_(qdata, scale, qdata, beta, alpha, rounding_seed, has_update=False)
        return

    mat2_contiguous = mat2.contiguous()
    rotated_mat2 = torch.empty_like(mat2_contiguous)
    rotate_input(mat2_contiguous, rotated_mat2, group_size, num_warps=4)
    rows, width = qdata.shape
    batch_rows = min(rows, max(1, _UPDATE_WORKSPACE_BYTES // (width * mat1.element_size())))
    if batch_rows == rows:
        update = torch.mm(mat1, rotated_mat2)
        _requantize_update_(qdata, scale, update, beta, alpha, rounding_seed, has_update=True)
        return
    for start in range(0, rows, batch_rows):
        stop = min(start + batch_rows, rows)
        batch_update = torch.mm(mat1[start:stop], rotated_mat2)
        _requantize_update_(
            qdata[start:stop],
            scale[start:stop],
            batch_update,
            beta,
            alpha,
            rounding_seed,
            has_update=True,
            start_row=start,
        )
        # Release before the next matmul allocates, keeping only one product live.
        del batch_update


def _requantize_update_(
    qdata, scale, update, beta, alpha, rounding_seed, *, has_update, start_row=0
):
    """Refill rowwise storage with one shared schedule on the weight's device."""
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
            0 if rounding_seed is None else rounding_seed,
            start_row,
            block_size=max(128, triton.next_power_of_2(in_features)),
            has_base=beta != 0,
            has_update=has_update,
            stochastic=rounding_seed is not None,
            reciprocal_scale=_reciprocal_scale(qdata),
            accelerator_backend="hip" if _reciprocal_scale(qdata) else qdata.device.type,
            num_warps=8,
        )


def add_(qdata, scale, update, group_size, beta, alpha, rounding_seed=None):
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
        beta,
        alpha,
        rounding_seed,
        has_update=has_update,
    )
