"""ConvRot INT8 rotation and quantization kernels, and weight conversion and update launchers.

ConvRot INT8 activation preparation and weight conversion share the row quantization kernels.
"""

# Triton's constexpr annotations reject the Python literals its JIT accepts, and its
# launcher accepts compile-time options outside its Python signature.
# pyright: reportArgumentType=false, reportCallIssue=false
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.gguf import triton as gguf_backend
from piper_kernels.stochastic_quantization.triton import stochastic_round_to_int
from piper_kernels.weights.convrot import triton as convrot_backend

from ._gguf_policy import select_conversion_chunks
from ._plan import AddmmExecutionPlan
from ._policy import select_addmm_plan

_GGUF_TILE_SIZE = 1024
_GGUF_MAXIMA_BYTES = 1024 * 1024
# Amortize GEMM/launch overhead while bounding scratch independently of output rows.
_UPDATE_WORKSPACE_BYTES = 32 * 1024 * 1024
# PyTorch's metadata-only alias predicate is missing from its type stubs.
_shares_storage: Callable[[torch.Tensor, torch.Tensor], bool] = torch._C._is_alias_of  # pyright: ignore[reportAttributeAccessIssue]


@triton.jit
def int8_scale_from_max(absolute_max, reciprocal_scale: tl.constexpr = False):
    """Construct an unclamped scale with the backend's scalar-division lowering.

    HIP PyTorch divides by a scalar using a rounded FP32 reciprocal. Request
    that explicitly on HIP; retain the established CUDA expression by default.
    """
    if reciprocal_scale:
        return absolute_max * (1.0 / 127.0)
    return absolute_max / 127.0


@triton.jit
def normalize_for_int8(values, scale):
    """Normalize in FP32 without rounding the scale or quotient."""
    return values.to(tl.float32) / scale


@triton.jit
def round_to_int8(scaled, accelerator_backend: tl.constexpr):
    """Saturate and round once to nearest-even INT8 using the target's conversion."""
    if accelerator_backend == "cuda":
        return libdevice.float2int_rn(tl.clamp(scaled, -128.0, 127.0)).to(tl.int8)  # pyright: ignore[reportAttributeAccessIssue]
    return tl.clamp(libdevice.rint(scaled), -128.0, 127.0).to(tl.int8)


@triton.jit
def _quantize_int8(values, scale, accelerator_backend: tl.constexpr):
    """Apply terminal INT8 rounding after FP32 normalization."""
    return round_to_int8(normalize_for_int8(values, scale), accelerator_backend)


@triton.jit
def _store_quantized_chunk(
    q_ptr,
    output_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    values,
    scale,
    accelerator_backend: tl.constexpr,
):
    offsets = chunk_start + chunk_offsets
    quantized = _quantize_int8(values, scale, accelerator_backend)
    tl.store(
        q_ptr + output_row_offset + offsets,
        quantized,
        mask=offsets < row_width,
    )


@triton.jit
def _load_weight_chunk(
    x_ptr,
    input_row_offset,
    packed_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
):
    if gguf_quant_type >= 0:
        return convrot_backend.load_gguf_rotated_chunk(
            x_ptr,
            packed_row_offset,
            row_width,
            chunk_start,
            chunk_offsets,
            chunk_size,
            group_size,
            inverse_sqrt_group,
            gguf_quant_type,
        )
    return convrot_backend.load_activated_rotated_chunk(
        x_ptr,
        input_row_offset,
        row_width,
        chunk_start,
        chunk_offsets,
        chunk_size,
        group_size,
        inverse_sqrt_group,
        activation_fn,
        accelerator_backend,
    )


@triton.jit
def rotate_quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    chunk_size: tl.constexpr,
    chunk_count: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    gguf_quant_type: tl.constexpr,
    static_scale_ptr=None,
):
    """Rotate and quantize a row held as one, two, or three equal chunks.

    Dynamic scaling keeps rotated chunks live until the row maximum is known.
    Static scaling quantizes and stores each chunk immediately.
    """
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    chunk_offsets = tl.arange(0, chunk_size)
    input_row_width = row_width * (2 if activation_fn == "swiglu" else 1)
    input_row_offset = row_i64 * input_row_width
    packed_row_offset = 0
    if gguf_quant_type >= 0:
        packed_row_offset = row_i64 * gguf_backend.packed_row_size(
            row_width,
            gguf_quant_type,
        )
    output_row_offset = row_i64 * row_width

    if static_scale_ptr is not None:
        scale = tl.load(static_scale_ptr)
        for chunk in tl.static_range(chunk_count):  # pyright: ignore[reportGeneralTypeIssues]
            values = _load_weight_chunk(
                x_ptr,
                input_row_offset,
                packed_row_offset,
                row_width,
                chunk * chunk_size,
                chunk_offsets,
                chunk_size,
                group_size,
                inverse_sqrt_group,
                activation_fn,
                accelerator_backend,
                gguf_quant_type,
            )
            _store_quantized_chunk(
                q_ptr,
                output_row_offset,
                row_width,
                chunk * chunk_size,
                chunk_offsets,
                values,
                scale,
                accelerator_backend,
            )
        tl.store(scale_ptr + row_i64, scale)
        return

    values0 = _load_weight_chunk(
        x_ptr,
        input_row_offset,
        packed_row_offset,
        row_width,
        0,
        chunk_offsets,
        chunk_size,
        group_size,
        inverse_sqrt_group,
        activation_fn,
        accelerator_backend,
        gguf_quant_type,
    )
    row_max = tl.max(tl.abs(values0).to(tl.float32), axis=0)
    if chunk_count >= 2:
        values1 = _load_weight_chunk(
            x_ptr,
            input_row_offset,
            packed_row_offset,
            row_width,
            chunk_size,
            chunk_offsets,
            chunk_size,
            group_size,
            inverse_sqrt_group,
            activation_fn,
            accelerator_backend,
            gguf_quant_type,
        )
        row_max = tl.maximum(
            row_max,
            tl.max(tl.abs(values1).to(tl.float32), axis=0),
        )
    if chunk_count >= 3:
        values2 = _load_weight_chunk(
            x_ptr,
            input_row_offset,
            packed_row_offset,
            row_width,
            2 * chunk_size,
            chunk_offsets,
            chunk_size,
            group_size,
            inverse_sqrt_group,
            activation_fn,
            accelerator_backend,
            gguf_quant_type,
        )
        row_max = tl.maximum(
            row_max,
            tl.max(tl.abs(values2).to(tl.float32), axis=0),
        )

    scale = tl.maximum(int8_scale_from_max(row_max, accelerator_backend == "hip"), 1e-30)
    _store_quantized_chunk(
        q_ptr,
        output_row_offset,
        row_width,
        0,
        chunk_offsets,
        values0,
        scale,
        accelerator_backend,
    )
    if chunk_count >= 2:
        _store_quantized_chunk(
            q_ptr,
            output_row_offset,
            row_width,
            chunk_size,
            chunk_offsets,
            values1,  # pyright: ignore[reportPossiblyUnboundVariable]
            scale,
            accelerator_backend,
        )
    if chunk_count >= 3:
        _store_quantized_chunk(
            q_ptr,
            output_row_offset,
            row_width,
            2 * chunk_size,
            chunk_offsets,
            values2,  # pyright: ignore[reportPossiblyUnboundVariable]
            scale,
            accelerator_backend,
        )
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def convert_gguf_tiles_kernel(
    data_ptr,
    q_ptr,
    scale_ptr,
    maxima_ptr,
    row_width,
    tiles_per_row,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    quant_type: tl.constexpr,
    write_maxima: tl.constexpr,
    accelerator_backend: tl.constexpr,
):
    """Decode/rotate twice, retaining only tile maxima between the two passes."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    offsets = tl.arange(0, block_size)
    start = tile * block_size
    row_bytes = gguf_backend.packed_row_size(row_width, quant_type)
    tile_bytes = gguf_backend.packed_row_size(block_size, quant_type)
    values = convrot_backend.load_gguf_rotated_chunk(
        data_ptr,
        row * row_bytes + tile.to(tl.int64) * tile_bytes,
        row_width - start,
        0,
        offsets,
        block_size,
        group_size,
        group_size**-0.5,
        quant_type,
    )
    if write_maxima:
        maximum = tl.max(tl.abs(values).to(tl.float32), axis=0)
        tl.store(maxima_ptr + row * tiles_per_row + tile, maximum)
    else:
        scale = tl.load(scale_ptr + row)
        quantized = _quantize_int8(values, scale, accelerator_backend)
        tl.store(q_ptr + row * row_width + start + offsets, quantized, start + offsets < row_width)


@triton.jit
def reduce_row_scales_kernel(
    maxima_ptr,
    scale_ptr,
    tiles_per_row,
    block_size: tl.constexpr,
    reciprocal_scale: tl.constexpr,
):
    """Reduce tile maxima to the weight's single scale per row."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block_size)
    maxima = tl.load(maxima_ptr + row * tiles_per_row + offsets, offsets < tiles_per_row, 0.0)
    scale = tl.maximum(int8_scale_from_max(tl.max(maxima, axis=0), reciprocal_scale), 1e-30)
    tl.store(scale_ptr + row, scale)


@triton.jit
def quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    accelerator_backend: tl.constexpr,
    reciprocal_scale: tl.constexpr = False,
    static_scale_ptr=None,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    row_offset = row_i64 * row_width
    values = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    if static_scale_ptr is None:
        scale = tl.maximum(
            int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), reciprocal_scale),
            1e-30,
        )
    else:
        scale = tl.load(static_scale_ptr)
    quantized = _quantize_int8(values, scale, accelerator_backend)
    tl.store(q_ptr + row_offset + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def requantize_update_rows_kernel(
    q_ptr,
    scale_ptr,
    update_ptr,
    row_width,
    stride_q_row,
    stride_q_col,
    stride_scale_row,
    stride_update_row,
    stride_update_col,
    beta,
    alpha,
    rounding_seed,
    start_row,
    block_size: tl.constexpr,
    has_base: tl.constexpr,
    has_update: tl.constexpr,
    stochastic: tl.constexpr,
    accelerator_backend: tl.constexpr,
    reciprocal_scale: tl.constexpr = False,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < row_width
    if has_base:
        quantized = tl.load(
            q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
            mask=mask,
            other=0,
        )
        old_scale = tl.load(scale_ptr + row_i64 * stride_scale_row)
        values = beta * quantized.to(tl.float32) * old_scale
    else:
        values = tl.zeros((block_size,), dtype=tl.float32)
    if has_update:
        update = tl.load(
            update_ptr + row_i64 * stride_update_row + offsets_i64 * stride_update_col,
            mask=mask,
            other=0.0,
        )
        values += alpha * update.to(tl.float32)
    scale = tl.maximum(
        int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), reciprocal_scale),
        1e-30,
    )
    quantized = _quantize_int8(values, scale, accelerator_backend)
    if stochastic:
        stochastic_scaled = values.to(tl.float32) / scale
        logical_offsets = (row_i64 + start_row) * row_width + offsets_i64
        quantized = stochastic_round_to_int(
            stochastic_scaled,
            quantized,
            rounding_seed,
            logical_offsets,
            -128,
            127,
        ).to(tl.int8)
    tl.store(
        q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row_i64 * stride_scale_row, scale)


@triton.jit
def merge_lora_tiles_kernel(
    q_ptr,
    scale_ptr,
    new_scale_ptr,
    mat1_ptr,
    mat2_ptr,
    maxima_ptr,
    rows_count,
    row_width: tl.constexpr,
    rank: tl.constexpr,
    stride_a_row: tl.constexpr,
    stride_a_col: tl.constexpr,
    stride_b_row: tl.constexpr,
    stride_b_col: tl.constexpr,
    beta,
    alpha,
    rounding_seed,
    group_size: tl.constexpr,
    matmul_dtype: tl.constexpr,
    has_base: tl.constexpr,
    stochastic: tl.constexpr,
    write_maxima: tl.constexpr,
    single_tile: tl.constexpr,
    accelerator_backend: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Rotate, multiply, and merge a tile; retain only maxima between passes."""
    rows = (tl.program_id(0) * block_m + tl.arange(0, block_m)).to(tl.int64)
    tile = tl.program_id(1)
    columns = (tile * block_n + tl.arange(0, block_n)).to(tl.int64)
    valid = (rows[:, None] < rows_count) & (columns[None, :] < row_width)
    reduction = tl.arange(0, block_k)
    product = tl.full((block_m, block_n), 0.0, tl.float32)
    logical_dtype: tl.constexpr = mat2_ptr.dtype.element_ty
    for start in range(tl.cdiv(rank, block_k)):
        k = (start * block_k + reduction).to(tl.int64)
        left = tl.load(
            mat1_ptr + rows[:, None] * stride_a_row + k[None, :] * stride_a_col,
            (rows[:, None] < rows_count) & (k[None, :] < rank),
            0.0,
        ).to(matmul_dtype)
        right = tl.load(
            mat2_ptr + k[:, None] * stride_b_row + columns[None, :] * stride_b_col,
            (k[:, None] < rank) & (columns[None, :] < row_width),
            0.0,
        ).to(tl.float32)
        rotated = convrot_backend.rotate_hadamard_groups(
            tl.reshape(right, (block_k * block_n,)), block_k * block_n, group_size
        )
        right = tl.reshape(rotated * (group_size**-0.5), (block_k, block_n))
        # Match materialized rotation, autocast, and GEMM-result rounding.
        right = right.to(logical_dtype).to(matmul_dtype)
        product = tl.dot(
            left,
            right,
            product,
            input_precision="tf32x3" if accelerator_backend == "cuda" else "ieee",
        )
    update = product.to(matmul_dtype).to(tl.float32)
    values = tl.full((block_m, block_n), 0.0, tl.float32)
    if has_base:
        base = tl.load(q_ptr + rows[:, None] * row_width + columns[None, :], valid, 0)
        old_scale = tl.load(scale_ptr + rows, rows < rows_count, 0.0)
        values = beta * base.to(tl.float32) * old_scale[:, None]
    values += alpha * update
    values = tl.where(valid, values, 0.0)
    if write_maxima:
        maximum = tl.max(tl.abs(values), axis=1)
        tl.store(
            maxima_ptr + rows * tl.cdiv(row_width, block_n) + tile,
            maximum,
            rows < rows_count,
        )
    else:
        if single_tile:
            scale = tl.maximum(
                int8_scale_from_max(tl.max(tl.abs(values), axis=1), accelerator_backend == "hip"),
                1e-30,
            )
        else:
            scale = tl.load(new_scale_ptr + rows, rows < rows_count, 1.0)
        quantized = _quantize_int8(values, scale[:, None], accelerator_backend)
        if stochastic:
            quantized = stochastic_round_to_int(
                values / scale[:, None],
                quantized,
                rounding_seed,
                rows[:, None] * row_width + columns[None, :],
                -128,
                127,
            ).to(tl.int8)
        tl.store(q_ptr + rows[:, None] * row_width + columns[None, :], quantized, valid)
        if single_tile:
            tl.store(scale_ptr + rows, scale, rows < rows_count)


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
    convrot_backend.rotate_input(mat2_contiguous, rotated_mat2, group_size, num_warps=4)
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
        convrot_backend.rotate_input(update_contiguous, rotated_update, group_size, num_warps=4)
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
