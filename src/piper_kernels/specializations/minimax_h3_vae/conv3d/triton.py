"""Triton kernels for the MiniMax-H3 VAE ConvRot INT8 encoder."""

# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot import triton as rotation_backend
from piper_kernels.linear.convrot.int8._kernels import triton as int8_kernels


@triton.jit
def _prepare_channelwise_kernel(
    input_ptr,
    output_ptr,
    token_count,
    token_volume: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    stride_batch,
    stride_channel,
    stride_frame,
    stride_height,
    stride_width,
    channels: tl.constexpr,
    group_size: tl.constexpr,
    input_scale,
    block_m: tl.constexpr,
):
    """Fuse NCTHW-to-NTHWC conversion, Hadamard rotation, and fixed quantization."""
    token_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    channel_offsets = tl.arange(0, channels)
    batch = token_offsets // token_volume
    position = token_offsets % token_volume
    frame = position // (input_height * input_width)
    spatial = position % (input_height * input_width)
    y = spatial // input_width
    x = spatial % input_width
    pointers = input_ptr + (
        batch[:, None] * stride_batch
        + channel_offsets[None, :] * stride_channel
        + frame[:, None] * stride_frame
        + y[:, None] * stride_height
        + x[:, None] * stride_width
    )
    valid = token_offsets < token_count
    values = tl.load(pointers, mask=valid[:, None], other=0.0).to(tl.float32)
    values = tl.reshape(values, (block_m * channels,))
    values = rotation_backend.rotate_hadamard_groups(
        values,
        block_m * channels,
        group_size,
    )
    values = tl.reshape(values, (block_m, channels))
    values = values * (group_size**-0.5)
    quantized = int8_kernels._quantize_int8(values, input_scale, "cuda")
    tl.store(
        output_ptr + token_offsets[:, None] * channels + channel_offsets[None, :],
        quantized,
        mask=valid[:, None],
    )


@triton.jit
def _group_norm_partial_stats_kernel(
    input_ptr,
    partial_sum_ptr,
    partial_square_sum_ptr,
    spatial_size: tl.constexpr,
    input_width: tl.constexpr,
    frames: tl.constexpr,
    groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    stride_batch,
    stride_channel,
    stride_frame,
    stride_height,
    stride_width,
    stats_block: tl.constexpr,
):
    """Accumulate frame-isolated GroupNorm statistics without a transpose."""
    stats_row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * stats_block + tl.arange(0, stats_block)
    reduction_size: tl.constexpr = channels_per_group * spatial_size
    valid = offsets < reduction_size
    group = stats_row % groups
    frame_row = stats_row // groups
    frame = frame_row % frames
    batch = frame_row // frames
    channel = group * channels_per_group + offsets // spatial_size
    spatial = offsets % spatial_size
    y = spatial // input_width
    x = spatial % input_width
    pointers = input_ptr + (
        batch * stride_batch
        + channel * stride_channel
        + frame * stride_frame
        + y * stride_height
        + x * stride_width
    )
    values = tl.load(pointers, mask=valid, other=0.0).to(tl.float32)
    chunk_count: tl.constexpr = tl.cdiv(reduction_size, stats_block)
    output_offset = stats_row * chunk_count + chunk
    tl.store(partial_sum_ptr + output_offset, tl.sum(values, axis=0))
    tl.store(partial_square_sum_ptr + output_offset, tl.sum(values * values, axis=0))


@triton.jit
def _group_norm_finalize_stats_kernel(
    partial_sum_ptr,
    partial_square_sum_ptr,
    mean_ptr,
    rstd_ptr,
    reduction_size: tl.constexpr,
    chunk_count: tl.constexpr,
    finalize_block: tl.constexpr,
    epsilon: tl.constexpr,
):
    stats_row = tl.program_id(0)
    offsets = tl.arange(0, finalize_block)
    valid = offsets < chunk_count
    partial_offsets = stats_row * chunk_count + offsets
    total = tl.sum(
        tl.load(partial_sum_ptr + partial_offsets, mask=valid, other=0.0),
        axis=0,
    )
    square_total = tl.sum(
        tl.load(partial_square_sum_ptr + partial_offsets, mask=valid, other=0.0),
        axis=0,
    )
    mean = total / reduction_size
    variance = tl.maximum(square_total / reduction_size - mean * mean, 0.0)
    tl.store(mean_ptr + stats_row, mean)
    tl.store(rstd_ptr + stats_row, tl.rsqrt(variance + epsilon))


@triton.jit
def _prepare_group_norm_silu_kernel(
    input_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_ptr,
    token_count,
    token_volume: tl.constexpr,
    frames: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    stride_batch,
    stride_channel,
    stride_frame,
    stride_height,
    stride_width,
    channels: tl.constexpr,
    groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    group_size: tl.constexpr,
    input_scale,
    block_m: tl.constexpr,
):
    """Fuse isolated GroupNorm, SiLU, channel rotation, and fixed quantization."""
    token_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    channel_offsets = tl.arange(0, channels)
    batch = token_offsets // token_volume
    position = token_offsets % token_volume
    frame = position // (input_height * input_width)
    spatial = position % (input_height * input_width)
    y = spatial // input_width
    x = spatial % input_width
    pointers = input_ptr + (
        batch[:, None] * stride_batch
        + channel_offsets[None, :] * stride_channel
        + frame[:, None] * stride_frame
        + y[:, None] * stride_height
        + x[:, None] * stride_width
    )
    valid = token_offsets < token_count
    values = tl.load(pointers, mask=valid[:, None], other=0.0).to(tl.float32)
    group = channel_offsets // channels_per_group
    stats_row = (batch[:, None] * frames + frame[:, None]) * groups + group[None, :]
    mean = tl.load(mean_ptr + stats_row, mask=valid[:, None], other=0.0)
    rstd = tl.load(rstd_ptr + stats_row, mask=valid[:, None], other=0.0)
    affine_weight = tl.load(affine_weight_ptr + channel_offsets)
    affine_bias = tl.load(affine_bias_ptr + channel_offsets)
    values = (values - mean) * rstd * affine_weight[None, :] + affine_bias[None, :]
    values = values * tl.sigmoid(values)
    values = tl.reshape(values, (block_m * channels,))
    values = rotation_backend.rotate_hadamard_groups(
        values,
        block_m * channels,
        group_size,
    )
    values = tl.reshape(values, (block_m, channels))
    values = values * (group_size**-0.5)
    quantized = int8_kernels._quantize_int8(values, input_scale, "cuda")
    tl.store(
        output_ptr + token_offsets[:, None] * channels + channel_offsets[None, :],
        quantized,
        mask=valid[:, None],
    )


@triton.jit
def _conv3d_kernel(  # noqa: PLR0915
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    residual_ptr,
    output_ptr,
    output_rows,
    output_channels,
    input_scale,
    input_channels: tl.constexpr,
    input_frames: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_frames: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    stride_frames: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    has_residual: tl.constexpr,
    symmetric_spatial_padding: tl.constexpr,
    right_spatial_padding: tl.constexpr,
    use_weight_descriptor: tl.constexpr,
    residual_stride_batch,
    residual_stride_channel,
    residual_stride_frame,
    residual_stride_height,
    residual_stride_width,
    loop_num_stages: tl.constexpr,
):
    """Apply causal 3x3x3 ConvRot with one flattened INT32 reduction."""
    row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    channel_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
    valid_rows = row_offsets < output_rows
    valid_channels = channel_offsets < output_channels
    output_volume: tl.constexpr = output_frames * output_height * output_width
    batch = row_offsets // output_volume
    output_position = row_offsets % output_volume
    output_frame = output_position // (output_height * output_width)
    output_spatial = output_position % (output_height * output_width)
    output_y = output_spatial // output_width
    output_x = output_spatial % output_width
    reduction: tl.constexpr = 27 * input_channels
    reduction_offsets = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for reduction_start in tl.range(
        0,
        reduction,
        block_k,
        num_stages=loop_num_stages,
    ):
        k = reduction_start + reduction_offsets
        kernel_position = k // input_channels
        input_channel = k % input_channels
        kernel_frame = kernel_position // 9
        kernel_spatial = kernel_position % 9
        kernel_y = kernel_spatial // 3
        kernel_x = kernel_spatial % 3
        input_frame = output_frame[:, None] * stride_frames + kernel_frame[None, :] - 2
        input_y = output_y[:, None] * stride_height + kernel_y[None, :]
        input_x = output_x[:, None] * stride_width + kernel_x[None, :]
        valid_input = (input_frame >= 0) & (input_frame < input_frames)
        if symmetric_spatial_padding:
            input_y -= 1
            input_x -= 1
            input_y = tl.where(
                input_y < 0,
                -input_y,
                tl.where(input_y >= input_height, 2 * input_height - 2 - input_y, input_y),
            )
            input_x = tl.where(
                input_x < 0,
                -input_x,
                tl.where(input_x >= input_width, 2 * input_width - 2 - input_x, input_x),
            )
        elif right_spatial_padding:
            input_y = tl.where(
                input_y >= input_height,
                2 * input_height - 2 - input_y,
                input_y,
            )
            input_x = tl.where(
                input_x >= input_width,
                2 * input_width - 2 - input_x,
                input_x,
            )
        else:
            valid_input &= (input_y >= 0) & (input_y < input_height)
            valid_input &= (input_x >= 0) & (input_x < input_width)
        input_token = (
            (batch[:, None] * input_frames + input_frame) * input_height + input_y
        ) * input_width + input_x
        input_values = tl.load(
            input_ptr + input_token * input_channels + input_channel[None, :],
            mask=valid_rows[:, None] & valid_input,
            other=0,
        )
        if use_weight_descriptor:
            weight_values = weight_ptr.load([tl.program_id(1) * block_n, reduction_start]).T
        else:
            weight_pointers = weight_ptr + channel_offsets[None, :] * reduction + k[:, None]
            weight_values = tl.load(
                weight_pointers,
                mask=(k[:, None] < reduction) & valid_channels[None, :],
                other=0,
            )
        accumulator += tl.dot(input_values, weight_values)

    output = accumulator.to(tl.float32) * input_scale
    weight_scale = tl.load(
        weight_scale_ptr + channel_offsets,
        mask=valid_channels,
        other=0.0,
    )
    output *= weight_scale[None, :]
    if has_bias:
        bias = tl.load(bias_ptr + channel_offsets, mask=valid_channels, other=0.0)
        output += bias[None, :]
    if has_residual:
        residual_pointers = residual_ptr + (
            batch[:, None] * residual_stride_batch
            + channel_offsets[None, :] * residual_stride_channel
            + output_frame[:, None] * residual_stride_frame
            + output_y[:, None] * residual_stride_height
            + output_x[:, None] * residual_stride_width
        )
        residual = tl.load(
            residual_pointers,
            mask=valid_rows[:, None] & valid_channels[None, :],
            other=0.0,
        )
        output += residual.to(tl.float32)
    output_pointers = output_ptr + (
        (batch[:, None] * output_channels + channel_offsets[None, :]) * output_volume
        + output_position[:, None]
    )
    tl.store(
        output_pointers,
        output,
        mask=valid_rows[:, None] & valid_channels[None, :],
    )


def _output_dimensions(input_shape, stride, *, symmetric_padding, right_padding):
    _, _, frames, height, width = input_shape
    spatial_padding = 2 if symmetric_padding else int(right_padding)
    return (
        (frames - 1) // stride[0] + 1,
        (height + spatial_padding - 3) // stride[1] + 1,
        (width + spatial_padding - 3) // stride[2] + 1,
    )


def _convolution_plan(input_qdata, output_shape):
    batch, _, _, _, channels = input_qdata.shape
    _, outputs, frames, height, width = output_shape
    rows = batch * frames * height * width
    plan = (64, 128, 128, 4, 3, 3)
    if channels == 128:
        plan = (128, 128, 64, 4, 3, 3)
    elif channels == 256 and rows >= 200_000:
        plan = (128, 128, 64, 4, 4, 4)
    elif channels == 256 and rows >= 30_000:
        plan = (128, 128, 64, 4, 2, 2)
    elif channels == 256 and outputs == 256:
        plan = (64, 128, 128, 4, 3, 3)
    elif channels == 256 or (channels == 512 and rows >= 5_000):
        plan = (128, 128, 128, 8, 3, 3)
    elif channels == 512 and outputs > 512:
        plan = (64, 128, 128, 4, 3, 3)
    elif channels == 512:
        plan = (32, 128, 256, 8, 3, 3)
    elif outputs <= 64:
        plan = (64, 64, 64, 4, 3, 3)
    return plan


def _preparation_plan(input, *, group_norm):  # noqa: A002
    batch, channels, frames, height, width = input.shape
    rows = batch * frames * height * width
    plan = (8, 8)
    if channels == 128:
        plan = (64, 4)
    elif channels == 256:
        if group_norm and rows >= 200_000:
            plan = (32, 4)
        elif group_norm:
            plan = (16, 4)
        elif rows >= 200_000:
            plan = (64, 4)
        else:
            plan = (32, 8)
    elif channels == 512 and group_norm and rows >= 5_000:
        plan = (16, 8)
    elif channels == 512 and not group_norm and rows >= 5_000:
        plan = (32, 8)
    return plan


def _prepare_input(input, group_size, input_scale):  # noqa: A002
    batch, channels, frames, height, width = input.shape
    token_volume = frames * height * width
    token_count = batch * token_volume
    qdata = torch.empty(
        (batch, frames, height, width, channels),
        device=input.device,
        dtype=torch.int8,
    )
    block_m, num_warps = _preparation_plan(input, group_norm=False)
    with device_context(input.device):
        _prepare_channelwise_kernel[(triton.cdiv(token_count, block_m),)](
            input,
            qdata,
            token_count,
            token_volume=token_volume,
            input_height=height,
            input_width=width,
            stride_batch=input.stride(0),
            stride_channel=input.stride(1),
            stride_frame=input.stride(2),
            stride_height=input.stride(3),
            stride_width=input.stride(4),
            channels=channels,
            group_size=group_size,
            input_scale=input_scale,
            block_m=block_m,
            num_warps=num_warps,
        )
    return qdata


def _prepare_group_norm_silu_input(
    input,  # noqa: A002
    norm_weight,
    norm_bias,
    norm_groups,
    norm_epsilon,
    group_size,
    input_scale,
):
    batch, channels, frames, height, width = input.shape
    channels_per_group = channels // norm_groups
    reduction_size = channels_per_group * height * width
    stats_rows = batch * frames * norm_groups
    stats_block = 4096
    chunk_count = int(triton.cdiv(reduction_size, stats_block))
    partial_sum = torch.empty(
        (stats_rows, chunk_count),
        device=input.device,
        dtype=torch.float32,
    )
    partial_square_sum = torch.empty_like(partial_sum)
    mean = torch.empty(stats_rows, device=input.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    with device_context(input.device):
        _group_norm_partial_stats_kernel[(stats_rows, chunk_count)](
            input,
            partial_sum,
            partial_square_sum,
            spatial_size=height * width,
            input_width=width,
            frames=frames,
            groups=norm_groups,
            channels_per_group=channels_per_group,
            stride_batch=input.stride(0),
            stride_channel=input.stride(1),
            stride_frame=input.stride(2),
            stride_height=input.stride(3),
            stride_width=input.stride(4),
            stats_block=stats_block,
            num_warps=8,
        )
        finalize_block = triton.next_power_of_2(chunk_count)
        _group_norm_finalize_stats_kernel[(stats_rows,)](
            partial_sum,
            partial_square_sum,
            mean,
            rstd,
            reduction_size=reduction_size,
            chunk_count=chunk_count,
            finalize_block=finalize_block,
            epsilon=norm_epsilon,
            num_warps=4,
        )
        token_volume = frames * height * width
        token_count = batch * token_volume
        qdata = torch.empty(
            (batch, frames, height, width, channels),
            device=input.device,
            dtype=torch.int8,
        )
        block_m, num_warps = _preparation_plan(input, group_norm=True)
        _prepare_group_norm_silu_kernel[(triton.cdiv(token_count, block_m),)](
            input,
            norm_weight,
            norm_bias,
            mean,
            rstd,
            qdata,
            token_count,
            token_volume=token_volume,
            frames=frames,
            input_height=height,
            input_width=width,
            stride_batch=input.stride(0),
            stride_channel=input.stride(1),
            stride_frame=input.stride(2),
            stride_height=input.stride(3),
            stride_width=input.stride(4),
            channels=channels,
            groups=norm_groups,
            channels_per_group=channels_per_group,
            group_size=group_size,
            input_scale=input_scale,
            block_m=block_m,
            num_warps=num_warps,
        )
    return qdata


def _conv3d_prepared(
    input_qdata,
    weight_qdata,
    weight_scale,
    bias,
    input_scale,
    stride,
    *,
    symmetric_spatial_padding,
    right_spatial_padding,
    residual,
    output_dtype,
):
    batch, input_frames, input_height, input_width, input_channels = input_qdata.shape
    output_frames, output_height, output_width = _output_dimensions(
        (batch, input_channels, input_frames, input_height, input_width),
        stride,
        symmetric_padding=symmetric_spatial_padding,
        right_padding=right_spatial_padding,
    )
    output_channels = weight_qdata.shape[0]
    output_shape = (
        batch,
        output_channels,
        output_frames,
        output_height,
        output_width,
    )
    output = torch.empty(output_shape, device=input_qdata.device, dtype=output_dtype)
    bias_pointer = bias if bias is not None else output
    residual_pointer = residual if residual is not None else output
    residual_strides = residual.stride() if residual is not None else output.stride()
    block_m, block_n, block_k, num_warps, num_stages, loop_num_stages = _convolution_plan(
        input_qdata, output_shape
    )
    target = AcceleratorTarget.from_device(input_qdata.device)
    use_weight_descriptor = (
        target.is_cuda_capability(12, 0)
        and input_channels == 128
        and output_channels % block_n == 0
        and (output_height >= 256 or output_channels > 128)
    )
    with device_context(input_qdata.device):
        weight_argument = (
            TensorDescriptor(
                base=weight_qdata,
                shape=[output_channels, 27 * input_channels],
                strides=[27 * input_channels, 1],
                block_shape=[block_n, block_k],
            )
            if use_weight_descriptor
            else weight_qdata
        )
        rows = batch * output_frames * output_height * output_width
        _conv3d_kernel[(triton.cdiv(rows, block_m), triton.cdiv(output_channels, block_n))](
            input_qdata,
            weight_argument,
            weight_scale,
            bias_pointer,
            residual_pointer,
            output,
            rows,
            output_channels,
            input_scale,
            input_channels=input_channels,
            input_frames=input_frames,
            input_height=input_height,
            input_width=input_width,
            output_frames=output_frames,
            output_height=output_height,
            output_width=output_width,
            stride_frames=stride[0],
            stride_height=stride[1],
            stride_width=stride[2],
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            has_bias=bias is not None,
            has_residual=residual is not None,
            symmetric_spatial_padding=symmetric_spatial_padding,
            right_spatial_padding=right_spatial_padding,
            use_weight_descriptor=use_weight_descriptor,
            residual_stride_batch=residual_strides[0],
            residual_stride_channel=residual_strides[1],
            residual_stride_frame=residual_strides[2],
            residual_stride_height=residual_strides[3],
            residual_stride_width=residual_strides[4],
            loop_num_stages=loop_num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output


def conv3d(
    input,  # noqa: A002
    weight_qdata,
    weight_scale,
    bias,
    group_size,
    input_scale,
    stride,
    *,
    symmetric_spatial_padding,
    right_spatial_padding,
    residual,
):
    prepared = _prepare_input(input, group_size, input_scale)
    return _conv3d_prepared(
        prepared,
        weight_qdata,
        weight_scale,
        bias,
        input_scale,
        stride,
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
        output_dtype=input.dtype,
    )


def group_norm_silu_conv3d(
    input,  # noqa: A002
    norm_weight,
    norm_bias,
    norm_groups,
    norm_epsilon,
    weight_qdata,
    weight_scale,
    bias,
    group_size,
    input_scale,
    stride,
    *,
    symmetric_spatial_padding,
    right_spatial_padding,
    residual,
):
    prepared = _prepare_group_norm_silu_input(
        input,
        norm_weight,
        norm_bias,
        norm_groups,
        norm_epsilon,
        group_size,
        input_scale,
    )
    return _conv3d_prepared(
        prepared,
        weight_qdata,
        weight_scale,
        bias,
        input_scale,
        stride,
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
        output_dtype=input.dtype,
    )


__all__ = ["conv3d", "group_norm_silu_conv3d"]
