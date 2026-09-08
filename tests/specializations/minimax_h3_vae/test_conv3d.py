"""Tests for the MiniMax-H3 VAE ConvRot INT8 Conv3D specialization."""

from __future__ import annotations

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.int8.reference import dynamic_quantize_rows
from piper_kernels.specializations.minimax_h3_vae.conv3d import (
    P995_ACTIVATION_SCALES,
    conv3d,
    group_norm_silu_conv3d,
    prepare_weight,
    reference,
)


def _nvidia_cuda_available() -> bool:
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).is_nvidia_cuda
    )


def _weight(
    output_channels: int,
    input_channels: int,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    weight = torch.randn(
        output_channels,
        input_channels,
        3,
        3,
        3,
        device=device,
        dtype=torch.float16,
    ).mul_(0.02)
    group_size = 64 if input_channels == 128 else 256
    return reference.quantize_weight(weight, group_size)


def test_calibration_covers_every_accelerated_encoder_convolution() -> None:
    assert len(P995_ACTIVATION_SCALES) == 29
    assert P995_ACTIVATION_SCALES["down_blocks.0.resnets.0.conv1"] == pytest.approx(
        0.008035494945943356
    )
    assert P995_ACTIVATION_SCALES["conv_out"] == pytest.approx(0.004467581398785114)
    assert all(scale > 0 for scale in P995_ACTIVATION_SCALES.values())


def test_prepare_weight_uses_h3_channel_layout_and_calibration() -> None:
    weight = torch.randn(32, 128, 3, 3, 3)
    qdata, weight_scale, group_size, input_scale = prepare_weight(
        "down_blocks.0.resnets.0.conv1",
        weight,
    )

    assert qdata.shape == (32, 3, 3, 3, 128)
    assert qdata.dtype is torch.int8
    assert weight_scale.shape == (32,)
    assert weight_scale.dtype is torch.float32
    assert group_size == 64
    assert input_scale == P995_ACTIVATION_SCALES["down_blocks.0.resnets.0.conv1"]


@pytest.mark.parametrize("group_size", [64, 256])
def test_cpu_weight_butterfly_matches_dense_convrot_quantization(group_size: int) -> None:
    torch.manual_seed(2468)
    weight = torch.randn(8, group_size, 3, 3, 3, dtype=torch.float16).mul_(0.02)
    channel_last = weight.permute(0, 2, 3, 4, 1).contiguous()
    expected_qdata, expected_scale = dynamic_quantize_rows(
        rotate_groups(channel_last.float(), group_size).flatten(1)
    )

    actual_qdata, actual_scale = reference.quantize_weight(weight, group_size)

    torch.testing.assert_close(actual_qdata.flatten(1), expected_qdata)
    torch.testing.assert_close(actual_scale, expected_scale.flatten())


def test_prepare_weight_rejects_uncalibrated_layer() -> None:
    with pytest.raises(ValueError, match="no calibrated scale"):
        prepare_weight("unknown", torch.empty(32, 128, 3, 3, 3))


@pytest.mark.parametrize(
    ("padding", "stride", "expected"),
    [
        ("reflect", (1, 1, 1), (2, 32, 5, 16, 18)),
        ("reflect_right", (1, 2, 2), (2, 32, 5, 8, 9)),
        ("none", (1, 1, 1), (2, 32, 5, 14, 16)),
    ],
)
def test_fake_conv3d_preserves_shape_and_dtype(
    padding: str,
    stride: tuple[int, int, int],
    expected: tuple[int, ...],
) -> None:
    input = torch.empty(2, 128, 5, 16, 18, device="meta", dtype=torch.float16)  # noqa: A001
    qdata = torch.empty(32, 3, 3, 3, 128, device="meta", dtype=torch.int8)
    scale = torch.empty(32, device="meta", dtype=torch.float32)

    output = conv3d(
        input,
        qdata,
        scale,
        None,
        64,
        0.01,
        stride,
        padding=padding,  # pyright: ignore[reportArgumentType]
    )

    assert output.shape == expected
    assert output.dtype is torch.float16


def test_conv3d_rejects_non_fp16_input() -> None:
    qdata = torch.empty(32, 3, 3, 3, 128, dtype=torch.int8)
    scale = torch.ones(32, dtype=torch.float32)
    with pytest.raises(ValueError, match="input must be float16"):
        conv3d(
            torch.empty(1, 128, 3, 8, 8),
            qdata,
            scale,
            None,
            64,
            0.01,
            (1, 1, 1),
            padding="reflect",
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _nvidia_cuda_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("padding", "stride", "residual_enabled"),
    [
        ("reflect", (1, 1, 1), True),
        ("reflect_right", (1, 2, 2), False),
    ],
)
def test_triton_conv3d_matches_reference(
    padding: str,
    stride: tuple[int, int, int],
    residual_enabled: bool,
) -> None:
    torch.manual_seed(5678)
    input = torch.randn(1, 128, 3, 16, 16, device="cuda", dtype=torch.float16)  # noqa: A001
    qdata, scale = _weight(128, 128, device="cuda")
    bias = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.01)
    symmetric = padding == "reflect"
    right = padding == "reflect_right"
    spatial = 2 if symmetric else int(right)
    output_height = (input.shape[3] + spatial - 3) // stride[1] + 1
    output_width = (input.shape[4] + spatial - 3) // stride[2] + 1
    residual = (
        torch.randn(
            1,
            128,
            3,
            output_height,
            output_width,
            device="cuda",
            dtype=torch.float16,
        ).mul_(0.1)
        if residual_enabled
        else None
    )

    with torch.no_grad():
        actual = conv3d(
            input,
            qdata,
            scale,
            bias,
            64,
            0.02,
            stride,
            padding=padding,  # pyright: ignore[reportArgumentType]
            residual=residual,
        )
        expected = reference.conv3d(
            input,
            qdata,
            scale,
            bias,
            64,
            0.02,
            stride,
            symmetric_spatial_padding=symmetric,
            right_spatial_padding=right,
            residual=residual,
        )

    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not _nvidia_cuda_available(), reason="requires NVIDIA CUDA")
def test_triton_group_norm_silu_conv3d_matches_reference() -> None:
    torch.manual_seed(9012)
    input = torch.randn(1, 128, 3, 16, 16, device="cuda", dtype=torch.float16)  # noqa: A001
    norm_weight = torch.randn(128, device="cuda")
    norm_bias = torch.randn(128, device="cuda")
    qdata, scale = _weight(128, 128, device="cuda")
    bias = torch.randn(128, device="cuda", dtype=torch.float16).mul_(0.01)

    with torch.no_grad():
        actual = group_norm_silu_conv3d(
            input,
            norm_weight,
            norm_bias,
            32,
            1e-6,
            qdata,
            scale,
            bias,
            64,
            0.02,
            (1, 1, 1),
            padding="reflect",
        )
        expected = reference.group_norm_silu_conv3d(
            input,
            norm_weight,
            norm_bias,
            32,
            1e-6,
            qdata,
            scale,
            bias,
            64,
            0.02,
            (1, 1, 1),
            symmetric_spatial_padding=True,
            right_spatial_padding=False,
            residual=None,
        )

    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)
