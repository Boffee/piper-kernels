"""Static-scale ConvRot INT8 convolution contracts and kernel correctness."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import conv3d, group_norm_silu_conv3d, reference
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor


def _nvidia_cuda_available() -> bool:
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(12, 0)


def _packed(qdata, scale, group_size=64, input_scale=0.02):
    return ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=group_size,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(input_scale, device=qdata.device),
    )


def _weight(
    output_channels: int,
    input_channels: int,
    *,
    device: str,
) -> ConvRotInt8Tensor:
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
    group_size = 64 if input_channels <= 128 else 256
    return ConvRotInt8Tensor.from_hp(
        weight, group_size=group_size, act_per_tensor_scale=torch.tensor(0.02, device=device)
    )


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
        _packed(qdata, scale, input_scale=0.01),
        stride=stride,
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
            _packed(qdata, scale, input_scale=0.01),
            padding="reflect",
        )


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    ("shape", "padding", "message"),
    [
        ((1, 128, 0, 4, 4), "reflect", "dimensions must be positive"),
        ((1, 128, 1, 1, 4), "reflect", "height and width > 1"),
        ((1, 128, 1, 4, 1), "reflect_right", "height and width > 1"),
        ((1, 128, 1, 2, 2), "none", "does not fit"),
        ((1, 768, 1, 4, 4), "reflect", "power-of-two"),
        ((1, 8192, 1, 4, 4), "reflect", "power-of-two"),
    ],
)
def test_eager_and_fake_reject_unsupported_shapes(device, shape, padding, message):
    with pytest.raises(ValueError, match=message):
        conv3d(
            torch.empty(shape, device=device, dtype=torch.float16),
            _packed(
                torch.empty(1, 3, 3, 3, shape[1], device=device, dtype=torch.int8),
                torch.ones(1, device=device),
            ),
            padding=padding,
        )


def test_custom_ops_schema_fake_and_dynamic_compile_contracts():
    input = torch.randn(1, 64, 2, 3, 3, dtype=torch.float16)  # noqa: A001
    qdata = torch.zeros(4, 3, 3, 3, 64, dtype=torch.int8)
    scale = torch.ones(4, 1)
    conv_args = (qdata, scale, None, 64, torch.tensor(0.02), [1, 1, 1], True, False, None)
    torch.library.opcheck(torch.ops.piper_kernels.convrot_int8_conv3d.default, (input, *conv_args))
    torch.library.opcheck(
        torch.ops.piper_kernels.convrot_int8_group_norm_silu_conv3d.default,
        (input, torch.ones(64), torch.zeros(64), 8, 1e-6, *conv_args),
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("gradient", ["input", "weight_scale", "bias", "residual"])
def test_public_convolution_rejects_autograd_before_custom_op_dispatch(fused, gradient):
    tensors = {
        "input": torch.zeros(1, 64, 1, 3, 3, dtype=torch.float16),
        "weight_scale": torch.ones(4),
        "bias": torch.zeros(4, dtype=torch.float16),
        "residual": torch.zeros(1, 4, 1, 3, 3, dtype=torch.float16),
    }
    tensors[gradient].requires_grad_()
    weight = torch.zeros(4, 3, 3, 3, 64, dtype=torch.int8)
    args = (_packed(weight, tensors["weight_scale"]), tensors["bias"])

    def run():
        if fused:
            return group_norm_silu_conv3d(
                tensors["input"],
                torch.ones(64),
                torch.zeros(64),
                8,
                1e-6,
                *args,
                padding="reflect",
                residual=tensors["residual"],
            )
        return conv3d(tensors["input"], *args, padding="reflect", residual=tensors["residual"])

    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        run()
    with torch.no_grad():
        assert not run().requires_grad


@pytest.mark.parametrize("gradient", ["weight", "bias"])
def test_public_fusion_rejects_trainable_normalization(gradient):
    affine = {"weight": torch.ones(64), "bias": torch.zeros(64)}
    affine[gradient].requires_grad_()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        group_norm_silu_conv3d(
            torch.zeros(1, 64, 1, 3, 3, dtype=torch.float16),
            affine["weight"],
            affine["bias"],
            8,
            1e-6,
            _packed(torch.zeros(4, 3, 3, 3, 64, dtype=torch.int8), torch.ones(4)),
            padding="reflect",
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _nvidia_cuda_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("shape", "outputs", "padding", "stride", "residual_enabled"),
    [
        ((1, 128, 3, 16, 16), 128, "reflect", (1, 1, 1), True),
        ((1, 128, 3, 17, 19), 256, "reflect_right", (1, 2, 2), False),
        ((2, 256, 5, 7, 9), 128, "none", (2, 2, 2), True),
        ((1, 512, 3, 5, 7), 65, "reflect", (2, 1, 1), False),
        ((1, 1024, 2, 3, 3), 32, "reflect", (1, 1, 1), True),
        ((1, 64, 1, 2, 2), 7, "reflect", (1, 1, 1), False),
    ],
)
def test_triton_conv3d_matches_reference(
    shape: tuple[int, ...],
    outputs: int,
    padding: str,
    stride: tuple[int, int, int],
    residual_enabled: bool,
) -> None:
    torch.manual_seed(5678)
    batch, channels, frames, height, width = shape
    input = torch.randn(  # noqa: A001
        batch, channels, frames, width, height, device="cuda", dtype=torch.float16
    ).transpose(3, 4)
    weight = _weight(outputs, channels, device="cuda")
    bias = torch.randn(outputs * 2, device="cuda", dtype=torch.float16)[::2].mul_(0.01)
    group_size = 64 if channels <= 128 else 256
    symmetric = padding == "reflect"
    right = padding == "reflect_right"
    spatial = 2 if symmetric else int(right)
    output_height = (input.shape[3] + spatial - 3) // stride[1] + 1
    output_width = (input.shape[4] + spatial - 3) // stride[2] + 1
    residual = (
        torch.randn(
            batch,
            outputs,
            (frames - 1) // stride[0] + 1,
            output_width,
            output_height,
            device="cuda",
            dtype=torch.float16,
        )
        .transpose(3, 4)
        .mul_(0.1)
        if residual_enabled
        else None
    )

    with torch.no_grad():
        actual = conv3d(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,  # pyright: ignore[reportArgumentType]
            residual=residual,
        )
        expected = reference.conv3d(
            input,
            weight.qdata,
            weight.scale,
            bias,
            group_size,
            weight.act_per_tensor_scale,
            stride,
            symmetric_spatial_padding=symmetric,
            right_spatial_padding=right,
            residual=residual,
        )

    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not _nvidia_cuda_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("channels", "height", "width"), [(128, 16, 16), (256, 33, 35), (512, 5, 7), (1024, 3, 3)]
)
def test_triton_group_norm_silu_conv3d_matches_reference(channels, height, width) -> None:
    torch.manual_seed(9012)
    input = torch.randn(  # noqa: A001
        2, channels, 3, width, height, device="cuda", dtype=torch.float16
    ).transpose(3, 4)
    norm_weight = torch.randn(channels * 2, device="cuda")[::2]
    norm_bias = torch.randn(channels * 2, device="cuda")[::2]
    weight = _weight(32, channels, device="cuda")
    bias = torch.randn(64, device="cuda", dtype=torch.float16)[::2].mul_(0.01)
    group_size = 64 if channels <= 128 else 256

    with torch.no_grad():
        actual = group_norm_silu_conv3d(
            input,
            norm_weight,
            norm_bias,
            32,
            1e-6,
            weight,
            bias,
            padding="reflect",
        )
        expected = reference.group_norm_silu_conv3d(
            input,
            norm_weight,
            norm_bias,
            32,
            1e-6,
            weight.qdata,
            weight.scale,
            bias,
            group_size,
            weight.act_per_tensor_scale,
            (1, 1, 1),
            symmetric_spatial_padding=True,
            right_spatial_padding=False,
            residual=None,
        )

    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not _nvidia_cuda_available(), reason="requires NVIDIA CUDA")
def test_group_norm_preserves_small_variance_at_large_frame_offsets():
    from piper_kernels.conv3d.convrot.int8 import triton as backend  # noqa: PLC0415

    torch.manual_seed(31415)
    offsets = torch.tensor([1000.0, -1000.0], device="cuda").view(1, 1, 2, 1, 1)
    activation = (offsets + 0.5 * torch.randn(1, 128, 2, 33, 35, device="cuda")).half()
    weight, bias = torch.ones(128, device="cuda"), torch.zeros(128, device="cuda")
    actual = backend._prepare_group_norm_silu_input(
        activation, weight, bias, 32, 1e-6, 64, torch.tensor(0.02, device="cuda")
    )
    expected = reference._prepare_group_norm_silu_input(
        activation, weight, bias, 32, 1e-6, 64, torch.tensor(0.02, device="cuda")
    )
    # Tiny FP32 reduction differences may cross a quantization rounding boundary.
    torch.testing.assert_close(actual, expected, atol=1, rtol=0)
    assert (actual != expected).float().mean().item() < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not _nvidia_cuda_available(), reason="requires SM120")
def test_unaligned_contiguous_weight_uses_pointer_loads():
    activation = torch.randn(1, 128, 2, 3, 3, device="cuda", dtype=torch.float16)
    weight = _weight(256, 128, device="cuda")
    qdata, scale = weight.qdata, weight.scale
    unaligned = torch.empty(qdata.numel() + 1, device="cuda", dtype=torch.int8)[1:].view_as(qdata)
    unaligned.copy_(qdata)
    assert unaligned.is_contiguous()
    assert unaligned.data_ptr() % 16 != 0
    args = (activation, unaligned, scale, None, 64, weight.act_per_tensor_scale, (1, 1, 1))
    actual = conv3d(activation, _packed(unaligned, scale), padding="reflect")
    expected = reference.conv3d(
        *args, symmetric_spatial_padding=True, right_spatial_padding=False, residual=None
    )
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)
