"""Standard Conv3D geometry and tensor dispatch must not assume causal padding."""

import pytest
import torch
from torch.nn import functional

from piper_kernels.conv3d.convrot.int8 import conv3d, reference
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ],
    ),
]


def _weight(device="cpu"):
    return ConvRotInt8Tensor.from_quantized(
        torch.randint(-8, 8, (16, 3, 3, 3, 64), device=device, dtype=torch.int8),
        torch.full((16,), 0.003, device=device),
        group_size=64,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(0.025, device=device),
    )


def _oracle(x, weight, bias=None, *, stride=1, padding=0):
    # Ordinary floating convolution on independently prepared quantized operands.
    # Its geometry and output shape are supplied by PyTorch, not Piper's fake/reference shape code.
    a = reference._prepare_input(x, 64, weight.act_per_tensor_scale)
    a = a.permute(0, 4, 1, 2, 3).float() * weight.act_per_tensor_scale
    w = (weight.qdata.float() * weight.scale.view(-1, 1, 1, 1, 1)).permute(0, 4, 1, 2, 3)
    with torch.autocast(x.device.type, enabled=False):
        return functional.conv3d(
            a, w, None if bias is None else bias.float(), stride, padding
        ).half()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("stride", "padding"),
    [
        (1, 0),
        ((2, 1, 2), (1, 2, 0)),
        (1, "same"),
        ((3, 2, 1), "valid"),
    ],
)
def test_standard_conv3d_matches_quantized_oracle(device, stride, padding):
    torch.manual_seed(177)
    weight = _weight(device)
    x = torch.randn(1, 64, 5, 7, 6, device=device).transpose(3, 4)
    bias = torch.randn(32, device=device)[::2] * 0.01
    pointers = [t.data_ptr() for t in (weight.qdata, weight.scale, weight.act_per_tensor_scale)]
    with torch.no_grad():
        actual = functional.conv3d(
            input=x, weight=weight, bias=bias, stride=stride, padding=padding
        )
        expected = _oracle(x, weight, bias, stride=stride, padding=padding)
    assert actual.dtype == torch.float16
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert pointers == [
        t.data_ptr() for t in (weight.qdata, weight.scale, weight.act_per_tensor_scale)
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("stride", "spatial"),
    [
        ((1, 1, 1), "reflect"),
        ((2, 2, 2), "reflect_right"),
        ((3, 1, 2), "none"),
    ],
)
def test_explicit_h3_padding_matches_causal_api(device, stride, spatial):
    weight = _weight(device)
    x = torch.randn(1, 64, 4, 5, 7, device=device, dtype=torch.float16)
    padded = x
    if spatial != "none":
        pads = (1, 1, 1, 1, 0, 0) if spatial == "reflect" else (0, 1, 0, 1, 0, 0)
        padded = functional.pad(x, pads, mode="reflect")
    padded = functional.pad(padded, (0, 0, 0, 0, 2, 0))
    with torch.inference_mode():
        actual = functional.conv3d(padded, weight, stride=stride, padding=0)
        expected = conv3d(x, weight, stride=stride, padding=spatial)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape[2] == (x.shape[2] - 1) // stride[0] + 1


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize(
    "operation", ["functional", "module", "conv3d", "padding", "convolution", "_convolution"]
)
def test_standard_dispatch_routes(operation, inference):
    weight = _weight()
    x = torch.randn(1, 64, 4, 5, 5, dtype=torch.float16)
    args = (x, weight, None, [1, 1, 1], [0, 0, 0], [1, 1, 1])
    with torch.no_grad(), torch.inference_mode(inference):
        if operation == "functional":
            actual = functional.conv3d(*args, 1)
        elif operation == "module":
            layer = torch.nn.Conv3d(64, 16, 3, bias=False, device="meta")
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            actual = layer(x)
            assert layer.weight.qdata.data_ptr() == weight.qdata.data_ptr()
        elif operation == "conv3d":
            actual = torch.ops.aten.conv3d.default(*args, 1)
        elif operation == "padding":
            actual = torch.ops.aten.conv3d.padding(
                x, weight, None, [1, 1, 1], "valid", [1, 1, 1], 1
            )
        elif operation == "convolution":
            actual = torch.ops.aten.convolution.default(*args, False, [0, 0, 0], 1)
        else:
            actual = torch.ops.aten._convolution.default(
                *args, False, [0, 0, 0], 1, False, False, True, True
            )
    torch.testing.assert_close(actual, _oracle(x, weight), atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_standard_autocast_casts_input_and_bias_without_repacking_weight(device):
    weight = _weight(device)
    x = torch.randn(1, 64, 3, 4, 5, device=device)
    bias = torch.randn(16, device=device)
    with torch.no_grad():
        expected = functional.conv3d(x.half(), weight, bias.half(), padding=1)
        with torch.autocast(device, dtype=torch.float16):
            actual = functional.conv3d(x, weight, bias, padding=1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert weight.qdata.dtype == torch.int8
    assert weight.scale.dtype == weight.act_per_tensor_scale.dtype == torch.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"groups": 2}, "groups=1"),
        ({"dilation": 2}, "dilation=1"),
        ({"padding": -1}, "padding"),
        ({"stride": 0}, "stride"),
        ({"padding": "same", "stride": 2}, "requires stride=1"),
    ],
)
def test_standard_rejects_unsupported_configurations(kwargs, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        functional.conv3d(torch.randn(1, 64, 3, 4, 4), _weight(), **kwargs)


def test_standard_rejects_transposed_convolution_and_autograd():
    weight = _weight()
    x = torch.randn(1, 64, 3, 4, 4)
    with pytest.raises(NotImplementedError, match="transposed"):
        torch.ops.aten.convolution.default(
            x, weight, None, [1] * 3, [0] * 3, [1] * 3, True, [0] * 3, 1
        )
    with pytest.raises(RuntimeError, match="inference-only"):
        functional.conv3d(x.requires_grad_(), weight)
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="float16 autocast"),
    ):
        functional.conv3d(x, weight)


def test_standard_fake_shape_and_too_short_input():
    weight = _weight("meta")
    x = torch.empty(2, 64, 7, 9, 11, device="meta")
    assert functional.conv3d(x, weight, stride=(2, 3, 2), padding=(0, 1, 2)).shape == (
        2,
        16,
        3,
        3,
        7,
    )
    with pytest.raises(ValueError, match="does not fit"):
        functional.conv3d(x[:, :, :2], weight, padding=0)


def test_mmap_weight_in_standard_module_keeps_storage(tmp_path):
    weight = _weight()
    path = tmp_path / "conv.pt"
    torch.save({"weight": weight}, path)
    with torch.serialization.safe_globals([ConvRotInt8Tensor]):
        state = torch.load(path, weights_only=True, mmap=True)
    layer = torch.nn.Conv3d(64, 16, 3, bias=False, device="meta")
    layer.load_state_dict(state, assign=True)
    layer.requires_grad_(False)
    with torch.no_grad():
        layer(torch.randn(1, 64, 3, 4, 4))
    for name in ("qdata", "scale", "act_per_tensor_scale"):
        assert getattr(layer.weight, name).data_ptr() == getattr(state["weight"], name).data_ptr()
        assert not getattr(layer.weight, name).untyped_storage().resizable()


@pytest.mark.parametrize("device", DEVICES)
def test_standard_compile_dynamic_shapes_without_fusion(device):
    weight = _weight(device)

    def run(x, w):
        return functional.conv3d(x, w, stride=(2, 1, 2), padding=(1, 0, 1))

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    with torch.no_grad():
        for frames in (3, 7):
            x = torch.randn(1, 64, frames, 4, 6, device=device)
            torch.testing.assert_close(compiled(x, weight), run(x, weight), atol=0, rtol=0)
