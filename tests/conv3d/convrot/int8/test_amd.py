"""RDNA4 preparation, exact integer accumulation, and graph-capture regressions."""

import pytest
import torch
from torch.nn import functional

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import _backend, conv3d, group_norm_silu_conv3d, reference
from piper_kernels.conv3d.convrot.int8 import triton as shared
from piper_kernels.conv3d.convrot.int8._amd import policy
from piper_kernels.conv3d.convrot.int8._amd import triton as amd
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not policy.supports_target(AcceleratorTarget.from_device(torch.device("cuda"))),
        reason="requires RDNA4 ROCm",
    ),
]


@pytest.mark.parametrize(
    ("channels", "group_size"),
    [
        (channels, group_size)
        for channels in (64, 128, 256, 512, 1024, 2048, 4096)
        for group_size in (16, 64, 256)
        if group_size <= channels
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_preparation_matches_reference(channels, group_size, dtype):
    torch.manual_seed(19)
    # Exact binary values exercise saturation and nearest-even quantization ties.
    activation = (torch.randint(-32, 33, (2, channels, 3, 5, 7), device="cuda") / 16).to(dtype)
    activation[:, :, 0, 0, 0] = 32
    activation[:, :, 0, 0, 1] = -32
    activation = activation.transpose(3, 4)
    scale = torch.tensor(0.125, device="cuda")
    actual = shared._prepare_input(
        activation, group_size, scale, policy=policy, accelerator_backend="hip"
    )
    expected = reference._prepare_input(activation, group_size, scale)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.min() == -128
    assert actual.max() == 127
    actual = shared._prepare_input(
        torch.zeros_like(activation), group_size, scale, policy=policy, accelerator_backend="hip"
    )
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("channels", [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("padding", ["reflect", "reflect_right", "none"])
def test_prepared_convolution_matches_exact_integer_accumulation(channels, padding):
    torch.manual_seed(23)
    batch, frames, height, width, outputs = 2, 3, 3, 5, 7
    prepared = torch.randint(
        -128, 128, (batch, frames, height, width, channels), device="cuda", dtype=torch.int8
    )
    weight = torch.randint(-128, 128, (outputs, 3, 3, 3, channels), device="cuda", dtype=torch.int8)
    input_scale = torch.tensor(0.003, device="cuda")
    weight_scale = torch.full((outputs, 1), 0.004, device="cuda")
    stride = (2, 2, 1)
    actual = shared._conv3d_prepared(
        prepared,
        weight,
        weight_scale,
        None,
        input_scale,
        stride,
        policy=policy,
        symmetric_spatial_padding=padding == "reflect",
        right_spatial_padding=padding == "reflect_right",
        residual=None,
    )
    padded = prepared.permute(0, 4, 1, 2, 3).float()
    if padding != "none":
        pads = (1, 1, 1, 1, 0, 0) if padding == "reflect" else (0, 1, 0, 1, 0, 0)
        padded = functional.pad(padded, pads, mode="reflect")
    padded = functional.pad(padded, (0, 0, 0, 0, 2, 0)).to(torch.int8)
    windows = padded.unfold(2, 3, stride[0]).unfold(3, 3, stride[1]).unfold(4, 3, stride[2])
    matrix = windows.permute(0, 2, 3, 4, 5, 6, 7, 1).reshape(-1, 27 * channels)
    rows = matrix.shape[0]
    accumulator = torch._int_mm(
        functional.pad(matrix, (0, 0, 0, (-rows) % 32)),
        functional.pad(weight.flatten(1), (0, 0, 0, (-outputs) % 8)).T,
    )[:rows, :outputs]
    expected = accumulator.float() * input_scale * weight_scale.T
    expected = expected.view(batch, *actual.shape[2:], outputs).permute(0, 4, 1, 2, 3).half()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("channels", [64, 2048, 4096])
def test_native_dispatch_and_graph_capture_use_live_scale(monkeypatch, fused, channels):
    torch.manual_seed(31)
    activation = torch.randn(1, channels, 2, 3, 3, device="cuda", dtype=torch.float32)
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(-16, 16, (7, 3, 3, 3, channels), device="cuda", dtype=torch.int8),
        torch.full((7,), 0.001, device="cuda"),
        group_size=64,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(0.02, device="cuda"),
    )
    norm_weight = torch.randn(channels, device="cuda")
    norm_bias = torch.randn(channels, device="cuda")
    assert _backend.select_backend(activation) is amd

    def run():
        if fused:
            return group_norm_silu_conv3d(
                activation, norm_weight, norm_bias, 8, 1e-6, weight, padding="reflect"
            )
        return conv3d(activation, weight, padding="reflect")

    operands = (weight.qdata, weight.scale, None, 64, weight.act_per_tensor_scale, (1, 1, 1))
    flags = {"symmetric_spatial_padding": True, "right_spatial_padding": False, "residual": None}
    expected = (
        reference.group_norm_silu_conv3d(
            activation, norm_weight, norm_bias, 8, 1e-6, *operands, **flags
        )
        if fused
        else reference.conv3d(activation, *operands, **flags)
    )
    torch.testing.assert_close(run(), expected, atol=4e-3, rtol=4e-3)

    def reject_fallback(*args, **kwargs):
        pytest.fail("native convolution fell back to the portable reference")

    monkeypatch.setattr(reference, "conv3d", reject_fallback)
    monkeypatch.setattr(reference, "group_norm_silu_conv3d", reject_fallback)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    graph.replay()
    torch.testing.assert_close(captured, run(), atol=0, rtol=0)
    before = captured.clone()
    weight.act_per_tensor_scale.fill_(0.05)
    graph.replay()
    torch.testing.assert_close(captured, run(), atol=0, rtol=0)
    assert not torch.equal(before, captured)


@pytest.mark.parametrize("fused", [False, True])
def test_dynamic_fullgraph_convolution(fused):
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(-8, 8, (7, 3, 3, 3, 64), device="cuda", dtype=torch.int8),
        torch.full((7,), 0.001, device="cuda"),
        group_size=64,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(0.02, device="cuda"),
    )
    norm_weight, norm_bias = torch.ones(64, device="cuda"), torch.zeros(64, device="cuda")

    def run(activation):
        if fused:
            return group_norm_silu_conv3d(
                activation, norm_weight, norm_bias, 8, 1e-6, weight, padding="reflect"
            )
        return conv3d(activation, weight, padding="reflect")

    compiled = torch.compile(run, dynamic=True, fullgraph=True)
    for frames, height, width in ((2, 5, 7), (3, 7, 9)):
        activation = torch.randn(1, 64, frames, height, width, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(compiled(activation), run(activation), atol=0, rtol=0)
