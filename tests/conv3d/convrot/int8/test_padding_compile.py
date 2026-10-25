"""Padding fusion is optional and preserves ordinary Conv3D geometry/numerics."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn import functional

from piper_kernels.conv3d.convrot.int8 import convrot_int8_conv3d_compile_options
from piper_kernels.conv3d.convrot.int8._compile import compile_pass, fuse_padding
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_CONV = torch.ops.piper_kernels.convrot_int8_conv3d.default


def _weight(device="cpu"):
    return ConvRotInt8Tensor.from_quantized(
        torch.randint(-8, 8, (16, 3, 3, 3, 64), device=device, dtype=torch.int8),
        torch.full((16,), 0.003, device=device),
        group_size=64,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(0.025, device=device),
    )


def _padded_conv(x, weight):
    x = functional.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
    x = functional.pad(x, (0, 0, 0, 0, 2, 0))
    return functional.conv3d(x, weight, stride=(2, 1, 2))


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize(
    "pattern", ["causal", "zero", "reflection", "shared", "nonzero", "crop", "mixed"]
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_padding_rewrite_preserves_outputs_and_unmatched_operations(pattern, device, static):
    def run(x, weight):
        if pattern == "causal":
            return _padded_conv(x, weight)
        if pattern == "zero":
            x = functional.pad(x, (1, 0, 0, 2, 0, 1))
            x = functional.pad(x, (0, 2, 1, 0, 2, 0))
        elif pattern == "reflection":
            x = functional.pad(x, (2, 1, 1, 2, 0, 0), mode="reflect")
        elif pattern == "mixed":
            x = functional.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
            x = functional.pad(x, (1, 1, 1, 1, 0, 0))
        else:
            pads = (-1, 0, 0, 0, 0, 0) if pattern == "crop" else (0, 0, 0, 0, 2, 0)
            x = functional.pad(x, pads, value=0.5 if pattern == "nonzero" else 0)
        output = functional.conv3d(x, weight)
        return (output, x) if pattern == "shared" else output

    x = torch.randn(1, 64, 4, 5, 6, device=device)
    weight = _weight(device)
    with torch.no_grad():
        gm = torch._dynamo.export(run, assume_static_by_default=static)(x, weight).graph_module
        expected = gm(x, weight)
        fuse_padding(gm.graph)
        gm.recompile()
        torch.testing.assert_close(gm(x, weight), expected, atol=0, rtol=0)
        pads = [n for n in gm.graph.nodes if n.target is torch._C._nn.pad]
        assert len(pads) == (0 if pattern in ("causal", "zero", "reflection") else 1)
        assert sum(n.target == _CONV for n in gm.graph.nodes) == 1


@pytest.mark.parametrize("scope", ["outer", "inner", "disabled"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_padding_fusion_preserves_autocast_rounding(device, scope):
    def run(x, weight, bias):
        def convolution():
            x_pad = functional.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
            x_pad = functional.pad(x_pad, (0, 0, 0, 0, 2, 0))
            return functional.conv3d(x_pad, weight, bias)

        if scope == "outer":
            return convolution()
        with torch.autocast(device, dtype=torch.float16, enabled=scope == "inner"):
            return convolution()

    x = torch.randn(1, 64, 2, 4, 6, device=device)
    bias = torch.randn(16, device=device)
    weight = _weight(device)
    compiled = torch.compile(run, fullgraph=True, options=convrot_int8_conv3d_compile_options())
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=scope != "inner"):
        expected = run(x, weight, bias)
        torch.testing.assert_close(compiled(x, weight, bias), expected, atol=0, rtol=0)
        gm = torch._dynamo.export(run, assume_static_by_default=True)(x, weight, bias).graph_module
        fuse_padding(gm.graph)
        assert all(n.target is not torch._C._nn.pad for n in gm.graph.nodes)


def test_padding_options_composition_and_no_float_weight_rewrite():
    def existing(graph):
        pass

    options = convrot_int8_conv3d_compile_options({"pre_grad_custom_pass": existing})
    assert convrot_int8_conv3d_compile_options(options)["pre_grad_custom_pass"] == (
        existing,
        compile_pass,
    )
    with torch.no_grad():
        gm = torch._dynamo.export(_padded_conv, assume_static_by_default=True)(
            torch.randn(1, 64, 2, 4, 4), torch.randn(16, 64, 3, 3, 3)
        ).graph_module
        original = str(gm.graph)
        fuse_padding(gm.graph)
        assert str(gm.graph) == original


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_standard_conv3d_cuda_graph_capture():
    x, weight = torch.randn(1, 64, 2, 4, 4, device="cuda"), _weight("cuda")
    with torch.no_grad():
        _padded_conv(x, weight)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _padded_conv(x, weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _padded_conv(x, weight)
        graph.replay()
        torch.testing.assert_close(captured, _padded_conv(x, weight), atol=0, rtol=0)


@pytest.mark.parametrize("symbolic", [False, True])
def test_padding_fusion_requires_proven_reflection_bounds(symbolic):
    shape_env = ShapeEnv()
    width = shape_env.create_unbacked_symint() if symbolic else 2
    with FakeTensorMode(shape_env=shape_env):
        value = torch.empty(1, 64, 3, 4, width)
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["example_value"] = value
    weight = graph.placeholder("weight")
    padded = graph.call_function(functional.pad, (source, (2, 0, 0, 0, 0, 0), "reflect"))
    conv = graph.call_function(
        _CONV, (padded, weight, weight, None, 64, weight, [1, 1, 1], [0] * 6, False, None)
    )
    graph.output(conv)
    guards = list(shape_env.guards)
    deferred_asserts = dict(shape_env.deferred_runtime_asserts)
    with torch.no_grad():
        fuse_padding(graph)
    # An invalid or unknown bound cannot become valid just because the tracing
    # example happens to fit. Matching must not introduce shape guards/asserts.
    assert conv.args[0] is padded
    assert shape_env.guards == guards
    assert shape_env.deferred_runtime_asserts == deferred_asserts
