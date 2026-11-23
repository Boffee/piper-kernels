"""Projection normalization preserves shapes, metadata, and external consumers."""

import pytest
import torch
from _accelerators import exact_sm120_available
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from piper_kernels import compile_options
from piper_kernels.linear._projection_views import normalize_projection_views
from piper_kernels.linear.convrot.int8 import _ops as int8_ops
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_nvfp4_ops
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops

_TARGETS = [
    pytest.param(int8_ops.linear, id="convrot_int8"),
    pytest.param(nvfp4_ops.linear, id="nvfp4"),
    pytest.param(convrot_nvfp4_ops.linear, id="convrot_nvfp4"),
]


@pytest.mark.parametrize("device", ["meta", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_mixed_linear_formats_share_one_batched_input_cast(device):
    if device == "cuda" and not exact_sm120_available():
        pytest.skip("CUDA execution requires SM120 for NVFP4")

    def projections(source, int8_weight, int8_scale, fp4_weight, fp4_scale):
        def project(target, *operands):
            cast = torch.ops.prims.convert_element_type.default(source, torch.bfloat16)
            return target(cast.flatten(0, 1), *operands).reshape(2, 16, 128)

        fp4_args = (fp4_weight, fp4_scale, None, None, None, True)
        return (
            project(int8_ops.linear, int8_weight, int8_scale, None, 64),
            project(nvfp4_ops.linear, *fp4_args),
            project(convrot_nvfp4_ops.linear, *fp4_args, 64),
        )

    with torch.device(device):
        args = (
            torch.randn(2, 16, 256),
            torch.randint(-127, 128, (128, 256), dtype=torch.int8),
            torch.full((128, 1), 0.01),
            torch.randint(256, (128, 128), dtype=torch.uint8),
            torch.ones(32, 64).to(torch.float8_e4m3fn),
        )
    module = make_fx(projections, tracing_mode="fake")(*args)
    (linear_pass,) = compile_options()["post_grad_custom_pre_pass"]
    linear_pass(module.graph, is_inference=True)
    module.graph.lint()
    (cast,) = module.graph.find_nodes(
        op="call_function", target=torch.ops.prims.convert_element_type.default
    )
    (output,) = module.graph.find_nodes(op="output")
    assert set(cast.users) == set(output.args[0])
    module.recompile()
    torch.testing.assert_close(module(*args), projections(*args), rtol=0, atol=0)


def _graph(target, source_shape=(2, 3, 64), *, reshape=torch.ops.aten.reshape.default):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(source_shape, dtype=torch.bfloat16)
    weight = graph.placeholder("weight")
    scale = graph.placeholder("scale")
    rows = source.meta["val"].numel() // source_shape[-1]
    flattened = graph.call_function(reshape, (source, (-1, source_shape[-1])))
    flattened.meta["val"] = source.meta["val"].reshape(rows, source_shape[-1])
    if target is int8_ops.linear:
        args = (flattened, weight, scale, None, 64)
    else:
        args = (flattened, weight, scale, None, None, None, True)
        args += (64, True) if target is convrot_nvfp4_ops.linear else (True,)
    linear = graph.call_function(target._opoverload, args)
    linear.meta["val"] = torch.empty((rows, 128), dtype=torch.bfloat16)
    leading_dimensions = tuple(
        graph.call_function(torch.ops.aten.sym_size.int, (source, index))
        if isinstance(size, torch.SymInt)
        else size
        for index, size in enumerate(source_shape[:-1])
    )
    restored = graph.call_function(reshape, (linear, (*leading_dimensions, 128)))
    restored.meta["val"] = linear.meta["val"].reshape(*source_shape[:-1], 128)
    restored.meta["eager_input_vals"] = (linear.meta["val"], restored.args[1])
    graph.output(restored)
    return graph, source, flattened, linear, restored


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("shape", [(6, 64), (2, 3, 64), (2, 3, 4, 64), (0, 3, 64)])
@pytest.mark.parametrize(
    "reshape",
    [
        pytest.param(torch.ops.aten.reshape.default, id="reshape"),
        pytest.param(torch.ops.aten.view.default, id="view"),
        pytest.param(torch.ops.aten._unsafe_view.default, id="unsafe_view"),
    ],
)
def test_normalization_preserves_projection_operands_and_output_metadata(target, shape, reshape):
    graph, source, flattened, linear, restored = _graph(target, shape, reshape=reshape)
    operands = linear.args[1:]
    output_value = restored.meta["val"]

    assert normalize_projection_views(graph)
    assert linear.args == (source, *operands)
    assert linear.meta["val"] is output_value
    assert "eager_input_vals" not in linear.meta
    assert list(graph.nodes)[-1].args == (linear,)
    assert flattened not in graph.nodes
    assert restored not in graph.nodes
    assert not normalize_projection_views(graph)
    graph.lint()


@pytest.mark.parametrize("target", _TARGETS)
def test_symbolic_leading_dimensions_do_not_add_guards(target):
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        batch, tokens = (shape_env.create_unbacked_symint() for _ in range(2))
        graph, source, _, linear, restored = _graph(target, (batch, tokens, 64))
        output_value = restored.meta["val"]
        guards = tuple(shape_env.guards)
        assert normalize_projection_views(graph)
        assert linear.args[0] is source
        assert linear.meta["val"] is output_value
        assert tuple(shape_env.guards) == guards


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("consumer", ["input_view", "linear", "restored"])
def test_external_consumers_keep_their_shapes(target, consumer):
    graph, source, flattened, linear, restored = _graph(target)
    exposed = {"input_view": flattened, "linear": linear, "restored": restored}[consumer]
    output = list(graph.nodes)[-1]
    with graph.inserting_before(output):
        extra = graph.call_function(torch.ops.aten.sigmoid.default, (exposed,))
    output.args = ((restored, extra),)
    shape = exposed.meta["val"].shape

    assert normalize_projection_views(graph) == (consumer != "linear")
    assert extra.args[0].meta["val"].shape == shape
    if consumer == "input_view":
        assert extra.args[0] is flattened
        assert linear.args[0] is source
    elif consumer == "restored":
        assert extra.args[0] is linear
        assert len(linear.users) == 2
    graph.lint()


@pytest.mark.parametrize(
    "failure", ["feature_axis", "leading_order", "noncontiguous", "dtype", "missing_metadata"]
)
def test_unsupported_reshape_layouts_remain_unchanged(failure):
    graph, source, flattened, linear, restored = _graph(int8_ops.linear)
    if failure == "feature_axis":
        flattened.args = (source, (3, 128))
        flattened.meta["val"] = source.meta["val"].reshape(3, 128)
    elif failure == "leading_order":
        restored.args = (linear, (3, 2, 128))
        restored.meta["val"] = linear.meta["val"].reshape(3, 2, 128)
    elif failure == "noncontiguous":
        source.meta["val"] = torch.empty(3, 2, 64, dtype=torch.bfloat16).transpose(0, 1)
    elif failure == "dtype":
        restored.meta["val"] = restored.meta["val"].float()
    else:
        del flattened.meta["val"]
    original = str(graph)
    assert not normalize_projection_views(graph)
    assert str(graph) == original


def test_attention_head_reshape_is_preserved():
    graph, _, _, linear, restored = _graph(int8_ops.linear)
    output = list(graph.nodes)[-1]
    with graph.inserting_before(output):
        heads = graph.call_function(torch.ops.aten.reshape.default, (restored, (2, 3, 2, 64)))
    heads.meta["val"] = restored.meta["val"].reshape(2, 3, 2, 64)
    output.args = (heads,)
    assert normalize_projection_views(graph)
    assert heads.args == (linear, (2, 3, 2, 64))
    graph.lint()
