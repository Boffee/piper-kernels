"""Projection transform grammar and metadata guards are independent of attention."""

import pytest
import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from piper_kernels.fusions.projected_qk import _compile, _pattern

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _transform_graph(dtype, affine, *, full_rotary=False):
    def transform(features, weight, cos, sin):
        projected = torch.ops.aten.clone.default(features)
        shaped = torch.ops.aten.reshape.default(projected, (1, 192, 2, 128))
        promoted = (
            torch.ops.prims.convert_element_type.default(shaped, torch.float32)
            if dtype is not torch.float32
            else shaped
        )
        squared = torch.ops.aten.pow.Tensor_Scalar(promoted, 2)
        mean = torch.ops.aten.mean.dim(squared, [3], True)
        variance = torch.ops.aten.add.Scalar(mean, 1e-5)
        inverse_rms = torch.ops.aten.rsqrt.default(variance)
        normalized = torch.ops.aten.mul.Tensor(promoted, inverse_rms)
        scaled = torch.ops.aten.mul.Tensor(normalized, weight) if affine else normalized
        rounded = (
            torch.ops.prims.convert_element_type.default(scaled, dtype)
            if dtype is not torch.float32
            else scaled
        )
        rotary = rounded if full_rotary else torch.ops.aten.slice.Tensor(rounded, 3, 0, 96)
        first, second = torch.ops.aten.split.Tensor(rotary, 64 if full_rotary else 48, -1)
        cos_table = (
            torch.ops.prims.convert_element_type.default(cos, dtype)
            if dtype is not torch.float32
            else cos
        )
        cos_table = torch.ops.aten.unsqueeze.default(cos_table, 0)
        cos_table = torch.ops.aten.unsqueeze.default(cos_table, 2)
        direct = torch.ops.aten.mul.Tensor(rotary, cos_table)
        rotated = torch.ops.aten.cat.default([torch.ops.aten.neg.default(second), first], -1)
        sin_table = (
            torch.ops.prims.convert_element_type.default(sin, dtype)
            if dtype is not torch.float32
            else sin
        )
        sin_table = torch.ops.aten.unsqueeze.default(sin_table, 0)
        sin_table = torch.ops.aten.unsqueeze.default(sin_table, 2)
        rotated = torch.ops.aten.mul.Tensor(rotated, sin_table)
        result = torch.ops.aten.add.Tensor(direct, rotated)
        if full_rotary:
            return result
        tail = torch.ops.aten.slice.Tensor(rounded, 3, 96, torch.iinfo(torch.int64).max)
        return torch.ops.aten.cat.default([result, tail], -1)

    return torch.fx.symbolic_trace(transform).graph


def _transform_pattern(dtype, affine, *, full_rotary=False):
    return _pattern.normalized_rope_pattern(
        CallFunction(torch.ops.aten.clone.default, KeywordArg("features"), _users=1),
        shape_name="shape",
        norm_weight_name="weight",
        norm_epsilon_name="epsilon",
        cos_name="cos",
        sin_name="sin",
        rotary_dim_name="width",
        half_rotary_dim_name="half_width",
        activation_dtype=dtype,
        affine=affine,
        full_rotary=full_rotary,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_transform_pattern_uses_caller_captures_for_every_dtype_and_norm(dtype, affine):
    graph = _transform_graph(dtype, affine)
    output = next(node for node in reversed(graph.nodes) if node.op == "call_function")
    match = _transform_pattern(dtype, affine).match(output)
    assert isinstance(match, Match)
    assert set(match.kwargs) == {
        "features",
        "shape",
        "epsilon",
        "cos",
        "sin",
        "width",
        "half_width",
    } | ({"weight"} if affine else set())
    assert match.kwargs["shape"] == (1, 192, 2, 128)
    assert match.kwargs["width"] == 96
    assert match.kwargs["half_width"] == 48


def test_transform_pattern_rejects_an_escaping_normalized_intermediate():
    graph = _transform_graph(torch.bfloat16, True)
    output = next(node for node in reversed(graph.nodes) if node.op == "call_function")
    normalization = next(node for node in graph.nodes if node.target is torch.ops.aten.mul.Tensor)
    graph.call_function(torch.ops.aten.neg.default, (normalization,))
    assert not _transform_pattern(torch.bfloat16, True).match(output)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_transform_pattern_supports_canonical_full_rope(dtype, affine):
    graph = _transform_graph(dtype, affine, full_rotary=True)
    output = next(node for node in reversed(graph.nodes) if node.op == "call_function")
    match = _transform_pattern(dtype, affine, full_rotary=True).match(output)
    assert isinstance(match, Match)
    assert "width" not in match.kwargs
    assert match.kwargs["half_width"] == 64


def _node(graph, name, value):
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


@pytest.mark.parametrize("dtype", [None, *_DTYPES])
def test_rmsnorm_guard_accepts_metadata_without_tensor_contents(dtype):
    graph = torch.fx.Graph()
    weight = (
        _node(graph, "weight", torch.empty(128, device="meta", dtype=dtype))
        if dtype is not None
        else None
    )
    assert _compile.valid_rmsnorm(
        weight,
        1e-5,
        head_dim=128,
        device=torch.device("meta"),
        supported_dtypes=_DTYPES,
    )


@pytest.mark.parametrize("epsilon", [True, 0, -1, float("inf"), float("nan"), None])
def test_rmsnorm_guard_rejects_invalid_python_epsilon(epsilon):
    assert not _compile.valid_rmsnorm(
        None,
        epsilon,
        head_dim=128,
        device=torch.device("meta"),
        supported_dtypes=_DTYPES,
    )


@pytest.mark.parametrize("invalid", ["dtype", "width", "stride", "device", "missing", "node"])
def test_rmsnorm_guard_rejects_invalid_weight_metadata(invalid):
    graph = torch.fx.Graph()
    with torch.device("meta"):
        tensor = torch.empty(128)
        if invalid == "dtype":
            tensor = tensor.to(torch.int8)
        elif invalid == "width":
            tensor = torch.empty(64)
        elif invalid == "stride":
            tensor = torch.empty(256)[::2]
    node = _node(graph, "weight", None if invalid == "missing" else tensor)
    assert not _compile.valid_rmsnorm(
        object() if invalid == "node" else node,
        1e-5,
        head_dim=128,
        device=torch.device("cpu" if invalid == "device" else "meta"),
        supported_dtypes=_DTYPES,
    )


@pytest.mark.parametrize(
    ("rotary_dim", "half_rotary_dim", "valid"),
    [
        (96, 48, True),
        (128, 64, True),
        (2, 1, True),
        (0, 0, False),
        (129, 65, False),
        (95, 48, False),
        (96, 47, False),
        (True, 1, False),
    ],
)
def test_rope_guard_preserves_static_width_checks(rotary_dim, half_rotary_dim, valid):
    graph = torch.fx.Graph()
    width = int(rotary_dim)
    cos = _node(graph, "cos", torch.empty((192, width), device="meta"))
    sin = _node(graph, "sin", torch.empty((192, width), device="meta"))
    assert (
        _compile.valid_rope_tables(
            cos,
            sin,
            rotary_dim,
            half_rotary_dim,
            sequence_length=192,
            head_dim=128,
            device=torch.device("meta"),
        )
        is valid
    )


@pytest.mark.parametrize(
    "invalid", ["dtype", "rows", "width", "stride", "rank", "device", "missing"]
)
def test_rope_guard_rejects_invalid_table_metadata(invalid):
    graph = torch.fx.Graph()
    with torch.device("meta"):
        table = torch.empty((192, 96))
        if invalid == "dtype":
            table = table.to(torch.bfloat16)
        elif invalid == "rows":
            table = torch.empty((193, 96))
        elif invalid == "width":
            table = torch.empty((192, 94))
        elif invalid == "stride":
            table = torch.empty((192, 192))[:, ::2]
        elif invalid == "rank":
            table = torch.empty(192)
    cos = _node(graph, "cos", None if invalid == "missing" else table)
    sin = _node(graph, "sin", torch.empty((192, 96), device="meta"))
    assert not _compile.valid_rope_tables(
        cos,
        sin,
        96,
        48,
        sequence_length=192,
        head_dim=128,
        device=torch.device("cpu" if invalid == "device" else "meta"),
    )


def test_rope_guard_preserves_symbolic_dimensions_without_adding_shape_guards():
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    table = mode.from_tensor(torch.empty((192, 96)), static_shapes=False)
    sequence, width = table.shape
    assert isinstance(sequence, torch.SymInt)
    assert isinstance(width, torch.SymInt)
    graph = torch.fx.Graph()
    cos = _node(graph, "cos", table)
    sin = _node(graph, "sin", table)
    width_node = _node(graph, "width", width)
    half_width_node = _node(graph, "half_width", (width + 1) // 2)
    guards_before = tuple(shape_env.guards)
    assert _compile.valid_rope_tables(
        cos,
        sin,
        width_node,
        half_width_node,
        sequence_length=sequence,
        head_dim=128,
        device=table.device,
    )
    assert tuple(shape_env.guards) == guards_before
    assert _compile.integer_scalar_argument(width_node) is width_node
