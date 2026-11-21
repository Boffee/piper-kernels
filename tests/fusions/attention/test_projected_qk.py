"""Projection transform grammar, head-width validation, and metadata guards."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from piper_kernels.fusions.attention import _projected_qk_compile, _validation
from piper_kernels.fusions.convrot_int8_sparse_piper import key as int8_key
from piper_kernels.fusions.convrot_int8_sparse_piper import query as int8_query
from piper_kernels.fusions.nvfp4_sparse_piper import key as nvfp4_key
from piper_kernels.fusions.nvfp4_sparse_piper import query as nvfp4_query

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


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_transform_matcher_captures_every_dtype_and_norm(dtype, affine):
    graph = _transform_graph(dtype, affine)
    output = next(node for node in reversed(graph.nodes) if node.op == "call_function")
    projected = next(node for node in graph.nodes if node.target is torch.ops.aten.clone.default)
    match = _projected_qk_compile.match_normalized_rope(output)
    assert isinstance(match, _projected_qk_compile.NormalizedRopeMatch)
    assert match.projection_node is projected
    assert match.shape == (1, 192, 2, 128)
    assert match.rotary_dim == 96
    assert match.half_rotary_dim == 48
    assert match.norm_epsilon == 1e-5
    assert (match.norm_weight is not None) is affine
    assert match.activation_dtype is (None if dtype is torch.float32 else dtype)


def test_transform_matcher_rejects_an_escaping_normalized_intermediate():
    graph = _transform_graph(torch.bfloat16, True)
    output = next(node for node in reversed(graph.nodes) if node.op == "call_function")
    normalization = next(node for node in graph.nodes if node.target is torch.ops.aten.mul.Tensor)
    graph.call_function(torch.ops.aten.neg.default, (normalization,))
    assert _projected_qk_compile.match_normalized_rope(output) is None


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_transform_matcher_supports_canonical_full_rope(dtype, affine):
    graph = _transform_graph(dtype, affine, full_rotary=True)
    output = next(node for node in reversed(graph.nodes) if node.op == "call_function")
    projected = next(node for node in graph.nodes if node.target is torch.ops.aten.clone.default)
    match = _projected_qk_compile.match_normalized_rope(output)
    assert isinstance(match, _projected_qk_compile.NormalizedRopeMatch)
    assert match.projection_node is projected
    assert match.rotary_dim is None
    assert match.half_rotary_dim == 64
    assert (match.norm_weight is not None) is affine
    assert match.activation_dtype is (None if dtype is torch.float32 else dtype)


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
    assert _projected_qk_compile.valid_rmsnorm(
        weight,
        1e-5,
        head_dim=128,
        device=torch.device("meta"),
        supported_dtypes=_DTYPES,
    )


@pytest.mark.parametrize("epsilon", [True, 0, -1, float("inf"), float("nan"), None])
def test_rmsnorm_guard_rejects_invalid_python_epsilon(epsilon):
    assert not _projected_qk_compile.valid_rmsnorm(
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
    assert not _projected_qk_compile.valid_rmsnorm(
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
        _projected_qk_compile.valid_rope_tables(
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
    assert not _projected_qk_compile.valid_rope_tables(
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
    assert _projected_qk_compile.valid_rope_tables(
        cos,
        sin,
        width_node,
        half_width_node,
        sequence_length=sequence,
        head_dim=128,
        device=table.device,
    )
    assert tuple(shape_env.guards) == guards_before
    assert _projected_qk_compile.integer_scalar_argument(width_node) is width_node


@pytest.mark.parametrize("head_dim", [64, 128])
def test_head_width_can_be_inferred_only_from_an_affine_norm(head_dim):
    norm = torch.empty(head_dim)
    assert _validation.resolve_head_dim(norm, None) == head_dim
    assert _validation.resolve_head_dim(norm, head_dim) == head_dim
    assert _validation.resolve_head_dim(None, head_dim) == head_dim


@pytest.mark.parametrize(
    ("shape", "head_dim", "message"),
    [
        (None, None, "explicit head_dim"),
        (None, True, "head_dim=64 or 128"),
        (None, 32, "head_dim=64 or 128"),
        (None, 64.0, "head_dim=64 or 128"),
        ((64,), 128, "must match head_dim"),
        ((2, 64), 64, "must match head_dim"),
        ((2, 64), None, "head_dim=64 or 128"),
    ],
)
def test_invalid_or_ambiguous_head_width_is_rejected(shape, head_dim, message):
    norm = torch.empty(shape) if shape is not None else None
    with pytest.raises(ValueError, match=message):
        _validation.resolve_head_dim(norm, head_dim)


@pytest.mark.parametrize("family", ["int8", "nvfp4"])
@pytest.mark.parametrize("operation", ["query", "key"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_weightless_projection_fake_kernel_keeps_the_explicit_head_width(
    family, operation, head_dim
):
    rows, heads, width = 193, 2, 256
    with torch.device("meta"):
        if family == "int8":
            prepared = (
                torch.empty((1, rows, width), dtype=torch.int8),
                torch.empty((1, rows), dtype=torch.float32),
                torch.empty((heads * head_dim, width), dtype=torch.int8),
                torch.empty((heads * head_dim, 1), dtype=torch.float32),
            )
            op = int8_query._project_query_op if operation == "query" else int8_key._project_key_op
        else:
            prepared = (
                torch.empty((rows, width // 2), dtype=torch.uint8),
                torch.empty((64, 64), dtype=torch.float8_e4m3fn),
                torch.empty((), dtype=torch.float32),
                torch.empty((heads * head_dim, width // 2), dtype=torch.uint8),
                torch.empty((64, 64), dtype=torch.float8_e4m3fn),
                None,
                None,
            )
            op = nvfp4_query.project_query if operation == "query" else nvfp4_key.project_key
        cos = torch.empty((rows, head_dim // 2), dtype=torch.float32)
        sin = torch.empty_like(cos)
    policy = (head_dim**-0.5,) if operation == "query" else ()
    policy += (128, 0) if family == "nvfp4" else (0,)
    args = (*prepared, None, cos, sin, 1e-5, *policy)
    with pytest.raises(ValueError, match="explicit head_dim"):
        op(*args)
    output = op(*args, head_dim=head_dim)
    assert output[0].shape == (1, heads, 256, head_dim)
    assert output[0].dtype is torch.int8
    if operation == "key":
        assert output[2].shape == (1, heads, 4, 2, head_dim)
        assert output[4].shape == (1, heads, 4, head_dim)
    else:
        assert output[2].shape == (1, heads, 4, head_dim)


def test_symbolic_head_width_is_validated_during_fake_execution():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    norm = mode.from_tensor(torch.empty(128), static_shapes=False)
    head_dim = norm.shape[0]
    assert isinstance(head_dim, torch.SymInt)
    assert _validation.resolve_head_dim(norm, None) == 128
    assert _validation.resolve_head_dim(None, head_dim) == 128
