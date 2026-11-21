"""Sparse fusion owns tensor contracts; selected operations own execution."""

import ast
import builtins
import importlib.util
import operator
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import triton
from _accelerators import rocm_fences_only_lds
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.attention import _output as attention_output
from piper_kernels.fusions.convrot_int8_projection import _validation as qk_validation
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    _backend,
    _compile,
    _kernels,
    _output_compile,
    key,
    output,
    query,
    value,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import triton as projection
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import gluon as amd_gluon
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import policy as amd_policy
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import triton as amd
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import resolve_row_block_launches
from piper_kernels.fusions.convrot_int8_sparse_piper._nvidia import dispatch as nvidia
from piper_kernels.fusions.nvfp4_sparse_piper import _output as nvfp4_output
from piper_kernels.fusions.sparse_piper import _output as output_common


def _operands(sequence=193, heads=3, head_dim=128, input_features=272):
    return (
        torch.empty((2, sequence, input_features), device="cuda:1", dtype=torch.int8),
        torch.empty((2, sequence), device="cuda:1"),
        torch.empty((heads * head_dim, input_features), device="cuda:1", dtype=torch.int8),
        torch.empty((heads * head_dim, 1), device="cuda:1"),
        torch.empty(head_dim, device="cuda:1", dtype=torch.bfloat16),
        torch.empty((sequence, head_dim * 3 // 4), device="cuda:1"),
        torch.empty((sequence, head_dim * 3 // 4), device="cuda:1"),
    )


def _call(
    operation,
    operands,
    routing=_MINMAX_ROUTING,
    *,
    bias=None,
    value_format=ValueFormat.TRANSPOSED,
):
    head_dim = operands[4].shape[0]
    if operation == "query":
        return query._launch_query_projection_range(
            *operands, 1e-6, head_dim**-0.5, routing, chunk_start=64, chunk_rows=129, bias=bias
        )
    if operation == "key":
        return key._launch_key_projection(*operands, 1e-6, routing, bias=bias)
    qdata, scale, weight, weight_scale, *_ = operands
    return value._launch_value_projection(
        qdata,
        scale,
        weight,
        weight_scale,
        None,
        value_format=int(value_format),
        head_dim=head_dim,
        bias=bias,
    )


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cuda", "sm121"),
        AcceleratorTarget("cuda", "sm89"),
        AcceleratorTarget("cuda", "sm86"),
        AcceleratorTarget("cuda", "sm80"),
        AcceleratorTarget("cuda", "sm90"),
        AcceleratorTarget("hip", "gfx1201"),
        AcceleratorTarget("hip", "gfx1200"),
        AcceleratorTarget("hip", "gfx1100"),
        AcceleratorTarget("cpu"),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_projection_selection_uses_operand_target_and_keeps_support_closed(
    monkeypatch, target, head_dim, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(
        torch.cuda, "current_device", Mock(side_effect=AssertionError("current GPU"))
    )
    operand = SimpleNamespace(device=torch.device("cuda:1"))
    expected = None
    if target.is_cuda_capability(12, 0) or target.is_cuda_capability(8):
        expected = nvidia
    elif (
        platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    ):
        expected = amd
    assert _backend.select_projection_backend(operand, head_dim=head_dim) is expected
    probe.assert_called_once_with(operand.device)


def test_missing_projection_backend_does_not_probe_device(monkeypatch):
    monkeypatch.setattr(_backend, "_nvidia_projection", None)
    monkeypatch.setattr(_backend, "_amd_projection", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    assert _backend.select_projection_backend(torch.empty(1)) is None


@pytest.mark.parametrize("missing", ["triton", "triton.language.extra.cuda", "triton_extra", None])
@pytest.mark.parametrize("vendor", ["_nvidia", "_amd"])
def test_projection_import_only_tolerates_absent_top_level_triton(monkeypatch, missing, vendor):
    original_import = builtins.__import__
    error = ModuleNotFoundError("missing dependency", name=missing)

    def import_with_missing_dependency(
        name,
        globals=None,  # noqa: A002 - match the builtin import signature
        locals=None,  # noqa: A002
        fromlist=(),
        level=0,
    ):
        implementation = "dispatch" if vendor == "_nvidia" else "triton"
        if name == vendor and fromlist == (implementation,) and level == 1:
            raise error
        return original_import(name, globals, locals, fromlist, level)

    spec = importlib.util.spec_from_file_location(_backend.__name__, _backend.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    if missing == "triton":
        spec.loader.exec_module(module)
        assert getattr(module, f"{vendor}_projection") is None
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            spec.loader.exec_module(module)
        assert caught.value is error


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("routing", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_projection_facades_forward_shared_buffers_without_execution_plans(
    monkeypatch, operation, routing
):
    execute = Mock()
    backend = SimpleNamespace(**{f"project_{operation}": execute})
    select = Mock(return_value=backend)
    monkeypatch.setattr(_backend, "require_projection_backend", select)
    with FakeTensorMode():
        operands = _operands()
        result = _call(operation, operands, routing)
    select.assert_called_once_with(operands[0], head_dim=128)
    execute.assert_called_once()
    assert execute.call_args.args[0] is operands[0]
    assert all(
        left is right for left, right in zip(result, execute.call_args.kwargs["out"], strict=True)
    )
    assert result[0].dtype is torch.int8
    assert (
        set(execute.call_args.kwargs)
        == {
            "query": {"out", "chunk_start", "chunk_rows", "bias"},
            "key": {"out", "bias"},
            "value": {"out", "bias", "value_format"},
        }[operation]
    )
    if operation == "query":
        assert all(tensor.dtype is torch.float32 for tensor in result[1:])
        assert result[0].shape == (2, 3, 192, 128)
        assert result[1].shape == (2, 3, 6)
        assert result[2].shape == (2, 3, 3, 128)
        assert execute.call_args.kwargs["chunk_start"] == 64
        assert execute.call_args.kwargs["chunk_rows"] == 129
    elif operation == "key":
        assert [tensor.dtype for tensor in result] == [
            torch.int8,
            torch.float32,
            torch.int8,
            torch.float32,
            torch.float32,
            torch.float32,
        ]
        assert result[0].shape == (2, 3, 256, 128)
        assert result[2].shape == (2, 3, 4, 2, 128)
        assert result[3].shape == (2, 3, 4, 2)
        assert result[4].shape == (2, 3, 4, 128)
        assert result[5].shape == (2, 3, 0 if routing == _MEAN_ROUTING else 4, 128)
    else:
        assert [tensor.dtype for tensor in result] == [torch.int8, torch.float32, torch.bfloat16]
        assert result[0].shape == (2, 3, 128, 256)
        assert result[2].shape == (2, 3, 4, 128)


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(("architecture", "head_dim"), [("gfx1100", 128), ("gfx1100", 64)])
def test_unvalidated_projection_rejects_before_output_allocation(
    monkeypatch, operation, architecture, head_dim
):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", architecture)
    )
    with FakeTensorMode():
        operands = _operands(head_dim=head_dim)
        # Allocate the test's own inputs before guarding: a cold FakeTensorMode
        # dispatch routes new_empty through torch.empty, so allocating after the
        # guard fails on test setup instead of on library allocations.
        monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocated outputs")))
        with pytest.raises(ValueError, match="sparse projections are unavailable"):
            _call(operation, operands)


@pytest.mark.parametrize("missing", [None, "attention", "linear"])
@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cuda", "sm89"),
        AcceleratorTarget("hip", "gfx1200"),
        AcceleratorTarget("hip", "gfx1201"),
    ],
)
@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_output_support_is_independent_of_qkv_projection_support(
    monkeypatch, missing, target, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(_backend, "_nvidia_projection", None)
    monkeypatch.setattr(_backend, "_amd_projection", None)
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(
        torch.cuda, "current_device", Mock(side_effect=AssertionError("current GPU"))
    )
    attention = Mock(return_value=None if missing == "attention" else object())
    linear = Mock(return_value=None if missing == "linear" else object())
    monkeypatch.setattr(_backend.attention_backend, "select_attention_backend", attention)
    monkeypatch.setattr(_backend.linear_backend, "select_linear_backend", linear)
    operand = SimpleNamespace(device=torch.device("cuda:1"))
    assert _backend.select_output_backend(operand) is (
        linear.return_value if missing is None else None
    )
    attention.assert_called_once_with(operand)
    probe.assert_called_once_with(operand.device)
    if missing == "attention":
        linear.assert_not_called()
    else:
        linear.assert_called_once_with(operand)


@pytest.mark.parametrize(
    ("platform", "target"),
    [
        ("linux", AcceleratorTarget("hip", "gfx1100")),
        ("linux", AcceleratorTarget("cuda", "sm121")),
        ("linux", AcceleratorTarget("cuda", "sm90")),
        ("win32", AcceleratorTarget("cuda", "sm90")),
        ("linux", AcceleratorTarget("cpu")),
        ("win32", AcceleratorTarget("hip", "gfx1100")),
        ("win32", AcceleratorTarget("cpu")),
        ("darwin", AcceleratorTarget("hip", "gfx1200")),
        ("darwin", AcceleratorTarget("hip", "gfx1201")),
    ],
)
def test_unvalidated_output_integration_rejects_before_resolving_operations(
    monkeypatch, platform, target
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(return_value=target))
    attention = Mock(side_effect=AssertionError("resolved unsupported attention"))
    linear = Mock(side_effect=AssertionError("resolved unsupported linear"))
    monkeypatch.setattr(_backend.attention_backend, "select_attention_backend", attention)
    monkeypatch.setattr(_backend.linear_backend, "select_linear_backend", linear)
    assert _backend.select_output_backend(torch.empty(1)) is None
    attention.assert_not_called()
    linear.assert_not_called()


def _run_output_fusion(projected_query, operands, storage):
    # These tests stub attention preparation and output projection. Only the
    # real Q facade consumes tensor contents/metadata; context slots are opaque.
    arguments = {
        "key": storage,
        "key_scale": storage,
        "key_center": storage,
        "key_center_scale": storage,
        "key_summary": storage,
        "key_aux": storage,
        "value": storage,
        "value_scale_multiplier": storage,
        "value_mean": storage,
        "ratio_profile": _SparsePiperRatioProfile((2, 2, 2)),
        "sparse_key_blocks": 4,
        "logical_sequence_length": 193,
        "routing_mode": _MINMAX_ROUTING,
        "weight_qdata": storage,
        "weight_scale": storage,
        "bias": None,
        "group_size": 16,
        "query_chunk_rows": 128,
    }
    if not projected_query:
        return output._run_attention_output(
            query=storage, query_scale=storage, query_summary=storage, **arguments
        )
    qdata, scale, weight, weight_scale, norm, cos, sin = operands
    return output._run_projected_query_attention_output(
        query_input=qdata,
        query_input_scale=scale,
        query_weight_qdata=weight,
        query_weight_scale=weight_scale,
        query_norm_weight=norm,
        cos=cos,
        sin=sin,
        query_norm_epsilon=1e-6,
        softmax_scale=128**-0.5,
        **arguments,
    )


@pytest.mark.parametrize(
    ("projected_query", "missing"),
    [(False, "output"), (True, "output"), (True, "projection")],
)
def test_output_fusion_rejects_before_attention_preparation_or_allocation(
    monkeypatch, projected_query, missing
):
    prepare = Mock(side_effect=AssertionError("prepared unsupported attention"))
    monkeypatch.setattr(output_common, "prepare_attention", prepare)
    monkeypatch.setattr(output_common, "prepare_attention_context", prepare)
    monkeypatch.setattr(
        _backend,
        "select_projection_backend",
        Mock(return_value=None if missing == "projection" else object()),
    )
    monkeypatch.setattr(_backend, "select_output_backend", Mock(return_value=None))
    with FakeTensorMode():
        operands = _operands()
        storage = torch.empty((2, 3, 256, 128), device="cuda:1", dtype=torch.int8)
        allocate = Mock(side_effect=AssertionError("allocated unsupported workspace"))
        monkeypatch.setattr(torch, "empty", allocate)
        message = (
            "sparse projections are unavailable"
            if missing == "projection"
            else "sparse output fusion is unavailable"
        )
        with pytest.raises(ValueError, match=message):
            _run_output_fusion(projected_query, operands, storage)
    prepare.assert_not_called()
    allocate.assert_not_called()


@pytest.mark.parametrize("projected_query", [False, True])
def test_output_fusion_selects_once_before_preparation_and_reuses_backend(
    monkeypatch, projected_query
):
    events = []
    projection = SimpleNamespace(project_query=Mock())
    linear = object()
    select_projection = Mock(
        side_effect=lambda operand, **kwargs: events.append("projection") or projection
    )
    select_output = Mock(side_effect=lambda operand: events.append("output") or linear)
    monkeypatch.setattr(_backend, "require_projection_backend", select_projection)
    monkeypatch.setattr(_backend, "require_output_backend", select_output)
    prepare = Mock(
        side_effect=lambda *args, **kwargs: (
            events.append("prepare") or SimpleNamespace(sequence_length=193)
        )
    )
    monkeypatch.setattr(output_common, "prepare_attention", prepare)
    monkeypatch.setattr(output_common, "prepare_attention_context", prepare)
    projector = Mock(return_value=(384, Mock(), ()))
    monkeypatch.setattr(output, "_prepare_output_chunk_projector", projector)
    result = object()

    def run(*args, **kwargs):
        if projected_query:
            project_query = args[3]
            project_query(0, 128)
            project_query(128, 65)
        return result

    monkeypatch.setattr(output_common, "run_chunked_attention_output", run)
    monkeypatch.setattr(output_common, "run_chunked_projected_query_attention_output", run)
    with FakeTensorMode():
        operands = _operands()
        storage = torch.empty((2, 3, 256, 128), device="cuda:1", dtype=torch.int8)
        assert _run_output_fusion(projected_query, operands, storage) is result
    assert events == (
        ["projection", "output", "prepare"] if projected_query else ["output", "prepare"]
    )
    select_output.assert_called_once_with(storage)
    assert projector.call_args.kwargs["backend"] is linear
    if projected_query:
        select_projection.assert_called_once_with(operands[0], head_dim=128)
        assert projection.project_query.call_count == 2
        assert [call.kwargs["chunk_start"] for call in projection.project_query.call_args_list] == [
            0,
            128,
        ]
        assert [call.kwargs["chunk_rows"] for call in projection.project_query.call_args_list] == [
            128,
            65,
        ]
    else:
        select_projection.assert_not_called()


@pytest.mark.parametrize("dynamic_activation_scale", [False, True])
def test_shared_output_validation_has_no_target_policy_but_nvfp4_still_does(
    monkeypatch, dynamic_activation_scale
):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    with FakeTensorMode():
        storage = torch.empty((1, 1, 64, 128), dtype=torch.int8, device="cuda:1")
        assert output_common.validate_attention_output(storage, 64, 128) == 128
        with pytest.raises(ValueError, match="requires exact NVIDIA SM120"):
            nvfp4_output._validate_output_projection(
                storage, None, None, None, None, None, 64, 128, dynamic_activation_scale
            )


def _assert_shared_fusion_boundary(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "is_cuda_capability",
                "from_device",
                "get_device_capability",
                "jit",
            }
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or ""
            for alias in node.names:
                assert not {"_nvidia", "_amd"}.intersection(f"{module}.{alias.name}".split("."))


@pytest.mark.parametrize(
    "module",
    [
        query,
        key,
        value,
        output,
        _compile,
        _output_compile,
        output_common,
        attention_output,
        qk_validation,
    ],
)
def test_shared_fusion_does_not_inspect_targets_or_launch_kernels(module):
    _assert_shared_fusion_boundary(Path(module.__file__).read_text())


@pytest.mark.parametrize("module", [projection])
def test_shared_projection_launcher_does_not_own_target_policy(module):
    _assert_shared_fusion_boundary(Path(module.__file__).read_text())


@pytest.mark.parametrize("vendor", ["_nvidia", "_amd"])
@pytest.mark.parametrize(
    "source",
    [
        "from .{vendor} import triton",
        "from . import {vendor}",
        "from ..{vendor}.triton import project_query",
        "from piper_kernels.fusions.convrot_int8_sparse_piper import {vendor}",
        "import piper_kernels.fusions.convrot_int8_sparse_piper.{vendor}.triton as implementation",
    ],
)
def test_shared_boundary_check_rejects_relative_and_absolute_vendor_imports(source, vendor):
    with pytest.raises(AssertionError):
        _assert_shared_fusion_boundary(source.format(vendor=vendor))


def _projection_match(head_dim, input_features=256):
    graph = torch.fx.Graph()

    def operand(name, shape, dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    input_node = operand("sparse_input", (1, 128, input_features), torch.bfloat16)
    projections = []
    for kind in ("q", "k", "v"):
        weight_qdata = operand(
            f"sparse_{kind}_weight_qdata", (2 * head_dim, input_features), torch.int8
        )
        weight_scale = operand(f"sparse_{kind}_weight_scale", (2 * head_dim, 1), torch.float32)
        projections.append(
            _compile._BoundInt8Projection(
                node=input_node,
                input=input_node,
                weight_qdata=weight_qdata,
                weight_scale=weight_scale,
                bias=None,
                group_size=16,
                input_scale=None,
            )
        )
    attention_output = operand("attention_output", (1, 128, 2, head_dim), torch.bfloat16)
    cos = operand("cos", (128, head_dim // 2), torch.float32)
    sin = operand("sin", (128, head_dim // 2), torch.float32)
    matched = _compile.sparse_piper_compile.SparsePiperProjectionMatch(
        output_node=attention_output,
        fine_node=attention_output,
        coarse_node=None,
        query_projection=input_node,
        key_projection=input_node,
        value_projection=input_node,
        attention_shape=(1, 128, 2, head_dim),
        q_norm_weight=None,
        q_norm_epsilon=1e-5,
        k_norm_weight=None,
        k_norm_epsilon=1e-5,
        cos=cos,
        sin=sin,
        rotary_dim=head_dim // 2,
        half_rotary_dim=head_dim // 4,
        activation_dtype=torch.bfloat16,
        ratio_profile=operand("ratio_profile", (2,), torch.int64),
        sparse_key_blocks=2,
        softmax_scale=head_dim**-0.5,
        routing_mode=_MINMAX_ROUTING,
        block_lengths=None,
        sparse_query_blocks=None,
        force_tile_ids=None,
        force_tile_counts=None,
        coarse_gate=None,
        coarse_key_blocks=None,
        coarse_scale=None,
        coarse_routing_mode=None,
    )
    return matched, tuple(projections), input_node.meta["val"]


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("input_features", [128, 256])
def test_amd_projection_compiler_selects_actual_attention_width(
    monkeypatch, head_dim, input_features
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    monkeypatch.setattr(
        _compile.linear_backend, "select_linear_backend", Mock(return_value=object())
    )
    validate = Mock(return_value=True)
    monkeypatch.setattr(_compile.sparse_piper_compile, "valid_sparse_piper_attention", validate)
    matched, projections, _ = _projection_match(head_dim, input_features)
    assert _compile._valid_sparse_piper_projection(matched, *projections)
    validate.assert_called_once()


@pytest.mark.parametrize("missing", [None, "projection", "linear", "attention"])
def test_projection_compiler_requires_each_emitted_operation(monkeypatch, missing):
    selectors = {
        "projection": (_backend, "select_projection_backend"),
        "linear": (_compile.linear_backend, "select_linear_backend"),
        "attention": (_compile.attention_backend, "select_attention_backend"),
    }
    probes = []
    for name, (module, attribute) in selectors.items():
        probe = Mock(return_value=None if name == missing else object())
        monkeypatch.setattr(module, attribute, probe)
        probes.append(probe)
    validate_attention = Mock(return_value=True)
    monkeypatch.setattr(
        _compile.sparse_piper_compile, "valid_sparse_piper_attention", validate_attention
    )
    matched, projections, input_value = _projection_match(128)
    assert _compile._valid_sparse_piper_projection(matched, *projections) is (missing is None)
    for probe in probes:
        if probe.called:
            if probe is probes[0]:
                probe.assert_called_once_with(input_value, head_dim=128)
            else:
                probe.assert_called_once_with(input_value)
    assert validate_attention.called is (missing is None)


@pytest.mark.parametrize("supported", [False, True])
def test_output_compiler_uses_selected_operation_not_device_family(monkeypatch, supported):
    select = Mock(return_value=object() if supported else None)
    monkeypatch.setattr(_backend, "select_output_backend", select)
    graph = torch.fx.Graph()

    def operand(name, shape, dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    query_node = operand("query", (1, 2, 128, 128), torch.int8)
    attention = operand("attention", (1, 128, 2, 128), torch.bfloat16)
    weight = operand("weight", (384, 256), torch.int8)
    scale = operand("scale", (384, 1), torch.float32)
    reshaped = graph.call_function(torch.reshape, (attention, (1, 128, 256)))
    reshaped.meta["val"] = torch.empty((1, 128, 256), dtype=torch.bfloat16)
    projected = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_linear.default, (reshaped, weight, scale, None, 16)
    )
    projected.meta["val"] = torch.empty((1, 128, 384), dtype=torch.bfloat16)
    reshaped_match = _compile.sparse_piper_compile.ReshapedQuantizedAttentionMatch(
        reshape_node=reshaped,
        attention_node=attention,
        attention_shape=[1, 128, 256],
        attention_args={"query": query_node},
        is_coarse=False,
        force_kwargs={},
    )
    assert (
        _output_compile._valid_attention_output(
            projected, reshaped_match, weight, scale, None, 16, None
        )
        is supported
    )
    select.assert_called_once_with(query_node.meta["val"])


def _capture_projection(
    monkeypatch,
    operation,
    implementation=nvidia,
    head_dim=128,
    *,
    with_bias=False,
    target=None,
    sequence=193,
    heads=3,
    input_features=272,
):
    monkeypatch.setattr(nvidia.linear_gluon, "operands_aligned", lambda *operands: True)
    if target is None:
        target = AcceleratorTarget("cuda", "sm120")
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    functions = {
        "query": _kernels._convrot_project_rmsnorm_rope_quantize_query_kernel,
        "key": _kernels._convrot_project_rmsnorm_rope_quantize_sparse_key_kernel,
        "value": _kernels._convrot_project_quantize_sparse_value_kernel,
    }
    function = functions[operation]
    kernel = MagicMock()
    monkeypatch.setattr(_kernels, function.__name__, kernel)
    monkeypatch.setattr(_backend, "require_projection_backend", Mock(return_value=implementation))
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(projection, "device_context", guard)
    with FakeTensorMode():
        operands = _operands(
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            input_features=input_features,
        )
        bias = operands[3].new_empty((heads * head_dim,)) if with_bias else None
        _call(
            operation,
            operands,
            bias=bias,
            value_format=ValueFormat.AMD_WMMA if target.is_amd_hip else ValueFormat.TRANSPOSED,
        )
    guard.assert_called_once_with(torch.device("cuda:1"))
    for call in kernel.__getitem__.return_value.call_args_list:
        assert call.kwargs["bias_ptr"] is bias
    return function, kernel


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cuda", "sm89"),
        AcceleratorTarget("hip", "gfx1201"),
    ],
)
def test_backend_launch_schedule_and_fp32_math_are_preserved(
    monkeypatch, operation, head_dim, target
):
    is_amd = target.is_amd_hip
    implementation = amd if is_amd else nvidia
    _, kernel = _capture_projection(monkeypatch, operation, implementation, head_dim, target=target)
    # These operands' 272 input features fall outside the Gluon kernels, so each target runs
    # its Triton schedule. SM8x's is 64 rows by one head for Q, K, and V.
    is_sm8x = target.is_cuda_capability(8)
    if is_sm8x:
        heads_per_program = 1
    elif is_amd:
        heads_per_program = 2 if operation == "value" and head_dim == 64 else 1
    else:
        heads_per_program = 2
    block_m = 64 if is_sm8x or operation == "query" or (is_amd and operation == "key") else 128
    warps = 4 if is_sm8x or (is_amd and operation != "value") else 8
    rows = 129 if operation == "query" else 193
    grid_heads = triton.cdiv(3, heads_per_program)
    grids = [call.args[0] for call in kernel.__getitem__.call_args_list]
    assert grids == [(rows // block_m, grid_heads, 2), (1, grid_heads, 2)]
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    for index, call in enumerate(calls):
        assert call.kwargs["block_m"] == block_m
        assert call.kwargs["block_n"] == head_dim * heads_per_program
        assert call.kwargs["block_k"] == (64 if is_amd else 128)
        assert call.kwargs["heads_per_program"] == heads_per_program
        assert call.kwargs["group_m"] == (8 if is_sm8x or is_amd else 0)
        assert call.kwargs["num_warps"] == warps
        assert call.kwargs["num_stages"] == (2 if is_amd else 3)
        assert call.kwargs["aligned_projection"] is False
        assert call.kwargs["mask_block_lengths"] is False
        if operation != "value":
            assert call.kwargs["round_rsqrt_to_nearest"] is (not is_amd)
            assert call.kwargs["mask_ragged_tail"] is (index == 1)
        if operation != "query":
            assert call.args[-1] == (0 if index == 0 else rows // block_m)


def _compile_projection_call(function, call, target, *, operation, affine=True):
    arguments = dict(zip(function.arg_names, call.args, strict=False))
    arguments.update(
        {name: item for name, item in call.kwargs.items() if name in function.arg_names}
    )
    arguments.setdefault("round_rsqrt_to_nearest", False)
    arguments.setdefault("group_m", 0)
    if not affine and operation != "value":
        arguments["norm_weight_ptr"] = None
    constants, signature = {}, {}
    types = {
        torch.int8: "*i8",
        torch.int32: "*i32",
        torch.float32: "*fp32",
        torch.bfloat16: "*bf16",
    }
    for parameter in function.params:
        argument = arguments[parameter.name]
        if parameter.is_constexpr:
            constants[parameter.name] = argument
        elif argument is None:
            constants[parameter.name] = None
            signature[parameter.name] = "constexpr"
        else:
            signature[parameter.name] = (
                types[argument.dtype] if isinstance(argument, torch.Tensor) else "i32"
            )
    return triton.compile(
        ASTSource(function, signature, constexprs=constants),
        target=target,
        options={
            name: call.kwargs[name]
            for name in ("num_warps", "num_stages", "llvm_fn_attrs")
            if name in call.kwargs
        },
    )


@pytest.mark.parametrize(
    ("operation", "affine"),
    [("query", True), ("query", False), ("key", True), ("key", False), ("value", True)],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "target",
    [
        GPUTarget("cuda", 120, 32),
        GPUTarget("cuda", 89, 32),
        GPUTarget("hip", "gfx1201", 32),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_production_launches_compile_with_expected_storage_precision(
    monkeypatch, operation, target, head_dim, affine, with_bias
):
    if target.backend == "hip" and sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    function, kernel = _capture_projection(
        monkeypatch,
        operation,
        nvidia if target.backend == "cuda" else amd,
        head_dim,
        with_bias=with_bias,
        target=AcceleratorTarget(
            target.backend, f"sm{target.arch}" if target.backend == "cuda" else target.arch
        ),
    )
    for call in kernel.__getitem__.return_value.call_args_list:
        compiled = _compile_projection_call(
            function, call, target, operation=operation, affine=affine
        )
        if target.backend == "cuda":
            assert compiled.asm["cubin"]
            if target.arch == 89:
                # SM89 caps one block's dynamic shared memory at 99 KiB.
                assert compiled.metadata.shared <= 99 * 1024
        else:
            assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]
            assert compiled.metadata.shared <= 65536
            if rocm_fences_only_lds():
                assert "global_inv" not in compiled.asm["amdgcn"]
        truncations = [
            line for line in compiled.asm["ttgir"].splitlines() if "arith.truncf" in line
        ]
        assert bool(truncations) is (operation == "value")
        assert all("f32" in line and "bf16" in line for line in truncations)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_amd_aligned_value_projection_compiles_without_vgpr_spills(monkeypatch, head_dim):
    if sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    monkeypatch.setattr(amd.policy, "supports_gluon", lambda *args, **kwargs: False)
    function, kernel = _capture_projection(
        monkeypatch,
        "value",
        amd,
        head_dim,
        target=AcceleratorTarget("hip", "gfx1201"),
        sequence=256,
        heads=4,
        input_features=256,
    )
    target = GPUTarget("hip", "gfx1201", 32)
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["aligned_projection"] is True
    compiled = _compile_projection_call(function, calls[0], target, operation="value")
    assert ".vgpr_spill_count: 0" in compiled.asm["amdgcn"]


_AMD_GLUON_POINTER_TYPES = {
    "input_ptr": "*i8",
    "weight_ptr": "*i8",
    "query_ptr": "*i8",
    "key_ptr": "*i8",
    "key_center_ptr": "*i8",
    "value_ptr": "*i8",
    "value_mean_ptr": "*bf16",
    "block_lengths_ptr": "*i32",
    "norm_weight_ptr": "*bf16",
    "bias_ptr": "*bf16",
}


def _compile_amd_gluon_kernel(
    architecture: str,
    operation: str,
    *,
    head_dim: int,
    rotary_dim: int,
    affine: bool,
    with_bias: bool,
    mean_pool_summary: bool,
    mask_rows: bool,
    mask_block_lengths: bool,
    input_features: int = 5376,
):
    function = getattr(amd_gluon, f"_{operation}_kernel")
    heads = 56 if head_dim == 128 else 32
    constants = {
        "input_features": input_features,
        "heads": heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "norm_epsilon": 1e-5,
        "softmax_scale": head_dim**-0.5,
        "mean_pool_summary": mean_pool_summary,
        "mask_block_lengths": mask_block_lengths,
        "mask_rows": mask_rows,
        "group_m": amd_policy.GLUON_GROUP_M,
    }
    constants = {name: item for name, item in constants.items() if name in function.arg_names}
    signature, attributes = {}, {}
    for index, name in enumerate(function.arg_names):
        if name in constants:
            continue
        if (name == "bias_ptr" and not with_bias) or (name == "norm_weight_ptr" and not affine):
            constants[name], signature[name] = None, "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = _AMD_GLUON_POINTER_TYPES.get(name, "*fp32")
            attributes[(index,)] = [["tt.divisibility", 16]]
        else:
            signature[name] = "i32"
    extra = (
        amd_gluon._build_key_kernel_options(
            head_dim,
            rotary_dim,
            mask_rows=mask_rows,
            mask_block_lengths=mask_block_lengths,
            mean_pool_summary=mean_pool_summary,
        )
        if operation == "key"
        else amd_gluon._build_kernel_options()
    )
    compiled = triton.compile(
        GluonASTSource(function, signature, constexprs=constants, attrs=attributes),
        target=GPUTarget("hip", architecture, 32),
        options={"num_warps": amd_policy.GLUON_NUM_WARPS, **extra},
    )
    amdgcn = compiled.asm["amdgcn"]
    assert "v_wmma_i32_16x16x16_iu8" in amdgcn
    if operation != "value":
        assert "v_permlanex16_b32" in amdgcn
    assert "buffer_load_b64" in amdgcn
    assert "buffer_store_b64" in amdgcn
    assert compiled.metadata.shared <= 16384
    assert ".vgpr_spill_count: 0" in amdgcn
    if rocm_fences_only_lds():
        assert "global_inv" not in amdgcn
    truncations = [line for line in compiled.asm["ttgir"].splitlines() if "arith.truncf" in line]
    assert bool(truncations) is (operation == "value")
    assert all("f32" in line and "bf16" in line for line in truncations)
    return compiled


def test_amd_policy_and_gluon_eligibility_contracts():
    assert amd_policy.supports_gluon(
        "query", head_dim=64, input_features=192, heads=2, rotary_dim=48
    )
    assert amd_policy.supports_gluon(
        "key", head_dim=128, input_features=256, heads=1, rotary_dim=96
    )
    assert amd_policy.supports_gluon(
        "value", head_dim=64, input_features=64, heads=2, value_format=ValueFormat.AMD_WMMA
    )
    # Unsupported shapes, counts, rotary widths, alignment, or value layout fall back to Triton.
    assert not amd_policy.supports_gluon("query", head_dim=64, input_features=192, heads=3)
    assert not amd_policy.supports_gluon("query", head_dim=64, input_features=272, heads=4)
    assert not amd_policy.supports_gluon("key", head_dim=64, input_features=32, heads=2)
    assert not amd_policy.supports_gluon(
        "query", head_dim=64, input_features=256, heads=4, rotary_dim=16
    )
    assert not amd_policy.supports_gluon(
        "key", head_dim=128, input_features=256, heads=3, rotary_dim=48
    )
    assert not amd_policy.supports_gluon("query", head_dim=32, input_features=256, heads=4)
    assert not amd_policy.supports_gluon(
        "value", head_dim=128, input_features=256, heads=4, value_format=ValueFormat.TRANSPOSED
    )
    assert not amd_policy.supports_gluon(
        "query", head_dim=128, input_features=256, heads=4, operands_aligned=False
    )

    with FakeTensorMode():
        q_in = torch.empty((1, 256, 256), device="cuda:1", dtype=torch.int8)
        w_d64_even = torch.empty((4 * 64, 256), device="cuda:1", dtype=torch.int8)
        w_d64_odd = torch.empty((3 * 64, 256), device="cuda:1", dtype=torch.int8)
        w_d128 = torch.empty((3 * 128, 256), device="cuda:1", dtype=torch.int8)

        assert amd_gluon.operands_aligned(
            q_in, w_d64_even, storage_sequence_length=256, rotary_dim=48, head_dim=64
        )
        assert not amd_gluon.operands_aligned(
            q_in[:, ::2], w_d64_even, storage_sequence_length=128, rotary_dim=48, head_dim=64
        )
        assert not amd_gluon.operands_aligned(
            q_in, w_d64_even[:, ::2], storage_sequence_length=256, rotary_dim=48, head_dim=64
        )
        # Signed 32-bit FP32 byte-offset check: 8_388_608 * 64 * 4 exceeds 2**31 bytes.
        q_large_fp32 = torch.empty((1, 8_388_608, 64), device="cuda:1", dtype=torch.int8)
        w_small = torch.empty((4 * 64, 64), device="cuda:1", dtype=torch.int8)
        assert not amd_gluon.operands_aligned(
            q_large_fp32, w_small, storage_sequence_length=8_388_608, rotary_dim=64, head_dim=64
        )

        q_plan = amd.default_execution_plan(
            q_in, w_d64_even, operation="query", head_dim=64, rotary_dim=48
        )
        assert q_plan == amd_policy._D64_GLUON_EXECUTION_PLAN
        assert q_plan.kernel == "gluon"
        assert q_plan.block_m == 128
        assert q_plan.heads_per_program == 2

        d128_plan = amd.default_execution_plan(
            q_in, w_d128, operation="key", head_dim=128, rotary_dim=96
        )
        assert d128_plan == amd_policy._D128_GLUON_EXECUTION_PLAN
        assert (
            amd.partial_row_block_size(
                q_in, w_d64_even, operation="key", head_dim=64, rotary_dim=48
            )
            == 128
        )
        assert (
            amd.partial_row_block_size(
                q_in, w_d128, operation="value", head_dim=128, value_format=ValueFormat.AMD_WMMA
            )
            == 128
        )
        assert amd.partial_row_block_size(q_in, w_d64_odd, operation="key", head_dim=64) == 64
        assert (
            amd.partial_row_block_size(
                q_in, w_d128, operation="value", head_dim=128, value_format=ValueFormat.TRANSPOSED
            )
            == 128
        )


def test_resolve_row_block_launches_covers_full_tail_and_partial_ranges():
    assert resolve_row_block_launches(0, 128) == ()
    assert resolve_row_block_launches(256, 128) == ((2, 0, True),)
    assert resolve_row_block_launches(205, 128) == ((1, 0, True), (1, 1, False))
    assert resolve_row_block_launches(384, 128, start_row_block=1, num_row_blocks=None) == (
        (2, 1, True),
    )
    assert resolve_row_block_launches(384, 128, start_row_block=1, num_row_blocks=0) == ()
    with pytest.raises(ValueError, match="non-negative"):
        resolve_row_block_launches(256, 128, start_row_block=-1)
    with pytest.raises(ValueError, match="alignment"):
        resolve_row_block_launches(205, 128, start_row_block=1, num_row_blocks=None)
    with pytest.raises(ValueError, match="full block_m"):
        resolve_row_block_launches(256, 128, start_row_block=1, num_row_blocks=2)


_AMD_QK_COMPILE_VARIANTS = [
    ("full_tiles_mean_routing", True, False, True, False, False, False),
    ("ragged_tail", True, False, False, True, False, False),
    ("block_lengths", True, False, True, True, True, False),
    ("bias_without_norm", False, True, False, True, False, False),
    ("full_rotary", True, True, False, False, False, True),
]
_AMD_V_COMPILE_VARIANTS = [
    ("ragged_tail", False, True, False),
    ("bias_block_lengths", True, True, True),
]


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_amd_gluon_kernels_compile_offline_for_rdna4(operation, head_dim):
    if sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    rotary_dim = 0 if operation == "value" else (48 if head_dim == 64 else 96)
    _compile_amd_gluon_kernel(
        "gfx1201",
        operation,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        affine=True,
        with_bias=False,
        mean_pool_summary=False,
        mask_rows=False,
        mask_block_lengths=False,
    )


@pytest.mark.parametrize("operation", ["query", "key"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    ("variant", "affine", "with_bias", "mean_pool_summary", "mask_rows", "mask_bl", "full_rotary"),
    _AMD_QK_COMPILE_VARIANTS,
)
def test_amd_gluon_qk_compile_variants_keep_zero_spills(
    operation,
    head_dim,
    variant,
    affine,
    with_bias,
    mean_pool_summary,
    mask_rows,
    mask_bl,
    full_rotary,
):
    if sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    rotary_dim = head_dim if full_rotary else (48 if head_dim == 64 else 96)
    _compile_amd_gluon_kernel(
        "gfx1201",
        operation,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        affine=affine,
        with_bias=with_bias,
        mean_pool_summary=mean_pool_summary,
        mask_rows=mask_rows,
        mask_block_lengths=mask_bl,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("variant", "with_bias", "mask_rows", "mask_bl"), _AMD_V_COMPILE_VARIANTS)
def test_amd_gluon_value_compile_variants_keep_zero_spills(
    head_dim,
    variant,
    with_bias,
    mask_rows,
    mask_bl,
):
    if sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    _compile_amd_gluon_kernel(
        "gfx1201",
        "value",
        head_dim=head_dim,
        rotary_dim=0,
        affine=True,
        with_bias=with_bias,
        mean_pool_summary=False,
        mask_rows=mask_rows,
        mask_block_lengths=mask_bl,
    )


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("invalid", ["size", "rank", "dtype", "device", "strides"])
def test_projection_rejects_invalid_bias_before_backend_selection(monkeypatch, operation, invalid):
    select = Mock(side_effect=AssertionError("backend must not be selected"))
    monkeypatch.setattr(_backend, "require_projection_backend", select)
    with FakeTensorMode():
        operands = _operands()
        bias = torch.empty(384, device="cuda:1", dtype=torch.float32)
        if invalid == "size":
            bias = torch.empty(383, device="cuda:1")
        elif invalid == "rank":
            bias = torch.empty((1, 384), device="cuda:1")
        elif invalid == "dtype":
            bias = bias.to(torch.int8)
        elif invalid == "device":
            bias = bias.to("cpu")
        elif invalid == "strides":
            bias = torch.empty_strided((384,), (2,), device="cuda:1")
        with pytest.raises(ValueError, match=r"bias|device|contiguous"):
            _call(operation, operands, bias=bias)
    select.assert_not_called()


@pytest.mark.parametrize("bias_argument", ["positional", "keyword", "duplicate"])
def test_output_folding_preserves_query_bias(bias_argument):
    graph = torch.fx.Graph()
    projection_tensors = tuple(
        graph.placeholder(name)
        for name in ("input", "input_scale", "weight", "weight_scale", "norm", "cos", "sin")
    )
    projection = (*projection_tensors, 1e-5, 64**-0.5)
    bias = graph.placeholder("bias")
    arguments = (*projection, _MINMAX_ROUTING, None)
    keywords = {"head_dim": 64}
    if bias_argument != "keyword":
        arguments += (bias,)
    if bias_argument != "positional":
        keywords["bias"] = bias
    producer = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default,
        args=arguments,
        kwargs=keywords,
    )
    outputs = tuple(graph.call_function(operator.getitem, args=(producer, i)) for i in range(3))
    outputs[0].meta["val"] = torch.empty((1, 2, 64, 64), dtype=torch.int8, device="meta")
    recovered = _output_compile._prepared_query_projection(*outputs, _MINMAX_ROUTING, None)
    if bias_argument == "duplicate":
        assert recovered is None
    else:
        assert recovered == (projection, bias)


@pytest.mark.parametrize("source_kind", ["fresh", "input", "alias", "escaped", "different-shape"])
def test_output_input_reuse_requires_exclusive_fresh_matching_storage(source_kind):
    graph = torch.fx.Graph()
    input_node = graph.placeholder("input")
    if source_kind == "input":
        source = input_node
    elif source_kind == "alias":
        source = graph.call_function(torch.ops.aten.alias.default, args=(input_node,))
    else:
        source = graph.call_function(torch.ops.aten.silu.default, args=(input_node,))
    source.meta["val"] = torch.empty(2, 257, 256, device="meta", dtype=torch.bfloat16)
    graph.call_function(torch.ops.aten.sym_size.int, args=(source, 1))
    graph.call_function(
        torch.ops.piper_kernels.convrot_int8_prepare_input.default, args=(source, 256, None)
    )
    if source_kind == "escaped":
        graph.call_function(torch.ops.aten.alias.default, args=(source,))
    original = graph.call_function(torch.ops.aten.clone.default, args=(input_node,))
    original.meta["val"] = torch.empty(
        2,
        257,
        320 if source_kind == "different-shape" else 256,
        device="meta",
        dtype=torch.bfloat16,
    )
    assert _output_compile._can_reuse_source_as_output(source, original) == (source_kind == "fresh")
