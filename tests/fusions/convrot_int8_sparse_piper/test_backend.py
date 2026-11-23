"""Sparse fusion owns tensor contracts; selected operations own execution."""

import importlib.abc
import importlib.util
import operator
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import triton
from _accelerators import (
    forbid_device_probes,
    forbid_triton_driver,
    mock_cuda_target,
    mock_rocm_target,
    rocm_fences_only_lds,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels import compile_options
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
    _SparsePiperRatioProfile,
)
from piper_kernels.fusions.attention import _sparse_output as output_common
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
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import gluon as amd_gluon
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import policy as amd_policy
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import triton as amd
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import resolve_row_block_launches
from piper_kernels.fusions.convrot_int8_sparse_piper._nvidia import dispatch as nvidia
from piper_kernels.fusions.nvfp4_sparse_piper import output as nvfp4_output


def _operands(sequence=193, heads=3, head_dim=128, input_features=272, device="cuda:1"):
    return (
        torch.empty((2, sequence, input_features), device=device, dtype=torch.int8),
        torch.empty((2, sequence), device=device),
        torch.empty((heads * head_dim, input_features), device=device, dtype=torch.int8),
        torch.empty((heads * head_dim, 1), device=device),
        torch.empty(head_dim, device=device, dtype=torch.bfloat16),
        torch.empty((sequence, head_dim * 3 // 4), device=device),
        torch.empty((sequence, head_dim * 3 // 4), device=device),
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
    operand = torch.empty(1)
    expected = None
    if target.is_cuda_capability(12, 0) or target.is_cuda_capability(8):
        expected = nvidia
    elif (
        platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    ):
        expected = amd
    assert _backend.select_projection_backend(operand, head_dim=head_dim, target=target) is expected


@pytest.mark.parametrize("is_hip", [False, True])
def test_selectors_resolve_operand_device_not_current_device(monkeypatch, is_hip):
    device = torch.device("cuda:1")
    monkeypatch.setattr(
        torch.cuda, "current_device", Mock(side_effect=AssertionError("current GPU"))
    )
    if is_hip:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(torch.version, "hip", "7.0")
        probe = Mock(return_value=SimpleNamespace(gcnArchName="gfx1201"))
        monkeypatch.setattr(torch.cuda, "get_device_properties", probe)
    else:
        monkeypatch.setattr(torch.version, "hip", None)
        probe = Mock(return_value=(12, 0))
        monkeypatch.setattr(torch.cuda, "get_device_capability", probe)

    operand = SimpleNamespace(device=device, ndim=0)
    assert _backend.select_projection_backend(operand) is (amd if is_hip else nvidia)  # type: ignore[arg-type]
    output_backend = _backend.select_output_backend(operand)  # type: ignore[arg-type]
    assert output_backend is not None
    assert probe.call_count == 2
    assert all(call.args == (device,) for call in probe.call_args_list)


class _MissingModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_name: str, error: ModuleNotFoundError) -> None:
        self._blocked_name = blocked_name
        self._error = error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._blocked_name:
            raise self._error


@pytest.mark.parametrize("missing", ["triton", "triton.language.extra.cuda", "triton_extra", None])
@pytest.mark.parametrize("vendor", ["_nvidia", "_amd"])
def test_projection_import_only_tolerates_absent_top_level_triton(monkeypatch, missing, vendor):
    package = "piper_kernels.fusions.convrot_int8_sparse_piper"
    implementation = "dispatch" if vendor == "_nvidia" else "triton"
    fullname = f"{package}.{vendor}.{implementation}"
    error = ModuleNotFoundError("missing dependency", name=missing)
    vendor_pkg = sys.modules.get(f"{package}.{vendor}")
    if vendor_pkg is not None:
        monkeypatch.delitem(vendor_pkg.__dict__, implementation, raising=False)
    monkeypatch.delitem(sys.modules, fullname, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_MissingModuleFinder(fullname, error), *sys.meta_path])
    spec = importlib.util.spec_from_file_location(_backend.__name__, _backend.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", Mock(side_effect=AssertionError("direct target probe"))
    )
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
@pytest.mark.parametrize("head_dim", [64, 128])
def test_unvalidated_projection_rejects_before_output_allocation(monkeypatch, operation, head_dim):
    with FakeTensorMode():
        operands = _operands(head_dim=head_dim, device="cpu")
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
    attention = Mock(return_value=None if missing == "attention" else object())
    linear = Mock(return_value=None if missing == "linear" else object())
    monkeypatch.setattr(_backend.attention_backend, "select_attention_backend", attention)
    monkeypatch.setattr(_backend.linear_backend, "select_linear_backend", linear)
    operand = torch.empty(1)
    assert _backend.select_output_backend(operand, target=target) is (
        linear.return_value if missing is None else None
    )
    attention.assert_called_once_with(operand, target=target)
    if missing == "attention":
        linear.assert_not_called()
    else:
        linear.assert_called_once_with(operand, target=target)


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
def test_unvalidated_output_integration_is_rejected(monkeypatch, platform, target):
    monkeypatch.setattr(sys, "platform", platform)
    forbid_device_probes(monkeypatch, "resolved unsupported device")
    assert _backend.select_output_backend(torch.empty(1), target=target) is None


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
        return output._attention_output_op._init_fn(
            query=storage, query_scale=storage, query_summary=storage, **arguments
        )
    qdata, scale, weight, weight_scale, norm, cos, sin = operands
    return output._projected_query_attention_output_op._init_fn(
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
    if projected_query and missing == "output":
        mock_cuda_target(monkeypatch, (12, 0))
        monkeypatch.setattr(_backend, "select_output_backend", Mock(return_value=None))
    else:
        mock_cuda_target(monkeypatch, (9, 0))
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
        with (
            forbid_triton_driver("prepared unsupported attention"),
            pytest.raises(ValueError, match=message),
        ):
            _run_output_fusion(projected_query, operands, storage)
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
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", Mock(side_effect=AssertionError("direct target probe"))
    )
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
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201"),
    )
    with FakeTensorMode():
        storage = torch.empty((1, 1, 64, 128), dtype=torch.int8, device="cuda:0")
        assert output_common.validate_attention_output(storage, 64, 128) == 128
        with pytest.raises(ValueError, match="requires exact NVIDIA SM120"):
            nvfp4_output._validate_output_projection(
                storage, None, None, None, None, None, 64, 128, dynamic_activation_scale
            )


def _projection_match(head_dim, input_features=256, device="cpu"):
    graph = torch.fx.Graph()

    def operand(name, shape, dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype, device=device)
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
    ratio_profile = graph.placeholder("ratio_profile")
    ratio_profile.meta["val"] = _SparsePiperRatioProfile((5_000, 10_000))
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
        ratio_profile=ratio_profile,
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
    mock_rocm_target(monkeypatch, "gfx1201")
    monkeypatch.setattr(sys, "platform", "linux")
    with FakeTensorMode():
        matched, projections, _ = _projection_match(head_dim, input_features, device="cuda:0")
        assert _compile._valid_sparse_piper_projection(matched, *projections)


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


@pytest.mark.parametrize(
    ("architecture", "supported"),
    [("sm120", True), ("gfx1201", True), ("sm90", False)],
)
def test_output_compiler_uses_selected_operation_not_device_family(
    monkeypatch, architecture, supported
):
    if architecture == "sm120":
        mock_cuda_target(monkeypatch, (12, 0))
    elif architecture == "gfx1201":
        mock_rocm_target(monkeypatch, "gfx1201")
        monkeypatch.setattr(sys, "platform", "linux")
    else:
        mock_cuda_target(monkeypatch, (9, 0))
    graph = torch.fx.Graph()

    with FakeTensorMode():

        def operand(name, shape, dtype):
            node = graph.placeholder(name)
            node.meta["val"] = torch.empty(shape, device="cuda:0", dtype=dtype)
            return node

        query_node = operand("query", (1, 2, 128, 128), torch.int8)
        attention = operand("attention", (1, 128, 2, 128), torch.bfloat16)
        weight = operand("weight", (384, 256), torch.int8)
        scale = operand("scale", (384, 1), torch.float32)
        reshaped = graph.call_function(torch.reshape, (attention, (1, 128, 256)))
        reshaped.meta["val"] = torch.empty((1, 128, 256), device="cuda:0", dtype=torch.bfloat16)
        projected = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_linear.default,
            (reshaped, weight, scale, None, 16),
        )
        projected.meta["val"] = torch.empty((1, 128, 384), device="cuda:0", dtype=torch.bfloat16)
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


@pytest.mark.parametrize(
    ("case", "group_size", "bad_bias"),
    [
        ("valid", 16, False),
        ("invalid-group-size", 0, False),
        ("invalid-bias", 16, True),
    ],
)
def test_output_compiler_rejects_invalid_metadata_before_backend_selection(
    monkeypatch, case, group_size, bad_bias
):
    graph = torch.fx.Graph()
    with FakeTensorMode():
        operand_specs = (
            ("query", (1, 2, 128, 128), torch.int8),
            ("query_scale", (1, 2, 4), torch.float32),
            ("query_summary", (1, 2, 2, 128), torch.float32),
            ("key", (1, 2, 128, 128), torch.int8),
            ("key_scale", (1, 2, 2), torch.float32),
            ("key_center", (1, 2, 2, 2, 128), torch.int8),
            ("key_center_scale", (1, 2, 2, 2), torch.float32),
            ("key_summary", (1, 2, 2, 128), torch.float32),
            ("key_aux", (1, 2, 2, 128), torch.float32),
            ("value", (1, 2, 128, 128), torch.int8),
            ("value_scale_multiplier", (1, 2, 2, 1), torch.float32),
            ("value_mean", (1, 2, 2, 128), torch.bfloat16),
        )
        operands = []
        for name, shape, dtype in operand_specs:
            node = graph.placeholder(name)
            node.meta["val"] = torch.empty(shape, dtype=dtype, device="cuda:0")
            operands.append(node)
        profile = graph.placeholder("ratio_profile")
        profile.meta["val"] = _SparsePiperRatioProfile((5_000, 10_000))
        attention = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
            args=(*operands, profile, 2, 128, _MINMAX_ROUTING),
        )
        attention.meta["val"] = torch.empty((1, 128, 2, 128), dtype=torch.bfloat16, device="cuda:0")
        reshaped = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(attention, (1, 128, 256)),
        )
        reshaped.meta["val"] = torch.empty((1, 128, 256), dtype=torch.bfloat16, device="cuda:0")
        weight = graph.placeholder("weight")
        weight.meta["val"] = torch.empty((384, 256), dtype=torch.int8, device="cuda:0")
        scale = graph.placeholder("scale")
        scale.meta["val"] = torch.empty((384, 1), dtype=torch.float32, device="cuda:0")
        bias = None
        if bad_bias:
            bias = graph.placeholder("bias")
            bias.meta["val"] = torch.empty((383,), dtype=torch.float32, device="cuda:0")
        projected = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_linear.default,
            (reshaped, weight, scale, bias, group_size),
        )
        projected.meta["val"] = torch.empty((1, 128, 384), dtype=torch.bfloat16, device="cuda:0")
        graph.output(projected)

    if case == "valid":
        mock_cuda_target(monkeypatch, (12, 0))
    else:
        forbid_device_probes(monkeypatch)

    passes = compile_options(linear=False, ffn=False, attention=True)["post_grad_custom_pre_pass"]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    attention_op = torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default
    fused_op = torch.ops.piper_kernels.convrot_int8_sparse_piper_attention_output.default
    linear_op = torch.ops.piper_kernels.convrot_int8_linear.default
    if case == "valid":
        assert targets.count(fused_op) == 1
        assert attention_op not in targets
        assert linear_op not in targets
    else:
        assert fused_op not in targets
        assert targets.count(attention_op) == 1
        assert targets.count(linear_op) == 1
    graph.lint()


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
    execution_plan=None,
):
    monkeypatch.setattr(nvidia.linear_gluon, "operands_aligned", lambda *operands: True)
    if target is None:
        target = AcceleratorTarget("cuda", "sm120")
    if target.is_nvidia_cuda:
        mock_cuda_target(monkeypatch, target.cuda_capability)
    elif target.is_amd_hip:
        mock_rocm_target(monkeypatch, target.architecture)
        monkeypatch.setattr(sys, "platform", "linux")
    functions = {
        "query": _kernels._convrot_project_rmsnorm_rope_quantize_query_kernel,
        "key": _kernels._convrot_project_rmsnorm_rope_quantize_sparse_key_kernel,
        "value": _kernels._convrot_project_quantize_sparse_value_kernel,
    }
    function = functions[operation]
    kernel = MagicMock()
    monkeypatch.setattr(_kernels, function.__name__, kernel)
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", guard)
    with FakeTensorMode():
        operands = _operands(
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            input_features=input_features,
        )
        bias = operands[3].new_empty((heads * head_dim,)) if with_bias else None
        if execution_plan is None:
            _call(
                operation,
                operands,
                bias=bias,
                value_format=ValueFormat.AMD_WMMA if target.is_amd_hip else ValueFormat.TRANSPOSED,
            )
        else:
            assert operation == "value"
            qdata, scale, weight, weight_scale, *_ = operands
            out = value.allocate_value_output(
                qdata, qdata.shape[0], heads, sequence, head_dim, ValueFormat.AMD_WMMA
            )
            implementation.project_value(
                qdata,
                scale,
                weight,
                weight_scale,
                None,
                value_format=ValueFormat.AMD_WMMA,
                out=out,
                bias=bias,
                execution_plan=execution_plan,
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
    rows = 129 if operation == "query" else 193
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    block_m = calls[0].kwargs["block_m"]
    heads_per_program = calls[0].kwargs["heads_per_program"]
    assert block_m in (64, 128)
    assert heads_per_program in (1, 2)
    if operation == "query":
        assert block_m == 64
    grid_heads = triton.cdiv(3, heads_per_program)
    grids = [call.args[0] for call in kernel.__getitem__.call_args_list]
    assert grids == [(rows // block_m, grid_heads, 2), (1, grid_heads, 2)]
    for index, call in enumerate(calls):
        assert call.kwargs["block_m"] == block_m
        assert call.kwargs["block_n"] == head_dim * heads_per_program
        assert call.kwargs["block_k"] in (32, 64, 128)
        assert call.kwargs["heads_per_program"] == heads_per_program
        assert call.kwargs["group_m"] >= 0
        assert call.kwargs["num_warps"] in (2, 4, 8)
        assert 1 <= call.kwargs["num_stages"] <= 4
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


# Covering compile matrix instead of the 5x2x3x2 (60-case) Cartesian product:
# - Every target family (SM120, SM89, RDNA4) compiles all three operands (Q, K, V) and
#   checks its target-specific codegen/shared-memory/precision contracts (including both
#   full-tile and ragged-tail Q/K launches via sequence=193 and TRANSPOSED vs AMD_WMMA V).
# - SM120 covers both head_dims (64, 128), affine/weightless RMSNorm, and bias/no-bias
#   across Q, K, and V; SM89 compiles worst-case D128 shared memory (<= 99 KiB) with both
#   affine and bias variants; RDNA4 compiles worst-case D128 across Q/K/V plus D64 V
#   (its unique two-head-per-program Triton schedule).
_PRODUCTION_COMPILE_CASES = [
    (GPUTarget("cuda", 120, 32), "query", 128, True, False),
    (GPUTarget("cuda", 120, 32), "query", 64, False, True),
    (GPUTarget("cuda", 120, 32), "key", 128, True, True),
    (GPUTarget("cuda", 120, 32), "key", 64, False, False),
    (GPUTarget("cuda", 120, 32), "value", 128, True, False),
    (GPUTarget("cuda", 120, 32), "value", 64, True, True),
    (GPUTarget("cuda", 89, 32), "query", 128, True, True),
    (GPUTarget("cuda", 89, 32), "key", 128, False, False),
    (GPUTarget("cuda", 89, 32), "value", 128, True, False),
    (GPUTarget("hip", "gfx1201", 32), "query", 128, False, False),
    (GPUTarget("hip", "gfx1201", 32), "key", 128, True, True),
    (GPUTarget("hip", "gfx1201", 32), "value", 128, True, False),
    (GPUTarget("hip", "gfx1201", 32), "value", 64, True, True),
]


@pytest.mark.parametrize(
    ("target", "operation", "head_dim", "affine", "with_bias"),
    _PRODUCTION_COMPILE_CASES,
)
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
    plan = amd.policy.select_execution_plan(
        "value",
        head_dim=head_dim,
        input_features=256,
        heads=4,
        operands_aligned=False,
        value_format=ValueFormat.AMD_WMMA,
    )
    function, kernel = _capture_projection(
        monkeypatch,
        "value",
        amd,
        head_dim,
        target=AcceleratorTarget("hip", "gfx1201"),
        sequence=256,
        heads=4,
        input_features=256,
        execution_plan=plan,
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
        q_in = torch.empty((1, 256, 256), device="cuda:0", dtype=torch.int8)
        w_d64_even = torch.empty((4 * 64, 256), device="cuda:0", dtype=torch.int8)
        w_d64_odd = torch.empty((3 * 64, 256), device="cuda:0", dtype=torch.int8)
        w_d128 = torch.empty((3 * 128, 256), device="cuda:0", dtype=torch.int8)

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
        q_large_fp32 = torch.empty((1, 8_388_608, 64), device="cuda:0", dtype=torch.int8)
        w_small = torch.empty((4 * 64, 64), device="cuda:0", dtype=torch.int8)
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
def test_projection_rejects_invalid_bias_before_backend_selection(operation, invalid):
    with FakeTensorMode():
        operands = _operands(device="cpu")
        bias = torch.empty(384, device="cpu", dtype=torch.float32)
        if invalid == "size":
            bias = torch.empty(383, device="cpu")
        elif invalid == "rank":
            bias = torch.empty((1, 384), device="cpu")
        elif invalid == "dtype":
            bias = bias.to(torch.int8)
        elif invalid == "device":
            bias = bias.to("meta")
        elif invalid == "strides":
            bias = torch.empty_strided((384,), (2,), device="cpu")
        with pytest.raises(ValueError, match=r"bias|device|contiguous"):
            _call(operation, operands, bias=bias)


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
