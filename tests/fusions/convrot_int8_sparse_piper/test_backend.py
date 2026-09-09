"""Sparse fusion owns tensor contracts; selected operations own execution."""

import ast
import builtins
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import triton
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.language.extra.cuda import libdevice

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._nvidia import policy as attention_policy
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.convrot_int8_sage_qk import _validation as qk_validation
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
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import triton as amd
from piper_kernels.fusions.convrot_int8_sparse_piper._nvidia import triton as nvidia
from piper_kernels.fusions.nvfp4_sparse_piper import _compile as nvfp4_compile
from piper_kernels.fusions.nvfp4_sparse_piper import _output as nvfp4_output
from piper_kernels.fusions.sparse_piper import _output as output_common


def _operands(sequence=193, heads=3, head_dim=128):
    return (
        torch.empty((2, sequence, 272), device="cuda:1", dtype=torch.int8),
        torch.empty((2, sequence), device="cuda:1"),
        torch.empty((heads * head_dim, 272), device="cuda:1", dtype=torch.int8),
        torch.empty((heads * head_dim, 1), device="cuda:1"),
        torch.empty(head_dim, device="cuda:1", dtype=torch.bfloat16),
        torch.empty((sequence, head_dim * 3 // 4), device="cuda:1"),
        torch.empty((sequence, head_dim * 3 // 4), device="cuda:1"),
    )


def _call(operation, operands, routing=_MINMAX_ROUTING, emit_block_mean=False):
    head_dim = operands[4].shape[0]
    if operation == "query":
        return query._launch_query_projection_range(
            *operands, 1e-6, head_dim**-0.5, routing, chunk_start=64, chunk_rows=129
        )
    if operation == "key":
        return key._launch_key_projection(*operands, 1e-6, routing)
    qdata, scale, weight, weight_scale, *_ = operands
    mean = qdata.new_empty((2, 272), dtype=torch.float32)
    return value._launch_value_projection(
        qdata,
        scale,
        mean,
        weight,
        weight_scale,
        None,
        emit_block_mean=emit_block_mean,
        head_dim=head_dim,
    )


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cuda", "sm121"),
        AcceleratorTarget("hip", "gfx1201"),
        AcceleratorTarget("hip", "gfx1200"),
        AcceleratorTarget("hip", "gfx1100"),
        AcceleratorTarget("cpu"),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_projection_selection_uses_operand_target_and_keeps_support_closed(
    monkeypatch, target, head_dim
):
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(
        torch.cuda, "current_device", Mock(side_effect=AssertionError("current GPU"))
    )
    operand = SimpleNamespace(device=torch.device("cuda:1"))
    expected = None
    if target.is_cuda_capability(12, 0):
        expected = nvidia
    elif (
        sys.platform == "linux"
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
        and head_dim == 128
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
        if name == vendor and fromlist == ("triton",) and level == 1:
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
        result = _call(operation, operands, routing, emit_block_mean=True)
    select.assert_called_once_with(operands[0], head_dim=128)
    execute.assert_called_once()
    assert execute.call_args.args[0] is operands[0]
    assert all(
        left is right for left, right in zip(result, execute.call_args.kwargs["out"], strict=True)
    )
    assert result[0].dtype is torch.int8
    assert all(tensor.dtype is torch.float32 for tensor in result[1:])
    assert (
        set(execute.call_args.kwargs)
        == {
            "query": {"out", "chunk_start", "chunk_rows"},
            "key": {"out"},
            "value": {"out", "emit_block_mean"},
        }[operation]
    )
    if operation == "query":
        assert result[0].shape == (2, 3, 192, 128)
        assert result[1].shape == (2, 3, 6)
        assert result[2].shape == (2, 3, 3, 128)
        assert execute.call_args.kwargs["chunk_start"] == 64
        assert execute.call_args.kwargs["chunk_rows"] == 129
    elif operation == "key":
        assert result[0].shape == (2, 3, 256, 128)
        assert result[3].shape == (2, 3, 0 if routing == _MEAN_ROUTING else 4, 128)
    else:
        assert result[0].shape == (2, 3, 128, 256)
        assert result[3].shape == (2, 3, 4, 128)


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(("architecture", "head_dim"), [("gfx1100", 128), ("gfx1201", 64)])
def test_unvalidated_projection_rejects_before_output_allocation(
    monkeypatch, operation, architecture, head_dim
):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", architecture)
    )
    with FakeTensorMode():
        operands = _operands(head_dim=head_dim)
        monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocated outputs")))
        with pytest.raises(ValueError, match="sparse projections are unavailable"):
            _call(operation, operands)


@pytest.mark.parametrize("missing", [None, "attention", "linear"])
@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("hip", "gfx1200"),
        AcceleratorTarget("hip", "gfx1201"),
    ],
)
def test_output_support_is_independent_of_qkv_projection_support(monkeypatch, missing, target):
    monkeypatch.setattr(sys, "platform", "linux")
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
        ("linux", AcceleratorTarget("cpu")),
        ("win32", AcceleratorTarget("hip", "gfx1200")),
        ("win32", AcceleratorTarget("hip", "gfx1201")),
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
        "key_summary": storage,
        "key_aux": storage,
        "value": storage,
        "value_scale_multiplier": storage,
        "value_mean": storage,
        "head_keep_ratio_units": [2, 2, 2],
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
        query_input_qdata=qdata,
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


def test_shared_output_validation_has_no_target_policy_but_nvfp4_still_does(monkeypatch):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    with FakeTensorMode():
        storage = torch.empty((1, 1, 64, 128), dtype=torch.int8, device="cuda:1")
        assert output_common.validate_attention_output(storage, 64, 128) == 128
        with pytest.raises(ValueError, match="requires exact NVIDIA SM120"):
            nvfp4_output._validate_output_projection(storage, None, None, None, None, None, 64, 128)


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
    "module", [query, key, value, output, _compile, _output_compile, output_common, qk_validation]
)
def test_shared_fusion_does_not_inspect_targets_or_launch_kernels(module):
    _assert_shared_fusion_boundary(Path(module.__file__).read_text())


def test_compiler_cache_keys_include_projection_validation_and_attention_policy():
    assert qk_validation.__file__ in _compile._source_files()
    for compiler in (_compile, nvfp4_compile):
        assert attention_policy.__file__ in compiler._source_files()


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
    arguments = {"sparse_routing_mode": _MINMAX_ROUTING, "sparse_group_size": 16}

    def operand(name, shape, dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype)
        arguments[name] = node
        return node.meta["val"]

    input_value = operand("sparse_input", (1, 128, input_features), torch.bfloat16)
    for kind in ("q", "k", "v"):
        operand(f"sparse_{kind}_weight_qdata", (2 * head_dim, input_features), torch.int8)
        operand(f"sparse_{kind}_weight_scale", (2 * head_dim, 1), torch.float32)
    operand("attention_output", (1, 128, 2, head_dim), torch.bfloat16)
    match = SimpleNamespace(kwargs=arguments, output_node=lambda: arguments["attention_output"])
    return match, input_value


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("input_features", [128, 256])
def test_amd_projection_compiler_selects_actual_attention_width(
    monkeypatch, head_dim, input_features
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    for attribute in ("select_linear_backend", "select_dequantized_mean"):
        monkeypatch.setattr(_compile.linear_backend, attribute, Mock(return_value=object()))
    validate = Mock(return_value=True)
    monkeypatch.setattr(_compile.sparse_piper_compile, "valid_sparse_piper_attention", validate)
    match, _ = _projection_match(head_dim, input_features)
    assert _compile._valid_sparse_piper_projection(match) is (head_dim == 128)
    assert validate.called is (head_dim == 128)


@pytest.mark.parametrize("missing", [None, "projection", "linear", "mean", "attention"])
def test_projection_compiler_requires_each_emitted_operation(monkeypatch, missing):
    selectors = {
        "projection": (_backend, "select_projection_backend"),
        "linear": (_compile.linear_backend, "select_linear_backend"),
        "mean": (_compile.linear_backend, "select_dequantized_mean"),
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
    match, input_value = _projection_match(128)
    assert _compile._valid_sparse_piper_projection(match) is (missing is None)
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
    match = SimpleNamespace(
        output_node=lambda: projected,
        kwargs={
            "output_query": query_node,
            "output_weight_qdata": weight,
            "output_weight_scale": scale,
            "output_attention_shape": [1, 128, 256],
            "output_group_size": 16,
            "output_bias": None,
        },
    )
    assert _output_compile._valid_attention_output(match) is supported
    select.assert_called_once_with(query_node.meta["val"])


def _capture_projection(monkeypatch, operation, implementation=nvidia):
    functions = {
        "query": _kernels._convrot_project_rmsnorm_rope_quantize_query_kernel,
        "key": _kernels._convrot_project_quantize_key_kernel,
        "value": _kernels._convrot_project_quantize_sparse_value_kernel,
    }
    function = functions[operation]
    kernel = MagicMock()
    monkeypatch.setattr(_kernels, function.__name__, kernel)
    monkeypatch.setattr(_kernels, "_project_prepared_input_mean_kernel", MagicMock())
    monkeypatch.setattr(_backend, "require_projection_backend", Mock(return_value=implementation))
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(implementation, "device_context", guard)
    with FakeTensorMode():
        _call(operation, _operands(), emit_block_mean=True)
    guard.assert_called_once_with(torch.device("cuda:1"))
    return function, kernel


@pytest.mark.parametrize("operation", ["query", "key", "value"])
def test_nvidia_launch_schedule_and_fp32_math_are_preserved(monkeypatch, operation):
    _, kernel = _capture_projection(monkeypatch, operation)
    grids = [call.args[0] for call in kernel.__getitem__.call_args_list]
    assert grids == ([(2, 2, 2), (1, 2, 2)] if operation == "query" else [(1, 2, 2)] * 2)
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    for index, call in enumerate(calls):
        assert call.kwargs["block_m"] == (64 if operation == "query" else 128)
        assert call.kwargs["block_n"] == 256
        assert call.kwargs["block_k"] == 128
        assert call.kwargs["num_warps"] == 8
        assert call.kwargs["num_stages"] == 3
        if operation != "value":
            assert call.kwargs["rsqrt_fn"] is libdevice.rsqrt_rn
            assert call.kwargs["mask_ragged_tail"] is (index == 1)


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(
    "target",
    [GPUTarget("cuda", 120, 32), GPUTarget("hip", "gfx1200", 32), GPUTarget("hip", "gfx1201", 32)],
)
@pytest.mark.parametrize("affine", [True, False])
def test_production_launches_compile_without_intermediate_bf16(
    monkeypatch, operation, target, affine
):
    if target.backend == "hip" and sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    function, kernel = _capture_projection(
        monkeypatch, operation, nvidia if target.backend == "cuda" else amd
    )
    for call in kernel.__getitem__.return_value.call_args_list:
        arguments = dict(zip(function.arg_names, call.args, strict=False))
        arguments.update(
            {name: item for name, item in call.kwargs.items() if name in function.arg_names}
        )
        arguments.setdefault("rsqrt_fn", None)
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
        compiled = triton.compile(
            ASTSource(function, signature, constexprs=constants),
            target=target,
            options={name: call.kwargs[name] for name in ("num_warps", "num_stages")},
        )
        if target.backend == "cuda":
            assert compiled.asm["cubin"]
        else:
            assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]
            assert compiled.metadata.shared <= 65536
        assert "arith.truncf" not in compiled.asm["ttgir"]
