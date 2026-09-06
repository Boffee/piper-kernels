"""Sparse orchestration selects operations, not vendor layouts or launch plans."""

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import (
    _backend,
    _quantized_dispatch,
    _routes,
    _routing_modes,
    _summaries,
    dispatch,
)
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._interfaces import AttentionBackend
from piper_kernels.attention.sparse_piper_attention._nvidia import policy
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
)


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cuda", "sm121"), False),
        (AcceleratorTarget("cuda", "sm89"), False),
        (AcceleratorTarget("hip", "gfx1201"), False),
        (AcceleratorTarget("hip", "gfx942"), False),
        (AcceleratorTarget("cpu"), False),
        (AcceleratorTarget("xpu"), False),
    ],
)
def test_attention_selection_uses_operand_target(monkeypatch, target, supported):
    query = SimpleNamespace(device=torch.device("cuda:1"))
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("wrong GPU")))
    backend = AttentionBackend(prepare=Mock(), launch=Mock())
    monkeypatch.setattr(_backend, "_nvidia_attention", backend)
    assert policy.supports_target(target) is supported
    assert _backend.select_attention_backend(query) is (backend if supported else None)
    probe.assert_called_once_with(query.device)


def test_missing_attention_implementation_does_not_probe_device(monkeypatch):
    monkeypatch.setattr(_backend, "_nvidia_attention", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1)
    assert _backend.select_attention_backend(query) is None
    with pytest.raises(RuntimeError, match="unavailable on cpu"):
        _backend.require_attention_backend(query)


@pytest.mark.parametrize("missing", ["_route_backend", "_summary_backend"])
def test_missing_auxiliary_implementation_does_not_probe_device(monkeypatch, missing):
    monkeypatch.setattr(_backend, missing, None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    if missing == "_route_backend":
        assert _backend.select_route_selector(torch.empty(1, 2, 1, dtype=torch.uint16)) is None
    else:
        assert _backend.select_sequence_summaries(query, query) is None


@pytest.mark.parametrize("supported", [False, True])
def test_auxiliary_selection_is_independent_of_attention(monkeypatch, supported):
    target = (
        AcceleratorTarget("cuda", "sm120") if supported else AcceleratorTarget("hip", "gfx1201")
    )
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(_backend, "_nvidia_attention", None)
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    routes = torch.empty(1, 2, 1, dtype=torch.uint16)
    assert (_backend.select_route_selector(routes) is not None) is supported
    assert (_backend.select_sequence_summaries(query, query) is not None) is supported
    assert all(call.args == (query.device,) for call in probe.call_args_list)


@pytest.mark.parametrize("invalid", ["dtype", "width", "stride", "key_dtype", "device"])
def test_summary_selection_preserves_tensor_constraints(monkeypatch, invalid):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("cuda", "sm120")
    )
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    if invalid == "dtype":
        query, key = query.float(), key.float()
    elif invalid == "width":
        query, key = query[..., :64], key[..., :64]
    elif invalid == "stride":
        query = query.transpose(-1, -2)
    elif invalid == "key_dtype":
        key = key.half()
    else:
        key = key.to("meta")
    assert _backend.select_sequence_summaries(query, key) is None


def test_dense_orchestration_uses_selected_operations(monkeypatch):
    query = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    key, value = torch.zeros_like(query), torch.zeros_like(query)
    state = object()
    prepare = Mock(return_value=state)
    launch = Mock(side_effect=lambda prepared, output: output.fill_(3))
    backend = AttentionBackend(prepare=prepare, launch=launch)
    select = Mock(return_value=backend)
    monkeypatch.setattr(_backend, "select_attention_backend", select)
    result = dispatch.SparsePiperAttention((0.5,))(query, key, value, sparse_key_blocks=2)
    assert result.shape == query.shape
    assert result.is_contiguous()
    assert (result == 3).all()
    assert result.device == query.device
    assert result.dtype == query.dtype
    select.assert_called_once()
    assert select.call_args.args[0].data_ptr() == query.data_ptr()
    assert prepare.call_args.args[0].data_ptr() == query.data_ptr()
    assert prepare.call_args.kwargs["combined_key"].data_ptr() == key.data_ptr()
    assert prepare.call_args.kwargs["combined_value"].data_ptr() == value.data_ptr()
    assert prepare.call_args.kwargs["sparse_key_blocks"] == 2
    assert launch.call_args.args[0] is state
    assert launch.call_args.args[1].data_ptr() == result.data_ptr()


def test_unsupported_attention_uses_reference_without_native_preparation(monkeypatch):
    query = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    select = Mock(return_value=None)
    reference = Mock(wraps=dispatch.reference_sparse_piper_attention)
    monkeypatch.setattr(_backend, "select_attention_backend", select)
    monkeypatch.setattr(dispatch, "reference_sparse_piper_attention", reference)
    result = dispatch.SparsePiperAttention((0.5,))(query, query, query, sparse_key_blocks=2)
    assert torch.equal(result, torch.zeros_like(query))
    reference.assert_called_once()
    select.assert_called_once()


def test_route_builder_uses_selected_operation_once_and_preserves_offsets(monkeypatch):
    layout = _resolve_route_layout(_normalize_head_keep_ratios((0.5,)), 2, torch.device("cpu"))
    selector = Mock()
    select = Mock(return_value=selector)
    monkeypatch.setattr(_backend, "select_route_selector", select)
    builder = _routes.PackedRouteBuilder(
        layout, batch=1, heads=1, query_blocks=3, sparse_key_blocks=2, device=torch.device("cpu")
    )
    scores = torch.ones(1, 1, 1, 2)
    builder.write(scores, query_block_offset=2)
    select.assert_called_once_with(builder.routes.indices)
    selector.assert_called_once_with(
        scores,
        builder.routes.indices,
        layout.head_keep_blocks,
        layout.route_head_offsets,
        query_block_offset=2,
    )
    assert builder._route_head_offsets is None
    assert builder._head_keep_block_values is None


def test_summaries_use_selected_operation_after_validation(monkeypatch):
    query = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)
    expected = (torch.empty(1, 1, 2, 128),) * 3
    summarize = Mock(return_value=expected)
    select = Mock(return_value=summarize)
    monkeypatch.setattr(_backend, "select_sequence_summaries", select)
    assert (
        _summaries.sequence_block_summaries(query, query, _routing_modes._MINMAX_ROUTING)
        is expected
    )
    select.assert_called_once_with(query, query)
    summarize.assert_called_once_with(query, query, _routing_modes._MINMAX_ROUTING, None)
    with pytest.raises(ValueError, match="rank-four"):
        _summaries.sequence_block_summaries(query[0], query, _routing_modes._MINMAX_ROUTING)
    select.assert_called_once()


def _quantized_context_arguments():
    return {
        "key": torch.zeros(1, 1, 128, 128, dtype=torch.int8),
        "key_scale": torch.ones(1, 1, 2),
        "key_summary": torch.zeros(1, 1, 2, 128),
        "key_aux": torch.zeros(1, 1, 2, 128),
        "value": torch.zeros(1, 1, 128, 128, dtype=torch.int8),
        "value_scale_multiplier": torch.ones(1, 1, 2, 1),
        "value_mean": torch.zeros(1, 1, 128),
        "head_keep_ratio_units": list(_normalize_head_keep_ratios((0.5,))),
        "sparse_key_blocks": 2,
        "logical_sequence_length": 128,
        "routing_mode": _routing_modes._MINMAX_ROUTING,
    }


@pytest.mark.parametrize("coarse", [False, True])
def test_quantized_orchestration_uses_shared_state_and_selected_launcher(monkeypatch, coarse):
    arguments = _quantized_context_arguments()
    if coarse:
        arguments.update(block_mean=torch.ones(1, 1, 2, 128), coarse_scale=0.1)
    launch = Mock()
    select = Mock(return_value=AttentionBackend(prepare=Mock(), launch=launch))
    monkeypatch.setattr(_backend, "select_attention_backend", select)
    context = _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
    query = torch.zeros(1, 1, 64, 128, dtype=torch.int8)
    prepared, pooled = _quantized_dispatch._prepare_quantized_sparse_piper_query(
        context, query, torch.ones(1, 1, 2), torch.zeros(1, 1, 1, 128), global_block_offset=1
    )
    assert prepared.context.key is arguments["key"]
    assert prepared.query.data is query
    assert prepared.query.global_block_offset == 1
    assert (pooled is not None) is coarse
    output = torch.empty(1, 1, 64, 128, dtype=torch.bfloat16)
    gate = torch.ones(1, 64, 1, 128, dtype=torch.bfloat16) if coarse else None
    _quantized_dispatch._launch_quantized_sparse_piper_attention(
        prepared, output, query_block_count=1, coarse_output=pooled, coarse_gate=gate
    )
    launch.assert_called_once_with(
        prepared,
        output,
        query_block_offset=0,
        query_block_count=1,
        coarse_output=pooled,
        coarse_gate=gate,
    )
    assert select.call_args_list[0].args[0] is arguments["key"]
    assert select.call_args_list[1].args[0] is query


def test_quantized_unsupported_backend_rejects_before_allocating_routes(monkeypatch):
    monkeypatch.setattr(_backend, "select_attention_backend", lambda query: None)
    monkeypatch.setattr(
        _quantized_dispatch, "_resolve_route_layout", Mock(side_effect=AssertionError("allocated"))
    )
    with pytest.raises(RuntimeError, match="unavailable on cpu"):
        _quantized_dispatch._prepare_quantized_sparse_piper_context(
            **_quantized_context_arguments()
        )


def test_shared_quantized_validation_rejects_cross_device_query():
    arguments = _quantized_context_arguments()
    context = _prepare_sparse_piper_context_from_quantized(
        arguments["key"],
        arguments["key_scale"],
        arguments["value"],
        arguments["value_scale_multiplier"],
        arguments["value_mean"],
        torch.ones(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        sparse_key_blocks=2,
        routes_per_query=1,
        logical_sequence_length=128,
    )
    with pytest.raises(ValueError, match="share a device"):
        _prepare_sparse_piper_query_from_quantized(
            torch.empty(1, 1, 64, 128, dtype=torch.int8, device="meta"),
            torch.ones(1, 1, 2),
            torch.zeros(1, 1, 1, dtype=torch.uint16),
            context,
        )


def test_sparse_orchestration_has_no_vendor_or_runtime_layout_knowledge():
    directory = Path(dispatch.__file__).parent
    for name in (
        "dispatch.py",
        "_quantized_dispatch.py",
        "_routes.py",
        "_summaries.py",
        "_prepared.py",
        "_routing_modes.py",
    ):
        source = (directory / name).read_text(encoding="utf-8")
        assert "sm120" not in source.lower()
        assert "AcceleratorTarget" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    word in (node.module or "") for word in ("_nvidia", "gluon", "triton")
                )


def test_public_fallback_and_quantized_state_work_without_triton():
    script = """
import importlib.abc
import sys
import torch

class WithoutTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton intentionally unavailable", name="triton")

sys.meta_path.insert(0, WithoutTriton())
from piper_kernels.attention.sparse_piper_attention import SparsePiperAttention, _backend
from piper_kernels.attention.sparse_piper_attention._prepared import _PreparedSparsePiperQuery
from piper_kernels._triton.targets import AcceleratorTarget
query = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
assert _backend.select_attention_backend(query) is None
assert torch.equal(SparsePiperAttention((0.5,))(query, query, query, sparse_key_blocks=2), query)
# Even a supported target cannot select absent optional implementations.
AcceleratorTarget.from_device = lambda device: AcceleratorTarget("cuda", "sm120")
assert _backend.select_route_selector(torch.empty(1, 2, 1, dtype=torch.uint16)) is None
head_major = query.transpose(1, 2)
assert _backend.select_sequence_summaries(head_major, head_major) is None
assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("sequence_length", [128, 193, 320])
def test_unsupported_gpu_retains_portable_attention(monkeypatch, routing, sequence_length):
    torch.manual_seed(987)
    operands = [torch.randn(1, sequence_length, 1, 128, dtype=torch.bfloat16) for _ in range(3)]
    device_operands = [operand.to("cuda") for operand in operands]
    if _backend.select_attention_backend(device_operands[0]) is not None:
        pytest.skip("this test exercises devices without a native sparse-attention backend")
    attention = dispatch.SparsePiperAttention((0.5,), routing=routing)
    expected = attention(*operands, sparse_key_blocks=2, sparse_query_blocks=1)
    reference = Mock(wraps=dispatch.reference_sparse_piper_attention)
    monkeypatch.setattr(dispatch, "reference_sparse_piper_attention", reference)
    actual = attention(*device_operands, sparse_key_blocks=2, sparse_query_blocks=1)
    reference.assert_called_once()
    assert actual.device == device_operands[0].device
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    error = (actual.cpu().float() - expected.float()).norm() / expected.float().norm()
    assert error < 0.02
