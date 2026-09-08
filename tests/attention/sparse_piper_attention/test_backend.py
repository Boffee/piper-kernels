"""Sparse orchestration selects operations, not vendor layouts or launch plans."""

import ast
import subprocess
import sys
from dataclasses import replace
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
    _routing,
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
    _PreparedSparsePiperAttention,
)


@pytest.mark.parametrize("head_dim", [64, 128])
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
def test_attention_selection_uses_operand_target(monkeypatch, target, supported, head_dim):
    query = SimpleNamespace(device=torch.device("cuda:1"), ndim=4, shape=(1, 1, 64, head_dim))
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("wrong GPU")))
    backend = AttentionBackend(prepare=Mock(), launch=Mock())
    d64_backend = replace(backend, skip_dense_routing=True)
    monkeypatch.setattr(_backend, "_nvidia_attention", backend)
    monkeypatch.setattr(_backend, "_nvidia_attention_skip_dense_routing", d64_backend)
    monkeypatch.setattr(_backend, "_amd_attention", None)
    assert policy.supports_target(target) is supported
    expected = d64_backend if head_dim == 64 else backend
    assert _backend.select_attention_backend(query) is (expected if supported else None)
    probe.assert_called_once_with(query.device)


def test_missing_attention_implementation_does_not_probe_device(monkeypatch):
    monkeypatch.setattr(_backend, "_nvidia_attention", None)
    monkeypatch.setattr(_backend, "_amd_attention", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1)
    assert _backend.select_attention_backend(query) is None
    with pytest.raises(RuntimeError, match="unavailable on cpu"):
        _backend.require_attention_backend(query)


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201", "gfx1100", "gfx942"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_amd_attention_selection_is_independent_and_uses_tensor_device(
    monkeypatch, architecture, head_dim
):
    target = AcceleratorTarget("hip", architecture)
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("wrong GPU")))
    monkeypatch.setattr(_backend, "_nvidia_attention", None)
    backend = AttentionBackend(prepare=Mock(), launch=Mock())
    monkeypatch.setattr(_backend, "_amd_attention", backend)
    query = SimpleNamespace(device=torch.device("cuda:1"), ndim=4, shape=(1, 1, 64, head_dim))
    assert _backend.select_attention_backend(query) is (
        backend if architecture in ("gfx1200", "gfx1201") and head_dim == 128 else None
    )
    probe.assert_called_once_with(query.device)


@pytest.mark.parametrize("shape", [(), (0,), (1, 128, 256)])
def test_amd_device_probes_and_projection_inputs_preserve_d128_support(monkeypatch, shape):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    backend = AttentionBackend(prepare=Mock(), launch=Mock())
    monkeypatch.setattr(_backend, "_amd_attention", backend)
    assert _backend.select_attention_backend(torch.empty(shape)) is backend


@pytest.mark.parametrize("missing", ["_route_backend", "_summary_backend"])
def test_missing_auxiliary_implementation_does_not_probe_device(monkeypatch, missing):
    monkeypatch.setattr(_backend, missing, None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    if missing == "_route_backend":
        assert _backend.select_route_selector(torch.empty(1, 2, 1, dtype=torch.uint16)) is None
    else:
        assert _backend.select_sequence_summaries(query, query) is None


def test_missing_score_implementation_does_not_probe_device(monkeypatch):
    monkeypatch.setattr(_backend, "_score_backend", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    summary = torch.empty(1, 1, 32, 128)
    assert _backend.select_minmax_scores(summary, summary, summary) is None


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("hip", "gfx1200"), True),
        (AcceleratorTarget("hip", "gfx1201"), True),
        (AcceleratorTarget("hip", "gfx1100"), False),
        (AcceleratorTarget("hip", "gfx942"), False),
        (AcceleratorTarget("cuda", "sm120"), False),
        (AcceleratorTarget("cpu"), False),
    ],
)
def test_score_selection_is_independent_and_uses_operand_target(monkeypatch, target, supported):
    score = Mock()
    probe = Mock(return_value=target)
    monkeypatch.setattr(_backend, "_score_backend", SimpleNamespace(minmax_scores=score))
    monkeypatch.setattr(_backend, "_amd_attention", None)
    monkeypatch.setattr(_backend, "_route_backend", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("wrong GPU")))
    summary = SimpleNamespace(
        device=torch.device("cuda:1"),
        ndim=4,
        shape=(1, 2, 32, 128),
        dtype=torch.float32,
        stride=lambda dim: 1,
        requires_grad=False,
    )
    selected = _backend.select_minmax_scores(summary, summary, summary)
    assert selected is (score if supported else None)
    probe.assert_called_once_with(summary.device)


@pytest.fixture
def score_selection_operands(monkeypatch):
    """Reject unsupported operands before any device probe or kernel launch."""
    score = Mock()
    monkeypatch.setattr(_backend, "_score_backend", SimpleNamespace(minmax_scores=score))
    monkeypatch.setattr(AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("probe")))
    query = torch.empty(1, 2, 32, 128)
    primary = torch.empty(1, 2, 17, 128)
    return [query, primary, torch.empty_like(primary)]


@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "width",
        "large_query",
        "empty_query",
        "empty_batch",
        "empty_heads",
        "empty_key",
        "heads",
        "auxiliary",
    ],
)
def test_score_selection_rejects_unsupported_shapes(score_selection_operands, invalid):
    query, primary, auxiliary = score_selection_operands
    if invalid == "rank":
        query = query[0]
    elif invalid == "width":
        query = query[..., :64]
    elif invalid == "large_query":
        query = torch.empty(1, 2, 65, 128)
    elif invalid == "empty_query":
        query = query[:, :, :0]
    elif invalid == "empty_batch":
        query, primary, auxiliary = query[:0], primary[:0], auxiliary[:0]
    elif invalid == "empty_heads":
        query, primary, auxiliary = query[:, :0], primary[:, :0], auxiliary[:, :0]
    elif invalid == "empty_key":
        primary, auxiliary = primary[:, :, :0], auxiliary[:, :, :0]
    elif invalid == "heads":
        query = query[:, :1]
    else:
        auxiliary = auxiliary[:, :, :3]
    assert _backend.select_minmax_scores(query, primary, auxiliary) is None


@pytest.mark.parametrize("invalid", ["dtype", "query_stride", "key_stride", "device"])
def test_score_selection_rejects_unsupported_storage(score_selection_operands, invalid):
    query, primary, auxiliary = score_selection_operands
    if invalid == "dtype":
        primary = primary.bfloat16()
    elif invalid == "query_stride":
        query = torch.empty(1, 2, 32, 256)[..., ::2]
    elif invalid == "key_stride":
        primary = torch.empty(1, 2, 17, 256)[..., ::2]
    else:
        auxiliary = auxiliary.to("meta")
    assert _backend.select_minmax_scores(query, primary, auxiliary) is None


@pytest.mark.parametrize("operand", [0, 1, 2], ids=["query", "primary", "auxiliary"])
def test_score_selection_preserves_autograd(score_selection_operands, operand):
    score_selection_operands[operand].requires_grad_()
    with torch.enable_grad():
        assert _backend.select_minmax_scores(*score_selection_operands) is None


def test_scoring_orchestration_selects_minmax_only_and_forwards_scale(monkeypatch):
    generator = torch.Generator().manual_seed(672)
    query, primary, auxiliary = [torch.randn(1, 2, 3, 128, generator=generator) for _ in range(3)]
    expected = torch.empty(1, 2, 3, 3)
    score = Mock(return_value=expected)
    select = Mock(return_value=score)
    monkeypatch.setattr(_backend, "select_minmax_scores", select)
    assert (
        _routing.routing_scores(
            query, primary, auxiliary, _routing_modes._MINMAX_ROUTING, score_scale=0.125
        )
        is expected
    )
    select.assert_called_once_with(query, primary, auxiliary)
    score.assert_called_once_with(query, primary, auxiliary, score_scale=0.125)
    mean_scores = _routing.routing_scores(
        query, primary, auxiliary[:, :, :0], _routing_modes._MEAN_ROUTING
    )
    torch.testing.assert_close(mean_scores, query @ primary.transpose(-1, -2))
    select.assert_called_once()


def test_score_fallback_keeps_autograd(monkeypatch):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    score = Mock(side_effect=AssertionError("non-differentiable kernel"))
    monkeypatch.setattr(_backend, "_score_backend", SimpleNamespace(minmax_scores=score))
    generator = torch.Generator().manual_seed(673)
    tensors = [torch.randn(1, 2, 3, 128, generator=generator, requires_grad=True) for _ in range(3)]
    result = _routing.routing_scores(*tensors, _routing_modes._MINMAX_ROUTING, score_scale=0.125)
    expected = torch.maximum(
        tensors[0] @ tensors[1].transpose(-1, -2) * 0.125,
        tensors[0] @ tensors[2].transpose(-1, -2) * 0.125,
    )
    expected_grad = torch.autograd.grad(expected.sum(), tensors)
    actual_grad = torch.autograd.grad(result.sum(), tensors)
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)
    score.assert_not_called()


@pytest.mark.parametrize(
    ("target", "routes_supported", "summaries_supported"),
    [
        (AcceleratorTarget("cuda", "sm120"), True, True),
        (AcceleratorTarget("hip", "gfx1200"), True, False),
        (AcceleratorTarget("hip", "gfx1201"), True, False),
        (AcceleratorTarget("cuda", "sm121"), False, False),
        (AcceleratorTarget("hip", "gfx1100"), False, False),
        (AcceleratorTarget("hip", "gfx942"), False, False),
        (AcceleratorTarget("cpu"), False, False),
        (AcceleratorTarget("xpu"), False, False),
    ],
)
def test_auxiliary_selection_is_independent_of_attention(
    monkeypatch, target, routes_supported, summaries_supported
):
    probe = Mock(return_value=target)
    monkeypatch.setattr(AcceleratorTarget, "from_device", probe)
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("wrong GPU")))
    monkeypatch.setattr(_backend, "_nvidia_attention", None)
    monkeypatch.setattr(_backend, "_amd_attention", None)
    route_selector, summarize = Mock(), Mock()
    monkeypatch.setattr(
        _backend, "_route_backend", SimpleNamespace(tiled_radix_select_packed_routes=route_selector)
    )
    monkeypatch.setattr(
        _backend, "_summary_backend", SimpleNamespace(sequence_block_summaries=summarize)
    )
    query = SimpleNamespace(
        device=torch.device("cuda:1"),
        shape=(1, 1, 128, 128),
        dtype=torch.bfloat16,
        stride=lambda dim: 1,
    )
    routes = SimpleNamespace(device=query.device)
    assert _backend.select_route_selector(routes) is (route_selector if routes_supported else None)
    assert _backend.select_sequence_summaries(query, query) is (
        summarize if summaries_supported else None
    )
    assert probe.call_count == 2
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
        query, key = query[..., :32], key[..., :32]
    elif invalid == "stride":
        query = query.transpose(-1, -2)
    elif invalid == "key_dtype":
        key = key.half()
    else:
        key = key.to("meta")
    assert _backend.select_sequence_summaries(query, key) is None


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("keep_ratio", [0.5, 1.0])
def test_dense_orchestration_uses_selected_operations(monkeypatch, head_dim, keep_ratio):
    query = torch.zeros(1, 128, 1, head_dim, dtype=torch.bfloat16)
    key, value = torch.zeros_like(query), torch.zeros_like(query)
    state = object()
    prepare = Mock(return_value=state)
    launch = Mock(side_effect=lambda prepared, output: output.fill_(3))
    backend = AttentionBackend(prepare=prepare, launch=launch, skip_dense_routing=head_dim == 64)
    select = Mock(return_value=backend)
    monkeypatch.setattr(_backend, "select_attention_backend", select)
    result = dispatch.SparsePiperAttention((keep_ratio,))(query, key, value, sparse_key_blocks=2)
    assert result.shape == query.shape
    assert result.is_contiguous()
    assert (result == 3).all()
    assert result.device == query.device
    assert result.dtype == query.dtype
    select.assert_called_once()
    assert select.call_args.args[0].data_ptr() == query.data_ptr()
    assert prepare.call_args.args[0].data_ptr() == query.data_ptr()
    route_count = 0 if head_dim == 64 and keep_ratio == 1.0 else int(2 * keep_ratio)
    assert prepare.call_args.args[1].shape[-1] == route_count
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
    monkeypatch.setattr(torch.Tensor, "cpu", Mock(side_effect=AssertionError("host readback")))
    monkeypatch.setattr(torch.Tensor, "tolist", Mock(side_effect=AssertionError("host readback")))
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


def _quantized_context_arguments(head_dim=128):
    return {
        "key": torch.zeros(1, 1, 128, head_dim, dtype=torch.int8),
        "key_scale": torch.ones(1, 1, 2),
        "key_summary": torch.zeros(1, 1, 2, head_dim),
        "key_aux": torch.zeros(1, 1, 2, head_dim),
        "value": torch.zeros(1, 1, head_dim, 128, dtype=torch.int8),
        "value_scale_multiplier": torch.ones(1, 1, 2, 1),
        "value_mean": torch.zeros(1, 1, head_dim),
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
    select.assert_called_once_with(arguments["key"])


def test_backend_context_binding_is_once_per_context_not_per_query(monkeypatch):
    launch = Mock()
    bind = Mock(return_value=launch)
    one_shot = Mock(side_effect=AssertionError("must reuse the bound context"))
    backend = AttentionBackend(prepare=Mock(), launch=one_shot, bind=bind)
    select = Mock(return_value=backend)
    monkeypatch.setattr(_backend, "select_attention_backend", select)
    arguments = _quantized_context_arguments()
    context = _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
    for offset in (0, 1):
        prepared, _ = _quantized_dispatch._prepare_quantized_sparse_piper_query(
            context,
            torch.zeros(1, 1, 64, 128, dtype=torch.int8),
            torch.ones(1, 1, 2),
            torch.zeros(1, 1, 1, 128),
            global_block_offset=offset,
        )
        _quantized_dispatch._launch_quantized_sparse_piper_attention(
            prepared, torch.empty(1, 1, 64, 128, dtype=torch.bfloat16)
        )
    bind.assert_called_once_with(context.kernel_context)
    select.assert_called_once_with(arguments["key"])
    assert launch.call_count == 2
    one_shot.assert_not_called()
    next_context = _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
    assert bind.call_count == 2
    assert bind.call_args.args[0] is next_context.kernel_context


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


def test_prepared_query_must_match_context_head_width():
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
    with pytest.raises(ValueError, match=r"compatible.*Q storage"):
        _prepare_sparse_piper_query_from_quantized(
            torch.empty(1, 1, 64, 64, dtype=torch.int8),
            torch.ones(1, 1, 2),
            torch.zeros(1, 1, 1, dtype=torch.uint16),
            context,
        )


def _prepared_attention(head_dim, routes_per_query):
    arguments = _quantized_context_arguments(head_dim)
    context = _prepare_sparse_piper_context_from_quantized(
        arguments["key"],
        arguments["key_scale"],
        arguments["value"],
        arguments["value_scale_multiplier"],
        arguments["value_mean"],
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        sparse_key_blocks=2,
        routes_per_query=routes_per_query,
        logical_sequence_length=128,
    )
    query = _prepare_sparse_piper_query_from_quantized(
        torch.zeros(1, 1, 128, head_dim, dtype=torch.int8),
        torch.ones(1, 1, 4),
        torch.zeros(1, 2, routes_per_query, dtype=torch.uint16),
        context,
    )
    return _PreparedSparsePiperAttention(context, query)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_shared_preparation_accepts_empty_routes_and_rejects_negative_counts(head_dim):
    prepared = _prepared_attention(head_dim, 0)
    assert prepared.context.routes_per_query == 0
    assert prepared.query.routes.shape == (1, 2, 0)
    with pytest.raises(ValueError, match="route count must be nonnegative"):
        _prepared_attention(head_dim, -1)


@pytest.mark.skipif(_backend.nvidia_gluon is None, reason="requires Triton import")
def test_nvidia_rejects_d128_empty_routes_before_device_execution(monkeypatch):
    native = _backend.nvidia_gluon
    prepared = _prepared_attention(128, 0)
    enter_device = Mock(side_effect=AssertionError("unsupported mode reached device execution"))
    monkeypatch.setattr(native, "device_context", enter_device)
    with pytest.raises(ValueError, match="skip_dense_routing requires NVIDIA D64"):
        native._launch_sparse_piper_attention(
            prepared, torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
        )
    enter_device.assert_not_called()


@pytest.mark.skipif(_backend.amd_gluon is None, reason="requires Triton import")
@pytest.mark.parametrize(
    ("head_dim", "route_count", "message"), [(64, 2, "D128"), (128, 0, "skip_dense_routing")]
)
def test_amd_rejects_unsupported_context_before_packing_or_launch(
    monkeypatch, head_dim, route_count, message
):
    native = _backend.amd_gluon
    prepared = _prepared_attention(head_dim, route_count)
    pack = Mock(side_effect=AssertionError("unsupported context reached packing"))
    enter_device = Mock(side_effect=AssertionError("unsupported context reached device execution"))
    monkeypatch.setattr(native, "pack_context", pack)
    monkeypatch.setattr(native, "device_context", enter_device)
    with pytest.raises(ValueError, match=message):
        native.bind_context(prepared.context)
    for packed in (None, SimpleNamespace(source=prepared.context)):
        with pytest.raises(ValueError, match=message):
            native._launch_sparse_piper_attention(
                prepared,
                torch.empty(1, 1, 128, head_dim, dtype=torch.bfloat16),
                packed=packed,
            )
    pack.assert_not_called()
    enter_device.assert_not_called()


def test_sparse_orchestration_has_no_vendor_or_runtime_layout_knowledge():
    directory = Path(dispatch.__file__).parent
    for name in (
        "dispatch.py",
        "_quantized_dispatch.py",
        "_routes.py",
        "_routing.py",
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
from piper_kernels.attention.sparse_piper_attention._nvidia import policy
assert policy.skip_dense_routing(64)
assert policy.select_attention_schedule(
    64, 32768, 32768, skip_dense_routing=True,
    has_coarse_residual=False, selected_key_rows=32768,
) == (128, 4)
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
