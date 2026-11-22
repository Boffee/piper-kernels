"""Sparse orchestration selects operations, not vendor layouts or launch plans."""

import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from _accelerators import forbid_triton_driver
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention import (
    _backend,
    _contracts,
    _quantized_dispatch,
    _routes,
    _routing,
    _summaries,
    dispatch,
)
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._contracts import (
    AttentionBackend,
    MinmaxScoreSchedule,
    _SparsePiperRatioProfile,
)
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)


@pytest.mark.skipif(
    _backend.nvidia_gluon is None or _backend.amd_gluon is None,
    reason="requires Triton import",
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    ("target", "kind"),
    [
        (AcceleratorTarget("cuda", "sm120"), "tma"),
        (AcceleratorTarget("cuda", "sm89"), "async_copy"),
        (AcceleratorTarget("cuda", "sm86"), "async_copy"),
        (AcceleratorTarget("cuda", "sm80"), "async_copy"),
        (AcceleratorTarget("hip", "gfx1200"), "amd"),
        (AcceleratorTarget("hip", "gfx1201"), "amd"),
        (AcceleratorTarget("cuda", "sm121"), None),
        (AcceleratorTarget("cuda", "sm100"), None),
        (AcceleratorTarget("cuda", "sm90"), None),
        (AcceleratorTarget("hip", "gfx1100"), None),
        (AcceleratorTarget("hip", "gfx942"), None),
        (AcceleratorTarget("cpu"), None),
        (AcceleratorTarget("xpu"), None),
    ],
)
def test_attention_selection_uses_operand_target(target, kind, head_dim):
    assert _backend.nvidia_gluon is not None
    assert _backend.amd_gluon is not None
    query = torch.empty(1, 1, 64, head_dim)
    selected = _backend.select_attention_backend(query, target=target)
    if kind is None:
        assert selected is None
        return
    assert selected is not None
    if kind == "amd":
        assert selected.value_format is ValueFormat.AMD_WMMA
        assert selected.skip_dense_routing is False
        assert selected.bind is _backend.amd_gluon.bind_context
    else:
        assert selected.value_format is ValueFormat.TRANSPOSED
        assert selected.skip_dense_routing is (head_dim == 64)
        expected_bind = (
            _backend.nvidia_gluon.bind_tma_context
            if kind == "tma"
            else _backend.nvidia_gluon.bind_async_copy_context
        )
        assert selected.bind is expected_bind


@pytest.mark.skipif(_backend.nvidia_gluon is None, reason="requires Triton import")
def test_sm8x_attention_uses_its_own_launcher_and_shared_preparation():
    assert _backend.nvidia_gluon is not None
    query = torch.empty(1, 1, 64, 128)
    sm8x = _backend.select_attention_backend(query, target=AcceleratorTarget("cuda", "sm89"))
    sm120 = _backend.select_attention_backend(query, target=AcceleratorTarget("cuda", "sm120"))
    assert sm8x is not None
    assert sm120 is not None
    assert sm8x.bind is _backend.nvidia_gluon.bind_async_copy_context
    assert sm120.bind is _backend.nvidia_gluon.bind_tma_context
    assert sm8x.bind is not sm120.bind
    assert sm8x.prepare is sm120.prepare


@pytest.mark.skipif(_backend.amd_gluon is None, reason="requires Triton import")
@pytest.mark.parametrize("shape", [(), (0,), (1, 128, 256)])
def test_amd_device_probes_and_projection_inputs_preserve_d128_support(shape):
    assert _backend.amd_gluon is not None
    selected = _backend.select_attention_backend(
        torch.empty(shape), target=AcceleratorTarget("hip", "gfx1201")
    )
    assert selected is not None
    assert selected.value_format is ValueFormat.AMD_WMMA
    assert selected.skip_dense_routing is False
    assert selected.bind is _backend.amd_gluon.bind_context


@pytest.mark.skipif(
    _backend.preparation is None or _backend.nvidia_gluon is None or _backend.amd_gluon is None,
    reason="requires Triton import",
)
@pytest.mark.parametrize(
    ("target", "expected_format"),
    [
        (AcceleratorTarget("cuda", "sm120"), ValueFormat.TRANSPOSED),
        (AcceleratorTarget("hip", "gfx1201"), ValueFormat.AMD_WMMA),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("emit_summaries", [False, True])
def test_selected_preparation_owns_backend_storage_contract(
    target,
    expected_format,
    head_dim,
    emit_summaries,
):
    query = torch.empty((0, 3, 65, head_dim), dtype=torch.bfloat16)
    kv = torch.empty((0, 1, 65, head_dim), dtype=torch.bfloat16)
    selected = _backend.select_attention_backend(query, target=target)
    assert selected is not None
    with forbid_triton_driver("empty-batch preparation launched a GPU kernel"):
        operands = selected.prepare(
            query,
            head_dim**-0.5,
            sparse_key_blocks=1,
            combined_key=kv,
            combined_value=kv,
            emit_summaries=emit_summaries,
        )
    assert operands.value_format is expected_format
    assert operands.query.shape == (0, 3, 128, head_dim)
    assert operands.query.dtype is torch.int8
    assert operands.query_scale.shape == (0, 3, 4)
    assert operands.query_scale.dtype is torch.float32
    assert operands.key.shape == (0, 1, 128, head_dim)
    assert operands.key.dtype is torch.int8
    assert operands.key_scale.shape == (0, 1, 2)
    assert operands.key_scale.dtype is torch.float32
    assert operands.key_center.shape == (0, 1, 2, 2, head_dim)
    assert operands.key_center.dtype is torch.int8
    assert operands.key_center_scale.shape == (0, 1, 2, 2)
    assert operands.key_center_scale.dtype is torch.float32
    assert operands.value.shape == expected_format.shape(0, 1, 128, head_dim)
    assert operands.value.dtype is torch.int8
    assert operands.value_scale_multiplier.shape == (0, 1, 2, 1)
    assert operands.value_scale_multiplier.dtype is torch.float32
    assert operands.value_mean.shape == (0, 1, 2, head_dim)
    assert operands.value_mean.dtype is torch.bfloat16
    assert operands.logical_sequence_length == 65
    assert operands.sparse_key_blocks == 1
    if emit_summaries:
        assert operands.query_summary is not None
        assert operands.query_summary.shape == (0, 3, 2, head_dim)
        assert operands.query_summary.dtype is torch.float32
        assert operands.key_summary is not None
        assert operands.key_summary.shape == (0, 1, 1, head_dim)
        assert operands.key_summary.dtype is torch.float32
        assert operands.key_aux is not None
        assert operands.key_aux.shape == (0, 1, 1, head_dim)
        assert operands.key_aux.dtype is torch.float32
    else:
        assert operands.query_summary is None
        assert operands.key_summary is None
        assert operands.key_aux is None
    bound = Mock(side_effect=lambda context: Mock(context=context))
    prepared = operands.with_routes(
        torch.empty((0, 2, 3), dtype=torch.uint16),
        torch.tensor([1, 1, 1], dtype=torch.int32),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        bind=bound,
    )
    bound.assert_called_once()
    assert prepared.context.value_format is expected_format
    assert prepared.context.value is operands.value


@pytest.mark.skipif(
    _backend.nvidia_gluon is None
    or _backend.amd_gluon is None
    or _backend._route_backend is None
    or _backend._score_backend is None
    or _backend._summary_backend is None,
    reason="requires Triton import",
)
@pytest.mark.parametrize("is_hip", [False, True])
def test_selectors_resolve_operand_device_not_current_device(monkeypatch, is_hip):
    device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("wrong GPU")))
    if is_hip:
        monkeypatch.setattr(torch.version, "hip", "7.0")
        probe = Mock(return_value=SimpleNamespace(gcnArchName="gfx1201"))
        monkeypatch.setattr(torch.cuda, "get_device_properties", probe)
    else:
        monkeypatch.setattr(torch.version, "hip", None)
        probe = Mock(return_value=(12, 0))
        monkeypatch.setattr(torch.cuda, "get_device_capability", probe)

    tensor = SimpleNamespace(
        device=device,
        ndim=4,
        shape=(1, 1, 128, 128),
        dtype=torch.float32,
        stride=lambda _dim: 1,
        requires_grad=False,
    )
    assert _backend.select_attention_backend(tensor) is not None  # type: ignore[arg-type]
    assert _backend.select_route_selector(SimpleNamespace(device=device)) is not None  # type: ignore[arg-type]
    assert _backend.select_minmax_scores(tensor, tensor, tensor) is not None  # type: ignore[arg-type]
    assert _backend.select_sequence_summaries(tensor, tensor) is not None  # type: ignore[arg-type]
    assert probe.call_count == 4
    assert all(call.args == (device,) for call in probe.call_args_list)


@pytest.mark.skipif(_backend._score_backend is None, reason="requires Triton import")
@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("hip", "gfx1200"), True),
        (AcceleratorTarget("hip", "gfx1201"), True),
        (AcceleratorTarget("hip", "gfx1100"), False),
        (AcceleratorTarget("hip", "gfx942"), False),
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cuda", "sm89"), False),
        (AcceleratorTarget("cuda", "sm80"), False),
        (AcceleratorTarget("cpu"), False),
    ],
)
def test_score_selection_is_independent_and_uses_operand_target(target, supported):
    assert _backend._score_backend is not None
    summary = torch.empty(1, 2, 32, 128)
    selected = _backend.select_minmax_scores(summary, summary, summary, target=target)
    if not supported:
        assert selected is None
    else:
        assert isinstance(selected, partial)
        assert selected.func is _backend._score_backend.minmax_scores
        schedule = selected.keywords["schedule"]
        assert isinstance(schedule, MinmaxScoreSchedule)
        assert schedule.lds_only_fences is target.is_amd_hip
        assert schedule.block_m > 0
        assert schedule.block_m & (schedule.block_m - 1) == 0
        assert schedule.block_n > 0
        assert schedule.block_n & (schedule.block_n - 1) == 0
        assert schedule.block_k > 0
        assert 128 % schedule.block_k == 0


@pytest.mark.skipif(_backend._score_backend is None, reason="requires Triton import")
@pytest.mark.parametrize(
    "target",
    [AcceleratorTarget("hip", "gfx1201"), AcceleratorTarget("cuda", "sm120")],
)
@pytest.mark.parametrize("query_heads_per_kv_head", [1, 3])
def test_score_selection_accepts_large_query_chunks(target, query_heads_per_kv_head):
    assert _backend._score_backend is not None
    primary = torch.empty(1, 2, 17, 128)
    block_ms = []
    for query_blocks in (1, 32, 65, 128, 384, 385):
        query = torch.empty(1, 2 * query_heads_per_kv_head, query_blocks, 128)
        selected = _backend.select_minmax_scores(query, primary, primary, target=target)
        assert isinstance(selected, partial)
        assert selected.func is _backend._score_backend.minmax_scores
        schedule = selected.keywords["schedule"]
        assert isinstance(schedule, MinmaxScoreSchedule)
        assert schedule.lds_only_fences is target.is_amd_hip
        block_ms.append(schedule.block_m)
    assert block_ms == sorted(block_ms)
    assert block_ms[-1] > block_ms[0]


@pytest.fixture(params=[32, 384], ids=["small_query", "large_query"])
def score_selection_operands(request):
    """Build valid summary shapes that fail validation only on explicit test defects."""
    if _backend._score_backend is None:
        pytest.skip("requires Triton import")
    query = torch.empty(1, 2, request.param, 128)
    primary = torch.empty(1, 2, 17, 128)
    return [query, primary, torch.empty_like(primary)]


@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "width",
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
    assert (
        _backend.select_minmax_scores(
            query, primary, auxiliary, target=AcceleratorTarget("cuda", "sm120")
        )
        is None
    )


@pytest.mark.parametrize("invalid", ["dtype", "query_stride", "key_stride", "device"])
def test_score_selection_rejects_unsupported_storage(score_selection_operands, invalid):
    query, primary, auxiliary = score_selection_operands
    if invalid == "dtype":
        primary = primary.bfloat16()
    elif invalid == "query_stride":
        query = torch.empty(*query.shape[:-1], 256)[..., ::2]
    elif invalid == "key_stride":
        primary = torch.empty(1, 2, 17, 256)[..., ::2]
    else:
        auxiliary = auxiliary.to("meta")
    assert (
        _backend.select_minmax_scores(
            query, primary, auxiliary, target=AcceleratorTarget("cuda", "sm120")
        )
        is None
    )


@pytest.mark.parametrize("operand", [0, 1, 2], ids=["query", "primary", "auxiliary"])
def test_score_selection_preserves_autograd(score_selection_operands, operand):
    score_selection_operands[operand].requires_grad_()
    with torch.enable_grad():
        assert (
            _backend.select_minmax_scores(
                *score_selection_operands, target=AcceleratorTarget("cuda", "sm120")
            )
            is None
        )


def test_scoring_orchestration_selects_minmax_only_and_forwards_scale(monkeypatch):
    generator = torch.Generator().manual_seed(672)
    query, primary, auxiliary = [torch.randn(1, 2, 3, 128, generator=generator) for _ in range(3)]
    expected = torch.empty(1, 2, 3, 3)
    score = Mock(return_value=expected)
    select = Mock(return_value=score)
    monkeypatch.setattr(_backend, "select_minmax_scores", select)
    assert (
        _routing.routing_scores(
            query, primary, auxiliary, _contracts._MINMAX_ROUTING, score_scale=0.125
        )
        is expected
    )
    select.assert_called_once_with(query, primary, auxiliary)
    score.assert_called_once_with(query, primary, auxiliary, score_scale=0.125)
    mean_scores = _routing.routing_scores(
        query, primary, auxiliary[:, :, :0], _contracts._MEAN_ROUTING
    )
    torch.testing.assert_close(mean_scores, query @ primary.transpose(-1, -2))
    select.assert_called_once()


@pytest.mark.parametrize("query_blocks", [3, 384])
def test_score_fallback_keeps_autograd(query_blocks):
    generator = torch.Generator().manual_seed(673)
    tensors = [
        torch.randn(1, 2, blocks, 128, generator=generator, requires_grad=True)
        for blocks in (query_blocks, 3, 3)
    ]
    assert (
        _backend.select_minmax_scores(*tensors, target=AcceleratorTarget("hip", "gfx1201")) is None
    )
    result = _routing.routing_scores(*tensors, _contracts._MINMAX_ROUTING, score_scale=0.125)
    expected = torch.maximum(
        tensors[0] @ tensors[1].transpose(-1, -2) * 0.125,
        tensors[0] @ tensors[2].transpose(-1, -2) * 0.125,
    )
    expected_grad = torch.autograd.grad(expected.sum(), tensors)
    actual_grad = torch.autograd.grad(result.sum(), tensors)
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(
    _backend._route_backend is None or _backend._summary_backend is None,
    reason="requires Triton import",
)
@pytest.mark.parametrize(
    ("target", "routes_supported", "summaries_supported"),
    [
        (AcceleratorTarget("cuda", "sm120"), True, True),
        (AcceleratorTarget("cuda", "sm89"), True, True),
        (AcceleratorTarget("cuda", "sm86"), True, True),
        (AcceleratorTarget("hip", "gfx1200"), True, True),
        (AcceleratorTarget("hip", "gfx1201"), True, True),
        (AcceleratorTarget("cuda", "sm121"), False, False),
        (AcceleratorTarget("cuda", "sm90"), False, False),
        (AcceleratorTarget("hip", "gfx1100"), False, False),
        (AcceleratorTarget("hip", "gfx942"), False, False),
        (AcceleratorTarget("cpu"), False, False),
        (AcceleratorTarget("xpu"), False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_auxiliary_selection_is_independent_of_attention(
    target, routes_supported, summaries_supported, dtype
):
    assert _backend._route_backend is not None
    assert _backend._summary_backend is not None
    query = torch.empty(1, 1, 128, 128, dtype=dtype)
    routes = torch.empty(1, 2, 1, dtype=torch.uint16)
    assert _backend.select_route_selector(routes, target=target) is (
        _backend._route_backend.tiled_radix_select_packed_routes if routes_supported else None
    )
    assert _backend.select_sequence_summaries(query, query, target=target) is (
        _backend._summary_backend.sequence_block_summaries if summaries_supported else None
    )


@pytest.mark.skipif(
    _backend.preparation is None or _backend._summary_backend is None,
    reason="requires Triton import",
)
def test_fused_operand_preparation_follows_summary_support():
    sm89 = AcceleratorTarget("cuda", "sm89")
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    selected = _backend.use_fused_operand_preparation(
        query, query, _contracts._MINMAX_ROUTING, target=sm89
    )
    assert selected is (_backend.select_sequence_summaries(query, query, target=sm89) is not None)
    unsupported = torch.empty(1, 1, 128, 32, dtype=torch.bfloat16)
    assert (
        _backend.use_fused_operand_preparation(
            unsupported, unsupported, _contracts._MINMAX_ROUTING, target=sm89
        )
        is False
    )


@pytest.mark.skipif(_backend.preparation is None, reason="requires Triton import")
def test_mean_pool_routing_keeps_the_separate_summary_pass():
    """Only min/max summaries reduce order-independently, so only they may fuse."""
    sm89 = AcceleratorTarget("cuda", "sm89")
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    assert (
        _backend.use_fused_operand_preparation(query, query, _contracts._MEAN_ROUTING, target=sm89)
        is False
    )
    assert (
        _backend.use_fused_operand_preparation(
            query, query, _contracts._MINMAX_ROUTING, target=sm89
        )
        is True
    )


@pytest.mark.skipif(_backend._summary_backend is None, reason="requires Triton import")
@pytest.mark.parametrize(
    "invalid", ["dtype", "width", "stride", "key_dtype", "device", "noncontiguous_block_lengths"]
)
def test_summary_selection_preserves_tensor_constraints(invalid):
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    block_lengths = None
    if invalid == "dtype":
        query, key = query.double(), key.double()
    elif invalid == "width":
        query, key = query[..., :32], key[..., :32]
    elif invalid == "stride":
        query = query.transpose(-1, -2)
    elif invalid == "key_dtype":
        key = key.half()
    elif invalid == "device":
        key = key.to("meta")
    else:
        block_lengths = torch.tensor([64, 17, 64, 17], dtype=torch.int32)[::2]
    assert (
        _backend.select_sequence_summaries(
            query, key, block_lengths, target=AcceleratorTarget("cuda", "sm120")
        )
        is None
    )


@pytest.mark.skipif(_backend._summary_backend is None, reason="requires Triton import")
@pytest.mark.parametrize("grad_operand", ["query", "key"])
def test_summary_selection_preserves_autograd(grad_operand):
    query = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16, requires_grad=grad_operand == "query")
    key = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16, requires_grad=grad_operand == "key")
    with torch.enable_grad():
        assert (
            _backend.select_sequence_summaries(
                query, key, target=AcceleratorTarget("cuda", "sm120")
            )
            is None
        )


@pytest.mark.parametrize(
    ("sparse_key_blocks", "expected_block_k"),
    [
        (1, 512),
        (512, 512),
        (513, 1024),
        (1024, 1024),
        (1025, 2048),
        (2048, 2048),
        (2049, None),
        (65_536, None),
    ],
)
def test_radix_route_dispatch_selects_single_load_or_tiled_kernel(
    monkeypatch, sparse_key_blocks, expected_block_k
):
    if _backend._route_backend is None:
        pytest.skip("requires Triton")
    routes_triton = _backend._route_backend
    assert routes_triton._select_single_load_block_k(sparse_key_blocks) == expected_block_k

    cuda_device = torch.device("cuda", 0)

    def fake_tensor(shape, dtype):
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= max(size, 1)
        strides.reverse()
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=cuda_device,
            is_contiguous=lambda: True,
            stride=lambda dim=None: strides[dim] if dim is not None else tuple(strides),
        )

    single_launch = Mock()
    single_kernel = MagicMock()
    single_kernel.__getitem__.return_value = single_launch
    tiled_launch = Mock()
    tiled_kernel = MagicMock()
    tiled_kernel.__getitem__.return_value = tiled_launch
    monkeypatch.setattr(torch.cuda, "device", lambda _device=None: nullcontext())
    monkeypatch.setattr(
        routes_triton, "_single_load_radix_select_packed_routes_kernel", single_kernel
    )
    monkeypatch.setattr(routes_triton, "_tiled_radix_select_packed_routes_kernel", tiled_kernel)

    routes_triton.tiled_radix_select_packed_routes(
        fake_tensor((1, 2, 3, sparse_key_blocks), torch.float32),
        fake_tensor((1, 3, 2), torch.uint16),
        fake_tensor((2,), torch.int32),
        fake_tensor((3,), torch.int32),
        query_block_offset=0,
    )
    if expected_block_k is not None:
        single_kernel.__getitem__.assert_called_once_with((3, 2, 1))
        assert single_launch.call_args.kwargs["block_k"] == expected_block_k
        tiled_kernel.__getitem__.assert_not_called()
    else:
        tiled_kernel.__getitem__.assert_called_once_with((3, 2, 1))
        assert tiled_launch.call_args.kwargs["selector_tile"] == routes_triton._SELECTOR_TILE
        single_kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("keep_ratio", [0.5, 1.0])
@pytest.mark.parametrize("fuse_summaries", [False, True])
def test_dense_orchestration_uses_selected_operations(
    monkeypatch, head_dim, keep_ratio, fuse_summaries
):
    query = torch.zeros(1, 128, 1, head_dim, dtype=torch.bfloat16)
    key, value = torch.zeros_like(query), torch.zeros_like(query)
    state = Mock()
    operands = Mock(with_routes=Mock(return_value=state))
    prepare = Mock(return_value=operands)
    launch = Mock(side_effect=lambda prepared, output: output.fill_(3))
    state.launch = launch
    backend = AttentionBackend(
        value_format=ValueFormat.TRANSPOSED,
        prepare=prepare,
        bind=Mock(),
        skip_dense_routing=head_dim == 64,
    )
    select = Mock(return_value=backend)
    monkeypatch.setattr(_backend, "select_attention_backend", select)
    monkeypatch.setattr(_backend, "use_fused_operand_preparation", lambda *args: fuse_summaries)
    # Both policy paths must use the backend already selected for this call.
    monkeypatch.setattr(
        dispatch,
        "packed_routes_from_summaries",
        Mock(
            return_value=_routing.packed_routes_from_sequences(
                query.transpose(1, 2),
                key.transpose(1, 2),
                _resolve_route_layout(_normalize_head_keep_ratios((keep_ratio,)), 2, query.device),
                _contracts._MINMAX_ROUTING,
                skip_dense_routing=backend.skip_dense_routing,
            )
        ),
    )
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
    assert prepare.call_args.args[1] == head_dim**-0.5
    operands.with_routes.assert_called_once()
    assert operands.with_routes.call_args.args[0].shape[-1] == route_count
    assert prepare.call_args.kwargs["combined_key"].data_ptr() == key.data_ptr()
    assert prepare.call_args.kwargs["combined_value"].data_ptr() == value.data_ptr()
    assert prepare.call_args.kwargs["sparse_key_blocks"] == 2
    assert prepare.call_args.kwargs.get("emit_summaries", False) is (
        fuse_summaries and keep_ratio < 1.0
    )
    assert launch.call_args.args[0] is state.query
    assert launch.call_args.args[1].data_ptr() == result.data_ptr()


def test_unsupported_attention_uses_reference_without_native_preparation(monkeypatch):
    query = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    reference = Mock(wraps=dispatch.reference_sparse_piper_attention)
    monkeypatch.setattr(dispatch, "reference_sparse_piper_attention", reference)
    result = dispatch.SparsePiperAttention((0.5,))(query, query, query, sparse_key_blocks=2)
    assert torch.equal(result, torch.zeros_like(query))
    reference.assert_called_once()


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
        force_tile_mask=None,
    )
    assert builder._route_head_offsets is None
    assert builder._head_keep_block_values is None


def test_summaries_use_selected_operation_after_validation(monkeypatch):
    query = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)
    expected = (torch.empty(1, 1, 2, 128),) * 3
    summarize = Mock(return_value=expected)
    select = Mock(return_value=summarize)
    monkeypatch.setattr(_backend, "select_sequence_summaries", select)
    assert _summaries.sequence_block_summaries(query, query, _contracts._MINMAX_ROUTING) is expected
    select.assert_called_once_with(query, query, None)
    summarize.assert_called_once_with(query, query, _contracts._MINMAX_ROUTING, None)
    with pytest.raises(ValueError, match="rank-four"):
        _summaries.sequence_block_summaries(query[0], query, _contracts._MINMAX_ROUTING)
    select.assert_called_once()


def _bind_mock(context, launch):
    launch.context = context
    return launch


def _quantized_context_arguments(head_dim=128, *, device="cpu"):
    return {
        "key": torch.zeros(1, 1, 128, head_dim, device=device, dtype=torch.int8),
        "key_scale": torch.ones(1, 1, 2, device=device),
        "key_center": torch.zeros(1, 1, 2, 2, head_dim, device=device, dtype=torch.int8),
        "key_center_scale": torch.ones(1, 1, 2, 2, device=device),
        "key_summary": torch.zeros(1, 1, 2, head_dim, device=device),
        "key_aux": torch.zeros(1, 1, 2, head_dim, device=device),
        "value": torch.zeros(1, 1, head_dim, 128, device=device, dtype=torch.int8),
        "value_scale_multiplier": torch.ones(1, 1, 2, 1, device=device),
        "value_mean": torch.zeros(1, 1, 2, head_dim, device=device, dtype=torch.bfloat16),
        "ratio_profile": _SparsePiperRatioProfile(_normalize_head_keep_ratios((0.5,))),
        "sparse_key_blocks": 2,
        "logical_sequence_length": 128,
        "routing_mode": _contracts._MINMAX_ROUTING,
    }


@pytest.mark.parametrize("coarse", [False, True])
def test_quantized_orchestration_uses_shared_state_and_selected_launcher(monkeypatch, coarse):
    arguments = _quantized_context_arguments()
    if coarse:
        arguments.update(coarse_scale=0.1, coarse_key_blocks=2)
    launch = Mock()
    select = Mock(
        return_value=AttentionBackend(
            value_format=ValueFormat.TRANSPOSED,
            prepare=Mock(),
            bind=lambda context: _bind_mock(context, launch),
        )
    )
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
        prepared.query,
        output,
        query_block_offset=0,
        query_block_count=1,
        coarse_output=pooled,
        coarse_gate=gate,
    )
    select.assert_called_once_with(arguments["key"])


def test_backend_context_binding_is_once_per_context_not_per_query(monkeypatch):
    launch = Mock()
    bind = Mock(side_effect=lambda context: _bind_mock(context, launch))
    backend = AttentionBackend(value_format=ValueFormat.TRANSPOSED, prepare=Mock(), bind=bind)
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
    next_context = _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
    assert bind.call_count == 2
    assert bind.call_args.args[0] is next_context.kernel_context


def test_quantized_unsupported_backend_rejects_before_allocating_routes(monkeypatch):
    arguments = _quantized_context_arguments()
    monkeypatch.setattr(torch, "tensor", Mock(side_effect=AssertionError("allocated")))
    with pytest.raises(RuntimeError, match="unavailable on cpu"):
        _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)


def test_shared_quantized_validation_rejects_cross_device_query():
    arguments = _quantized_context_arguments()
    context = _prepare_sparse_piper_context_from_quantized(
        arguments["key"],
        arguments["key_scale"],
        arguments["key_center"],
        arguments["key_center_scale"],
        arguments["value"],
        arguments["value_scale_multiplier"],
        arguments["value_mean"],
        torch.ones(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        sparse_key_blocks=2,
        value_format=ValueFormat.TRANSPOSED,
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
        arguments["key_center"],
        arguments["key_center_scale"],
        arguments["value"],
        arguments["value_scale_multiplier"],
        arguments["value_mean"],
        torch.ones(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        sparse_key_blocks=2,
        value_format=ValueFormat.TRANSPOSED,
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


@pytest.mark.parametrize("value_format", list(ValueFormat))
@pytest.mark.parametrize("defect", [None, "shape", "dtype", "strides"])
def test_context_requires_the_consumers_value_storage(monkeypatch, value_format, defect):
    arguments = _quantized_context_arguments()
    value = torch.empty(value_format.shape(1, 1, 128, 128), dtype=torch.int8)
    if defect == "shape":
        value = value.flatten()
    elif defect == "dtype":
        value = value.float()
    elif defect == "strides":
        value = value.transpose(-1, -2).contiguous().transpose(-1, -2)
    arguments["value"] = value
    bind = Mock(side_effect=lambda context: Mock(context=context))
    backend = AttentionBackend(value_format=value_format, prepare=Mock(), bind=bind)
    monkeypatch.setattr(_backend, "select_attention_backend", lambda query: backend)
    if defect is not None:
        with pytest.raises(ValueError, match=r"V must use|contiguous"):
            _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
        bind.assert_not_called()
    else:
        prepared = _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
        assert prepared.kernel_context.value is value
        assert prepared.kernel_context.value_format is value_format
        bind.assert_called_once_with(prepared.kernel_context)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("width", "K64-aligned D64/D128"),
        ("rows", "K64-aligned D64/D128"),
        ("logical_length", "padded logical sequence"),
        ("scales", "one value per K64"),
    ],
)
def test_context_rejects_invalid_storage_before_binding(monkeypatch, defect, message):
    arguments = _quantized_context_arguments()
    if defect == "width":
        arguments["key"] = torch.zeros(1, 1, 128, 32, dtype=torch.int8)
    elif defect == "rows":
        arguments["key"] = torch.zeros(1, 1, 65, 128, dtype=torch.int8)
    elif defect == "logical_length":
        arguments["logical_sequence_length"] = 64
    else:
        arguments["value_scale_multiplier"] = torch.ones(1, 1, 2, 2)
    bind = Mock(side_effect=AssertionError("invalid storage reached binding"))
    backend = AttentionBackend(value_format=ValueFormat.TRANSPOSED, prepare=Mock(), bind=bind)
    monkeypatch.setattr(_backend, "select_attention_backend", lambda query: backend)
    with pytest.raises(ValueError, match=message):
        _quantized_dispatch._prepare_quantized_sparse_piper_context(**arguments)
    bind.assert_not_called()


@pytest.mark.skipif(_backend.nvidia_gluon is None, reason="requires Triton import")
def test_tma_reuses_context_descriptors_across_query_windows(monkeypatch):
    native = _backend.nvidia_gluon
    prepared = _prepared_attention(128, 2)
    descriptors = []

    def make_descriptor(*args):
        descriptor = object()
        descriptors.append(descriptor)
        return descriptor

    describe = Mock(side_effect=make_descriptor)
    kernel = MagicMock()
    monkeypatch.setattr(native, "_descriptor", describe)
    monkeypatch.setattr(native, "_sparse_piper_attention_kernel", kernel)
    monkeypatch.setattr(native, "install_uint8_int8_dot_hook", lambda: None)
    launch = native.bind_tma_context(prepared.context)
    assert describe.call_count == 2
    assert describe.call_args_list[0].args[0] is prepared.context.key
    assert describe.call_args_list[1].args[0] is prepared.context.value
    key_descriptor, value_descriptor = descriptors[0], descriptors[1]
    for offset in (0, 1):
        local = _prepare_sparse_piper_query_from_quantized(
            prepared.query.data[:, :, offset * 64 : (offset + 1) * 64].contiguous(),
            prepared.query.scale[:, :, offset * 2 : (offset + 1) * 2].contiguous(),
            prepared.query.routes[:, offset : offset + 1].contiguous(),
            prepared.context,
            global_block_offset=offset,
        )
        launch(local, torch.empty(1, 1, 64, 128, dtype=torch.bfloat16))
        assert describe.call_args.args[0] is local.data
        assert kernel.__getitem__.return_value.call_args.args[1] is key_descriptor
        assert kernel.__getitem__.return_value.call_args.args[2] is value_descriptor
    assert describe.call_count == 4  # Two K/V descriptors and one per Q window.


def test_backend_rejects_mismatched_producer_format_before_binding():
    prepared = _prepared_attention(128, 1)
    bind = Mock(side_effect=AssertionError("incompatible context reached backend"))
    backend = AttentionBackend(value_format=ValueFormat.AMD_WMMA, prepare=Mock(), bind=bind)
    with pytest.raises(ValueError, match="requires AMD_WMMA"):
        backend.bind_context(prepared.context)
    bind.assert_not_called()


def _prepared_attention(head_dim, routes_per_query, *, device="cpu"):
    arguments = _quantized_context_arguments(head_dim, device=device)
    context = _prepare_sparse_piper_context_from_quantized(
        arguments["key"],
        arguments["key_scale"],
        arguments["key_center"],
        arguments["key_center_scale"],
        arguments["value"],
        arguments["value_scale_multiplier"],
        arguments["value_mean"],
        torch.tensor([2], device=device, dtype=torch.int32),
        torch.tensor([0, 2], device=device, dtype=torch.int32),
        sparse_key_blocks=2,
        value_format=ValueFormat.TRANSPOSED,
        routes_per_query=routes_per_query,
        logical_sequence_length=128,
    )
    query = _prepare_sparse_piper_query_from_quantized(
        torch.zeros(1, 1, 128, head_dim, device=device, dtype=torch.int8),
        torch.ones(1, 1, 4, device=device),
        torch.zeros(1, 2, routes_per_query, device=device, dtype=torch.uint16),
        context,
    )
    return _PreparedSparsePiperAttention(Mock(context=context), query)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_shared_preparation_accepts_empty_routes_and_rejects_negative_counts(head_dim):
    prepared = _prepared_attention(head_dim, 0)
    assert prepared.context.routes_per_query == 0
    assert prepared.query.routes.shape == (1, 2, 0)
    with pytest.raises(ValueError, match="route count must be nonnegative"):
        _prepared_attention(head_dim, -1)


@pytest.mark.skipif(_backend.nvidia_gluon is None, reason="requires Triton import")
@pytest.mark.parametrize("binder", ["bind_tma_context", "bind_async_copy_context"])
def test_nvidia_rejects_d128_empty_routes_before_device_execution(monkeypatch, binder):
    native = _backend.nvidia_gluon
    with FakeTensorMode():
        prepared = _prepared_attention(128, 0, device="cuda:0")
        output = torch.empty(1, 1, 128, 128, device="cuda:0", dtype=torch.bfloat16)
    enter_device = Mock(side_effect=AssertionError("unsupported mode reached device execution"))
    monkeypatch.setattr(torch.cuda, "device", enter_device)
    with pytest.raises(ValueError, match="skip_dense_routing requires NVIDIA D64"):
        getattr(native, binder)(prepared.context)(prepared.query, output)
    enter_device.assert_not_called()


@pytest.mark.skipif(_backend.nvidia_gluon is None, reason="requires Triton import")
@pytest.mark.parametrize("operand", ["query", "key", "value"])
@pytest.mark.parametrize("defect", ["noncontiguous", "misaligned"])
def test_async_copy_rejects_uncopyable_storage_before_device_execution(
    monkeypatch, operand, defect
):
    native = _backend.nvidia_gluon
    prepared = _prepared_attention(128, 2)
    tensor = {
        "query": prepared.query.data,
        "key": prepared.context.key,
        "value": prepared.context.value,
    }[operand]
    if defect == "noncontiguous":
        replacement = torch.empty_like(tensor).transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        replacement = torch.empty(tensor.numel() + 1, dtype=tensor.dtype)[1:].view(tensor.shape)
    if operand == "query":
        prepared = replace(prepared, query=replace(prepared.query, data=replacement))
    else:
        prepared = replace(
            prepared, launch=Mock(context=replace(prepared.context, **{operand: replacement}))
        )
    with forbid_triton_driver("unsupported storage reached device execution") as enter_device:
        monkeypatch.setattr(torch.cuda, "device", enter_device)
        with pytest.raises(ValueError, match="16-byte-aligned"):
            native.bind_async_copy_context(prepared.context)(
                prepared.query, torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
            )
    enter_device.assert_not_called()


@pytest.mark.skipif(_backend.amd_gluon is None, reason="requires Triton import")
@pytest.mark.parametrize(
    ("head_dim", "route_count", "message"),
    [(64, 0, "skip_dense_routing"), (128, 0, "skip_dense_routing")],
)
def test_amd_rejects_unsupported_context_before_parameters_or_launch(
    monkeypatch, head_dim, route_count, message
):
    native = _backend.amd_gluon
    with FakeTensorMode():
        prepared = _prepared_attention(head_dim, route_count, device="cuda:0")
    parameters = Mock(
        side_effect=AssertionError("unsupported context reached parameter preparation")
    )
    enter_device = Mock(side_effect=AssertionError("unsupported context reached device execution"))
    monkeypatch.setattr(native, "prepare_parameters", parameters)
    monkeypatch.setattr(torch.cuda, "device", enter_device)
    with pytest.raises(ValueError, match=message):
        native.bind_context(prepared.context)
    parameters.assert_not_called()
    enter_device.assert_not_called()


def test_public_fallback_and_quantized_state_work_without_triton():
    script = """
import importlib.abc
import pytest
import sys
import torch

class WithoutTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton intentionally unavailable", name="triton")

sys.meta_path.insert(0, WithoutTriton())
torch.cuda.get_device_capability = lambda *_a: (_ for _ in ()).throw(AssertionError("probe"))
torch.cuda.get_device_properties = lambda *_a: (_ for _ in ()).throw(AssertionError("probe"))
from piper_kernels.attention.sparse_piper_attention import (
    SparsePiperAttention,
    _backend,
    _contracts,
    _prepared,
    _quantized_dispatch,
    _routes,
    _routing,
    _summaries,
    dispatch,
)
from piper_kernels._triton.targets import AcceleratorTarget
assert all(
    module is not None
    for module in (
        dispatch, _quantized_dispatch, _routes, _routing, _summaries, _prepared, _contracts
    )
)
# Shared orchestration imports without Triton or any vendor kernel modules.
assert not any(
    name == "triton"
    or name.startswith("triton.")
    or ".gluon" in name
    or "._nvidia.gluon" in name
    or "._amd.gluon" in name
    for name in sys.modules
)
query = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
sm120 = AcceleratorTarget("cuda", "sm120")
assert _backend.select_attention_backend(query) is None
assert _backend.select_attention_backend(query, target=sm120) is None
with pytest.raises(RuntimeError, match="unavailable on cpu"):
    _backend.require_attention_backend(query)
from piper_kernels.attention.sparse_piper_attention._nvidia import policy
from piper_kernels.attention.sparse_piper_attention._contracts import AttentionSchedule
assert policy.skip_dense_routing(64)
schedule = policy.select_sm120_attention_schedule(
    64, 32768, 32768, skip_dense_routing=True,
    has_coarse_residual=False, selected_key_rows=32768,
)
assert isinstance(schedule, AttentionSchedule)
assert schedule.block_m in (64, 128) and schedule.num_warps in (2, 4)
assert torch.equal(SparsePiperAttention((0.5,))(query, query, query, sparse_key_blocks=2), query)
assert (
    _backend.select_route_selector(torch.empty(1, 2, 1, dtype=torch.uint16), target=sm120) is None
)
head_major = query.transpose(1, 2)
assert _backend.select_sequence_summaries(head_major, head_major, target=sm120) is None
summary = torch.empty(1, 1, 32, 128)
assert _backend.select_minmax_scores(summary, summary, summary, target=sm120) is None
assert (
    _backend.use_fused_operand_preparation(
        head_major, head_major, _contracts._MINMAX_ROUTING, target=sm120
    )
    is False
)
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
