"""Semantic ratio and physical route-layout tests."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_head_keep_blocks,
    _resolve_route_layout,
)


def test_h3_ratio_profile_resolves_exact_physical_budget() -> None:
    units = _normalize_head_keep_ratios((0.2, 0.4, 0.6))

    layout = _resolve_route_layout(units, 1036, torch.device("cpu"))

    assert layout.head_keep_blocks.tolist() == [207, 414, 622]
    assert layout.route_head_offsets.tolist() == [0, 207, 621, 1243]
    assert layout.routes_per_query == 1243


def test_head_range_keeps_the_blocks_its_heads_keep_among_all_heads() -> None:
    units = _normalize_head_keep_ratios((0.3,) * 4)

    second = _resolve_route_layout(units, 8, torch.device("cpu"), start_head=2, end_head=4)

    # Rounding gives the two extra blocks to the first two of all four heads; the last
    # two heads alone would take one of them.
    assert _resolve_head_keep_blocks(units, 8) == (3, 3, 2, 2)
    assert _resolve_head_keep_blocks(units[2:], 8) == (3, 2)
    assert second.head_keep_blocks.tolist() == [2, 2]
    assert second.route_head_offsets.tolist() == [0, 2, 4]
    assert second.routes_per_query == 4


def test_largest_remainder_rounding_preserves_nearest_total() -> None:
    attention = SparsePiperAttention((0.5, 0.5))

    assert attention.resolve_head_keep_blocks(3) == (2, 1)


def test_minimum_route_does_not_reduce_a_dense_head() -> None:
    attention = SparsePiperAttention((0.001, 1.0))

    assert attention.resolve_head_keep_blocks(100) == (1, 100)


def test_budget_query_uses_host_configuration_only() -> None:
    class NoTensorOperations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"budget query performed a tensor operation: {func}")

    attention = SparsePiperAttention((0.2, 0.4, 0.6))
    with NoTensorOperations():
        assert attention.resolve_head_keep_blocks(1036) == (207, 414, 622)
        assert attention.resolve_head_keep_blocks(1) == (1, 1, 1)
        assert attention.resolve_head_keep_blocks(65536) == (13107, 26214, 39322)


@pytest.mark.parametrize("sparse_key_blocks", [0, 65537])
def test_budget_query_rejects_out_of_range_prefix(sparse_key_blocks):
    with pytest.raises(ValueError, match="sparse key block count"):
        SparsePiperAttention((0.5,)).resolve_head_keep_blocks(sparse_key_blocks)


@pytest.mark.parametrize("sparse_key_blocks", [True, 1.5])
def test_budget_query_requires_integer_prefix(sparse_key_blocks):
    with pytest.raises(TypeError, match="must be an integer"):
        SparsePiperAttention((0.5,)).resolve_head_keep_blocks(sparse_key_blocks)


@pytest.mark.parametrize("units", [(), (0,), (1_000_001,)])
def test_physical_budget_rejects_invalid_fixed_point_profiles(units: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="ratio profile"):
        _resolve_head_keep_blocks(units, 100)


@pytest.mark.parametrize(
    "ratios",
    [(), (0.0,), (1.1,), (float("nan"),), (float("inf"),)],
)
def test_ratio_profile_rejects_invalid_values(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="head keep ratios"):
        _normalize_head_keep_ratios(ratios)


def test_ratio_profile_rejects_integer_tensor() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        _normalize_head_keep_ratios(torch.tensor([1], dtype=torch.int32))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_route_metadata_survives_graph_capture_and_host_allocation_reuse() -> None:
    graphs, layouts = [], []
    for units in ((250_000, 500_000), (750_000, 1_000_000)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layout = _resolve_route_layout(units, 64, torch.device("cuda"))
        graphs.append(graph)
        layouts.append(layout)
    for _ in range(16):
        torch.empty(5, dtype=torch.int32, pin_memory=True).fill_(-1)
    for graph in reversed(graphs):
        graph.replay()
    assert layouts[0].head_keep_blocks.tolist() == [16, 32]
    assert layouts[0].route_head_offsets.tolist() == [0, 16, 48]
    assert layouts[1].head_keep_blocks.tolist() == [48, 64]
    assert layouts[1].route_head_offsets.tolist() == [0, 48, 112]
