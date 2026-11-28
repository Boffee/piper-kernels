"""Public compile-time SymFloat scalar validation contracts under dynamic=True."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Literal

import pytest
import torch

from piper_kernels import (
    SparsePiperAttention,
    piper_attention,
    sage_attention_2pp,
    sparse_piper_coarse_residual,
)
from piper_kernels.conv3d.convrot.int8 import group_norm_silu_conv3d
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_INVALID_POSITIVE_FLOATS = (0.0, -1.0, math.inf, -math.inf, math.nan)
_INVALID_FINITE_FLOATS = (math.inf, -math.inf, math.nan)
_COMPILED_REJECTION_ERRORS = (ValueError, RuntimeError, torch._dynamo.exc.TorchDynamoException)


@pytest.fixture(autouse=True)
def _reset_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _capture_backend() -> tuple[list[torch.fx.GraphModule], object]:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph: torch.fx.GraphModule, _example_inputs: list[object]):
        graphs.append(graph)
        return graph.forward

    return graphs, backend


def _assert_rejects_invalid_positive_scalars(
    invoke: Callable[[Callable[..., torch.Tensor], float], torch.Tensor],
    target: Callable[..., torch.Tensor],
    *,
    warm_values: tuple[float, float] = (0.5, 0.25),
) -> None:
    """Verify invalid positive floats fail eagerly and under cold/warm dynamic=True compile."""
    for bad_value in _INVALID_POSITIVE_FLOATS:
        with pytest.raises(ValueError, match="finite and positive"):
            invoke(target, bad_value)

        torch._dynamo.reset()
        cold = torch.compile(target, backend="eager", fullgraph=True, dynamic=True)
        with pytest.raises(_COMPILED_REJECTION_ERRORS):
            invoke(cold, bad_value)

    torch._dynamo.reset()
    warm = torch.compile(target, backend="eager", fullgraph=True, dynamic=True)
    for valid_value in warm_values:
        invoke(warm, valid_value)
    for bad_value in _INVALID_POSITIVE_FLOATS:
        with pytest.raises(_COMPILED_REJECTION_ERRORS):
            invoke(warm, bad_value)


@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_coarse_residual_compiles_with_dynamic_scales(
    routing: Literal["mean", "minmax"],
    requires_grad: bool,
) -> None:
    generator = torch.Generator().manual_seed(1201)
    shape = (1, 129, 1, 4)
    base_tensors = [torch.randn(shape, generator=generator) for _ in range(4)]

    graphs, capture = _capture_backend()
    compiled = torch.compile(
        sparse_piper_coarse_residual,
        backend=capture if requires_grad else "eager",
        fullgraph=True,
        dynamic=True,
    )

    outputs: list[torch.Tensor] = []
    for scale in (0.5, 0.25, 0.125):
        eager_tensors = [t.clone().requires_grad_(requires_grad) for t in base_tensors]
        compiled_tensors = [t.clone().requires_grad_(requires_grad) for t in base_tensors]

        expected = sparse_piper_coarse_residual(
            *eager_tensors,
            routing=routing,
            coarse_scale=scale,
        )
        actual = compiled(
            *compiled_tensors,
            routing=routing,
            coarse_scale=scale,
        )
        torch.testing.assert_close(actual, expected)
        outputs.append(actual.detach())

        if requires_grad:
            expected.square().sum().backward()
            actual.square().sum().backward()
            for compiled_tensor, eager_tensor in zip(compiled_tensors, eager_tensors, strict=True):
                assert compiled_tensor.grad is not None
                assert eager_tensor.grad is not None
                assert bool(torch.isfinite(compiled_tensor.grad).all())
                torch.testing.assert_close(compiled_tensor.grad, eager_tensor.grad)

    assert not torch.equal(outputs[0], outputs[1])
    if requires_grad:
        assert len(graphs) == 1


@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_coarse_residual_rejects_invalid_scales_eager_and_dynamic(
    routing: Literal["mean", "minmax"],
    requires_grad: bool,
) -> None:
    shape = (1, 65, 1, 4)
    tensors = [torch.randn(shape, requires_grad=requires_grad) for _ in range(4)]

    _assert_rejects_invalid_positive_scalars(
        lambda fn, scale: fn(*tensors, routing=routing, coarse_scale=scale),
        sparse_piper_coarse_residual,
    )


@pytest.mark.parametrize("bad_value", [None, 1j, True, False])
def test_coarse_residual_rejects_non_real_scales(bad_value: object) -> None:
    tensors = [torch.randn(1, 65, 1, 4) for _ in range(4)]

    with pytest.raises(TypeError, match="finite and positive"):
        sparse_piper_coarse_residual(*tensors, routing="mean", coarse_scale=bad_value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "attention",
    [piper_attention, sage_attention_2pp],
    ids=["piper_attention", "sage_attention_2pp"],
)
def test_dense_attention_compiles_single_graph_and_rejects_invalid_dynamic_scales(
    attention,
) -> None:
    generator = torch.Generator().manual_seed(1202)
    query = torch.randn(1, 1, 2, 64, dtype=torch.float16, generator=generator)
    key = torch.randn(1, 1, 4, 64, dtype=torch.float16, generator=generator)
    value = torch.randn(1, 1, 4, 64, dtype=torch.float16, generator=generator)
    for tensor in (query, key, value):
        torch._dynamo.mark_static(tensor)

    # Eager and cold dynamic=True rejection before warming the graph.
    for bad_scale in _INVALID_POSITIVE_FLOATS:
        with pytest.raises(ValueError, match="finite and positive"):
            attention(query, key, value, scale=bad_scale)

        torch._dynamo.reset()
        cold = torch.compile(attention, backend="eager", fullgraph=True, dynamic=True)
        with pytest.raises(_COMPILED_REJECTION_ERRORS):
            cold(query, key, value, scale=bad_scale)

    for bad_value in (1j, True, False):
        with pytest.raises(TypeError, match="finite and positive"):
            attention(query, key, value, scale=bad_value)

    # Single compiled graph across distinct dynamic scales, plus warm rejection.
    torch._dynamo.reset()
    graphs, capture = _capture_backend()
    compiled = torch.compile(attention, backend=capture, fullgraph=True, dynamic=True)

    outputs: list[torch.Tensor] = []
    for scale in (0.5, 0.25, 0.125):
        expected = attention(query, key, value, scale=scale)
        actual = compiled(query, key, value, scale=scale)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        outputs.append(actual)

    assert len(graphs) == 1
    assert not torch.equal(outputs[0], outputs[1])

    for bad_scale in _INVALID_POSITIVE_FLOATS:
        with pytest.raises(_COMPILED_REJECTION_ERRORS):
            compiled(query, key, value, scale=bad_scale)


def test_piper_attention_compiles_single_graph_with_dynamic_window_bounds_and_lse() -> None:
    generator = torch.Generator().manual_seed(1209)
    query = torch.randn(1, 2, 17, 64, dtype=torch.bfloat16, generator=generator)
    key = torch.randn(1, 2, 17, 64, dtype=torch.bfloat16, generator=generator)
    value = torch.randn(1, 2, 17, 64, dtype=torch.bfloat16, generator=generator)
    for tensor in (query, key, value):
        torch._dynamo.mark_static(tensor)

    for bad_left in (-1, -5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            piper_attention(query, key, value, window_left=bad_left)
        torch._dynamo.reset()
        cold = torch.compile(
            lambda q, k, v, wl: piper_attention(q, k, v, window_left=wl, return_lse=True),
            backend="eager",
            fullgraph=True,
            dynamic=True,
        )
        with pytest.raises(_COMPILED_REJECTION_ERRORS):
            cold(query, key, value, bad_left)

    torch._dynamo.reset()
    graphs, capture = _capture_backend()
    compiled = torch.compile(
        lambda q, k, v, wl, wr: piper_attention(
            q, k, v, window_left=wl, window_right=wr, return_lse=True
        ),
        backend=capture,
        fullgraph=True,
        dynamic=True,
    )

    outputs: list[torch.Tensor] = []
    for wl, wr in ((0, 0), (2, 1), (8, 4)):
        exp_out, exp_lse = piper_attention(
            query, key, value, window_left=wl, window_right=wr, return_lse=True
        )
        act_out, act_lse = compiled(query, key, value, wl, wr)
        torch.testing.assert_close(act_out, exp_out, atol=0, rtol=0)
        torch.testing.assert_close(act_lse, exp_lse, atol=0, rtol=0)
        outputs.append(act_out)

    assert len(graphs) == 1
    assert not torch.equal(outputs[0], outputs[1])

    for bad_left in (-1, -3):
        with pytest.raises(_COMPILED_REJECTION_ERRORS):
            compiled(query, key, value, bad_left, 1)


@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_sparse_piper_attention_compiles_with_dynamic_scales(
    routing: Literal["mean", "minmax"],
) -> None:
    generator = torch.Generator().manual_seed(1203)
    shape = (1, 128, 2, 64)
    query = torch.randn(shape, dtype=torch.float16, generator=generator)
    key = torch.randn(shape, dtype=torch.float16, generator=generator)
    value = torch.randn(shape, dtype=torch.float16, generator=generator)
    module = SparsePiperAttention((0.5, 1.0), routing=routing)

    compiled = torch.compile(module, backend="eager", fullgraph=True, dynamic=True)
    outputs: list[torch.Tensor] = []
    for scale in (0.5, 0.25, 0.125):
        expected = module(query, key, value, sparse_key_blocks=2, scale=scale)
        actual = compiled(query, key, value, sparse_key_blocks=2, scale=scale)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        outputs.append(actual)
    assert not torch.equal(outputs[0], outputs[1])


def test_sparse_piper_attention_rejects_invalid_scales_eager_and_dynamic() -> None:
    shape = (1, 128, 2, 64)
    query = torch.randn(shape, dtype=torch.float16)
    key = torch.randn(shape, dtype=torch.float16)
    value = torch.randn(shape, dtype=torch.float16)
    module = SparsePiperAttention((0.5, 1.0), routing="minmax")

    _assert_rejects_invalid_positive_scalars(
        lambda fn, scale: fn(query, key, value, sparse_key_blocks=2, scale=scale),
        module,
    )


def test_group_norm_silu_conv3d_compiles_with_dynamic_epsilon() -> None:
    generator = torch.Generator().manual_seed(1204)
    input_tensor = torch.randn(1, 64, 1, 3, 3, dtype=torch.float16, generator=generator)
    norm_weight = torch.ones(64, dtype=torch.float32)
    norm_bias = torch.zeros(64, dtype=torch.float32)
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(4, 64, 3, 3, 3, dtype=torch.float16, generator=generator),
        group_size=64,
        act_per_tensor_scale=torch.tensor(0.25, dtype=torch.float32),
    )

    def run(inp: torch.Tensor, epsilon: float) -> torch.Tensor:
        return group_norm_silu_conv3d(
            inp,
            norm_weight,
            norm_bias,
            4,
            epsilon,
            weight,
            padding="none",
        )

    compiled = torch.compile(run, backend="eager", fullgraph=True, dynamic=True)
    for epsilon in (1e-5, 1e-4, 1e-3):
        expected = run(input_tensor, epsilon)
        actual = compiled(input_tensor, epsilon)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_group_norm_silu_conv3d_rejects_invalid_epsilon_eager_and_dynamic() -> None:
    input_tensor = torch.randn(1, 64, 1, 3, 3, dtype=torch.float16)
    norm_weight = torch.ones(64, dtype=torch.float32)
    norm_bias = torch.zeros(64, dtype=torch.float32)
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(4, 64, 3, 3, 3, dtype=torch.float16),
        group_size=64,
        act_per_tensor_scale=torch.tensor(0.25, dtype=torch.float32),
    )

    def run(inp: torch.Tensor, epsilon: float) -> torch.Tensor:
        return group_norm_silu_conv3d(
            inp,
            norm_weight,
            norm_bias,
            4,
            epsilon,
            weight,
            padding="none",
        )

    _assert_rejects_invalid_positive_scalars(
        lambda fn, epsilon: fn(input_tensor, epsilon),
        run,
        warm_values=(1e-5, 1e-4),
    )


@pytest.mark.parametrize("bad_value", _INVALID_FINITE_FLOATS)
def test_weight_update_rejects_non_finite_coefficients(bad_value: float) -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(4, 16), group_size=16)
    update = torch.ones(4, 16)
    with pytest.raises(ValueError, match="must be finite"):
        weight.add_(update, alpha=bad_value)


@pytest.mark.parametrize("bad_value", [None, 1j, True, False])
def test_weight_update_rejects_non_real_coefficients(bad_value: object) -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(4, 16), group_size=16)
    update = torch.ones(4, 16)
    with pytest.raises(TypeError, match="must be a real number"):
        weight.add_(update, alpha=bad_value)  # type: ignore[arg-type]
