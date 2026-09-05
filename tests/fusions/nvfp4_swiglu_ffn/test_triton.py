"""Tests for the direct chunked NVFP4 SwiGLU FFN operations."""

import pytest
import torch

from piper_kernels.fusions.nvfp4_swiglu_ffn.triton import (
    _chunked_swiglu_ffn_gated_updates_op,
    _chunked_swiglu_ffn_op,
)
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

from ._helpers import dense_reference, down_affine_reference, make_operands, materialized


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("rows", [127, 128, 129, 385])
@pytest.mark.parametrize("chunk_rows", [128, 256])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_static_chunked_ffn_matches_down_affine_reference(
    rows: int,
    chunk_rows: int,
    with_bias: bool,
) -> None:
    operands = make_operands(
        rows=rows,
        dynamic=False,
        bias_dtype=torch.bfloat16 if with_bias else None,
    )

    expected = down_affine_reference(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(chunk_rows))

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
def test_static_chunked_ffn_supports_mixed_precision_bias(
    bias_dtype: torch.dtype,
) -> None:
    operands = make_operands(
        rows=129,
        dynamic=False,
        bias_dtype=bias_dtype,
        seed=909,
    )

    expected = down_affine_reference(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(128))

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_high_first_chunked_ffn_matches_materialized_reference() -> None:
    operands = make_operands(rows=129, dynamic=False, high_first=True, seed=908)

    actual = _chunked_swiglu_ffn_op(*operands.arguments(128))

    assert torch.equal(actual, down_affine_reference(operands))


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("chunk_rows", [128, 256])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_dynamic_chunked_ffn_retains_dense_accuracy(
    chunk_rows: int,
    with_bias: bool,
) -> None:
    operands = make_operands(
        dynamic=True,
        bias_dtype=torch.bfloat16 if with_bias else None,
        seed=902,
    )

    expected = materialized(operands)
    dense = dense_reference(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(chunk_rows))

    relative_to_materialized = (actual.float() - expected.float()).norm() / expected.float().norm()
    materialized_error = (expected.float() - dense.float()).norm() / dense.float().norm()
    chunked_error = (actual.float() - dense.float()).norm() / dense.float().norm()
    assert relative_to_materialized < 0.07
    assert chunked_error <= materialized_error + 0.002


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_up_preparation_uses_global_scale_and_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operands = make_operands(rows=385, dynamic=True, seed=906)
    scale_rows = []
    prepared_rows = []
    original_dynamic_scale = nvfp4_backend.dynamic_scale
    original_prepare_static_out = nvfp4_backend.prepare_static_out

    def dynamic_scale(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        scale_rows.append(input.shape[0])
        return original_dynamic_scale(input)

    def prepare_static_out(
        input: torch.Tensor,  # noqa: A002
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
        high_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared_rows.append(input.shape[0])
        return original_prepare_static_out(
            input,
            per_tensor_scale,
            out,
            high_first=high_first,
        )

    monkeypatch.setattr(nvfp4_backend, "dynamic_scale", dynamic_scale)
    monkeypatch.setattr(nvfp4_backend, "prepare_static_out", prepare_static_out)

    _chunked_swiglu_ffn_op(*operands.arguments(128))

    assert scale_rows == [385]
    assert prepared_rows == [128, 128, 128, 1]


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_chunked_ffn_gated_updates_retains_materialized_accuracy() -> None:
    operands = make_operands(
        rows=385,
        output_features=384,
        dynamic=True,
        seed=905,
    )
    rows, output_features = operands.input.shape[0], operands.down.weight.shape[0]
    base = torch.randn(rows, output_features, device="cuda", dtype=torch.bfloat16)
    reusable_update = torch.randn_like(base)
    gate_storage = torch.randn(7, 6 * output_features, device="cuda", dtype=torch.bfloat16)
    update_gate = gate_storage[:, 2 * output_features : 3 * output_features]
    ffn_gate = gate_storage[:, 5 * output_features :]
    gate_indices = torch.randint(0, 7, (rows,), device="cuda", dtype=torch.int64)
    expected = _materialized_gated_updates(
        materialized(operands),
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
    )
    actual = reusable_update.clone()

    result = _chunked_swiglu_ffn_gated_updates_op(
        *operands.arguments(128)[:-1],
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        False,
        128,
    )

    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert result is None
    assert relative_error < 0.07


def _materialized_gated_updates(
    ffn: torch.Tensor,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
) -> torch.Tensor:
    hidden = base.float() + update_gate[gate_indices].float() * reusable_update.float()
    return (hidden + ffn_gate[gate_indices].float() * ffn.float()).to(base.dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_static_chunked_ffn_gated_updates_matches_materialized() -> None:
    operands = make_operands(
        rows=385,
        output_features=384,
        dynamic=False,
        seed=903,
    )
    rows, output_features = operands.input.shape[0], operands.down.weight.shape[0]
    base = torch.randn(rows, output_features, device="cuda", dtype=torch.bfloat16)
    reusable_update = torch.randn_like(base)
    gate_storage = torch.randn(7, 6 * output_features, device="cuda", dtype=torch.bfloat16)
    update_gate = gate_storage[:, 2 * output_features : 3 * output_features]
    ffn_gate = gate_storage[:, 5 * output_features :]
    gate_indices = torch.randint(0, 7, (rows,), device="cuda", dtype=torch.int64)
    expected = _materialized_gated_updates(
        materialized(operands),
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
    )
    actual = reusable_update.clone()

    result = _chunked_swiglu_ffn_gated_updates_op(
        *operands.arguments(128)[:-1],
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        False,
        128,
    )

    assert result is None
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_chunked_ffn_rejects_misaligned_chunk_rows() -> None:
    operands = make_operands(rows=129, dynamic=False, seed=904)

    with pytest.raises(ValueError, match="multiple of 128"):
        _chunked_swiglu_ffn_op(*operands.arguments(129))
