"""Tests for direct semantic chunked ConvRot NVFP4 FFN operations."""

import pytest
import torch

from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn.triton import (
    _chunked_swiglu_ffn_gated_updates_op,
    _chunked_swiglu_ffn_op,
)
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_backend

from ._helpers import make_operands, materialized


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("rows", [127, 385], ids=["short", "ragged-multi-chunk"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
def test_chunked_ffn_matches_materialized(
    rows: int,
    dynamic: bool,
    bias_dtype: torch.dtype | None,
) -> None:
    operands = make_operands(
        rows=rows,
        dynamic=dynamic,
        bias_dtype=bias_dtype,
        seed=931 + rows + dynamic,
    )

    expected = materialized(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(128))

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert actual.dtype is torch.bfloat16
    # The independent reference includes portable rotation and FP4 preparation.
    assert relative_l2 < (0.1 if dynamic else 0.06)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("high_first", [False, True])
def test_static_chunked_ffn_supports_nibble_order_and_distinct_input_scales(
    high_first: bool,
) -> None:
    operands = make_operands(
        rows=385,
        dynamic=False,
        high_first=high_first,
        distinct_input_scales=True,
        seed=938 + high_first,
    )

    expected = materialized(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(128))

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.06


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_source_preparation_uses_one_global_scale_and_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operands = make_operands(rows=385, dynamic=True, seed=935)
    scale_rows: list[int] = []
    prepared_rows: list[int] = []
    original_dynamic_scale = convrot_nvfp4_backend.dynamic_scale
    original_prepare_static_out = convrot_nvfp4_backend.prepare_static_out

    def dynamic_scale(input: torch.Tensor, group_size: int) -> torch.Tensor:  # noqa: A002
        scale_rows.append(input.shape[0])
        return original_dynamic_scale(input, group_size)

    def prepare_static_out(
        input: torch.Tensor,  # noqa: A002
        per_tensor_scale: torch.Tensor,
        group_size: int,
        out: tuple[torch.Tensor, torch.Tensor],
        high_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared_rows.append(input.shape[0])
        return original_prepare_static_out(
            input,
            per_tensor_scale,
            group_size,
            out,
            high_first=high_first,
        )

    monkeypatch.setattr(convrot_nvfp4_backend, "dynamic_scale", dynamic_scale)
    monkeypatch.setattr(convrot_nvfp4_backend, "prepare_static_out", prepare_static_out)

    _chunked_swiglu_ffn_op(*operands.arguments(128))

    assert scale_rows == [385]
    assert prepared_rows == [128, 128, 128, 1]


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
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_chunked_ffn_gated_updates_matches_materialized(dynamic: bool) -> None:
    operands = make_operands(rows=385, dynamic=dynamic, seed=934 + dynamic)
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

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert result is None
    assert relative_l2 < (0.1 if dynamic else 0.06)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_chunked_ffn_rejects_mismatched_gate_value_group_sizes() -> None:
    operands = make_operands(rows=129, dynamic=False, seed=936)
    arguments = list(operands.arguments(128))
    arguments[15] = 64

    with pytest.raises(ValueError, match="share a group size"):
        _chunked_swiglu_ffn_op(*arguments)
