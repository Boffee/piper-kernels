"""Tests for the semantic chunked NVFP4 SwiGLU FFN operations."""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.fusions.nvfp4_swiglu_ffn import _core
from piper_kernels.fusions.nvfp4_swiglu_ffn.triton import (
    _chunked_swiglu_ffn_gated_updates_op,
    _chunked_swiglu_ffn_op,
)
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

from ._helpers import make_operands, materialized, precise_linear


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("with_weight_global_scale", [False, True])
def test_source_affine_precision_in_strided_workspace(
    bias_dtype: torch.dtype | None,
    with_weight_global_scale: bool,
) -> None:
    operands = make_operands(rows=127, dynamic=False, bias_dtype=bias_dtype, seed=987)
    linear = operands.gate
    if not with_weight_global_scale:
        linear.weight.per_tensor_scale = None
    qdata, scale, global_scale = nvfp4_ops._prepare_compiled(
        operands.input, linear.activation_scale, linear.dynamic
    )
    width = linear.weight.shape[0]
    workspace = torch.full((127, 2 * width), float("nan"), device="cuda", dtype=torch.bfloat16)
    actual = workspace[:, width:]
    _core._project_affine_source_chunk(
        qdata, scale, global_scale, _core.LinearOperands(*linear.arguments()), 127, actual
    )
    if with_weight_global_scale and (bias_dtype is None or bias_dtype is torch.bfloat16):
        reference = precise_linear(operands.input, linear)
        error = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert error < 0.0001
    else:
        # Fallbacks retain the ordinary linear's rounding, including mixed-bias
        # scaling/addition together in FP32.
        assert torch.equal(actual, nvfp4_ops.linear(operands.input, *linear.arguments()))
    assert workspace[:, :width].isnan().all()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("rows", [1, 127, 1536])
def test_compiled_swiglu_scale_matches_fp32_reference(rows: int) -> None:
    torch.manual_seed(989)
    projections = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    value, gate = projections.chunk(2, dim=-1)
    expected = per_tensor_amax_to_scale((value.float() * F.silu(gate.float())).abs().amax())

    torch.testing.assert_close(_core.dynamic_swiglu_scale(projections), expected, rtol=1e-6, atol=0)


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
        seed=901 + rows + dynamic,
    )

    expected = materialized(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(128))

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert actual.dtype is torch.bfloat16
    assert relative_l2 < (0.07 if dynamic else 0.03)


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
        seed=908 + high_first,
    )

    expected = materialized(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(128))

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.03


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_source_preparation_uses_one_global_scale_and_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operands = make_operands(rows=385, dynamic=True, seed=906)
    scale_rows: list[int] = []
    prepared_rows: list[int] = []
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
    operands = make_operands(rows=385, output_features=384, dynamic=dynamic, seed=903 + dynamic)
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
    assert relative_l2 < (0.07 if dynamic else 0.03)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_chunked_ffn_rejects_misaligned_chunk_rows() -> None:
    operands = make_operands(rows=129, dynamic=False, seed=904)

    with pytest.raises(ValueError, match="multiple of 128"):
        _chunked_swiglu_ffn_op(*operands.arguments(129))
