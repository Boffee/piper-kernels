"""Residual GEMMs preserve projection rounding and chunk storage ownership."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from _accelerators import exact_sm120_available

from piper_kernels._residual import ResidualAdd, ReusableUpdate, validate_residual_add
from piper_kernels._residual import triton as residual_kernels
from piper_kernels.linear.convrot.int8._nvidia import dispatch, policy

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not exact_sm120_available(), reason="requires SM120"),
]


def _case(dtype, rows, width, features, kind, bias_dtype):
    torch.manual_seed(301)
    start = 17
    total = start + rows + 9
    qdata = torch.randint(-127, 128, (rows, width), device="cuda", dtype=torch.int8)
    weight = torch.randint(-127, 128, (features, width), device="cuda", dtype=torch.int8)
    scales = torch.rand(rows, device="cuda") * 0.02
    weight_scales = torch.rand(features, 1, device="cuda") * 0.03
    bias = None if bias_dtype is None else torch.randn(features, device="cuda", dtype=bias_dtype)
    base = torch.randn(2, total, features, device="cuda", dtype=dtype)
    # Process the second batch to exercise both batch and window offsets.
    start += total
    indices = torch.randint(-3, 3, (2 * total,), device="cuda", dtype=torch.int64)
    gate = torch.randn(3, 3 * features, device="cuda", dtype=dtype)[:, features : 2 * features]
    reusable = None
    if kind == "two":
        previous_gate = torch.randn(5, 2 * features, device="cuda", dtype=dtype)[:, :features]
        reusable = ReusableUpdate(torch.randn_like(base), previous_gate)
    residual = ResidualAdd(base, gate, indices, True, reusable)
    validate_residual_add(base, residual, features)
    return (qdata, scales, weight, weight_scales, bias, dtype), residual, start


def _clone_residual(residual):
    return replace(
        residual,
        base=residual.base.clone(),
        reusable=None
        if residual.reusable is None
        else replace(residual.reusable, update=residual.reusable.update.clone()),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("rows", "width", "features", "kind", "bias_dtype"),
    [
        (64, 256, 128, "gated", None),
        (64, 256, 128, "two", None),
        (129, 192, 97, "gated", torch.float32),
        (129, 192, 97, "two", torch.float16),
        (32, 128, 64, "gated", torch.bfloat16),
    ],
)
def test_residual_projection_preserves_math_and_storage(
    dtype, rows, width, features, kind, bias_dtype, monkeypatch
):
    args, residual, start = _case(dtype, rows, width, features, kind, bias_dtype)
    original_base = residual.base.clone()
    original_output = residual.output.clone()
    qdata, scales, weight, weight_scales, bias, _ = args
    # Double precision supplies an independent oracle for integer accumulation,
    # dequantization, and the residual arithmetic before the final storage cast.
    projection = (qdata.double() @ weight.double().T) * scales.double()[:, None]
    projection *= weight_scales.double().T
    if bias is not None:
        projection += bias.double()
    projection = projection.to(dtype).double()
    materialized = dispatch.linear_prepared(*args)
    reference = _clone_residual(residual)
    expected = reference.output
    residual_kernels.apply_residual_add(materialized, reference, start)
    wanted = original_base.view(-1, features)[start : start + rows].double()
    if residual.reusable is not None:
        previous = original_output.view(-1, features)[start : start + rows].double()
        previous *= residual.reusable.gate[residual.gate_indices[start : start + rows]].double()
        wanted += previous
    projection *= residual.gate[residual.gate_indices[start : start + rows]].double()
    wanted = (wanted + projection).to(dtype)
    standalone = (
        Mock(wraps=residual_kernels.apply_residual_add)
        if dtype is torch.float32
        else Mock(side_effect=AssertionError("residual launched separately"))
    )
    monkeypatch.setattr(dispatch, "apply_residual_add", standalone)
    actual = dispatch.linear_prepared_residual(*args[:-1], residual, row_offset=start)
    assert actual.data_ptr() == residual.output.view(-1, features)[start:].data_ptr()
    torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=max(2e-5, torch.finfo(dtype).eps))
    if dtype is not torch.float32:
        torch.testing.assert_close(residual.output, expected, atol=0, rtol=0)
    actual_rows = residual.output.view(-1, features)
    original_rows = original_output.view(-1, features)
    torch.testing.assert_close(actual_rows[:start], original_rows[:start], rtol=0, atol=0)
    torch.testing.assert_close(
        actual_rows[start + rows :], original_rows[start + rows :], rtol=0, atol=0
    )
    if residual.reusable is not None:
        torch.testing.assert_close(residual.base, original_base, rtol=0, atol=0)
    if dtype is torch.float32:
        standalone.assert_called_once()
    else:
        standalone.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_residual_preserves_projection_rounding(dtype):
    qdata = torch.ones((64, 128), device="cuda", dtype=torch.int8)
    weight = torch.ones_like(qdata)
    scale = torch.ones(64, device="cuda")
    weight_scale = torch.full((64, 1), 1.0002 / 128, device="cuda")
    # A unit gate keeps the product exact, so only projection rounding decides the sum.
    residual = ResidualAdd(
        torch.full((64, 64), -1.0, device="cuda", dtype=dtype),
        torch.ones(1, 64, device="cuda", dtype=dtype),
        torch.zeros(64, device="cuda", dtype=torch.int64),
        False,
    )
    args = (qdata, scale, weight, weight_scale, None, dtype)
    rounded = dispatch.linear_prepared(*args)
    assert torch.equal(rounded, torch.ones_like(rounded))
    actual = dispatch.linear_prepared_residual(*args[:-1], residual)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["gated", "two"])
def test_backend_fallback_keeps_residual_semantics(kind, monkeypatch):
    args, residual, start = _case(torch.bfloat16, 129, 192, 97, kind, torch.float32)
    reference = _clone_residual(residual)
    projection = dispatch.linear_prepared(*args)
    rows, features = projection.shape
    residual_kernels.apply_residual_add(projection, reference, start)
    monkeypatch.setattr(policy, "supports_residual_epilogue", lambda *args: False)
    standalone = Mock(wraps=residual_kernels.apply_residual_add)
    monkeypatch.setattr(dispatch, "apply_residual_add", standalone)
    scratch = torch.empty(rows, features + 8, device="cuda", dtype=args[-1])[:, 4:-4]
    output = residual.output.view(-1, features)[start : start + rows]
    actual = dispatch.linear_prepared_residual(
        *args[:-1],
        residual,
        row_offset=start,
        projection_scratch=scratch,
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(residual.output, reference.output, rtol=0, atol=0)
    standalone.assert_called_once()
    assert standalone.call_args.args[0].data_ptr() == scratch.data_ptr()
    assert standalone.call_args.args[0].stride() == scratch.stride()
