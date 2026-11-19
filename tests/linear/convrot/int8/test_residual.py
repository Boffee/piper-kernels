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
        (1024, 256, 2560, "gated", None),
        (1024, 256, 2560, "two", torch.float32),
        (2305, 272, 1025, "two", torch.float16),
    ],
)
def test_residual_projection_preserves_math_and_storage(
    dtype, rows, width, features, kind, bias_dtype, monkeypatch
):
    args, residual, start = _case(dtype, rows, width, features, kind, bias_dtype)
    original_base = residual.base.clone()
    original_output = residual.output.clone()
    qdata, scales, weight, weight_scales, bias, _ = args
    expected_kernel = "gluon_tma" if rows >= 1024 else "triton"
    assert dispatch.default_execution_plan(weight, rows=rows).matmul_kernel == expected_kernel
    # Double precision supplies an independent oracle for integer accumulation,
    # dequantization, and the residual arithmetic before the final storage cast.
    projection = (qdata.double() @ weight.double().T) * scales.double()[:, None]
    projection *= weight_scales.double().T
    if bias is not None:
        projection += bias.double()
    materialized = dispatch.linear_prepared(*args)
    torch.testing.assert_close(
        materialized,
        projection.to(dtype),
        atol=2e-5,
        rtol=max(2e-5, 2 * torch.finfo(dtype).eps),
    )
    reference = _clone_residual(residual)
    expected = reference.output
    residual_kernels.apply_residual_add(materialized, reference, start)
    wanted = original_base.view(-1, features)[start : start + rows].double()
    if residual.reusable is not None:
        previous = original_output.view(-1, features)[start : start + rows].double()
        previous *= residual.reusable.gate[residual.gate_indices[start : start + rows]].double()
        wanted += previous
    wanted = (
        wanted
        + materialized.double()
        * residual.gate[residual.gate_indices[start : start + rows]].double()
    ).to(dtype)
    standalone = (
        Mock(wraps=residual_kernels.apply_residual_add)
        if dtype is torch.float32
        else Mock(side_effect=AssertionError("residual launched separately"))
    )
    tma_launch = Mock(wraps=dispatch.gluon_tma.launch_int8_residual_matmul)
    monkeypatch.setattr(dispatch, "apply_residual_add", standalone)
    monkeypatch.setattr(dispatch.gluon_tma, "launch_int8_residual_matmul", tma_launch)
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
    if expected_kernel == "gluon_tma" and dtype is not torch.float32:
        tma_launch.assert_called_once()
    else:
        tma_launch.assert_not_called()
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["offset", "width_tail", "non_contiguous"])
def test_sm120_tma_residual_falls_back_to_triton_for_unaligned_operands(dtype, case, monkeypatch):
    rows, features = 1024, 2560
    width = 264 if case == "width_tail" else 256
    args, residual, start = _case(dtype, rows, width, features, "two", torch.float32)
    qdata, scales, weight, weight_scales, bias, _ = args
    if case == "offset":
        storage = torch.randint(-127, 128, (rows * width + 8,), device="cuda", dtype=torch.int8)
        storage[8:].copy_(qdata.reshape(-1))
        qdata = storage[8:].view(rows, width)
    elif case == "non_contiguous":
        weight_storage = torch.randint(
            -127, 128, (features, width + 16), device="cuda", dtype=torch.int8
        )
        weight = weight_storage[:, :width]
    assert dispatch.default_execution_plan(weight, rows=rows).matmul_kernel == "gluon_tma"
    original_base = residual.base.clone()
    reference = _clone_residual(residual)
    materialized = dispatch.linear_prepared(qdata, scales, weight, weight_scales, bias, dtype)
    residual_kernels.apply_residual_add(materialized, reference, start)
    monkeypatch.setattr(
        dispatch.gluon_tma,
        "launch_int8_residual_matmul",
        Mock(side_effect=AssertionError("TMA residual launched on unaligned operands")),
    )
    monkeypatch.setattr(
        dispatch,
        "apply_residual_add",
        Mock(side_effect=AssertionError("standalone residual launched")),
    )
    actual = dispatch.linear_prepared_residual(
        qdata, scales, weight, weight_scales, bias, residual, row_offset=start
    )
    assert actual.data_ptr() == residual.output.view(-1, features)[start:].data_ptr()
    torch.testing.assert_close(residual.output, reference.output, atol=0, rtol=0)
    torch.testing.assert_close(residual.base, original_base, atol=0, rtol=0)


def test_sm120_tma_residual_compiles_once_across_row_counts_and_offsets():
    kernel = dispatch.gluon_tma._int8_tma_residual_matmul_kernel
    kernel.device_caches.clear()
    width, features = 256, 2560
    torch.manual_seed(704)
    weight = torch.randint(-127, 128, (features, width), device="cuda", dtype=torch.int8)
    weight_scales = torch.rand(features, 1, device="cuda") * 0.03
    bias = torch.randn(features, device="cuda", dtype=torch.float32)
    for rows, row_offset in ((1024, 0), (1025, 17), (2048, 1024), (2049, 4096)):
        total = row_offset + rows + 9
        qdata = torch.randint(-127, 128, (rows, width), device="cuda", dtype=torch.int8)
        scales = torch.rand(rows, device="cuda") * 0.02
        base = torch.randn(total, features, device="cuda", dtype=torch.bfloat16)
        indices = torch.randint(-3, 3, (total,), device="cuda", dtype=torch.int64)
        gate = torch.randn(3, 3 * features, device="cuda", dtype=torch.bfloat16)[
            :, features : 2 * features
        ]
        previous_gate = torch.randn(5, 2 * features, device="cuda", dtype=torch.bfloat16)[
            :, :features
        ]
        residual = ResidualAdd(
            base, gate, indices, True, ReusableUpdate(torch.randn_like(base), previous_gate)
        )
        validate_residual_add(base, residual, features)
        assert dispatch.default_execution_plan(weight, rows=rows).matmul_kernel == "gluon_tma"
        dispatch.linear_prepared_residual(
            qdata, scales, weight, weight_scales, bias, residual, row_offset=row_offset
        )
    compiled = sum(len(cache[0]) for cache in kernel.device_caches.values())
    assert compiled == 1


def test_sm120_tma_residual_supports_cuda_graph_capture_and_replay(monkeypatch):
    rows, width, features = 1024, 256, 2560
    args, residual, start = _case(torch.bfloat16, rows, width, features, "two", torch.float32)
    qdata, scales, weight, weight_scales, bias, dtype = args
    assert dispatch.default_execution_plan(weight, rows=rows).matmul_kernel == "gluon_tma"
    tma_launch = Mock(wraps=dispatch.gluon_tma.launch_int8_residual_matmul)
    monkeypatch.setattr(dispatch.gluon_tma, "launch_int8_residual_matmul", tma_launch)
    initial_base = residual.base.clone()
    initial_update = residual.reusable.update.clone()
    dispatch.linear_prepared_residual(
        qdata, scales, weight, weight_scales, bias, residual, row_offset=start
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatch.linear_prepared_residual(
            qdata, scales, weight, weight_scales, bias, residual, row_offset=start
        )
    assert tma_launch.call_count == 2
    qdata.random_(-127, 128)
    residual.base.copy_(initial_base)
    residual.reusable.update.copy_(initial_update)
    reference = _clone_residual(residual)
    materialized = dispatch.linear_prepared(qdata, scales, weight, weight_scales, bias, dtype)
    residual_kernels.apply_residual_add(materialized, reference, start)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(residual.output, reference.output, atol=0, rtol=0)
    actual_rows = residual.output.view(-1, features)
    initial_rows = initial_update.view(-1, features)
    torch.testing.assert_close(actual_rows[:start], initial_rows[:start], rtol=0, atol=0)
    torch.testing.assert_close(
        actual_rows[start + rows :], initial_rows[start + rows :], rtol=0, atol=0
    )
    torch.testing.assert_close(residual.base, initial_base, rtol=0, atol=0)
