"""RDNA4 residual GEMMs preserve projection rounding and chunk storage ownership."""

import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._residual import ResidualAdd, ReusableUpdate, validate_residual_add
from piper_kernels._residual import triton as residual_kernels
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import _backend
from piper_kernels.linear.convrot.int8._amd import policy

amd = _backend._amd_backend

_ROCM_GFX1201 = (
    sys.platform in ("linux", "win32")
    and torch.version.hip is not None
    and torch.cuda.is_available()
    and AcceleratorTarget.from_device(torch.device("cuda")).is_architecture("gfx1201")
)

_gpu_gfx1201 = pytest.mark.skipif(not _ROCM_GFX1201, reason="requires ROCm gfx1201")


@pytest.mark.parametrize(
    ("arch", "dtype", "block_n", "block_k", "rows", "width", "features", "expected"),
    [
        ("gfx1201", torch.float16, 256, 64, 65, 256, 512, True),
        ("gfx1201", torch.bfloat16, 256, 64, 129, 256, 512, True),
        ("gfx1201", torch.bfloat16, 128, 128, 1025, 8192, 2048, True),
        ("gfx1200", torch.bfloat16, 256, 64, 129, 256, 512, False),
        ("gfx1201", torch.float32, 256, 64, 129, 256, 512, False),
        ("gfx1201", torch.bfloat16, 256, 64, 64, 256, 512, False),
        ("gfx1201", torch.bfloat16, 256, 64, 129, 256, 335, False),
        ("gfx1201", torch.bfloat16, 256, 64, 129, 224, 512, False),
        ("gfx1201", torch.bfloat16, 64, 64, 129, 256, 512, False),
        ("gfx1201", torch.bfloat16, 128, 128, 300000, 8192, 2048, False),
    ],
)
def test_gfx1201_residual_epilogue_selection_boundaries(
    arch, dtype, block_n, block_k, rows, width, features, expected
):
    target = AcceleratorTarget("hip", arch)
    plan = replace(
        policy.select_execution_plan(target, in_features=width, rows=rows, out_features=features),
        matmul_block_m=128,
        matmul_block_n=block_n,
        matmul_block_k=block_k,
        matmul_num_warps=8,
        matmul_num_stages=2,
    )
    assert (
        policy.supports_residual_epilogue(
            target,
            plan,
            dtype,
            rows=rows,
            in_features=width,
            out_features=features,
        )
        is expected
    )


def _case(dtype, rows, width, features, kind, bias_dtype):
    torch.manual_seed(307)
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


@pytest.mark.gpu
@_gpu_gfx1201
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("rows", "width", "features", "kind", "bias_dtype", "aligned"),
    [
        (129, 256, 512, "gated", None, True),
        (129, 256, 512, "two", torch.float32, True),
        (1025, 8192, 2048, "two", torch.bfloat16, True),
        (129, 192, 335, "gated", torch.float16, False),
        (64, 256, 512, "gated", None, False),
    ],
)
def test_amd_residual_projection_preserves_math_and_storage(
    dtype, rows, width, features, kind, bias_dtype, aligned, monkeypatch
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
    materialized = amd.linear_prepared(*args)
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
    projected = materialized.double()
    projected *= residual.gate[residual.gate_indices[start : start + rows]].double()
    wanted = (wanted + projected).to(dtype)
    fused = aligned and dtype in (torch.float16, torch.bfloat16)
    standalone = (
        Mock(side_effect=AssertionError("residual launched separately"))
        if fused
        else Mock(wraps=residual_kernels.apply_residual_add)
    )
    monkeypatch.setattr(_backend, "apply_residual_add", standalone)
    actual = amd.linear_prepared_residual(*args[:-1], residual, row_offset=start)
    assert actual.data_ptr() == residual.output.view(-1, features)[start:].data_ptr()
    torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=max(2e-5, torch.finfo(dtype).eps))
    if fused:
        torch.testing.assert_close(residual.output, expected, atol=0, rtol=0)
    actual_rows = residual.output.view(-1, features)
    original_rows = original_output.view(-1, features)
    torch.testing.assert_close(actual_rows[:start], original_rows[:start], rtol=0, atol=0)
    torch.testing.assert_close(
        actual_rows[start + rows :], original_rows[start + rows :], rtol=0, atol=0
    )
    if residual.reusable is not None:
        torch.testing.assert_close(residual.base, original_base, rtol=0, atol=0)
    if fused:
        standalone.assert_not_called()
    else:
        standalone.assert_called_once()


@pytest.mark.gpu
@_gpu_gfx1201
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amd_fused_residual_preserves_projection_rounding(dtype):
    qdata = torch.ones((128, 128), device="cuda", dtype=torch.int8)
    weight = torch.ones((256, 128), device="cuda", dtype=torch.int8)
    scale = torch.ones(128, device="cuda")
    weight_scale = torch.full((256, 1), 1.0002 / 128, device="cuda")
    # A unit gate keeps the product exact, so only projection rounding decides the sum.
    residual = ResidualAdd(
        torch.full((128, 256), -1.0, device="cuda", dtype=dtype),
        torch.ones(1, 256, device="cuda", dtype=dtype),
        torch.zeros(128, device="cuda", dtype=torch.int64),
        False,
    )
    args = (qdata, scale, weight, weight_scale, None, dtype)
    rounded = amd.linear_prepared(*args)
    assert torch.equal(rounded, torch.ones_like(rounded))
    actual = amd.linear_prepared_residual(*args[:-1], residual)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.gpu
@_gpu_gfx1201
@pytest.mark.parametrize("kind", ["gated", "two"])
def test_amd_backend_fallback_keeps_residual_semantics(kind, monkeypatch):
    args, residual, start = _case(torch.bfloat16, 129, 256, 512, kind, torch.float32)
    reference = _clone_residual(residual)
    projection = amd.linear_prepared(*args)
    rows, features = projection.shape
    residual_kernels.apply_residual_add(projection, reference, start)
    monkeypatch.setattr(policy, "supports_residual_epilogue", lambda *args, **kwargs: False)
    standalone = Mock(wraps=residual_kernels.apply_residual_add)
    monkeypatch.setattr(_backend, "apply_residual_add", standalone)
    scratch = torch.empty(rows, features + 8, device="cuda", dtype=args[-1])[:, 4:-4]
    output = residual.output.view(-1, features)[start : start + rows]
    actual = amd.linear_prepared_residual(
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
