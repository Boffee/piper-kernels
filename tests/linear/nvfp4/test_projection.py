"""Precision and bounded-workspace checks for the shared affine GEMM."""

import pytest
import torch
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.prototype.mx_formats.utils import from_blocked

from piper_kernels.linear.nvfp4 import _layout, _ops, _projection, reference

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
        reason="requires exact NVIDIA SM120",
    ),
]


@pytest.mark.parametrize(("rows", "seed"), [(257, 417), (127, 1028)])
def test_independent_preparation_disagreements_are_rounding_boundaries(rows, seed):
    torch.manual_seed(seed)
    input = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    expected_qdata, expected_scale, global_scale = reference.prepare_input(input, None, True)
    actual_qdata, actual_scale, _ = _ops._prepare_compiled(input, global_scale, False)
    expected_scale = from_blocked(expected_scale, rows, 16).float()
    actual_scale = from_blocked(actual_scale, rows, 16).float()
    scale_changed = actual_scale != expected_scale
    unrounded_scale = input.float().reshape(rows, 16, 16).abs().amax(-1) / 6 / global_scale
    scale_midpoint = (actual_scale[scale_changed] + expected_scale[scale_changed]) / 2
    torch.testing.assert_close(unrounded_scale[scale_changed], scale_midpoint, atol=1e-4, rtol=0)

    expected = f4_unpacked_to_f32(unpack_uint4(expected_qdata))
    actual = f4_unpacked_to_f32(unpack_uint4(actual_qdata))
    changed = (actual != expected) & (~scale_changed).repeat_interleave(16, dim=-1)
    scale = expected_scale.repeat_interleave(16, dim=-1)
    normalized = input.float() * ((1.0 / global_scale) / scale)
    midpoint = (expected[changed] + actual[changed]) / 2
    # An approximate reciprocal can move a value across a rounding boundary.
    # This must not conceal changes away from those FP4 midpoints.
    torch.testing.assert_close(normalized[changed], midpoint, atol=1e-6, rtol=0)
    assert scale_changed.any() or changed.any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [None, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("two_level", [False, True])
def test_affine_preserves_accumulator_and_bias_precision(dtype, bias_dtype, two_level):
    rows, features, outputs = 129, 256, 128
    qdata = torch.full((rows, features // 2), 0x22, device="cuda", dtype=torch.uint8)
    qdata[:, 0] = 0x12
    weight = torch.full((outputs, features // 2), 0x22, device="cuda", dtype=torch.uint8)
    scale = torch.ones(_layout.scale_shape(rows, features), device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.ones(_layout.scale_shape(outputs, features), device="cuda").to(
        torch.float8_e4m3fn,
    )
    bias = (
        None
        if bias_dtype is None
        else torch.full((2 * outputs,), -1.0009765625, device="cuda", dtype=bias_dtype)[::2]
    )
    if bias is not None:
        # Distinct skipped lanes expose consumers that ignore the bias stride.
        bias.as_strided((outputs,), (2,), storage_offset=1).fill_(17)
    arguments = (
        qdata,
        scale,
        torch.tensor(1 / 256, device="cuda"),
        weight,
        weight_scale,
        torch.tensor(1.0, device="cuda") if two_level else None,
        bias,
        dtype,
    )
    expected = torch.full((rows, outputs), 255.5 / 256, device="cuda")
    if bias is not None:
        expected += bias.float()
    actual = _ops._execute_prepared(*arguments)

    assert torch.equal(actual, expected.to(dtype))
    assert torch.equal(actual, reference.linear_prepared(*arguments))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_bias_chunks_preserve_strided_output_and_ragged_rows(dtype):
    torch.manual_seed(733)
    rows, features, outputs = 4097, 256, 2048
    input = torch.randn(rows, features, device="cuda", dtype=dtype)  # noqa: A001
    weight = torch.randn(outputs, features, device="cuda", dtype=dtype)
    prepared = reference.prepare_input(input, None, True)
    prepared_weight = reference.prepare_input(weight, None, True)
    bias = torch.randn(outputs, device="cuda", dtype=torch.float32)
    expected = reference.linear_prepared(*prepared, *prepared_weight, bias, dtype)
    storage = torch.full((rows, 2 * outputs), float("nan"), device="cuda", dtype=dtype)
    output = storage[:, outputs:]
    actual = _projection.matmul_prepared_chunk_affine_out(
        *prepared,
        *prepared_weight,
        bias,
        0,
        rows,
        output,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert storage[:, :outputs].isnan().all()
    torch.testing.assert_close(actual, expected)
