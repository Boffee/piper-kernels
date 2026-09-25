"""CPU INT8 multiplication preserves the portable ConvRot accumulation contract."""

import math

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels.linear.convrot.int8.reference import linear_prepared


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "out_features"),
    [((32,), 1), ((0, 32), 11), ((2, 0, 64), 11), ((2, 7, 64), 11), ((3, 256), 0)],
)
def test_cpu_prepared_linear_matches_integer_accumulation(dtype, shape, out_features):
    torch.manual_seed(2026)
    # Exercise strided input and an unaligned output width with the full signed range.
    qdata = torch.randint(-128, 128, (*shape[:-1], shape[-1] * 2), dtype=torch.int8)[..., ::2]
    weight = torch.randint(-128, 128, (out_features, shape[-1]), dtype=torch.int8)
    input_scale = torch.full(shape[:-1], 0.125)
    weight_scale = torch.full((out_features, 1), 0.0625)
    bias = torch.arange(out_features, dtype=torch.float32) / 4
    accumulated = qdata.reshape(math.prod(shape[:-1]), shape[-1]).long() @ weight.T.long()
    expected = (accumulated.float() * 0.125 * 0.0625 + bias).to(dtype)

    actual = linear_prepared(qdata, input_scale, weight, weight_scale, dtype, bias)

    assert actual.shape == (*shape[:-1], out_features)
    assert actual.dtype is dtype
    torch.testing.assert_close(actual, expected.reshape(actual.shape), rtol=0, atol=0)


@pytest.mark.parametrize("mkldnn", [False, True])
@pytest.mark.parametrize(("input_value", "weight_value"), [(-128, -128), (-128, 127), (127, 127)])
def test_cpu_prepared_linear_preserves_full_range_accumulation(mkldnn, input_value, weight_value):
    # A byte result or a saturating INT16 intermediate cannot represent these sums.
    width = 1024
    qdata = torch.full((3, width), input_value, dtype=torch.int8)
    weight = torch.full((11, width), weight_value, dtype=torch.int8)
    with torch.backends.mkldnn.flags(enabled=mkldnn):
        actual = linear_prepared(qdata, torch.ones(3), weight, torch.ones(11, 1), torch.float32)

    expected = torch.full((3, 11), float(width * input_value * weight_value))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_prepared_linear_compiles_with_int8_operands():
    torch.manual_seed(2027)
    qdata = torch.randint(-128, 128, (2, 7, 64), dtype=torch.int8)
    weight = torch.randint(-128, 128, (11, 64), dtype=torch.int8)
    input_scale = torch.full((2, 7), 0.125)
    weight_scale = torch.full((11, 1), 0.0625)
    capture = TargetCapturePass()
    compiled = torch.compile(
        linear_prepared, fullgraph=True, options={"post_grad_custom_post_pass": capture}
    )

    with torch.inference_mode():
        actual = compiled(qdata, input_scale, weight, weight_scale, torch.float32)

    accumulated = qdata.reshape(-1, 64).long() @ weight.T.long()
    expected = (accumulated.float() * 0.125 * 0.0625).reshape(2, 7, 11)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.ops.aten._int_mm.default in capture.targets
