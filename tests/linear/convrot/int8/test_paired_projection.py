"""Paired prepared projections preserve the ordinary INT8 GEMM semantics."""

import pytest
import torch

from piper_kernels.linear.convrot.int8 import triton as backend


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("shape", [(1, 256, 384), (385, 512, 300), (129, 5376, 768)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "bias_dtypes",
    [
        (None, None),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
        (None, torch.float32),
        (torch.float32, None),
    ],
)
def test_paired_projection_matches_separate(shape, dtype, bias_dtypes) -> None:
    torch.manual_seed(891)
    rows, features, width = shape
    qdata = torch.randint(-127, 128, (rows, features), device="cuda", dtype=torch.int8)
    scale = torch.rand(rows, device="cuda") * 0.01
    weights = [
        torch.randint(-127, 128, (width, features), device="cuda", dtype=torch.int8)
        for _ in range(2)
    ]
    scales = [torch.rand(width, device="cuda") * 0.01 for _ in range(2)]
    biases = [
        None if bias_dtype is None else torch.randn(width, device="cuda", dtype=bias_dtype)
        for bias_dtype in bias_dtypes
    ]
    plan = backend.default_execution_plan(weights[0])
    expected = torch.cat(
        [
            backend._execute_prepared_linear(qdata, scale, w, s, b, dtype, plan)
            for w, s, b in zip(weights, scales, biases, strict=True)
        ],
        dim=-1,
    )
    arguments = (qdata, scale, weights[0], scales[0], biases[0], dtype, plan)
    second = (weights[1], scales[1], biases[1])
    actual = backend._execute_prepared_linear(*arguments, second_projection=second)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    storage = torch.full((rows, 2 * width + 32), torch.nan, device="cuda", dtype=dtype)
    out = storage[:, 16:-16]
    result = backend._execute_prepared_linear(*arguments, second_projection=second, out=out)
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert storage[:, :16].isnan().all()
    assert storage[:, -16:].isnan().all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_paired_projection_rejects_mismatched_weights() -> None:
    qdata = torch.zeros((1, 256), device="cuda", dtype=torch.int8)
    scale = torch.ones(1, device="cuda")
    weight = torch.zeros((384, 256), device="cuda", dtype=torch.int8)
    weight_scale = torch.ones(384, device="cuda")
    with pytest.raises(ValueError, match="matching weight shapes"):
        backend._execute_prepared_linear(
            qdata,
            scale,
            weight,
            weight_scale,
            None,
            torch.bfloat16,
            backend.default_execution_plan(weight),
            second_projection=(weight[:128], weight_scale[:128], None),
        )
