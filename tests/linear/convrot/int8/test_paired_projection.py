"""Paired prepared projections preserve the ordinary INT8 GEMM semantics."""

import pytest
import torch
from _accelerators import exact_sm120_available
from torch.nn import functional as F  # noqa: N812

from piper_kernels.linear.convrot.int8._backend import require_linear_backend


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
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
    implementation = require_linear_backend(qdata)
    expected = torch.cat(
        [
            implementation.linear_prepared(qdata, scale, w, s, b, dtype)
            for w, s, b in zip(weights, scales, biases, strict=True)
        ],
        dim=-1,
    )
    arguments = (qdata, scale, weights[0], scales[0], biases[0], dtype)
    second = (weights[1], scales[1], biases[1])
    actual = implementation.linear_prepared(*arguments, second_projection=second)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    storage = torch.full((rows, 2 * width + 32), torch.nan, device="cuda", dtype=dtype)
    out = storage[:, 16:-16]
    result = implementation.linear_prepared(*arguments, second_projection=second, out=out)
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert storage[:, :16].isnan().all()
    assert storage[:, -16:].isnan().all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_paired_projection_rejects_mismatched_weights() -> None:
    qdata = torch.zeros((1, 256), device="cuda", dtype=torch.int8)
    scale = torch.ones(1, device="cuda")
    weight = torch.zeros((384, 256), device="cuda", dtype=torch.int8)
    weight_scale = torch.ones(384, device="cuda")
    with pytest.raises(ValueError, match="matching weight shapes"):
        require_linear_backend(qdata).linear_prepared(
            qdata,
            scale,
            weight,
            weight_scale,
            None,
            torch.bfloat16,
            second_projection=(weight[:128], weight_scale[:128], None),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize(
    "shape",
    [(1, 256, 384), (129, 512, 300), (1024, 256, 2560), (1024, 264, 2560), (1041, 272, 2577)],
)
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
def test_prepared_swiglu_matches_paired_reference(shape, dtype, bias_dtypes) -> None:
    torch.manual_seed(893)
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
    implementation = require_linear_backend(qdata)
    expected_fused = (
        exact_sm120_available()
        and rows >= 1_024
        and features % 16 == 0
        and dtype in (torch.float16, torch.bfloat16)
    )
    assert (
        implementation.supports_prepared_swiglu(weights[0], weights[1], dtype, rows=rows)
        is expected_fused
    )
    arguments = (qdata, scale, weights[0], scales[0], biases[0])
    second = (weights[1], scales[1], biases[1])
    paired = implementation.linear_prepared(*arguments, dtype, second_projection=second)
    up_part, gate_part = paired.chunk(2, dim=-1)
    expected = (up_part.float() * F.silu(gate_part.float())).to(dtype)

    actual = implementation.linear_prepared_swiglu(*arguments, second, dtype)
    if expected_fused:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2 * torch.finfo(dtype).eps,
            atol=torch.finfo(dtype).smallest_normal,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    storage = torch.full((rows, width + 32), torch.nan, device="cuda", dtype=dtype)
    out = storage[:, 16:-16]
    result = implementation.linear_prepared_swiglu(*arguments, second, dtype, out=out)
    assert result is out
    torch.testing.assert_close(result, actual, rtol=0, atol=0)
    assert storage[:, :16].isnan().all()
    assert storage[:, -16:].isnan().all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_prepared_swiglu_validates_outputs_handles_3d_and_empty() -> None:
    torch.manual_seed(895)
    batch, seq, features, width = 2, 512, 256, 2_560
    dtype = torch.bfloat16
    qdata = torch.randint(-127, 128, (batch, seq, features), device="cuda", dtype=torch.int8)
    scale = torch.rand((batch, seq), device="cuda") * 0.01
    weights = [
        torch.randint(-127, 128, (width, features), device="cuda", dtype=torch.int8)
        for _ in range(2)
    ]
    scales = [torch.rand(width, device="cuda") * 0.01 for _ in range(2)]
    implementation = require_linear_backend(qdata)
    second = (weights[1], scales[1], None)
    args = (qdata, scale, weights[0], scales[0], None, second, dtype)

    with pytest.raises(ValueError, match="matching weight shapes"):
        implementation.linear_prepared_swiglu(
            qdata,
            scale,
            weights[0],
            scales[0],
            None,
            (weights[1][:128], scales[1][:128], None),
            dtype,
        )

    wrong_shape = torch.empty((batch, seq, width + 1), device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="expected shape"):
        implementation.linear_prepared_swiglu(*args, out=wrong_shape)

    wrong_dtype = torch.full((batch, seq, width), float("nan"), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="expected shape"):
        implementation.linear_prepared_swiglu(*args, out=wrong_dtype)
    assert wrong_dtype.isnan().all()

    transposed = (
        torch.empty((width, batch * seq), device="cuda", dtype=dtype).t().view(batch, seq, width)
    )
    with pytest.raises(ValueError, match="column-contiguous"):
        implementation.linear_prepared_swiglu(*args, out=transposed)

    nonviewable = torch.full(
        (batch, seq + 2, width + 32), float("nan"), device="cuda", dtype=dtype
    )[:, :seq, 16:-16]
    with pytest.raises(ValueError, match="viewable as a 2-D row-strided matrix"):
        implementation.linear_prepared_swiglu(*args, out=nonviewable)
    assert nonviewable.isnan().all()

    paired = implementation.linear_prepared(
        qdata, scale, weights[0], scales[0], None, dtype, second_projection=second
    )
    up_part, gate_part = paired.chunk(2, dim=-1)
    expected = (up_part.float() * F.silu(gate_part.float())).to(dtype)
    actual = implementation.linear_prepared_swiglu(*args)
    assert actual.shape == (batch, seq, width)
    if exact_sm120_available():
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2 * torch.finfo(dtype).eps,
            atol=torch.finfo(dtype).smallest_normal,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    storage = torch.full((batch, seq, width + 32), float("nan"), device="cuda", dtype=dtype)
    strided_out = storage[:, :, 16:-16]
    returned = implementation.linear_prepared_swiglu(*args, out=strided_out)
    assert returned is strided_out
    torch.testing.assert_close(strided_out, actual, rtol=0, atol=0)
    assert storage[:, :, :16].isnan().all()
    assert storage[:, :, -16:].isnan().all()

    empty = implementation.linear_prepared_swiglu(
        qdata[:0].view(0, features),
        scale[:0].view(0),
        weights[0],
        scales[0],
        None,
        second,
        dtype,
    )
    assert empty.shape == (0, width)
