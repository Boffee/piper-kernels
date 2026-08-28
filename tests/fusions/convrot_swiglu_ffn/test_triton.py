"""Tests for the direct chunked ConvRot SwiGLU FFN operation."""

import pytest
import torch

from piper_kernels.fusions.convrot_swiglu_ffn.triton import (
    _chunked_swiglu_ffn_op,
)
from piper_kernels.linear.convrot.int8 import triton as convrot_backend


def _weight(
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    return qdata, scale


def _materialized_ffn(
    activation: torch.Tensor,
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    packed = convrot_backend.run_linear(
        activation,
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        group_size,
    )
    return convrot_backend.run_linear(
        packed,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        group_size,
        activation_fn="swiglu",
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("rows", [127, 128, 129, 385])
@pytest.mark.parametrize("chunk_rows", [64, 128, 256])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_chunked_ffn_matches_materialized_rows(
    rows: int,
    chunk_rows: int,
    with_bias: bool,
) -> None:
    torch.manual_seed(201)
    input_features, intermediate_features, output_features = 256, 512, 384
    activation = torch.randn(rows, input_features, dtype=torch.bfloat16, device="cuda")
    up_qdata, up_scale = _weight(
        2 * intermediate_features,
        input_features,
    )
    down_qdata, down_scale = _weight(
        output_features,
        intermediate_features,
    )
    up_bias = (
        torch.randn(2 * intermediate_features, dtype=activation.dtype, device="cuda")
        if with_bias
        else None
    )
    down_bias = (
        torch.randn(output_features, dtype=activation.dtype, device="cuda") if with_bias else None
    )

    expected = _materialized_ffn(
        activation,
        up_qdata,
        up_scale,
        up_bias,
        down_qdata,
        down_scale,
        down_bias,
        256,
    )
    actual = _chunked_swiglu_ffn_op(
        activation,
        up_qdata,
        up_scale,
        up_bias,
        256,
        down_qdata,
        down_scale,
        down_bias,
        256,
        chunk_rows,
    )

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_preserves_leading_shape_and_noncontiguous_input() -> None:
    torch.manual_seed(202)
    input_features, intermediate_features, output_features = 256, 512, 384
    storage = torch.randn(2, 193, 2 * input_features, dtype=torch.bfloat16, device="cuda")
    activation = storage[..., ::2]
    assert not activation.is_contiguous()
    up_qdata, up_scale = _weight(
        2 * intermediate_features,
        input_features,
    )
    down_qdata, down_scale = _weight(
        output_features,
        intermediate_features,
    )

    expected = _materialized_ffn(
        activation,
        up_qdata,
        up_scale,
        None,
        down_qdata,
        down_scale,
        None,
        256,
    )
    actual = _chunked_swiglu_ffn_op(
        activation,
        up_qdata,
        up_scale,
        None,
        256,
        down_qdata,
        down_scale,
        None,
        256,
        128,
    )

    assert actual.shape == (2, 193, output_features)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_runs_under_dynamic_fullgraph_compile() -> None:
    torch.manual_seed(203)
    input_features, intermediate_features, output_features = 256, 512, 384
    up_qdata, up_scale = _weight(
        2 * intermediate_features,
        input_features,
    )
    down_qdata, down_scale = _weight(
        output_features,
        intermediate_features,
    )

    @torch.compile(fullgraph=True, dynamic=True)
    def run(activation: torch.Tensor) -> torch.Tensor:
        return _chunked_swiglu_ffn_op(
            activation,
            up_qdata,
            up_scale,
            None,
            256,
            down_qdata,
            down_scale,
            None,
            256,
            128,
        )

    for rows in (257, 385):
        activation = torch.randn(rows, input_features, dtype=torch.bfloat16, device="cuda")
        expected = _materialized_ffn(
            activation,
            up_qdata,
            up_scale,
            None,
            down_qdata,
            down_scale,
            None,
            256,
        )
        assert torch.equal(run(activation), expected)
