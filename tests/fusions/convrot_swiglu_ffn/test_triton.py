"""Tests for the direct chunked ConvRot SwiGLU FFN operation."""

import subprocess
import sys
import textwrap

import pytest
import torch

from piper_kernels.fusions.convrot_swiglu_ffn.triton import (
    _chunked_swiglu_ffn_gated_updates_op,
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
def test_chunked_ffn_rejects_noncontiguous_input() -> None:
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

    with pytest.raises(ValueError, match="contiguous"):
        _chunked_swiglu_ffn_op(
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_rejects_noncontiguous_bias() -> None:
    torch.manual_seed(204)
    input_features, intermediate_features, output_features = 256, 512, 384
    activation = torch.randn(17, input_features, dtype=torch.bfloat16, device="cuda")
    up_qdata, up_scale = _weight(2 * intermediate_features, input_features)
    down_qdata, down_scale = _weight(output_features, intermediate_features)
    bias_storage = torch.randn(
        4 * intermediate_features,
        dtype=activation.dtype,
        device=activation.device,
    )
    up_bias = bias_storage[::2]
    assert not up_bias.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        _chunked_swiglu_ffn_op(
            activation,
            up_qdata,
            up_scale,
            up_bias,
            256,
            down_qdata,
            down_scale,
            None,
            256,
            128,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("output_features", [384, 1280], ids=["packed", "separate"])
def test_chunked_ffn_gated_updates_matches_materialized_epilogue(
    output_features: int,
) -> None:
    torch.manual_seed(203)
    rows = 385
    input_features, intermediate_features = 256, 512
    activation = torch.randn(rows, input_features, dtype=torch.bfloat16, device="cuda")
    base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    reusable_update = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    up_qdata, up_scale = _weight(2 * intermediate_features, input_features)
    down_qdata, down_scale = _weight(output_features, intermediate_features)
    gate_storage = torch.randn(
        7,
        6 * output_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    update_gate = gate_storage[:, 2 * output_features : 3 * output_features]
    ffn_gate = gate_storage[:, 5 * output_features :]
    assert update_gate.stride() == (6 * output_features, 1)
    assert ffn_gate.stride() == (6 * output_features, 1)
    gate_indices = torch.randint(
        0,
        7,
        (rows,),
        dtype=torch.int64,
        device="cuda",
    )
    ffn = _materialized_ffn(
        activation,
        up_qdata,
        up_scale,
        None,
        down_qdata,
        down_scale,
        None,
        256,
    )
    expected = _materialized_gated_updates(
        ffn,
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
    )
    actual = reusable_update.clone()
    result = _chunked_swiglu_ffn_gated_updates_op(
        activation,
        up_qdata,
        up_scale,
        None,
        256,
        down_qdata,
        down_scale,
        None,
        256,
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        False,
        128,
    )

    assert result is None
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_gated_updates_preserves_negative_python_indices() -> None:
    torch.manual_seed(205)
    rows = 10
    input_features, intermediate_features, output_features = 256, 512, 384
    activation = torch.randn(rows, input_features, dtype=torch.bfloat16, device="cuda")
    base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    reusable_update = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    up_qdata, up_scale = _weight(2 * intermediate_features, input_features)
    down_qdata, down_scale = _weight(output_features, intermediate_features)
    update_gate = torch.randn(7, output_features, dtype=torch.bfloat16, device="cuda")
    ffn_gate = torch.randn(5, output_features, dtype=torch.bfloat16, device="cuda")
    gate_indices = torch.tensor(
        [-1, -2, -3, -4, -5, 0, 1, 2, 3, 4],
        dtype=torch.int64,
        device="cuda",
    )
    ffn = _materialized_ffn(
        activation,
        up_qdata,
        up_scale,
        None,
        down_qdata,
        down_scale,
        None,
        256,
    )
    expected = _materialized_gated_updates(
        ffn,
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
    )
    actual = reusable_update.clone()
    result = _chunked_swiglu_ffn_gated_updates_op(
        activation,
        up_qdata,
        up_scale,
        None,
        256,
        down_qdata,
        down_scale,
        None,
        256,
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        True,
        8,
    )

    assert result is None
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_gated_updates_rejects_out_of_bounds_indices() -> None:
    program = textwrap.dedent(
        """
        import torch

        from piper_kernels.fusions.convrot_swiglu_ffn.triton import (
            _chunked_swiglu_ffn_gated_updates_op,
        )


        def weight(out_features: int, in_features: int):
            return (
                torch.randint(
                    -127,
                    128,
                    (out_features, in_features),
                    dtype=torch.int8,
                    device="cuda",
                ),
                torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01,
            )


        rows, input_features, intermediate_features, output_features = 1, 256, 256, 128
        activation = torch.randn(rows, input_features, dtype=torch.bfloat16, device="cuda")
        base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
        update = torch.randn_like(base)
        update_gate = torch.randn(1, output_features, dtype=torch.bfloat16, device="cuda")
        ffn_gate = torch.randn_like(update_gate)
        gate_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        up_qdata, up_scale = weight(2 * intermediate_features, input_features)
        down_qdata, down_scale = weight(output_features, intermediate_features)
        _chunked_swiglu_ffn_gated_updates_op(
            activation,
            up_qdata,
            up_scale,
            None,
            256,
            down_qdata,
            down_scale,
            None,
            256,
            base,
            update,
            update_gate,
            ffn_gate,
            gate_indices,
            False,
            1,
        )
        torch.cuda.synchronize()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "gate index out of bounds" in result.stderr or "device-side assert" in result.stderr


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
