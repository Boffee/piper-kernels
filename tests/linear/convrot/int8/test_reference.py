"""Tests for the portable INT8 ConvRot reference implementation."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_linear
from piper_kernels.linear.convrot._rotation import build_hadamard, rotate_groups
from piper_kernels.linear.convrot.int8.reference import dynamic_quantize_rows, linear


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm"),
            ],
        ),
    ],
)
def test_int8_normalization_avoids_bfloat16_double_rounding(device) -> None:
    # Both inputs are exact BF16 values. The smaller value is 2/3 of the
    # maximum, so its terminal INT8 code is round(127 * 2/3) = 85, not 84.
    values = torch.tensor([[0.02001953125, 0.030029296875]], dtype=torch.bfloat16, device=device)
    expected = torch.tensor([[85, 127]], dtype=torch.int8, device=device)
    actual, scale = dynamic_quantize_rows(values)
    assert torch.equal(actual, expected)
    if device == "cuda":
        from piper_kernels.linear.convrot.int8._kernels.triton import (  # noqa: PLC0415
            quantize_rows_kernel,
        )

        output = torch.empty_like(expected)
        output_scale = torch.empty_like(scale)
        quantize_rows_kernel[(1,)](
            values,
            output,
            output_scale,
            2,
            block_size=128,
            reciprocal_scale=torch.version.hip is not None,
            accelerator_backend="hip" if torch.version.hip is not None else "cuda",
        )
        assert torch.equal(output, expected)
        torch.testing.assert_close(output_scale, scale)


def test_cpu_linear_matches_explicit_w8a8_reference() -> None:
    torch.manual_seed(4)
    group_size, in_features, out_features = 16, 32, 11
    weight = torch.randn(out_features, in_features)
    rotated_weight = rotate_groups(weight, group_size)
    weight_scale = (rotated_weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    weight_qdata = (rotated_weight / weight_scale).round().clamp(-128, 127).to(torch.int8)
    wrapped = ConvRotInt8Tensor.from_quantized(
        weight_qdata,
        weight_scale,
        group_size=group_size,
        logical_dtype=torch.float32,
    )
    activation = torch.randn(7, in_features)
    bias = torch.randn(out_features)

    rotated_input = rotate_groups(activation, group_size)
    input_qdata, input_scale = dynamic_quantize_rows(rotated_input)
    accumulated = input_qdata.to(torch.int32) @ weight_qdata.T.to(torch.int32)
    expected = accumulated.float() * input_scale * weight_scale.T + bias

    assert torch.equal(torch.nn.functional.linear(activation, wrapped, bias), expected)


def test_cpu_linear_supports_pytorch_keyword_and_mixed_argument_forms() -> None:
    torch.manual_seed(5)
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    activation = torch.randn(7, 32)
    bias = torch.randn(11)
    expected = torch.nn.functional.linear(activation, weight, bias)
    expected_without_bias = torch.nn.functional.linear(activation, weight)

    results = (
        torch.nn.functional.linear(input=activation, weight=weight, bias=bias),
        torch.nn.functional.linear(activation, weight=weight, bias=bias),
        torch.nn.functional.linear(activation, weight, bias=bias),
    )

    assert all(torch.equal(result, expected) for result in results)
    assert torch.equal(
        torch.nn.functional.linear(input=activation, weight=weight),
        expected_without_bias,
    )


@pytest.mark.parametrize("with_bias", [False, True])
def test_cpu_public_linear_matches_materialized_up_gate_swiglu(with_bias: bool) -> None:
    torch.manual_seed(45)
    in_features, out_features = 32, 11
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(out_features, in_features),
        group_size=16,
    )
    up = torch.randn(2, 7, in_features)
    gate = torch.randn(2, 7, in_features)
    activation = torch.cat((up, gate), dim=-1)
    bias = torch.randn(out_features) if with_bias else None

    expected = torch.nn.functional.linear(up * torch.nn.functional.silu(gate), weight, bias)
    actual = convrot_int8_linear(activation, weight, bias, activation_fn="swiglu")

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("with_bias", [False, True])
def test_cpu_public_linear_matches_materialized_gelu_tanh(with_bias: bool) -> None:
    torch.manual_seed(47)
    in_features, out_features = 32, 11
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(out_features, in_features),
        group_size=16,
    )
    activation = torch.randn(2, 7, in_features)
    bias = torch.randn(out_features) if with_bias else None

    expected = torch.nn.functional.linear(
        torch.nn.functional.gelu(activation, approximate="tanh"),
        weight,
        bias,
    )
    actual = convrot_int8_linear(activation, weight, bias, activation_fn="gelu_tanh")

    assert torch.equal(actual, expected)


def test_public_linear_supports_keyword_arguments() -> None:
    torch.manual_seed(124)
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    input_tensor = torch.randn(2, 64)
    bias = torch.randn(11)

    expected = linear(
        input_tensor,
        weight.qdata,
        weight.scale,
        weight.group_size,
        bias,
        activation_fn="swiglu",
    )
    actual = convrot_int8_linear(
        input=input_tensor,
        weight=weight,
        bias=bias,
        activation_fn="swiglu",
    )

    assert torch.equal(actual, expected)


def test_public_linear_propagates_meta_metadata() -> None:
    weight = ConvRotInt8Tensor.from_quantized(
        torch.empty(11, 32, dtype=torch.int8, device="meta"),
        torch.empty(11, 1, dtype=torch.float32, device="meta"),
        group_size=16,
        logical_dtype=torch.bfloat16,
    )
    activation = torch.empty(2, 7, 64, dtype=torch.bfloat16, device="meta")
    bias = torch.empty(11, dtype=torch.bfloat16, device="meta")

    result = convrot_int8_linear(activation, weight, bias, activation_fn="swiglu")

    assert result.shape == (2, 7, 11)
    assert result.dtype is torch.bfloat16
    assert result.device.type == "meta"


def test_public_linear_supports_fake_tensors() -> None:
    build_hadamard(16)
    with FakeTensorMode():
        weight = ConvRotInt8Tensor.from_quantized(
            torch.empty(11, 32, dtype=torch.int8),
            torch.empty(11, 1, dtype=torch.float32),
            group_size=16,
            logical_dtype=torch.float32,
        )
        activation = torch.empty(2, 7, 64)
        result = convrot_int8_linear(activation, weight, activation_fn="swiglu")

    assert isinstance(result, FakeTensor)
    assert result.shape == (2, 7, 11)
    assert result.dtype is torch.float32


def test_meta_linear_and_input_activation_run_under_torch_compile() -> None:
    weight = ConvRotInt8Tensor.from_quantized(
        torch.empty(11, 32, dtype=torch.int8, device="meta"),
        torch.empty(11, 1, dtype=torch.float32, device="meta"),
        group_size=16,
        logical_dtype=torch.bfloat16,
    )
    activation = torch.empty(2, 7, 64, dtype=torch.bfloat16, device="meta")
    bias = torch.empty(11, dtype=torch.bfloat16, device="meta")

    def apply_both(
        value: torch.Tensor,
        convrot_weight: ConvRotInt8Tensor,
        linear_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regular = torch.nn.functional.linear(value[..., :32], convrot_weight, linear_bias)
        fused = convrot_int8_linear(
            value,
            convrot_weight,
            linear_bias,
            activation_fn="swiglu",
        )
        return regular, fused

    regular, fused = torch.compile(apply_both, backend="eager", fullgraph=True)(
        activation,
        weight,
        bias,
    )

    assert regular.shape == (2, 7, 11)
    assert fused.shape == (2, 7, 11)
    assert regular.device.type == fused.device.type == "meta"


@pytest.mark.parametrize("activation_fn", ["gelu_tanh", "swiglu"])
def test_cpu_public_linear_runs_under_torch_compile(activation_fn: str) -> None:
    torch.manual_seed(46)
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    activation = torch.randn(7, 64 if activation_fn == "swiglu" else 32)
    bias = torch.randn(11)

    def apply_activation(
        value: torch.Tensor,
        convrot_weight: ConvRotInt8Tensor,
        linear_bias: torch.Tensor,
    ) -> torch.Tensor:
        return convrot_int8_linear(
            value,
            convrot_weight,
            linear_bias,
            activation_fn=activation_fn,  # type: ignore[arg-type]
        )

    expected = apply_activation(activation, weight, bias)
    actual = torch.compile(apply_activation, backend="eager", fullgraph=True)(
        activation,
        weight,
        bias,
    )

    assert torch.equal(actual, expected)


def test_public_linear_rejects_unknown_activation() -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)

    with pytest.raises(ValueError, match="expected 'gelu_tanh', 'swiglu', or None"):
        convrot_int8_linear(
            torch.randn(7, 64),
            weight,
            activation_fn="gelu",  # type: ignore[arg-type]
        )


def test_public_linear_defaults_to_ordinary_linear() -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    activation = torch.randn(7, 32)
    bias = torch.randn(11)

    expected = torch.nn.functional.linear(activation, weight, bias)
    actual = convrot_int8_linear(activation, weight, bias)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "activation",
    [torch.tensor(1.0), torch.empty(7, 63)],
)
def test_public_linear_rejects_wrong_raw_width(activation: torch.Tensor) -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)

    with pytest.raises(ValueError, match=r"input has .* features, expected 64"):
        convrot_int8_linear(activation, weight, activation_fn="swiglu")


def test_dynamic_quantize_rows_handles_underflowing_float16_scale() -> None:
    value = torch.zeros(2, 16, dtype=torch.float16)
    value[0, 0] = 1e-6

    qdata, scale = dynamic_quantize_rows(value)

    assert qdata[0, 0] == 127
    assert torch.count_nonzero(qdata[0, 1:]) == 0
    assert torch.count_nonzero(qdata[1]) == 0
    assert scale[0, 0] > 0
    assert scale[0, 0].to(torch.float16) == 0
    assert scale[1, 0] == pytest.approx(1e-30)
