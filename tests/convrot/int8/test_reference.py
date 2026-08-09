"""Tests for the portable INT8 ConvRot reference implementation."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from piper_kernels.convrot import ConvRotInt8Tensor, convrot_linear
from piper_kernels.convrot._rotation import build_hadamard, rotate_groups
from piper_kernels.convrot.int8.reference import (
    dynamic_quantize_rows,
)


def test_cpu_linear_matches_explicit_w8a8_reference() -> None:
    torch.manual_seed(4)
    group_size, in_features, out_features = 16, 32, 11
    weight = torch.randn(out_features, in_features)
    rotated_weight = rotate_groups(weight, group_size)
    weight_scale = (rotated_weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    weight_qdata = (rotated_weight / weight_scale).round().clamp(-128, 127).to(torch.int8)
    wrapped = ConvRotInt8Tensor.from_packed(
        weight_qdata,
        weight_scale,
        group_size=group_size,
        dtype=torch.float32,
    )
    activation = torch.randn(7, in_features)
    bias = torch.randn(out_features)

    rotated_activation = rotate_groups(activation, group_size)
    activation_qdata, activation_scale = dynamic_quantize_rows(rotated_activation)
    accumulated = activation_qdata.to(torch.int32) @ weight_qdata.T.to(torch.int32)
    expected = accumulated.float() * activation_scale * weight_scale.T + bias

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
def test_cpu_convrot_linear_matches_materialized_up_gate_swiglu(with_bias: bool) -> None:
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
    actual = convrot_linear(activation, weight, bias, input_activation="swiglu")

    assert torch.equal(actual, expected)


def test_convrot_linear_handles_empty_rows() -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    activation = torch.empty(2, 0, 64)

    result = convrot_linear(activation, weight, input_activation="swiglu")

    assert result.shape == (2, 0, 11)
    assert result.dtype is activation.dtype
    assert result.device == activation.device


@pytest.mark.parametrize("prefix", [(), (4,), (2, 3), (2, 0)])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_linear_handles_zero_input_features(
    prefix: tuple[int, ...],
    with_bias: bool,
    input_activation: str | None,
) -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(3, 0, dtype=torch.int8),
        torch.ones(3, 1, dtype=torch.float32),
        group_size=16,
        dtype=torch.float16,
    )
    activation = torch.empty((*prefix, 0), dtype=torch.float16)
    bias = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float16) if with_bias else None

    if input_activation is None:
        result = torch.nn.functional.linear(activation, weight, bias)
    else:
        result = convrot_linear(
            activation,
            weight,
            bias,
            input_activation="swiglu",
        )

    expected = activation.new_zeros((*prefix, 3))
    if bias is not None:
        expected += bias
    assert torch.equal(result, expected)


@pytest.mark.parametrize("prefix", [(2, 3), (2, 0)])
@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_linear_preserves_zero_output_and_row_dimensions(
    prefix: tuple[int, ...],
    input_activation: str | None,
) -> None:
    in_features = 16
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(0, in_features, dtype=torch.int8),
        torch.empty(0, 1, dtype=torch.float32),
        group_size=16,
        dtype=torch.float32,
    )
    input_factor = 1 if input_activation is None else 2
    activation = torch.empty((*prefix, input_factor * in_features))

    if input_activation is None:
        result = torch.nn.functional.linear(activation, weight)
    else:
        result = convrot_linear(activation, weight, input_activation="swiglu")

    assert result.shape == (*prefix, 0)
    assert result.dtype is activation.dtype
    assert result.device == activation.device


def test_zero_input_features_run_under_fullgraph_compile() -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(3, 0, dtype=torch.int8),
        torch.ones(3, 1, dtype=torch.float32),
        group_size=16,
        dtype=torch.float32,
    )
    activation = torch.empty(2, 4, 0)
    bias = torch.arange(3, dtype=torch.float32)

    def apply_both(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.nn.functional.linear(value, weight, bias),
            convrot_linear(value, weight, bias, input_activation="swiglu"),
        )

    expected = apply_both(activation)
    actual = torch.compile(apply_both, backend="eager", fullgraph=True)(activation)

    assert all(
        torch.equal(item, reference) for item, reference in zip(actual, expected, strict=True)
    )


def test_convrot_linear_propagates_meta_metadata() -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(11, 32, dtype=torch.int8, device="meta"),
        torch.empty(11, 1, dtype=torch.float32, device="meta"),
        group_size=16,
        dtype=torch.bfloat16,
    )
    activation = torch.empty(2, 7, 64, dtype=torch.bfloat16, device="meta")
    bias = torch.empty(11, dtype=torch.bfloat16, device="meta")

    result = convrot_linear(activation, weight, bias, input_activation="swiglu")

    assert result.shape == (2, 7, 11)
    assert result.dtype is torch.bfloat16
    assert result.device.type == "meta"


def test_convrot_linear_supports_fake_tensors() -> None:
    build_hadamard(16)
    with FakeTensorMode():
        weight = ConvRotInt8Tensor.from_packed(
            torch.empty(11, 32, dtype=torch.int8),
            torch.empty(11, 1, dtype=torch.float32),
            group_size=16,
            dtype=torch.float32,
        )
        activation = torch.empty(2, 7, 64)
        result = convrot_linear(activation, weight, input_activation="swiglu")

    assert isinstance(result, FakeTensor)
    assert result.shape == (2, 7, 11)
    assert result.dtype is torch.float32


def test_meta_linear_and_input_activation_run_under_torch_compile() -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(11, 32, dtype=torch.int8, device="meta"),
        torch.empty(11, 1, dtype=torch.float32, device="meta"),
        group_size=16,
        dtype=torch.bfloat16,
    )
    activation = torch.empty(2, 7, 64, dtype=torch.bfloat16, device="meta")
    bias = torch.empty(11, dtype=torch.bfloat16, device="meta")

    def apply_both(
        value: torch.Tensor,
        convrot_weight: ConvRotInt8Tensor,
        linear_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regular = torch.nn.functional.linear(value[..., :32], convrot_weight, linear_bias)
        fused = convrot_linear(
            value,
            convrot_weight,
            linear_bias,
            input_activation="swiglu",
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


def test_cpu_convrot_linear_runs_under_torch_compile() -> None:
    torch.manual_seed(46)
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    activation = torch.randn(7, 64)
    bias = torch.randn(11)

    def apply_swiglu(
        value: torch.Tensor,
        convrot_weight: ConvRotInt8Tensor,
        linear_bias: torch.Tensor,
    ) -> torch.Tensor:
        return convrot_linear(
            value,
            convrot_weight,
            linear_bias,
            input_activation="swiglu",
        )

    expected = apply_swiglu(activation, weight, bias)
    actual = torch.compile(apply_swiglu, backend="eager", fullgraph=True)(
        activation,
        weight,
        bias,
    )

    assert torch.equal(actual, expected)


def test_convrot_linear_rejects_unknown_activation() -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)

    with pytest.raises(ValueError, match="input_activation must be 'swiglu'"):
        convrot_linear(
            torch.randn(7, 64),
            weight,
            input_activation="gelu",  # type: ignore[arg-type]
        )


def test_convrot_linear_requires_explicit_input_activation() -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)

    with pytest.raises(TypeError, match="required keyword-only argument: 'input_activation'"):
        convrot_linear(torch.randn(7, 64), weight)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "activation",
    [torch.tensor(1.0), torch.empty(7, 63)],
)
def test_convrot_linear_rejects_wrong_raw_width(activation: torch.Tensor) -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)

    with pytest.raises(ValueError, match=r"input has .* features, expected 64"):
        convrot_linear(activation, weight, input_activation="swiglu")


def test_convrot_linear_rejects_invalid_argument_types() -> None:
    weight = ConvRotInt8Tensor.from_hp(torch.randn(11, 32), group_size=16)
    activation = torch.randn(7, 64)

    with pytest.raises(TypeError, match="input must be a tensor"):
        convrot_linear(None, weight, input_activation="swiglu")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="weight must be ConvRotInt8Tensor"):
        convrot_linear(
            activation,
            torch.randn(11, 32),  # type: ignore[arg-type]
            input_activation="swiglu",
        )
    with pytest.raises(TypeError, match="bias must be a tensor or None"):
        convrot_linear(
            activation,
            weight,
            1.0,  # type: ignore[arg-type]
            input_activation="swiglu",
        )


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


@pytest.mark.parametrize("shape", [(3, 0), (2, 0, 0)])
def test_dynamic_quantize_rows_handles_zero_features(shape: tuple[int, ...]) -> None:
    value = torch.empty(shape, dtype=torch.bfloat16)

    qdata, scale = dynamic_quantize_rows(value)

    assert qdata.shape == value.shape
    assert qdata.dtype is torch.int8
    assert scale.shape == (*shape[:-1], 1)
    assert scale.dtype is torch.float32
    assert torch.all(scale == 1e-30)
