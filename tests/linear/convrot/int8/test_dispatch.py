"""Tests for the INT8 ConvRot inference contract and semantic operators."""

import pytest
import torch

from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_linear


def _weight(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    scale_requires_grad: bool = False,
) -> ConvRotInt8Tensor:
    return ConvRotInt8Tensor.from_quantized(
        torch.arange(-128, 96, dtype=torch.int8, device=device).reshape(7, 32),
        torch.full(
            (7, 1),
            0.01,
            dtype=torch.float32,
            device=device,
            requires_grad=scale_requires_grad,
        ),
        group_size=16,
        logical_dtype=dtype,
    )


def _call_linear(
    input_activation: str | None,
    activation: torch.Tensor,
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if input_activation is None:
        return torch.nn.functional.linear(activation, weight, bias)
    return convrot_int8_linear(
        activation,
        weight,
        bias,
        activation_fn=input_activation,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
def test_linear_rejects_activation_weight_dtype_mismatch(
    input_activation: str | None,
) -> None:
    weight = _weight(dtype=torch.bfloat16)
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.empty(3, input_factor * 32, dtype=torch.float16)

    with pytest.raises(ValueError, match="logical dtype"):
        _call_linear(input_activation, activation, weight, None)


@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("shape", "bias must have shape"),
        ("device", "must share a device"),
        ("dtype", "must be FP16, BF16, or FP32"),
        ("layout", "bias must use strided layout"),
    ],
)
def test_linear_rejects_invalid_bias_contract(
    input_activation: str | None,
    violation: str,
    message: str,
) -> None:
    weight = _weight()
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.empty(3, input_factor * 32)
    if violation == "shape":
        bias = torch.empty(7, 1)
    elif violation == "device":
        bias = torch.empty(7, device="meta")
    elif violation == "dtype":
        bias = torch.empty(7, dtype=torch.float64)
    else:
        bias = torch.ones(7).to_sparse()

    with pytest.raises(ValueError, match=message):
        _call_linear(input_activation, activation, weight, bias)


@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_linear_accepts_independent_bias_dtype(
    input_activation: str | None,
    bias_dtype: torch.dtype,
) -> None:
    torch.manual_seed(122)
    weight = _weight(dtype=torch.bfloat16)
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.randn(3, input_factor * 32, dtype=torch.bfloat16)
    bias = torch.randn(7, dtype=bias_dtype)

    result = _call_linear(input_activation, activation, weight, bias)

    assert result.shape == (3, 7)
    assert result.dtype is torch.bfloat16


@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
def test_meta_linear_validates_before_propagating_metadata(
    input_activation: str | None,
) -> None:
    weight = _weight(device="meta", dtype=torch.bfloat16)
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.empty(
        3,
        input_factor * 32,
        dtype=torch.bfloat16,
        device="meta",
    )
    malformed_bias = torch.empty(8, dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match="bias must have shape"):
        _call_linear(input_activation, activation, weight, malformed_bias)


@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
@pytest.mark.parametrize("grad_input", ["activation", "bias", "scale"])
def test_linear_rejects_autograd_inputs_but_allows_no_grad(
    input_activation: str | None,
    grad_input: str,
) -> None:
    weight = _weight(scale_requires_grad=grad_input == "scale")
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.randn(
        3,
        input_factor * 32,
        requires_grad=grad_input == "activation",
    )
    bias = torch.randn(7, requires_grad=grad_input == "bias")

    with pytest.raises(RuntimeError, match="does not support autograd"):
        _call_linear(input_activation, activation, weight, bias)

    with torch.no_grad():
        result = _call_linear(input_activation, activation, weight, bias)

    assert result.shape == (3, 7)


@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
def test_linear_accepts_noncontiguous_vector_bias(input_activation: str | None) -> None:
    torch.manual_seed(121)
    weight = _weight()
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.randn(3, input_factor * 32)
    bias = torch.randn(14)[::2]
    assert not bias.is_contiguous()

    expected = _call_linear(input_activation, activation, weight, bias.contiguous())
    actual = _call_linear(input_activation, activation, weight, bias)

    assert torch.equal(actual, expected)


def test_addmm_rejects_grad_enabled_scale_but_allows_no_grad() -> None:
    weight = _weight(scale_requires_grad=True)
    mat1 = torch.randn(7, 3)
    mat2 = torch.randn(3, 32)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        weight.addmm_(mat1, mat2)

    with torch.no_grad():
        assert weight.addmm_(mat1, mat2) is weight


def test_add_rejects_grad_enabled_scale_but_allows_no_grad() -> None:
    weight = _weight(scale_requires_grad=True)
    update = torch.randn(7, 32)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        weight.add_(update)

    with torch.no_grad():
        assert weight.add_(update) is weight


@pytest.mark.parametrize("storage_name", ["qdata", "scale"])
@pytest.mark.parametrize("operation", ["linear", "swiglu", "addmm", "add"])
def test_operations_revalidate_canonical_storage_layout(
    storage_name: str,
    operation: str,
) -> None:
    weight = _weight()
    if storage_name == "qdata":
        weight.qdata = torch.empty(7, 64, dtype=torch.int8)[:, ::2]
    else:
        weight.scale = torch.empty(7, 2, dtype=torch.float32)[:, ::2]
    assert not getattr(weight, storage_name).is_contiguous()

    def call() -> object:
        if operation == "linear":
            return torch.nn.functional.linear(torch.empty(3, 32), weight)
        if operation == "swiglu":
            return convrot_int8_linear(
                torch.empty(3, 64),
                weight,
                activation_fn="swiglu",
            )
        if operation == "addmm":
            return weight.addmm_(torch.empty(7, 3), torch.empty(3, 32))
        return weight.add_(torch.empty(7, 32))

    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        call()
