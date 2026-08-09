"""Tests for the INT8 ConvRot inference contract and semantic operators."""

import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from piper_kernels.convrot import ConvRotInt8Tensor, convrot_linear
from piper_kernels.convrot.int8 import dispatch as convrot_dispatch


def test_semantic_operator_schemas_and_fake_kernels_exist_without_triton() -> None:
    script = textwrap.dedent(
        """
        import sys

        sys.modules["triton"] = None

        import torch
        from piper_kernels.convrot.int8 import dispatch

        assert dispatch._triton_addmm_ is None
        assert dispatch._triton_linear is None
        assert dispatch._triton_swiglu_linear is None

        activation = torch.empty(2, 32, device="meta")
        swiglu_activation = torch.empty(2, 64, device="meta")
        qdata = torch.empty(7, 32, dtype=torch.int8, device="meta")
        scale = torch.empty(7, 1, device="meta")
        bias = torch.empty(7, device="meta")

        ordinary = dispatch._convrot_int8_linear_op(
            activation, qdata, scale, bias, 16
        )
        swiglu = dispatch._convrot_int8_swiglu_linear_op(
            swiglu_activation, qdata, scale, bias, 16
        )
        assert ordinary.shape == (2, 7)
        assert swiglu.shape == (2, 7)
        assert dispatch._convrot_int8_addmm_op(
            qdata,
            scale,
            torch.empty(7, 3, device="meta"),
            torch.empty(3, 32, device="meta"),
            16,
            1.0,
            1.0,
        ) is None
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def _weight(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    scale_requires_grad: bool = False,
) -> ConvRotInt8Tensor:
    return ConvRotInt8Tensor.from_packed(
        torch.arange(-128, 96, dtype=torch.int8, device=device).reshape(7, 32),
        torch.full(
            (7, 1),
            0.01,
            dtype=torch.float32,
            device=device,
            requires_grad=scale_requires_grad,
        ),
        group_size=16,
        dtype=dtype,
    )


def _call_linear(
    input_activation: str | None,
    activation: torch.Tensor,
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if input_activation is None:
        return torch.nn.functional.linear(activation, weight, bias)
    return convrot_linear(
        activation,
        weight,
        bias,
        input_activation=input_activation,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_linear_rejects_activation_weight_dtype_mismatch(
    input_activation: str | None,
) -> None:
    weight = _weight(dtype=torch.bfloat16)
    input_factor = 1 if input_activation is None else 2
    activation = torch.empty(3, input_factor * 32, dtype=torch.float16)

    with pytest.raises(ValueError, match="logical dtype"):
        _call_linear(input_activation, activation, weight, None)


@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("shape", "bias must have shape"),
        ("device", "must share a device"),
        ("dtype", "logical dtype"),
        ("layout", "bias must use strided layout"),
    ],
)
def test_linear_rejects_invalid_bias_contract(
    input_activation: str | None,
    violation: str,
    message: str,
) -> None:
    weight = _weight()
    input_factor = 1 if input_activation is None else 2
    activation = torch.empty(3, input_factor * 32)
    if violation == "shape":
        bias = torch.empty(7, 1)
    elif violation == "device":
        bias = torch.empty(7, device="meta")
    elif violation == "dtype":
        bias = torch.empty(7, dtype=torch.float16)
    else:
        bias = torch.ones(7).to_sparse()

    with pytest.raises(ValueError, match=message):
        _call_linear(input_activation, activation, weight, bias)


@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_meta_linear_validates_before_propagating_metadata(
    input_activation: str | None,
) -> None:
    weight = _weight(device="meta", dtype=torch.bfloat16)
    input_factor = 1 if input_activation is None else 2
    activation = torch.empty(
        3,
        input_factor * 32,
        dtype=torch.bfloat16,
        device="meta",
    )
    malformed_bias = torch.empty(8, dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match="bias must have shape"):
        _call_linear(input_activation, activation, weight, malformed_bias)


@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
@pytest.mark.parametrize("grad_input", ["activation", "bias", "scale"])
def test_linear_rejects_autograd_inputs_but_allows_no_grad(
    input_activation: str | None,
    grad_input: str,
) -> None:
    weight = _weight(scale_requires_grad=grad_input == "scale")
    input_factor = 1 if input_activation is None else 2
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


@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_linear_accepts_noncontiguous_vector_bias(input_activation: str | None) -> None:
    torch.manual_seed(121)
    weight = _weight()
    input_factor = 1 if input_activation is None else 2
    activation = torch.randn(3, input_factor * 32)
    bias = torch.randn(14)[::2]
    assert not bias.is_contiguous()

    expected = _call_linear(input_activation, activation, weight, bias.contiguous())
    actual = _call_linear(input_activation, activation, weight, bias)

    assert torch.equal(actual, expected)


def test_public_fake_cuda_paths_do_not_require_a_physical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_capability(_device: torch.device) -> tuple[int, int]:
        raise AssertionError("synthetic CUDA device has no physical capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable_capability)
    with FakeTensorMode():
        device = torch.device("cuda:137")
        weight = ConvRotInt8Tensor.from_packed(
            torch.empty(7, 256, dtype=torch.int8, device=device),
            torch.empty(7, 1, dtype=torch.float32, device=device),
            group_size=256,
        )
        ordinary = torch.nn.functional.linear(
            torch.empty(4, 256, dtype=torch.bfloat16, device=device),
            weight,
        )
        swiglu = convrot_linear(
            torch.empty(512, 512, dtype=torch.bfloat16, device=device),
            weight,
            input_activation="swiglu",
        )
        updated = weight.addmm_(
            torch.empty(7, 3, dtype=torch.bfloat16, device=device),
            torch.empty(3, 256, dtype=torch.bfloat16, device=device),
        )

    assert isinstance(ordinary, FakeTensor)
    assert isinstance(swiglu, FakeTensor)
    assert ordinary.shape == (4, 7)
    assert swiglu.shape == (512, 7)
    assert ordinary.device == swiglu.device == torch.device("cuda:137")
    assert updated is weight


def test_public_swiglu_fake_cuda_runs_under_fullgraph_compile_without_a_target() -> None:
    unavailable_index = torch.cuda.device_count()
    with FakeTensorMode():
        device = torch.device(f"cuda:{unavailable_index}")
        weight = ConvRotInt8Tensor.from_packed(
            torch.empty(7, 256, dtype=torch.int8, device=device),
            torch.empty(7, 1, dtype=torch.float32, device=device),
            group_size=256,
        )
        activation = torch.empty(512, 512, dtype=torch.bfloat16, device=device)

        def apply_swiglu(value: torch.Tensor) -> torch.Tensor:
            return convrot_linear(value, weight, input_activation="swiglu")

        result = torch.compile(apply_swiglu, backend="eager", fullgraph=True)(activation)

    assert isinstance(result, FakeTensor)
    assert result.shape == (512, 7)
    assert result.dtype is torch.bfloat16
    assert result.device == torch.device(f"cuda:{unavailable_index}")


def test_addmm_rejects_grad_enabled_scale_but_allows_no_grad() -> None:
    weight = _weight(scale_requires_grad=True)
    mat1 = torch.randn(7, 3)
    mat2 = torch.randn(3, 32)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        weight.addmm_(mat1, mat2)

    with torch.no_grad():
        assert weight.addmm_(mat1, mat2) is weight


@pytest.mark.parametrize("storage_name", ["qdata", "scale"])
@pytest.mark.parametrize("operation", ["linear", "swiglu", "addmm"])
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
            return convrot_linear(
                torch.empty(3, 64),
                weight,
                input_activation="swiglu",
            )
        return weight.addmm_(torch.empty(7, 3), torch.empty(3, 32))

    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        call()


@pytest.mark.parametrize(
    ("operator", "input_factor"),
    [
        (convrot_dispatch._convrot_int8_linear_op, 1),
        (convrot_dispatch._convrot_int8_swiglu_linear_op, 2),
    ],
)
def test_semantic_linear_custom_ops_pass_opcheck(
    operator: Callable[..., torch.Tensor],
    input_factor: int,
) -> None:
    torch.manual_seed(122)
    qdata = torch.randint(-127, 128, (7, 32), dtype=torch.int8)
    scale = torch.rand(7, 1, dtype=torch.float32) * 0.01
    activation = torch.randn(3, input_factor * 32)
    bias = torch.randn(7)

    result = torch.library.opcheck(operator, (activation, qdata, scale, bias, 16))

    assert set(result.values()) == {"SUCCESS"}


def test_semantic_addmm_custom_op_passes_opcheck() -> None:
    torch.manual_seed(123)
    qdata = torch.randint(-127, 128, (7, 32), dtype=torch.int8)
    scale = torch.rand(7, 1, dtype=torch.float32) * 0.01
    mat1 = torch.randn(7, 5)
    mat2 = torch.randn(5, 32)

    result = torch.library.opcheck(
        convrot_dispatch._convrot_int8_addmm_op,
        (qdata, scale, mat1, mat2, 16, 0.5, 1.25),
    )

    assert set(result.values()) == {"SUCCESS"}
