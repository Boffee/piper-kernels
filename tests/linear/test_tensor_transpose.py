"""Quantized transposes preserve the represented weight and its matrix semantics."""

import io

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor
from piper_kernels.linear.nvfp4._layout import swap_packed_pairs

_WEIGHT_FORMATS = [
    pytest.param(ConvRotInt8Tensor, False, id="int8"),
    pytest.param(PiperNVFP4Tensor, False, id="nvfp4_low_first"),
    pytest.param(PiperNVFP4Tensor, True, id="nvfp4_high_first"),
    pytest.param(ConvRotNVFP4Tensor, False, id="convrot_nvfp4_low_first"),
    pytest.param(ConvRotNVFP4Tensor, True, id="convrot_nvfp4_high_first"),
]


def _weight(cls, high_first=False):
    torch.manual_seed(106)
    # The output dimension deliberately cannot be a ConvRot feature dimension.
    source = torch.randn(7, 64, dtype=torch.bfloat16)
    result = cls.from_hp(source, **({} if cls is PiperNVFP4Tensor else {"group_size": 64}))
    if high_first:
        result.qdata = swap_packed_pairs(result.qdata)
        result.high_first = True
    return result


@pytest.fixture(params=[ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotNVFP4Tensor])
def weight(request):
    return _weight(request.param)


@pytest.mark.parametrize(("cls", "high_first"), _WEIGHT_FORMATS)
@pytest.mark.parametrize("operation", ["t", "transpose", "negative", "mT", "permute", "aten"])
def test_transpose_preserves_weight_metadata_and_aliases(cls, high_first, operation):
    weight = _weight(cls, high_first)
    operations = {
        "t": lambda w: w.t(),
        "transpose": lambda w: w.transpose(0, 1),
        "negative": lambda w: w.transpose(-1, -2),
        "mT": lambda w: w.mT,
        "permute": lambda w: w.permute(1, 0),
        "aten": torch.ops.aten.t.default,
    }
    transposed = operations[operation](weight)
    assert type(transposed) is type(weight)
    assert transposed.transposed
    assert transposed.shape == weight.shape[::-1]
    assert transposed.stride() == weight.stride()[::-1]
    assert torch._C._is_alias_of(transposed, weight)
    for name in weight.__tensor_flatten__()[0]:
        assert torch._C._is_alias_of(getattr(transposed, name), getattr(weight, name))
    if isinstance(weight, PiperNVFP4Tensor):
        assert transposed.high_first == high_first
    if hasattr(weight, "group_size"):
        assert transposed.group_size == weight.group_size
    assert torch.equal(transposed.dequantize(), weight.dequantize().t())
    assert torch.equal(transposed.dequantize(torch.float32), weight.dequantize(torch.float32).t())
    round_trip = transposed.t()
    assert not round_trip.transposed
    assert round_trip.shape == weight.shape
    assert round_trip.stride() == weight.stride()
    assert torch.equal(round_trip.dequantize(), weight.dequantize())


@pytest.mark.parametrize("transposed", [False, True])
def test_noop_transpose_preserves_orientation(weight, transposed):
    weight = weight.t() if transposed else weight
    for actual in (weight.transpose(0, -2), weight.permute(0, 1)):
        assert actual.transposed == transposed
        assert actual.stride() == weight.stride()
        assert torch._C._is_alias_of(actual, weight)
        assert torch.equal(actual.dequantize(), weight.dequantize())


@pytest.mark.parametrize("operation", ["clone", "detach", "view", "dtype", "copy", "save"])
def test_transposed_metadata_survives_reconstruction(weight, operation):
    transposed = weight.t()
    if operation == "clone":
        actual = transposed.clone()
    elif operation == "detach":
        actual = transposed.detach()
    elif operation == "view":
        actual = transposed.view(transposed.shape)
    elif operation == "dtype":
        actual = transposed.to(torch.float16)
    elif operation == "copy":
        actual = transposed.to(dtype=torch.float16, copy=True)
    else:
        buffer = io.BytesIO()
        torch.save(transposed, buffer)
        buffer.seek(0)
        actual = torch.load(buffer, weights_only=False)
    assert type(actual) is type(weight)
    assert actual.transposed
    assert actual.shape == transposed.shape
    assert actual.stride() == transposed.stride()
    assert torch.equal(actual.dequantize(torch.float32), transposed.dequantize(torch.float32))


@pytest.mark.parametrize("operation", ["transpose", "as_strided", "round_trip"])
def test_compiled_transpose_preserves_aliasing_and_dequantization(weight, operation):
    torch.compiler.reset()

    def view(w):
        if operation == "as_strided":
            return w.as_strided(w.shape[::-1], w.stride()[::-1])
        if operation == "round_trip":
            return w.t().t()
        return w.t()

    actual = torch.compile(view, backend="aot_eager", fullgraph=True)(weight)
    expected = view(weight)
    assert type(actual) is type(weight)
    assert actual.stride() == expected.stride()
    assert torch._C._is_alias_of(actual, weight)
    assert torch.equal(actual.dequantize(), expected.dequantize())


@pytest.mark.parametrize(
    "operation", ["linear", "contiguous", "clone", "to", "to_copy", "add_", "addmm_", "mm"]
)
def test_unsupported_transposed_operations_raise(weight, operation):
    transposed = weight.t()

    def run():
        if operation == "linear":
            F.linear(torch.zeros(3, 7, dtype=weight.dtype), transposed)
        elif operation == "contiguous":
            transposed.contiguous()
        elif operation == "clone":
            transposed.clone(memory_format=torch.contiguous_format)
        elif operation == "to":
            transposed.to(copy=True, memory_format=torch.contiguous_format)
        elif operation == "to_copy":
            torch.ops.aten._to_copy.default(transposed, memory_format=torch.contiguous_format)
        elif operation == "add_":
            torch.ops.aten.add_.Tensor(transposed, torch.zeros_like(transposed.dequantize()))
        elif operation == "addmm_":
            torch.ops.aten.addmm_.default(transposed, torch.zeros(64, 2), torch.zeros(2, 7))
        else:
            torch.mm(torch.zeros(3, 7, dtype=weight.dtype), weight)

    with pytest.raises(NotImplementedError, match="transposed"):
        run()


@pytest.mark.parametrize("operation", ["slice", "select"])
def test_unsupported_slicing_cannot_drop_wrapper(weight, operation):
    index = slice(3) if operation == "slice" else 0
    with pytest.raises(NotImplementedError):
        weight[index]


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_portable_addmm_preserves_linear_output_dtype(bias_dtype):
    weight = _weight(ConvRotInt8Tensor)
    activation = torch.randn(3, 64, dtype=weight.dtype)
    bias = torch.randn(7, dtype=bias_dtype)
    actual = torch.addmm(bias, activation, weight.t())
    expected = F.linear(activation, weight, bias)
    assert actual.dtype is weight.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("alpha", "beta", "bias_shape"),
    [
        (0.5, 1, (7,)),
        (1, 0, (7,)),
        (1, 2, (7,)),
        (1 + 0j, 1, (7,)),
        (1, 1, ()),
        (1, 1, (1,)),
        (1, 1, (3, 1)),
        (1, 1, (3, 7)),
    ],
)
def test_addmm_rejects_non_linear_arguments(weight, alpha, beta, bias_shape):
    activation = torch.randn(3, 64, dtype=weight.dtype)
    bias = torch.randn(bias_shape, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="quantized addmm requires"):
        torch.addmm(bias, activation, weight.t(), alpha=alpha, beta=beta)


@pytest.mark.parametrize("cls", [ConvRotInt8Tensor, PiperNVFP4Tensor])
@pytest.mark.parametrize("shape", [(64,), (3, 64), (2, 3, 64)])
@pytest.mark.parametrize("compiled", [False, True])
def test_portable_matmul_with_transposed_input(cls, shape, compiled):
    torch.manual_seed(106)
    weight = cls.from_hp(
        torch.randn(7, 64), **({"group_size": 64} if cls is ConvRotInt8Tensor else {})
    )
    activation = torch.randn((*shape[:-1], 128))[..., ::2]
    matmul = torch.matmul
    if compiled:
        torch.compiler.reset()
        matmul = torch.compile(matmul, backend="aot_eager", fullgraph=True)
    actual = matmul(activation, weight.t())
    expected = F.linear(activation, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_int8_checkpoint_without_transpose_metadata_still_reconstructs():
    weight = ConvRotInt8Tensor.from_hp(torch.randn(7, 64), group_size=64)
    del weight.transposed
    buffer = io.BytesIO()
    torch.save(weight, buffer)
    buffer.seek(0)
    actual = torch.load(buffer, weights_only=False).clone()
    assert actual.transposed is False
    assert torch.equal(actual.dequantize(), weight.dequantize())


@pytest.mark.parametrize(("cls", "high_first"), _WEIGHT_FORMATS)
@pytest.mark.parametrize("rows", [0, 1])
def test_transpose_empty_and_single_row_weights(cls, rows, high_first):
    if cls is ConvRotInt8Tensor:
        weight = cls.from_hp(torch.randn(rows, 64, dtype=torch.bfloat16), group_size=64)
    else:
        # TorchAO also accepts scales stored as a flat tensor.
        weight = cls(
            torch.randint(0, 256, (rows, 32), dtype=torch.uint8),
            torch.ones(rows * 4, dtype=torch.float8_e4m3fn),
            16,
            torch.bfloat16,
            high_first=high_first,
            **({"group_size": 64} if cls is ConvRotNVFP4Tensor else {}),
        )
    transposed = weight.t()
    assert transposed.stride() == weight.stride()[::-1]
    assert torch.equal(transposed.dequantize(), weight.dequantize().t())
    if isinstance(weight, PiperNVFP4Tensor):
        assert torch.equal(transposed.get_hp_scales(), weight.get_hp_scales())
