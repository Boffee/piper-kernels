"""Tests for direct GGUF-to-ConvRot-INT8 conversion."""

import pytest
import torch
from gguf_format._fixtures import dequantize_reference, finite_packed

from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
def test_from_gguf_matches_materialized_reference(quant_type: GGUFQuantizationType) -> None:
    torch.manual_seed(820 + int(quant_type))
    packed = finite_packed(quant_type)
    dense = dequantize_reference(packed, quant_type, dtype=torch.bfloat16).cuda()

    expected = ConvRotInt8Tensor.from_hp(dense, group_size=64)
    actual = ConvRotInt8Tensor.from_gguf(
        packed.cuda(),
        quant_type=quant_type,
        group_size=64,
    )

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_from_gguf_refills_existing_storage() -> None:
    first = finite_packed(GGUFQuantizationType.Q4_K)
    second = finite_packed(GGUFQuantizationType.Q4_K)
    weight = ConvRotInt8Tensor.from_gguf(
        first.cuda(),
        quant_type=GGUFQuantizationType.Q4_K,
        group_size=64,
    )
    expected = ConvRotInt8Tensor.from_gguf(
        second.cuda(),
        quant_type=GGUFQuantizationType.Q4_K,
        group_size=64,
    )
    qdata = weight.qdata
    scale = weight.scale

    result = weight.copy_from_gguf_(second.cuda(), quant_type=GGUFQuantizationType.Q4_K)

    assert result is weight
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale, expected.scale)


def test_from_gguf_rejects_cpu_storage() -> None:
    packed = finite_packed(GGUFQuantizationType.Q5_1)
    attributed = packed.as_subclass(torch.Tensor)
    attributed.quant_type = int(GGUFQuantizationType.Q5_1)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="requires CUDA"):
        ConvRotInt8Tensor.from_gguf(attributed, group_size=64)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_from_gguf_reads_quant_type_attribute() -> None:
    packed = finite_packed(GGUFQuantizationType.Q5_1).cuda()
    packed.quant_type = int(GGUFQuantizationType.Q5_1)  # type: ignore[attr-defined]

    actual = ConvRotInt8Tensor.from_gguf(packed, group_size=64)
    expected = ConvRotInt8Tensor.from_gguf(
        packed,
        quant_type=GGUFQuantizationType.Q5_1,
        group_size=64,
    )

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
