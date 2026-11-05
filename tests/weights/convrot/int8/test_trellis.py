"""Tests for decoding trellis-coded storage into ConvRot INT8 weights."""

import pytest
import torch

from piper_kernels.trellis._layout import TrellisLayout
from piper_kernels.trellis._reference import decode_reference
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
_LAYOUT = TrellisLayout(256, 1144)


def _storage(
    rows: int = 3, blocks_per_row: int = 2, seed: int = 0, layout: TrellisLayout = _LAYOUT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (rows, blocks_per_row, layout.bytes_per_block)
    bitstreams = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    block_scales = (torch.rand(rows, blocks_per_row, generator=generator) + 0.5).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(rows, generator=generator) * 1e-3 + 1e-4
    return bitstreams, block_scales, scale


def _cuda(storage: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    return [tensor.cuda() for tensor in storage]


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("layout", [_LAYOUT, TrellisLayout(224, 776)], ids=str)
def test_from_trellis_decodes_rotated_int8_storage(layout: TrellisLayout) -> None:
    bitstreams, block_scales, scale = _storage(layout=layout)
    expected = decode_reference(bitstreams, block_scales, layout).reshape(3, -1)
    scale = scale.cuda()

    weight = ConvRotInt8Tensor.from_trellis(
        bitstreams.cuda(),
        block_scales.cuda(),
        scale,
        block_size=layout.block_size,
        group_size=16,
        logical_dtype=torch.float16,
    )

    assert weight.shape == (3, 2 * layout.block_size)
    assert weight.dtype is torch.float16
    assert weight.group_size == 16
    assert torch.equal(weight.qdata.cpu(), expected)
    assert weight.scale.shape == (3, 1)
    assert weight.scale.data_ptr() == scale.data_ptr()  # contiguous row scale is reused


@pytest.mark.gpu
@_gpu
def test_copy_from_trellis_refills_existing_storage_at_another_rate() -> None:
    first = _cuda(_storage(seed=1))
    second_layout = TrellisLayout(256, 888)
    second = _cuda(_storage(seed=2, layout=second_layout))
    weight = ConvRotInt8Tensor.from_trellis(*first, block_size=256, group_size=256)
    expected = ConvRotInt8Tensor.from_trellis(*second, block_size=256, group_size=256)
    qdata, scale = weight.qdata, weight.scale

    result = weight.copy_from_trellis_(*second)

    assert result is weight
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale, expected.scale)


@pytest.mark.gpu
@_gpu
def test_copy_from_trellis_rejects_another_shape() -> None:
    weight = ConvRotInt8Tensor.from_trellis(*_cuda(_storage()), block_size=256, group_size=256)

    with pytest.raises(ValueError, match="output storage is incompatible"):
        weight.copy_from_trellis_(*_cuda(_storage(rows=2)))
    with pytest.raises(ValueError, match="output storage is incompatible"):
        weight.copy_from_trellis_(*_cuda(_storage(blocks_per_row=3)))


def test_from_trellis_rejects_cpu_storage() -> None:
    with pytest.raises(ValueError, match="requires a supported Triton accelerator"):
        ConvRotInt8Tensor.from_trellis(*_storage(), block_size=256, group_size=256)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b, s, r: (b[..., :95], s, r), "take 96 to 192 whole bytes"),
        (lambda b, s, r: (b.view(3, -1), s, r), "must be a 3-D uint8 tensor"),
        (lambda b, s, r: (b, s.float(), r), "block_scales must be float8_e4m3fn"),
        (lambda b, s, r: (b, s[:, :1], r), r"block_scales must be .* with shape \(3, 2\)"),
        (lambda b, s, r: (b, s, r.double()), "scale must be float32"),
        (lambda b, s, r: (b, s, r[:2]), r"scale must have shape \(3,\) or \(3, 1\)"),
    ],
)
def test_from_trellis_validates_storage_metadata(change, message) -> None:
    bitstreams, block_scales, scale = change(*_storage())

    with pytest.raises(ValueError, match=message):
        ConvRotInt8Tensor.from_trellis(
            bitstreams, block_scales, scale, block_size=256, group_size=256
        )


def test_from_trellis_rejects_unsupported_block_sizes() -> None:
    with pytest.raises(ValueError, match="hold 16 to 256 weights, got 512"):
        ConvRotInt8Tensor.from_trellis(*_storage(), block_size=512, group_size=256)
