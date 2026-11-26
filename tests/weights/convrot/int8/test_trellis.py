"""Tests for decoding trellis-coded storage into ConvRot INT8 weights."""

import pytest
import torch

from piper_kernels.trellis import TrellisLayout, TrellisStorage
from piper_kernels.trellis._reference import decode_reference
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="trellis encoding requires an NVIDIA GPU",
)
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
@pytest.mark.parametrize(
    "layout", [TrellisLayout(256, 512), _LAYOUT, TrellisLayout(224, 776)], ids=str
)
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
        (lambda b, s, r: (b[..., :63], s, r), "take 64 to 192 whole bytes"),
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


def _variable_storage(
    block_size: int = 256,
    widths: tuple[int, ...] = (96, 128, 176),
    blocks_per_row: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rows = len(widths)
    row_widths = torch.tensor(widths, dtype=torch.uint8)
    bitstreams = torch.randint(
        0, 256, (blocks_per_row * sum(widths),), dtype=torch.uint8, generator=generator
    )
    block_scales = (torch.rand(rows, blocks_per_row, generator=generator) + 0.5).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(rows, generator=generator) * 1e-3 + 1e-4
    return bitstreams, block_scales, scale, row_widths


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize(
    ("block_size", "widths", "blocks_per_row", "noncontiguous"),
    [
        (256, (64, 79, 96, 128, 143, 176), 2, False),
        (224, (56, 70, 84, 112, 140, 168), 2, False),
        (256, (144,), 3, False),
        (256, (), 2, False),
        (256, (64, 144, 192), 2, True),
    ],
)
def test_from_trellis_variable_rows_matches_rectangular_and_reference(
    block_size: int, widths: tuple[int, ...], blocks_per_row: int, noncontiguous: bool
) -> None:
    bitstreams, block_scales, scale, row_widths = _variable_storage(
        block_size=block_size, widths=widths, blocks_per_row=blocks_per_row, seed=block_size
    )

    gpu_bitstreams = bitstreams.cuda()
    gpu_block_scales = block_scales.cuda()
    gpu_scale = scale.cuda()
    gpu_row_widths = row_widths.cuda()
    if noncontiguous:
        gpu_bitstreams = torch.stack((gpu_bitstreams, gpu_bitstreams), dim=1)[:, 0]
        gpu_block_scales = gpu_block_scales.t().contiguous().t()
        gpu_row_widths = torch.stack((gpu_row_widths, gpu_row_widths), dim=1)[:, 0]
        gpu_scale = torch.stack((gpu_scale, gpu_scale), dim=1)[:, 0]

    weight = ConvRotInt8Tensor.from_trellis(
        gpu_bitstreams,
        gpu_block_scales,
        gpu_scale,
        block_size=block_size,
        group_size=16,
        logical_dtype=torch.float16,
        row_widths=gpu_row_widths,
    )

    assert weight.shape == (len(widths), blocks_per_row * block_size)
    assert weight.dtype is torch.float16
    assert weight.group_size == 16
    assert weight.scale.shape == (len(widths), 1)
    if not noncontiguous:
        assert weight.scale.data_ptr() == gpu_scale.data_ptr()

    offset = 0
    for row, width in enumerate(widths):
        row_bytes = blocks_per_row * width
        row_streams = bitstreams[offset : offset + row_bytes].view(1, blocks_per_row, width)
        row_layout = TrellisLayout.from_storage(block_size, width)
        row_expected = decode_reference(
            row_streams, block_scales[row : row + 1], row_layout
        ).reshape(1, -1)
        assert torch.equal(weight.qdata[row : row + 1].cpu(), row_expected)
        row_weight = ConvRotInt8Tensor.from_trellis(
            row_streams.cuda(),
            block_scales[row : row + 1].cuda(),
            scale[row : row + 1].cuda(),
            block_size=block_size,
            group_size=16,
            logical_dtype=torch.float16,
        )
        assert torch.equal(weight.qdata[row : row + 1], row_weight.qdata)
        offset += row_bytes


@pytest.mark.gpu
@_gpu
def test_copy_from_trellis_refills_between_rectangular_and_variable_rows() -> None:
    rect_first = _cuda(_storage(seed=1))
    var_a = _cuda(_variable_storage(widths=(96, 144, 192), seed=2))
    var_b = _cuda(_variable_storage(widths=(176, 112, 128), seed=3))
    rect_last = _cuda(_storage(seed=4, layout=TrellisLayout(256, 888)))

    weight = ConvRotInt8Tensor.from_trellis(*rect_first, block_size=256, group_size=256)
    qdata, scale = weight.qdata, weight.scale

    for args, row_widths in (
        (var_a[:3], var_a[3]),
        (var_b[:3], var_b[3]),
        (rect_last, None),
    ):
        expected = ConvRotInt8Tensor.from_trellis(
            *args, block_size=256, group_size=256, row_widths=row_widths
        )
        result = weight.copy_from_trellis_(*args, row_widths=row_widths)
        assert result is weight
        assert weight.qdata is qdata
        assert weight.scale is scale
        assert torch.equal(weight.qdata, expected.qdata)
        assert torch.equal(weight.scale, expected.scale)


@pytest.mark.gpu
@_gpu
def test_copy_from_trellis_variable_rows_rejects_incompatible_shape_without_mutating() -> None:
    weight = ConvRotInt8Tensor.from_trellis(*_cuda(_storage()), block_size=256, group_size=256)
    expected_qdata = weight.qdata.clone()
    expected_scale = weight.scale.clone()

    wrong_rows = _cuda(_variable_storage(widths=(96, 144), seed=5))
    with pytest.raises(ValueError, match="output storage is incompatible"):
        weight.copy_from_trellis_(*wrong_rows[:3], row_widths=wrong_rows[3])

    wrong_blocks = _cuda(_variable_storage(widths=(96, 144, 192), blocks_per_row=3, seed=6))
    with pytest.raises(ValueError, match="output storage is incompatible"):
        weight.copy_from_trellis_(*wrong_blocks[:3], row_widths=wrong_blocks[3])

    assert torch.equal(weight.qdata, expected_qdata)
    assert torch.equal(weight.scale, expected_scale)


@pytest.mark.gpu
@_nvidia
def test_from_hp_bytes_per_block_round_trips_through_variable_row_trellis() -> None:
    weight = (torch.randn(8, 512, generator=torch.Generator().manual_seed(7)) * 0.02).cuda()
    weight = weight.to(torch.bfloat16)
    widths = [64, 79, 96, 128, 144, 176, 64, 192]
    row_widths = torch.tensor(widths, dtype=torch.uint8, device="cuda")
    blocks_per_row = 2

    ref_qdata = torch.empty(8, 512, dtype=torch.int8, device="cuda")
    ref_scale = torch.empty(8, 1, dtype=torch.float32, device="cuda")
    block_scales = torch.empty(8, blocks_per_row, dtype=torch.float8_e4m3fn, device="cuda")
    bitstreams = torch.empty(blocks_per_row * sum(widths), dtype=torch.uint8, device="cuda")

    locations = (row_widths.cumsum(0, dtype=torch.int64) - row_widths) * blocks_per_row
    for width in sorted(set(widths)):
        ids = (row_widths == width).nonzero().flatten()
        encoded = TrellisStorage.from_hp(weight[ids], group_size=64, bytes_per_block=width)
        rect = ConvRotInt8Tensor.from_trellis(
            encoded.bitstreams,
            encoded.block_scales,
            encoded.scale,
            block_size=256,
            group_size=64,
        )
        ref_qdata[ids] = rect.qdata
        ref_scale[ids] = rect.scale
        block_scales[ids] = encoded.block_scales
        for row_idx, stream in zip(ids.tolist(), encoded.bitstreams, strict=True):
            start = int(locations[row_idx].item())
            bitstreams[start : start + blocks_per_row * width] = stream.reshape(-1)

    decoded = ConvRotInt8Tensor.from_trellis(
        bitstreams,
        block_scales,
        ref_scale.clone(),
        block_size=256,
        group_size=64,
        row_widths=row_widths,
    )
    assert torch.equal(decoded.qdata, ref_qdata)
    assert torch.equal(decoded.scale, ref_scale)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b, s, r, w: (b[None], s, r, w), "must be a 1-D uint8 tensor"),
        (lambda b, s, r, w: (b.to(torch.int8), s, r, w), "must be a 1-D uint8 tensor"),
        (lambda b, s, r, w: (b, s.float(), r, w), "must be a 2-D float8_e4m3fn tensor"),
        (lambda b, s, r, w: (b, s.reshape(-1), r, w), "must be a 2-D float8_e4m3fn tensor"),
        (lambda b, s, r, w: (b, s, r, w.to(torch.int32)), "row_widths must be uint8"),
        (lambda b, s, r, w: (b, s, r, w[:2]), r"row_widths must be uint8 with shape \(3,\)"),
        (lambda b, s, r, w: (b, s, r[:2], w), r"scale must have shape \(3,\) or \(3, 1\)"),
    ],
)
def test_from_trellis_variable_rows_validates_metadata(change, message) -> None:
    bitstreams, block_scales, scale, row_widths = change(*_variable_storage())

    with pytest.raises(ValueError, match=message):
        ConvRotInt8Tensor.from_trellis(
            bitstreams,
            block_scales,
            scale,
            block_size=256,
            group_size=256,
            row_widths=row_widths,
        )


def test_from_trellis_variable_rows_rejects_bad_block_size_and_cpu() -> None:
    bitstreams, block_scales, scale, row_widths = _variable_storage()
    with pytest.raises(ValueError, match="hold 16 to 256 weights, got 512"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams,
            block_scales,
            scale,
            block_size=512,
            group_size=256,
            row_widths=row_widths,
        )
    with pytest.raises(ValueError, match="requires a supported Triton accelerator"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams,
            block_scales,
            scale,
            block_size=256,
            group_size=256,
            row_widths=row_widths,
        )


@pytest.mark.gpu
@_gpu
def test_from_trellis_variable_rows_rejects_mismatched_device() -> None:
    bitstreams, block_scales, scale, row_widths = _variable_storage()
    with pytest.raises(ValueError, match="must share a device"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams.cuda(),
            block_scales.cuda(),
            scale.cuda(),
            block_size=256,
            group_size=256,
            row_widths=row_widths,
        )


def test_from_trellis_validates_group_size_dtype_divisibility_and_layout_before_alloc() -> None:
    bitstreams, block_scales, scale = _storage(blocks_per_row=1)
    # Invalid group_size, logical_dtype, in_features divisibility, and non-strided layout must all
    # reject on CPU storage before checking device support or allocating decoded storage.
    with pytest.raises(ValueError, match="group size"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams, block_scales, scale, block_size=256, group_size=3
        )
    with pytest.raises(ValueError, match="logical dtype"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams,
            block_scales,
            scale,
            block_size=256,
            group_size=256,
            logical_dtype=torch.int32,
        )
    with pytest.raises(ValueError, match="divisible by group size"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams[:, :1, :64],
            block_scales[:, :1],
            scale,
            block_size=128,
            group_size=256,
        )
    with pytest.raises(ValueError, match="strided layout"):
        ConvRotInt8Tensor.from_trellis(
            bitstreams.to_sparse(),
            block_scales,
            scale,
            block_size=256,
            group_size=256,
        )
