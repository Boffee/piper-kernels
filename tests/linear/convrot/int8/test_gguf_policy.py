"""GGUF policy owns hardware choices; shared execution follows its schedule."""

from unittest.mock import MagicMock, Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.convrot.int8 import _gguf_policy
from piper_kernels.linear.convrot.int8._generic import triton as generic
from piper_kernels.linear.convrot.int8._nvidia import policy as nvidia_policy
from piper_kernels.linear.convrot.int8._plan import fused_preparation_chunks


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm70"),
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cuda"),
        AcceleratorTarget("hip", "gfx1036"),
        AcceleratorTarget("hip", "gfx942"),
        AcceleratorTarget("hip", "gfx1201"),
        AcceleratorTarget("hip", "gfx9999"),
        AcceleratorTarget("other", "unvalidated"),
    ],
)
def test_policy_preserves_previous_schedules_across_group_aligned_widths(target):
    for width in range(16, 100001, 16):
        previous = fused_preparation_chunks(width)
        if not target.is_nvidia_cuda and width > 8192:
            previous = None
        assert _gguf_policy.select_conversion_chunks(target, width) == previous


@pytest.mark.parametrize(
    ("width", "nvidia_chunks", "default_chunks"),
    [
        (16, (1, 128), (1, 128)),
        (4096, (1, 4096), (1, 4096)),
        (6144, (3, 2048), (3, 2048)),
        (8192, (1, 8192), (1, 8192)),
        (8208, (3, 4096), None),
        (16384, (2, 8192), None),
        (32768, (2, 16384), None),
        (49152, (3, 16384), None),
        (49168, None, None),
    ],
)
def test_schedule_boundaries(width, nvidia_chunks, default_chunks):
    assert (
        _gguf_policy.select_conversion_chunks(AcceleratorTarget("cuda", "sm70"), width)
        == nvidia_chunks
    )
    assert (
        _gguf_policy.select_conversion_chunks(AcceleratorTarget("hip", "gfx1201"), width)
        == default_chunks
    )


def test_nvidia_conversion_does_not_require_a_matrix_backend():
    target = AcceleratorTarget("cuda", "sm70")
    assert not nvidia_policy.supports_target(target)
    assert _gguf_policy.select_conversion_chunks(target, 16384) == (2, 8192)


@pytest.mark.parametrize("chunks", [(2, 8192), None])
def test_shared_converter_executes_selected_schedule_without_vendor_rules(monkeypatch, chunks):
    target = AcceleratorTarget("hip", "gfx1201")
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    select = Mock(return_value=chunks)
    fused = MagicMock()
    tiled = Mock()
    monkeypatch.setattr(generic, "select_conversion_chunks", select)
    monkeypatch.setattr(generic, "rotate_quantize_rows_kernel", fused)
    monkeypatch.setattr(generic, "_convert_gguf_tiled_out", tiled)
    # Force fusion above the former inline HIP cutoff, or tiles below it.
    width = 16384 if chunks is not None else 4096
    data = torch.empty(2, 16, dtype=torch.uint8)
    qdata, scale = torch.empty(2, width, dtype=torch.int8), torch.empty(2, 1)
    arguments = (data, int(GGUFQuantizationType.Q4_K), 256, torch.bfloat16, qdata, scale)
    generic.convert_gguf_out(*arguments)
    select.assert_called_once_with(target, width)
    if chunks is None:
        fused.__getitem__.assert_not_called()
        tiled.assert_called_once_with(*arguments)
    else:
        tiled.assert_not_called()
        fused.__getitem__.assert_called_once_with((2,))
        fused.__getitem__.return_value.assert_called_once_with(
            data,
            qdata,
            scale,
            width,
            chunk_count=2,
            chunk_size=8192,
            group_size=256,
            inverse_sqrt_group=256**-0.5,
            logical_dtype_code=2,
            activation_fn=None,
            accelerator_backend="hip",
            gguf_quant_type=int(GGUFQuantizationType.Q4_K),
            num_warps=4,
        )


@pytest.mark.parametrize("shape", [(0, 256), (3, 0)])
def test_empty_shared_conversion_does_not_select_a_schedule(monkeypatch, shape):
    unexpected = Mock(side_effect=AssertionError("empty conversion queried hardware or policy"))
    monkeypatch.setattr(AcceleratorTarget, "from_device", unexpected)
    monkeypatch.setattr(generic, "select_conversion_chunks", unexpected)
    qdata, scale = torch.empty(shape, dtype=torch.int8), torch.empty(shape[0], 1)
    generic.convert_gguf_out(
        torch.empty(shape), int(GGUFQuantizationType.F32), 16, torch.float32, qdata, scale
    )
    assert (scale == 1e-30).all()
    unexpected.assert_not_called()
