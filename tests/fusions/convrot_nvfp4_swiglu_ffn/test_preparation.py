"""FP32 SwiGLU preparation inside full ConvRot NVFP4 FFN fusion."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import _preparation
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import triton as ffn_backend
from piper_kernels.linear.convrot.nvfp4 import _ops

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)


def _assert_storage_equal(actual, expected):
    for left, right in zip(actual[:2], expected[:2], strict=True):
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))
    # Different FP32 schedules can change the global scale by a final FP32 ULP.
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-6, atol=0)


def _prepare(projections, group, high_first=False, per_tensor_scale=None):
    backend = ffn_backend._preparation(group, group, group, high_first, high_first, high_first)
    return backend.prepare_down(projections, per_tensor_scale, per_tensor_scale is None)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("dynamic", "materialize"),
    [(False, False), (True, False), (True, True)],
    ids=["static", "dynamic-recompute", "dynamic-reuse"],
)
@pytest.mark.parametrize(
    ("shape", "group", "high_first"),
    [
        ((2, 17, 256), 256, False),
        ((129, 80), 16, False),
        ((129, 5_376), 16, True),
        ((3, 13_824), 64, False),
        ((1_797, 8_192), 256, True),
    ],
)
def test_preparation_matches_existing_fp32_arithmetic(
    monkeypatch, dtype, materialize, dynamic, shape, group, high_first
):
    monkeypatch.setattr(_preparation, "_ROTATED_WORKSPACE_BYTES", 1 << 30 if materialize else 0)
    torch.manual_seed(941)
    gate = torch.randn(shape, device="cuda", dtype=dtype) * 3
    value = torch.randn_like(gate)
    projections = torch.cat((value, gate), dim=-1)
    per_tensor_scale = None if dynamic else torch.tensor(0.01, device="cuda")
    with torch.inference_mode():
        expected = _ops.prepare_input(
            projections, per_tensor_scale, dynamic, group, "swiglu", high_first
        )
        actual = _prepare(projections, group, high_first, per_tensor_scale)
    if not dynamic:
        assert actual[2] is per_tensor_scale
    _assert_storage_equal(actual, expected)


class _RejectCopies(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func not in (torch.ops.aten.clone.default, torch.ops.aten._to_copy.default)
        return func(*args, **(kwargs or {}))


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_preparation_reads_private_workspace_without_copies_or_mutation(zero, dynamic):
    torch.manual_seed(943)
    # Match a short final chunk of the runner's larger reusable allocation.
    workspace = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
    projections = workspace[:17]
    if zero:
        projections[:, 512:].zero_()
    before = workspace.clone()
    per_tensor_scale = None if dynamic else torch.tensor(0.01, device="cuda")
    with torch.inference_mode():
        expected = _ops.prepare_input(projections, per_tensor_scale, dynamic, 256, "swiglu")
        with _RejectCopies():
            actual = _prepare(projections, 256, per_tensor_scale=per_tensor_scale)
    _assert_storage_equal(actual, expected)
    assert torch.equal(workspace, before)
    assert torch.isfinite(actual[2])


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("materialize", [False, True])
def test_fp32_intermediates_avoid_fp16_product_overflow(monkeypatch, materialize):
    monkeypatch.setattr(_preparation, "_ROTATED_WORKSPACE_BYTES", 1 << 30 if materialize else 0)
    gate = torch.full((17, 256), 512.0, dtype=torch.float16, device="cuda")
    value = torch.full_like(gate, 512.0)
    assert not torch.isfinite(torch.nn.functional.silu(gate) * value).all()
    with torch.inference_mode():
        projections = torch.cat((value, gate), dim=-1)
        expected = _ops.prepare_input(projections, None, True, 256, "swiglu")
        actual = _prepare(projections, 256)
    _assert_storage_equal(actual, expected)
    assert torch.isfinite(actual[1].float()).all()
    assert torch.isfinite(actual[2])
    assert actual[2] > 0
