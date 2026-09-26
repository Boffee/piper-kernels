"""Output fusion validation and fake paths inspect only tensor metadata."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import _backend, output

from .test_context_validation import _MetadataOnly, _OutputOnly
from .test_output import _arguments


@pytest.mark.parametrize(
    "invalid",
    ["width", "bias", "scale", "group", "device", "causal", "dtype", "chunk", "grad"],
)
def test_output_rejects_invalid_metadata_without_tensor_operations(invalid):
    args = list(_arguments(device="meta", sequence=65))
    chunk_rows = 128
    if invalid == "width":
        args[19] = torch.empty(80, 64, device="meta", dtype=torch.int8)
    elif invalid == "bias":
        args[21] = torch.empty(79, device="meta")
    elif invalid == "scale":
        args[23] = torch.empty(1, device="meta")
    elif invalid == "group":
        args[22] = 32
    elif invalid == "device":
        args[20] = torch.empty(80, 1)
    elif invalid == "causal":
        args[17] = True
    elif invalid == "dtype":
        args[18] = torch.float32
    elif invalid == "chunk":
        chunk_rows = 64
    elif invalid == "grad":
        args[20].requires_grad_()
    with _MetadataOnly(), pytest.raises((ValueError, TypeError, RuntimeError)):
        output._validate_inputs(*args, head_dim=64, query_chunk_rows=chunk_rows)


@pytest.mark.parametrize("batch", [0, 2])
def test_output_fake_allocates_only_final_output_and_empty_batch_skips_hardware(monkeypatch, batch):
    args = list(_arguments(device="meta", sequence=65))
    for index in (0, 1, 10, 11, 12, 13, 14, 15):
        args[index] = args[index][:batch]

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata path probed hardware")

    monkeypatch.setattr(AcceleratorTarget, "from_device", forbidden)
    monkeypatch.setattr(_backend, "select_projection_backend", forbidden)
    with _OutputOnly():
        result = output._projected_query_attention_output_op_fake(*args, head_dim=64)
    assert result.shape == (batch, 65, 80)
    assert result.dtype is torch.bfloat16
    if batch == 0:
        # Call the implementation directly so a meta tensor cannot bypass it.
        with _OutputOnly():
            result = output._projected_query_attention_output_op._init_fn(*args, head_dim=64)
        assert result.shape == (0, 65, 80)
