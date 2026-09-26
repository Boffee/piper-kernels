"""Metadata and output ownership for dense Piper's prepared NVIDIA launch."""

from contextlib import nullcontext
from math import prod
from unittest.mock import Mock, call

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels.attention.piper_attention._nvidia import triton as backend
from piper_kernels.attention.piper_attention._nvidia.policy import PiperAttentionExecutionPlan
from piper_kernels.attention.piper_attention._validation import validate_output_buffer

_QUERY_SHAPE = (2, 6, 65, 64)


class _NoTensorOperations(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"metadata validation or prepared launch performed {func}")


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _prepared(dtype=torch.bfloat16):
    plan = PiperAttentionExecutionPlan(
        block_m=64,
        grouped_qk=True,
        split_pv_head_dim=False,
        use_tensor_descriptors=False,
    )
    with torch.device("meta"):
        context = backend._PreparedPiperContext(
            key=torch.empty((2, 2, 97, 64), dtype=torch.int8),
            value=torch.empty((2, 2, 64, 97), dtype=torch.int8),
            key_scale=torch.empty((2, 2, 2)),
            value_scale_multiplier=torch.empty((2, 2, 97)),
            value_log_scale=torch.empty((2, 2, 97), dtype=torch.float16),
            value_mean=torch.empty((2, 2, 64)),
            key_length=97,
            is_causal=False,
            plan=plan,
        )
        query = backend._PreparedPiperQuery(
            data=torch.empty(_QUERY_SHAPE, dtype=torch.int8),
            scale=torch.empty((2, 6, 4)),
            descriptor=None,
            shape=_QUERY_SHAPE,
            dtype=dtype,
        )
    return context, query


def _offset_output(device, dtype):
    storage = torch.empty(prod(_QUERY_SHAPE) + 19, device=device, dtype=dtype)
    return storage[19:].view(_QUERY_SHAPE)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_validator_accepts_contiguous_offset_views_without_tensor_operations(device, dtype):
    output = _offset_output(device, dtype)
    assert output.is_contiguous()
    assert output.storage_offset() == 19

    with _NoTensorOperations():
        validate_output_buffer(output, shape=_QUERY_SHAPE, dtype=dtype, device=torch.device(device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_launch_reuses_caller_outputs_and_logical_query_metadata_without_allocations(
    monkeypatch, dtype
):
    context, query = _prepared(dtype)
    outputs = [_offset_output("meta", dtype) for _ in range(2)]
    kernel = _RecordingKernel()
    guard = Mock(return_value=nullcontext())
    monkeypatch.setattr(backend, "_piper_attention_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", guard)

    with _NoTensorOperations():
        for output in outputs:
            assert backend._launch_piper_attention_into(context, query, output) is output

    assert guard.call_args_list == [call(query.data.device), call(query.data.device)]
    assert len(kernel.calls) == 2
    for output, (grid, args, kwargs) in zip(outputs, kernel.calls, strict=True):
        assert grid == (2, 6, 2)
        expected_operands = (
            query.data,
            query.data,
            context.key,
            context.value,
            query.scale,
            context.key_scale,
            context.value_scale_multiplier,
            context.value_log_scale,
            context.value_mean,
            output,
        )
        operands = zip(args[:10], expected_operands, strict=True)
        assert all(actual is expected for actual, expected in operands)
        assert args[10:] == (65, 97)
        assert kwargs["heads"] == 6
        assert kwargs["head_groups"] == 3
        assert kwargs["head_dim"] == 64
        assert kwargs["aligned_queries"] is False
        assert kwargs["unmasked_key_tiles"] is False


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride", "layout"])
def test_incompatible_output_metadata_rejected_before_device_or_kernel_entry(monkeypatch, invalid):
    context, query = _prepared()
    output = torch.empty(
        (2, 6, 128, 64) if invalid == "shape" else query.shape,
        dtype=torch.float32 if invalid == "dtype" else query.dtype,
        device="cpu" if invalid == "device" else "meta",
        layout=torch.sparse_coo if invalid == "layout" else torch.strided,
    )
    if invalid == "stride":
        output = torch.empty((2, 65, 6, 64), device="meta", dtype=query.dtype).transpose(1, 2)
    kernel = _RecordingKernel()
    guard = Mock(side_effect=AssertionError("invalid output entered a device context"))
    monkeypatch.setattr(backend, "_piper_attention_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", guard)

    with _NoTensorOperations(), pytest.raises(ValueError, match="output must match the query"):
        backend._launch_piper_attention_into(context, query, output)

    assert not kernel.calls
    guard.assert_not_called()
