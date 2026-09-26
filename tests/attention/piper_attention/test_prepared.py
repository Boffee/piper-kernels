"""Metadata and output ownership for dense Piper's prepared NVIDIA launch."""

from contextlib import nullcontext
from dataclasses import replace
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


def _prepared(dtype=torch.bfloat16, *, query_length=65, key_length=97, block_m=64):
    plan = PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=True,
        split_pv_head_dim=False,
        use_tensor_descriptors=False,
    )
    with torch.device("meta"):
        context = backend._PreparedPiperContext(
            key=torch.empty((2, 2, key_length, 64), dtype=torch.int8),
            value=torch.empty((2, 2, 64, key_length), dtype=torch.int8),
            key_scale=torch.empty((2, 2, (key_length + 63) // 64)),
            value_scale_multiplier=torch.empty((2, 2, key_length)),
            value_log_scale=torch.empty((2, 2, key_length), dtype=torch.float16),
            value_mean=torch.empty((2, 2, 64)),
            key_length=key_length,
            is_causal=False,
            plan=plan,
        )
        query = backend._PreparedPiperQuery(
            data=torch.empty((2, 6, query_length, 64), dtype=torch.int8),
            scale=torch.empty((2, 6, (query_length + 31) // 32)),
            descriptor=None,
            shape=(2, 6, query_length, 64),
            dtype=dtype,
        )
    return context, query


def _offset_output(device, dtype):
    storage = torch.empty(prod(_QUERY_SHAPE) + 19, device=device, dtype=dtype)
    return storage[19:].view(_QUERY_SHAPE)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("layout", ["offset", "bhsd_slice", "bshd", "pingpong_tail"])
def test_output_validator_accepts_strided_views_without_tensor_operations(device, layout):
    dtype = torch.bfloat16
    if layout == "offset":
        output = _offset_output(device, dtype)
    elif layout == "bhsd_slice":
        output = torch.empty((2, 6, 69, 64), device=device, dtype=dtype)[:, :, 2:67]
    elif layout == "bshd":
        output = torch.empty((2, 69, 6, 64), device=device, dtype=dtype)[:, 2:67].transpose(1, 2)
    else:
        slots = torch.empty((2, 2, 69, 6, 64), device=device, dtype=dtype)
        output = slots[1, :, :65].transpose(1, 2)
    assert output.storage_offset() > 0

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
        assert args[10:] == (65, 97, 0, 65, 0, *output.stride()[:3])
        assert kwargs["heads"] == 6
        assert kwargs["head_groups"] == 3
        assert kwargs["head_dim"] == 64
        assert kwargs["aligned_queries"] is False
        assert kwargs["unmasked_key_tiles"] is False


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "device", "feature_stride", "overlap", "broadcast", "layout"]
)
def test_incompatible_output_metadata_rejected_before_device_or_kernel_entry(monkeypatch, invalid):
    context, query = _prepared()
    output = torch.empty(
        (2, 6, 128, 64) if invalid == "shape" else query.shape,
        dtype=torch.float32 if invalid == "dtype" else query.dtype,
        device="cpu" if invalid == "device" else "meta",
        layout=torch.sparse_coo if invalid == "layout" else torch.strided,
    )
    if invalid == "feature_stride":
        output = torch.empty((2, 6, 65, 128), device="meta", dtype=query.dtype)[..., ::2]
    elif invalid == "overlap":
        output = output.as_strided(query.shape, (6 * 65 * 64, 65 * 64, 32, 1))
    elif invalid == "broadcast":
        output = output[:1].expand(query.shape)
    kernel = _RecordingKernel()
    guard = Mock(side_effect=AssertionError("invalid output entered a device context"))
    monkeypatch.setattr(backend, "_piper_attention_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", guard)

    with _NoTensorOperations(), pytest.raises(ValueError, match="Piper Attention output"):
        backend._launch_piper_attention_into(context, query, output)

    assert not kernel.calls
    guard.assert_not_called()


@pytest.mark.parametrize(
    ("block_m", "origin", "start", "rows"),
    [(64, 0, 64, 17), (64, 64, 64, None), (128, 128, 128, 17)],
)
def test_window_launch_keeps_local_query_and_global_causal_coordinates(
    monkeypatch, block_m, origin, start, rows
):
    context, query = _prepared(
        query_length=2 * block_m + 17, key_length=4 * block_m + 17, block_m=block_m
    )
    context = replace(context, is_causal=True)
    query = replace(query, global_row_offset=origin)
    output_rows = query.shape[2] - start if rows is None else rows
    output = torch.empty((2, output_rows + 7, 6, 64), device="meta", dtype=query.dtype)
    output = output[:, 3 : output_rows + 3].transpose(1, 2)
    kernel = _RecordingKernel()
    monkeypatch.setattr(backend, "_piper_attention_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", lambda _device: nullcontext())

    with _NoTensorOperations():
        actual = backend._launch_piper_attention_into(
            context, query, output, query_start=start, query_rows=rows
        )

    assert actual is output
    [(grid, args, kwargs)] = kernel.calls
    assert grid == ((output_rows + block_m - 1) // block_m, 6, 2)
    assert args[0] is query.data
    assert args[9] is output
    assert args[10:] == (
        query.shape[2],
        context.key_length,
        start,
        output_rows,
        origin + start,
        *output.stride()[:3],
    )
    assert kwargs["is_causal"] is True


@pytest.mark.parametrize(
    ("start", "rows", "origin", "causal", "error"),
    [
        (-64, 1, 0, False, ValueError),
        (1, 1, 0, False, ValueError),
        (True, 1, 0, False, TypeError),
        (0, 0, 0, False, ValueError),
        (0, -1, 0, False, ValueError),
        (0, True, 0, False, TypeError),
        (0, 1.5, 0, False, TypeError),
        (256, None, 0, False, ValueError),
        (128, 66, 0, False, ValueError),
        (0, 1, -64, False, ValueError),
        (0, 1, 1, False, ValueError),
        (0, 1, True, False, TypeError),
        (128, 65, 128, True, ValueError),
    ],
)
def test_invalid_window_rejected_before_device_or_tensor_operations(
    monkeypatch, start, rows, origin, causal, error
):
    context, query = _prepared(query_length=193, key_length=257)
    context = replace(context, is_causal=causal)
    query = replace(query, global_row_offset=origin)
    output = torch.empty((2, 6, 1, 64), device="meta", dtype=query.dtype)
    kernel = _RecordingKernel()
    guard = Mock(side_effect=AssertionError("invalid window entered a device context"))
    monkeypatch.setattr(backend, "_piper_attention_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", guard)

    window_error = (
        "Piper Attention (query_start|query_rows|global_row_offset|query window|causal query)"
    )
    with _NoTensorOperations(), pytest.raises(error, match=window_error):
        backend._launch_piper_attention_into(
            context, query, output, query_start=start, query_rows=rows
        )

    assert not kernel.calls
    guard.assert_not_called()


@pytest.mark.parametrize("origin", [-128, 64, True])
def test_query_preparation_rejects_unaligned_origin_before_tensor_operations(monkeypatch, origin):
    context, query = _prepared(block_m=128)
    floating_query = torch.empty(query.shape, device="meta", dtype=query.dtype)
    guard = Mock(side_effect=AssertionError("invalid query origin entered a device context"))
    monkeypatch.setattr(backend, "device_context", guard)

    with _NoTensorOperations(), pytest.raises((TypeError, ValueError), match="global_row_offset"):
        backend._prepare_piper_query(
            floating_query, 0.125, execution_plan=context.plan, global_row_offset=origin
        )

    guard.assert_not_called()
