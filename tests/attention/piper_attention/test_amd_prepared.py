"""RDNA4 prepared operands and output ownership can be checked without hardware."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from piper_kernels.attention.piper_attention._amd import gluon as backend


def _prepared(
    batch=2,
    *,
    query_length=65,
    key_length=97,
    global_row_offset=0,
    causal=False,
):
    query_storage = (query_length + 63) // 64 * 64
    key_storage = (key_length + 63) // 64 * 64
    with torch.device("meta"):
        context = backend.PreparedContext(
            key=torch.empty((batch, 2, key_storage, 64), dtype=torch.int8),
            value=torch.empty((batch, 2, key_storage // 64, 64, 64), dtype=torch.int8),
            key_scale=torch.empty((batch, 2, key_storage // 64)),
            multiplier=torch.empty((batch, 2, key_storage)),
            log_scale=torch.empty((batch, 2, key_storage)),
            value_mean=torch.empty((batch, 2, 64)),
            key_length=key_length,
            is_causal=causal,
        )
        query = backend.PreparedQuery(
            data=torch.empty((batch, 6, query_storage, 64), dtype=torch.int8),
            scale=torch.empty((batch, 6, query_storage // 32)),
            shape=(batch, 6, query_length, 64),
            dtype=torch.bfloat16,
            global_row_offset=global_row_offset,
        )
    return context, query


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def test_launch_reuses_operands_and_logical_metadata_with_independent_outputs(monkeypatch):
    context, query = _prepared()
    outputs = [torch.empty(query.shape, dtype=query.dtype, device="meta") for _ in range(2)]
    kernel = _RecordingKernel()
    monkeypatch.setattr(backend, "_dense_piper_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", lambda _device: nullcontext())
    for output in outputs:
        assert backend.launch_attention_into(context, query, output) is output
    assert len(kernel.calls) == 2
    for output, (grid, args, _kwargs) in zip(outputs, kernel.calls, strict=True):
        assert grid == (2, 6, 2)
        expected_operands = (
            query.data,
            context.key,
            context.value,
            query.scale,
            context.key_scale,
            context.multiplier,
            context.log_scale,
            context.value_mean,
            output,
        )
        assert all(
            actual is expected for actual, expected in zip(args[:9], expected_operands, strict=True)
        )
        assert args[9:21] == (0, 65, 0, 97, 128, 128, 6, *output.stride()[:3], 3, 64)


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "device", "feature_stride", "overlap", "layout"]
)
def test_launch_rejects_incompatible_output_metadata_before_kernel_entry(monkeypatch, invalid):
    context, query = _prepared()
    shape = query.shape if invalid != "shape" else (2, 6, 128, 64)
    output = torch.empty(
        shape,
        dtype=torch.float16 if invalid == "dtype" else query.dtype,
        device="cpu" if invalid == "device" else "meta",
        layout=torch.sparse_coo if invalid == "layout" else torch.strided,
    )
    if invalid == "feature_stride":
        output = torch.empty((2, 6, 65, 128), dtype=query.dtype, device="meta")[..., ::2]
    elif invalid == "overlap":
        output = torch.empty((1, 6, 65, 64), dtype=query.dtype, device="meta").expand(query.shape)
    kernel = _RecordingKernel()
    guard = Mock(side_effect=AssertionError("invalid output entered a device context"))
    monkeypatch.setattr(backend, "_dense_piper_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", guard)
    with pytest.raises(ValueError, match="Piper Attention output"):
        backend.launch_attention_into(context, query, output)
    assert not kernel.calls
    guard.assert_not_called()


def test_empty_batch_returns_output_without_device_or_kernel_entry(monkeypatch):
    context, query = _prepared(batch=0)
    output = torch.empty(query.shape, dtype=query.dtype, device="meta")
    kernel = _RecordingKernel()
    guard = Mock(side_effect=AssertionError("empty batch entered a device context"))
    monkeypatch.setattr(backend, "_dense_piper_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", guard)
    assert backend.launch_attention_into(context, query, output) is output
    assert not kernel.calls
    guard.assert_not_called()


@pytest.mark.parametrize("global_row_offset", [0, 128])
def test_independent_preparation_keeps_quantization_policy_without_allocating_output(
    monkeypatch,
    global_row_offset,
):
    expected_context, expected_query = _prepared()
    with torch.device("meta"):
        query = torch.empty(expected_query.shape, dtype=expected_query.dtype)
        key = torch.empty((2, 2, 97, 64), dtype=query.dtype)
        value = torch.empty_like(key)
        key_mean = torch.empty((2, 2, 64))
    means = Mock(return_value=(key_mean, expected_context.value_mean))
    key_quantization = Mock(return_value=(expected_context.key, expected_context.key_scale))
    query_quantization = Mock(return_value=(expected_query.data, expected_query.scale))
    value_quantization = Mock(
        return_value=(
            expected_context.value,
            expected_context.multiplier,
            expected_context.log_scale,
        )
    )
    monkeypatch.setattr(backend, "compute_kv_means", means)
    monkeypatch.setattr(backend.qk_quantization, "prepare_key", key_quantization)
    monkeypatch.setattr(backend.qk_quantization, "prepare_query", query_quantization)
    monkeypatch.setattr(backend, "prepare_value", value_quantization)
    monkeypatch.setattr(backend, "device_context", lambda _device: nullcontext())
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocated an output")))
    context = backend.prepare_context(key, value, is_causal=False)
    prepared_query = backend.prepare_query(query, 0.125, global_row_offset=global_row_offset)
    assert context.key is expected_context.key
    assert context.value is expected_context.value
    assert context.key_length == 97
    assert prepared_query.data is expected_query.data
    assert prepared_query.scale is expected_query.scale
    assert prepared_query.shape == expected_query.shape
    assert prepared_query.dtype is expected_query.dtype
    assert prepared_query.global_row_offset == global_row_offset
    means.assert_called_once_with(key, value, is_causal=False)
    key_quantization.assert_called_once_with(key, key_mean, grouped=True, storage_key_length=128)
    value_quantization.assert_called_once_with(
        value,
        expected_context.value_mean,
        is_causal=False,
        storage_length=128,
    )
    query_quantization.assert_called_once_with(query, 0.125, grouped=True, storage_query_length=128)


def test_benchmark_wrapper_allocates_output_during_preparation_and_reuses_it(monkeypatch):
    context, query = _prepared()
    with torch.device("meta"):
        input_query = torch.empty(query.shape, dtype=query.dtype)
        key = torch.empty((2, 2, 97, 64), dtype=query.dtype)
    prepare_context = Mock(return_value=context)
    prepare_query = Mock(return_value=query)
    launch_into = Mock(side_effect=lambda _context, _query, output: output)
    monkeypatch.setattr(backend, "prepare_context", prepare_context)
    monkeypatch.setattr(backend, "prepare_query", prepare_query)
    monkeypatch.setattr(backend, "launch_attention_into", launch_into)
    prepared = backend.prepare_attention(input_query, key, key, 0.125, False)
    assert prepared.context is context
    assert prepared.query is query
    assert prepared.output.shape == query.shape
    assert prepared.output.dtype is query.dtype
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocated during launch")))
    assert backend.launch_attention(prepared) is prepared.output
    assert backend.launch_attention(prepared) is prepared.output
    assert launch_into.call_count == 2
    prepare_context.assert_called_once_with(key, key, is_causal=False)
    prepare_query.assert_called_once_with(input_query, 0.125)


@pytest.mark.parametrize("layout", ["bhsd", "bshd", "padded_bshd"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(("origin", "start", "rows"), [(0, 64, 65), (64, 64, 1), (0, 128, None)])
def test_query_windows_pass_local_and_global_offsets_and_output_strides(
    monkeypatch,
    layout,
    causal,
    origin,
    start,
    rows,
):
    context, query = _prepared(
        query_length=193, key_length=257, global_row_offset=origin, causal=causal
    )
    resolved_rows = 193 - start if rows is None else rows
    with torch.device("meta"):
        if layout == "bhsd":
            output = torch.empty((2, 6, resolved_rows, 64), dtype=query.dtype)
        else:
            capacity = resolved_rows + (17 if layout == "padded_bshd" else 0)
            output = torch.empty((2, capacity, 6, 64), dtype=query.dtype)[
                :, :resolved_rows
            ].transpose(1, 2)
    kernel = _RecordingKernel()
    monkeypatch.setattr(backend, "_dense_piper_kernel", kernel)
    monkeypatch.setattr(backend, "device_context", lambda _device: nullcontext())
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocated during launch")))
    assert (
        backend.launch_attention_into(context, query, output, query_start=start, query_rows=rows)
        is output
    )
    ((grid, args, _kwargs),) = kernel.calls
    assert grid == ((resolved_rows + 63) // 64, 6, 2)
    assert args[8] is output
    assert args[9:12] == (start, resolved_rows, origin + start)
    assert args[16:19] == output.stride()[:3]


@pytest.mark.parametrize(
    ("origin", "error"),
    [(-64, ValueError), (1, ValueError), (63, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_query_preparation_rejects_invalid_origins_before_device_entry(monkeypatch, origin, error):
    query = torch.empty((2, 6, 65, 64), dtype=torch.bfloat16, device="meta")
    guard = Mock(side_effect=AssertionError("invalid origin entered a device context"))
    monkeypatch.setattr(backend, "device_context", guard)
    with pytest.raises(error, match="global_row_offset"):
        backend.prepare_query(query, 0.125, global_row_offset=origin)
    guard.assert_not_called()


@pytest.mark.parametrize(
    ("origin", "start", "rows", "causal"),
    [
        (0, 1, 1, False),
        (0, -64, 1, False),
        (0, 0, 0, False),
        (0, 64, 2, False),
        (64, 0, 65, True),
        (1, 0, 1, False),
    ],
)
def test_launch_rejects_invalid_windows_before_device_entry(
    monkeypatch, origin, start, rows, causal
):
    context, query = _prepared(global_row_offset=origin, causal=causal)
    output = torch.empty(query.shape, dtype=query.dtype, device="meta")
    guard = Mock(side_effect=AssertionError("invalid window entered a device context"))
    monkeypatch.setattr(backend, "device_context", guard)
    with pytest.raises(ValueError, match=r"query|global_row_offset|causal"):
        backend.launch_attention_into(context, query, output, query_start=start, query_rows=rows)
    guard.assert_not_called()
