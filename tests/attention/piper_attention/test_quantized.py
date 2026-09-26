"""Quantized dense Q keeps metadata validation and native attention semantics."""

from unittest.mock import Mock

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.attention.piper_attention import _quantized_dispatch as dispatch
from piper_kernels.attention.piper_attention._amd import gluon as amd
from piper_kernels.attention.piper_attention._amd import policy as amd_policy
from piper_kernels.attention.piper_attention._nvidia import triton as nvidia


class _NoTensorOperations(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"metadata validation performed {func}")


class _OutputAllocationOnly(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.allocations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func in (
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.empty_like.default,
            torch.ops.aten.new_empty.default,
        ), f"fake execution performed {func}"
        self.allocations.append(func)
        return func(*args, **(kwargs or {}))


def _inputs(
    *,
    device="meta",
    dtype=torch.bfloat16,
    query_length=65,
    key_length=97,
    batch=2,
    heads=6,
    kv_heads=2,
    head_dim=64,
    causal=False,
):
    storage = ((query_length + 63) // 64) * 64
    query = torch.empty((batch, heads, storage, head_dim), device=device, dtype=torch.int8)
    scale = torch.empty((batch, heads, storage // 32), device=device, dtype=torch.float32)
    # Floating K/V retain projection-style outer strides.
    key = torch.empty((batch, key_length, kv_heads, head_dim), device=device, dtype=dtype)
    key = key.transpose(1, 2)
    value = torch.empty_like(key)
    return [query, scale, key, value, query_length, causal]


def _forbid_runtime(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("metadata path entered native execution"))
    monkeypatch.setattr(AcceleratorTarget, "from_device", forbidden)
    monkeypatch.setattr(nvidia, "_prepare_piper_context", forbidden)
    monkeypatch.setattr(amd, "prepare_context", forbidden)
    monkeypatch.setattr(qk_quantization, "prepare_query", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_metadata_validation_accepts_ragged_gqa_without_tensor_operations(
    monkeypatch, device, dtype, causal
):
    operands = _inputs(device=device, dtype=dtype, key_length=65 if causal else 97, causal=causal)
    _forbid_runtime(monkeypatch)
    with _NoTensorOperations():
        shape = dispatch._validate_quantized_query(*operands)
    assert shape == (2, 6, 65, 64)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [0, 2])
def test_fake_impl_only_allocates_logical_output(monkeypatch, device, dtype, batch):
    operands = _inputs(device=device, dtype=dtype, batch=batch)
    _forbid_runtime(monkeypatch)
    with _OutputAllocationOnly() as operations:
        output = dispatch._piper_attention_from_quantized_query_fake(*operands)
    assert len(operations.allocations) == 1
    assert output.shape == (batch, 6, 65, 64)
    assert output.dtype is dtype
    assert output.device == operands[0].device
    assert output.is_contiguous()


def test_registered_fake_boundary_does_not_select_or_prepare_a_backend(monkeypatch):
    with FakeTensorMode():
        operands = _inputs(device="cpu")
        _forbid_runtime(monkeypatch)
        output = dispatch._piper_attention_from_quantized_query_op(*operands)
    assert isinstance(output, FakeTensor)
    assert output.shape == (2, 6, 65, 64)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()


@pytest.mark.parametrize("scale_value", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_contents_are_caller_preconditions(monkeypatch, scale_value):
    operands = _inputs(device="cpu")
    operands[1].fill_(scale_value)
    _forbid_runtime(monkeypatch)
    with _OutputAllocationOnly() as operations:
        output = dispatch._piper_attention_from_quantized_query_fake(*operands)
    assert len(operations.allocations) == 1
    assert output.shape == (2, 6, 65, 64)


@pytest.mark.parametrize(
    "invalid",
    [
        "query_rank",
        "query_dtype",
        "query_stride",
        "query_layout",
        "query_storage_short",
        "query_storage_extra",
        "scale_rank",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "scale_device",
        "key_rank",
        "key_dtype",
        "key_feature_stride",
        "key_layout",
        "value_shape",
        "value_dtype",
        "value_device",
        "batch",
        "head_dim",
        "unsupported_head_dim",
        "head_groups",
        "empty_query_heads",
        "empty_kv_heads",
        "empty_keys",
        "causal_length",
    ],
)
def test_malformed_metadata_rejected_before_device_or_tensor_operations(  # noqa: PLR0912, PLR0915
    monkeypatch, invalid
):
    operands = _inputs()
    query, scale, key, value, *_ = operands
    with torch.device("meta"):
        if invalid == "query_rank":
            operands[0] = query[0]
        elif invalid == "query_dtype":
            operands[0] = query.to(torch.uint8)
        elif invalid == "query_stride":
            operands[0] = torch.empty((2, 128, 6, 64), dtype=torch.int8).transpose(1, 2)
        elif invalid == "query_layout":
            operands[0] = torch.empty(query.shape, dtype=query.dtype, layout=torch.sparse_coo)
        elif invalid == "query_storage_short":
            operands[0] = torch.empty((2, 6, 64, 64), dtype=torch.int8)
        elif invalid == "query_storage_extra":
            operands[0] = torch.empty((2, 6, 192, 64), dtype=torch.int8)
            operands[1] = torch.empty((2, 6, 6), dtype=torch.float32)
        elif invalid == "scale_rank":
            operands[1] = scale[0]
        elif invalid == "scale_dtype":
            operands[1] = scale.to(torch.float16)
        elif invalid == "scale_shape":
            operands[1] = torch.empty((2, 6, 3))
        elif invalid == "scale_stride":
            operands[1] = torch.empty((2, 6, 8))[..., ::2]
        elif invalid == "scale_device":
            operands[1] = torch.empty(scale.shape, device="cpu")
        elif invalid == "key_rank":
            operands[2] = key[0]
        elif invalid == "key_dtype":
            operands[2] = key.to(torch.float32)
            operands[3] = value.to(torch.float32)
        elif invalid == "key_feature_stride":
            operands[2] = torch.empty((2, 2, 97, 128), dtype=key.dtype)[..., ::2]
        elif invalid == "key_layout":
            operands[2] = torch.empty(key.shape, dtype=key.dtype, layout=torch.sparse_coo)
        elif invalid == "value_shape":
            operands[3] = value[:, :, :-1]
        elif invalid == "value_dtype":
            operands[3] = value.to(torch.float16)
        elif invalid == "value_device":
            operands[3] = torch.empty(value.shape, device="cpu", dtype=value.dtype)
        elif invalid == "batch":
            operands[2] = key[:1]
            operands[3] = value[:1]
        elif invalid == "head_dim":
            operands[2] = torch.empty((2, 2, 97, 128), dtype=key.dtype)
            operands[3] = torch.empty_like(operands[2])
        elif invalid == "unsupported_head_dim":
            operands = _inputs(head_dim=32)
        elif invalid == "head_groups":
            operands = _inputs(heads=5)
        elif invalid == "empty_query_heads":
            operands = _inputs(heads=0)
        elif invalid == "empty_kv_heads":
            operands = _inputs(kv_heads=0)
        elif invalid == "empty_keys":
            operands = _inputs(key_length=0)
        elif invalid == "causal_length":
            operands[-1] = True
    _forbid_runtime(monkeypatch)
    with _NoTensorOperations(), pytest.raises((ValueError, TypeError)):
        dispatch._piper_attention_from_quantized_query_fake(*operands)


@pytest.mark.parametrize("query_length", [0, -1, 129, True, 65.0])
def test_invalid_logical_length_rejected_without_tensor_operations(monkeypatch, query_length):
    operands = _inputs()
    operands[-2] = query_length
    _forbid_runtime(monkeypatch)
    with _NoTensorOperations(), pytest.raises((ValueError, TypeError)):
        dispatch._piper_attention_from_quantized_query_fake(*operands)


@pytest.mark.parametrize("operand", [1, 2, 3])
def test_gradient_flags_follow_inference_contract(monkeypatch, operand):
    operands = _inputs()
    operands[operand].requires_grad_(True)
    _forbid_runtime(monkeypatch)
    with _NoTensorOperations(), pytest.raises(RuntimeError, match="inference"):
        dispatch._validate_quantized_query(*operands)
    with torch.no_grad(), _NoTensorOperations():
        assert dispatch._validate_quantized_query(*operands) == (2, 6, 65, 64)


@pytest.mark.parametrize("architecture", ["sm80", "sm89", "sm90", "sm100", "sm120", "sm121"])
def test_native_grouped_q32_target_gate(architecture):
    assert dispatch.supports_quantized_query(AcceleratorTarget("cuda", architecture)) is (
        architecture in ("sm120", "sm121")
    )


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201", "gfx1100", "gfx942"])
def test_native_amd_target_gate(architecture):
    target = AcceleratorTarget("hip", architecture)
    assert dispatch.supports_quantized_query(target) is amd_policy.supports_target(target)


@pytest.mark.parametrize("architecture", ["sm80", "sm89", "sm90", "sm100"])
def test_unsupported_native_target_fails_before_kv_preparation(monkeypatch, architecture):
    operands = _inputs(device="cpu")
    _forbid_runtime(monkeypatch)
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda _device: AcceleratorTarget("cuda", architecture)
    )
    with pytest.raises(RuntimeError):
        dispatch._piper_attention_from_quantized_query_op(*operands)


def test_empty_batch_returns_output_without_a_backend_probe(monkeypatch):
    operands = _inputs(device="cpu", batch=0)
    _forbid_runtime(monkeypatch)
    output = dispatch._piper_attention_from_quantized_query_op(*operands)
    assert output.shape == (0, 6, 65, 64)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()


def _native_available():
    return torch.cuda.is_available() and dispatch.supports_quantized_query(
        AcceleratorTarget.from_device(torch.device("cuda"))
    )


_native_only = pytest.mark.skipif(
    not _native_available(), reason="requires native grouped-Q32 Piper"
)


def _quantized_query(query, scale):
    return qk_quantization.prepare_query(
        query,
        scale,
        grouped=True,
        storage_query_length=((query.shape[2] + 63) // 64) * 64,
    )


@pytest.mark.gpu
@_native_only
@pytest.mark.parametrize("query_length", [1, 31, 63, 64, 65, 127, 128, 193])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_quantized_query_matches_regular_dense_attention(
    query_length, head_dim, causal, dtype
):
    torch.manual_seed(617 + query_length)
    query = torch.randn(2, query_length, 6, head_dim, device="cuda", dtype=dtype).transpose(1, 2)
    key = torch.randn(
        2, query_length if causal else query_length + 17, 2, head_dim, device="cuda", dtype=dtype
    ).transpose(1, 2)
    value = torch.randn_like(key)
    scale = 0.17
    with torch.no_grad():
        query_int8, query_scale = _quantized_query(query, scale)
        actual = dispatch._piper_attention_from_quantized_query_op(
            query_int8, query_scale, key, value, query_length, causal
        )
        expected = piper_attention(query, key, value, scale=scale, is_causal=causal)
    assert actual.shape == query.shape
    assert actual.dtype is dtype
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@_native_only
@pytest.mark.parametrize("query_length", [1023, 1024, 1025])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_causal_d128_quantized_storage_across_descriptor_threshold(query_length, dtype):
    torch.manual_seed(620 + query_length)
    query = torch.randn(2, 6, query_length, 128, device="cuda", dtype=dtype)
    key = torch.randn(2, 2, query_length, 128, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    with torch.no_grad():
        query_int8, query_scale = _quantized_query(query, 128**-0.5)
        actual = dispatch._piper_attention_from_quantized_query_op(
            query_int8, query_scale, key, value, query_length, True
        )
        expected = piper_attention(query, key, value, is_causal=True)
    assert actual.shape == query.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@_native_only
def test_native_query_padding_does_not_contribute_to_real_rows():
    torch.manual_seed(618)
    query = torch.randn(2, 6, 65, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 97, 128, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    query_int8, query_scale = _quantized_query(query, 128**-0.5)
    expected = dispatch._piper_attention_from_quantized_query_op(
        query_int8, query_scale, key, value, 65, False
    )
    query_int8[:, :, 65:].fill_(127)
    query_scale[:, :, 3:].fill_(float("nan"))
    actual = dispatch._piper_attention_from_quantized_query_op(
        query_int8, query_scale, key, value, 65, False
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@_native_only
@pytest.mark.parametrize("causal", [False, True])
def test_quantized_boundary_reuses_one_dynamic_fullgraph(causal):
    torch._dynamo.reset()
    capture = TargetCapturePass()

    def attention(query, scale, key, value, query_length):
        return dispatch._piper_attention_from_quantized_query_op(
            query, scale, key, value, query_length, causal
        )

    compiled = torch.compile(
        attention, fullgraph=True, dynamic=True, options={"post_grad_custom_pre_pass": capture}
    )
    with torch.inference_mode():
        for query_length in (65, 129, 193):
            query = torch.randn(2, 6, query_length, 64, device="cuda", dtype=torch.bfloat16)
            key = torch.randn(
                2,
                2,
                query_length if causal else query_length + 17,
                64,
                device="cuda",
                dtype=query.dtype,
            )
            value = torch.randn_like(key)
            query_int8, query_scale = _quantized_query(query, 0.125)
            actual = compiled(query_int8, query_scale, key, value, query_length)
            expected = piper_attention(query, key, value, scale=0.125, is_causal=causal)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert capture.calls == 1


@pytest.mark.gpu
@_native_only
@pytest.mark.parametrize("causal", [False, True])
def test_quantized_boundary_graph_capture_recomputes_live_kv_and_query(causal):
    torch.manual_seed(619)
    query = torch.randn(2, 6, 65, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 2, 65 if causal else 97, 128, device="cuda", dtype=query.dtype)
    value = torch.randn_like(key)
    query_int8, query_scale = _quantized_query(query, 128**-0.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        return dispatch._piper_attention_from_quantized_query_op(
            query_int8, query_scale, key, value, 65, causal
        )

    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        query_int8.zero_()
        key.mul_(0.5).add_(1)
        value.mul_(0.25).add_(3)
        graph.replay()
        expected = run()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
