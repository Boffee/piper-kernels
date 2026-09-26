"""K/V fusion validation uses metadata, including fake and empty-batch execution."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _quantized_dispatch as dispatch
from piper_kernels.fusions.convrot_int8_piper import _backend, key, value


class _MetadataOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"validation performed a tensor operation: {func}")


class _OutputOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func is torch.ops.aten.new_empty.default
        return func(*args, **(kwargs or {}))


def _inputs(*, batch=2, device="cpu"):
    x = torch.empty(batch, 65, 256, device=device, dtype=torch.int8)
    xs = torch.empty(batch, 65, device=device)
    w = torch.empty(128, 256, device=device, dtype=torch.int8)
    ws = torch.empty(128, 1, device=device)
    return [x, xs, w, ws]


def _native_inputs(*, batch=2):
    q = torch.empty(batch, 4, 128, 64, dtype=torch.int8)
    qs = torch.empty(batch, 4, 4)
    k = torch.empty(batch, 2, 128, 64, dtype=torch.int8)
    ks = torch.empty(batch, 2, 2)
    v = torch.empty(batch, 2, 64, 128, dtype=torch.int8)
    mult, logs, mean = (
        torch.empty(batch, 2, 128),
        torch.empty(batch, 2, 128),
        torch.empty(batch, 2, 64),
    )
    return [q, qs, k, ks, v, mult, logs, mean, 65, 97, False, torch.bfloat16]


@pytest.mark.parametrize(
    "invalid",
    [
        "input_dtype",
        "scale_shape",
        "weight_shape",
        "weight_dtype",
        "weight_scale",
        "bias",
        "stride",
        "device",
        "heads",
        "causal",
        "grad",
    ],
)
def test_value_rejects_bad_metadata_without_tensor_operations(invalid):
    operands = _inputs()
    bias, head_dim, causal = None, 64, False
    if invalid == "input_dtype":
        operands[0] = operands[0].float()
    elif invalid == "scale_shape":
        operands[1] = torch.empty(2, 64)
    elif invalid == "weight_shape":
        operands[2] = torch.empty(128, 255, dtype=torch.int8)
    elif invalid == "weight_dtype":
        operands[2] = operands[2].float()
    elif invalid == "weight_scale":
        operands[3] = operands[3].half()
    elif invalid == "bias":
        bias = torch.empty(64)
    elif invalid == "stride":
        operands[0] = torch.empty(2, 65, 512, dtype=torch.int8)[..., ::2]
    elif invalid == "device":
        operands[1] = operands[1].to("meta")
    elif invalid == "heads":
        head_dim = 32
    elif invalid == "causal":
        causal = 1
    elif invalid == "grad":
        operands[3].requires_grad_()
    with _MetadataOnly(), pytest.raises((ValueError, TypeError, RuntimeError)):
        value._validate_inputs(*operands, bias, head_dim, causal)


@pytest.mark.parametrize(
    "invalid",
    [
        "key_dtype",
        "key_shape",
        "scale_shape",
        "value_shape",
        "mult_dtype",
        "log_shape",
        "mean_shape",
        "stride",
        "device",
        "length",
        "causal",
        "output_dtype",
        "grad",
    ],
)
def test_quantized_context_rejects_bad_metadata_without_tensor_operations(invalid):  # noqa: PLR0912
    operands = _native_inputs()
    if invalid == "key_dtype":
        operands[2] = operands[2].float()
    elif invalid == "key_shape":
        operands[2] = torch.empty(2, 3, 128, 64, dtype=torch.int8)
    elif invalid == "scale_shape":
        operands[3] = torch.empty(2, 2, 3)
    elif invalid == "value_shape":
        operands[4] = operands[4].transpose(2, 3).contiguous()
    elif invalid == "mult_dtype":
        operands[5] = operands[5].half()
    elif invalid == "log_shape":
        operands[6] = torch.empty(1)
    elif invalid == "mean_shape":
        operands[7] = torch.empty(1)
    elif invalid == "stride":
        operands[5] = torch.empty(2, 2, 256)[..., ::2]
    elif invalid == "device":
        operands[3] = operands[3].to("meta")
    elif invalid == "length":
        operands[9] = 0
    elif invalid == "causal":
        operands[10] = True
    elif invalid == "output_dtype":
        operands[11] = torch.float32
    elif invalid == "grad":
        operands[5].requires_grad_()
    with _MetadataOnly(), pytest.raises((ValueError, TypeError, RuntimeError)):
        dispatch._validate_quantized(*operands)


@pytest.mark.parametrize("number", [0.0, -1.0, float("nan"), float("inf")])
def test_tensor_contents_remain_caller_preconditions(number):
    operands = _native_inputs()
    for index in (1, 3, 5, 6, 7):
        operands[index].fill_(number)
    with _MetadataOnly():
        assert dispatch._validate_quantized(*operands) == (2, 4, 65, 64)
    projection = _inputs()
    projection[1].fill_(number)
    projection[3].fill_(number)
    with _MetadataOnly():
        assert value._validate_inputs(*projection, None, 64, False) == (2, 65, 2, 64)


@pytest.mark.parametrize("batch", [0, 2])
def test_fake_outputs_only_and_empty_execution_skips_target(monkeypatch, batch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata path probed hardware")

    monkeypatch.setattr(_backend, "select_projection_backend", forbidden)
    monkeypatch.setattr(AcceleratorTarget, "from_device", forbidden)
    operands = _inputs(batch=batch)
    cos = torch.empty(65, 48)
    norm = torch.empty(64, requires_grad=True)
    with torch.no_grad(), _OutputOnly():
        prepared_key = key._project_key_op_fake(*operands, norm, cos, cos, 1e-5, head_dim=64)
        prepared_value = value._project_value_op_fake(*operands, head_dim=64, is_causal=False)
        assert prepared_key[0].shape == (batch, 2, 128, 64)
        assert prepared_value[0].shape == (batch, 2, 64, 128)
    native = _native_inputs(batch=batch)
    with _OutputOnly():
        assert dispatch._piper_attention_from_quantized_fake(*native).shape == (batch, 4, 65, 64)
    if batch == 0:
        with torch.no_grad():
            assert key._project_key_op(*operands, norm, cos, cos, 1e-5, head_dim=64)[0].numel() == 0
            assert value._project_value_op(*operands, head_dim=64, is_causal=False)[0].numel() == 0
            assert dispatch._piper_attention_from_quantized_op(*native).numel() == 0
