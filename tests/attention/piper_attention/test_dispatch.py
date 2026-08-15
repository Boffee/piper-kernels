"""Public API and validation tests for Piper Attention."""

import pytest
import torch

import piper_kernels
from piper_kernels import piper_attention
from piper_kernels.attention.piper_attention import triton as piper_attention_backend


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    return query, torch.randn_like(query), torch.randn_like(query)


def test_package_root_exports_piper_attention() -> None:
    assert piper_kernels.piper_attention is piper_attention
    assert "piper_attention" in piper_kernels.__all__


def test_public_api_uses_portable_reference_on_cpu() -> None:
    torch.manual_seed(50)
    query = torch.randn(1, 1, 9, 64, dtype=torch.float16)
    key = torch.randn(1, 1, 11, 64, dtype=torch.float16)
    value = torch.randn_like(key)

    with torch.no_grad():
        output = piper_attention(query, key, value, scale=0.2)

    assert output.shape == query.shape
    assert output.dtype is query.dtype
    assert torch.isfinite(output).all()


def test_native_mixed_int8_hook_uses_query_device_before_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    guarded_devices: list[torch.device] = []

    class PreprocessingReachedError(RuntimeError):
        pass

    class DeviceGuard:
        def __init__(self, device: torch.device) -> None:
            guarded_devices.append(device)

        def __enter__(self) -> None:
            events.append("device-enter")

        def __exit__(self, *_args: object) -> None:
            events.append("device-exit")

    def record_hook() -> None:
        events.append("hook")

    def stop_at_preprocessing(*_args: object, **_kwargs: object) -> None:
        events.append("preprocessing")
        raise PreprocessingReachedError

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(torch.cuda, "device", DeviceGuard)
    monkeypatch.setattr(
        piper_attention_backend,
        "install_uint8_int8_dot_hook",
        record_hook,
    )
    monkeypatch.setattr(
        piper_attention_backend,
        "_compute_kv_means",
        stop_at_preprocessing,
    )
    query, key, value = _inputs()
    plan = piper_attention_backend._default_piper_attention_execution_plan(
        query,
        True,
    )

    with pytest.raises(PreprocessingReachedError):
        piper_attention_backend._prepare_piper_attention(
            query,
            key,
            value,
            64**-0.5,
            True,
            execution_plan=plan,
        )

    assert guarded_devices == [query.device]
    assert events == ["device-enter", "hook", "device-exit", "preprocessing"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_rejects_unsupported_dtype(dtype: torch.dtype) -> None:
    query = torch.zeros((1, 1, 8, 64), dtype=dtype)

    with pytest.raises(ValueError, match="float16 or bfloat16"):
        piper_attention(query, query, query)


def test_rejects_mismatched_dtypes() -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="share a dtype"):
        piper_attention(query, key.to(torch.bfloat16), value)


def test_rejects_unsupported_head_dimension() -> None:
    query = torch.randn(1, 1, 8, 32, dtype=torch.float16)

    with pytest.raises(ValueError, match="head dimensions 64 and 128"):
        piper_attention(query, query, query)


def test_rejects_mismatched_key_value_lengths() -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="key/value lengths"):
        piper_attention(query, key, value[:, :, :-1])


def test_rejects_rectangular_causal_inputs() -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="equal query and key lengths"):
        piper_attention(query[:, :, :-1], key, value, is_causal=True)


@pytest.mark.parametrize(("query_length", "key_length"), [(0, 8), (8, 0)])
def test_rejects_empty_sequences(query_length: int, key_length: int) -> None:
    query = torch.empty((1, 1, query_length, 64), dtype=torch.float16)
    key = torch.empty((1, 1, key_length, 64), dtype=torch.float16)

    with pytest.raises(ValueError, match="does not accept empty"):
        piper_attention(query, key, key)


def test_rejects_noncontiguous_head_dimension() -> None:
    storage = torch.randn(1, 1, 8, 128, dtype=torch.float16)
    query = storage[..., ::2]

    with pytest.raises(ValueError, match="head dimension must be contiguous"):
        piper_attention(query, query, query)


def test_rejects_autograd() -> None:
    query, key, value = _inputs()
    query.requires_grad_(True)

    with pytest.raises(RuntimeError, match="inference-only"):
        piper_attention(query, key, value)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_scale(scale: float) -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="finite and positive"):
        piper_attention(query, key, value, scale=scale)
