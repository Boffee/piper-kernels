"""Public API and validation tests for Piper Attention."""

import pytest
import torch

from piper_kernels.attention.piper_attention import _quantization as piper_quantization
from piper_kernels.attention.piper_attention._nvidia import triton as piper_attention_backend


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    return query, torch.randn_like(query), torch.randn_like(query)


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
    monkeypatch.setattr(piper_attention_backend, "device_context", DeviceGuard)
    monkeypatch.setattr(
        piper_attention_backend,
        "install_uint8_int8_dot_hook",
        record_hook,
    )
    monkeypatch.setattr(
        piper_quantization,
        "compute_kv_means",
        stop_at_preprocessing,
    )
    query, key, value = _inputs()
    plan = piper_attention_backend.default_execution_plan(
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
    assert events == ["device-enter", "hook", "preprocessing", "device-exit"]
