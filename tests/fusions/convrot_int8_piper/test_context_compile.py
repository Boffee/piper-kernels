"""Compile the actual K/V producer launch configurations for RDNA4."""

import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import _backend, _kernels, key, value
from piper_kernels.fusions.convrot_int8_piper import triton as projection
from piper_kernels.fusions.convrot_int8_sage_qk import _key_kernels as key_kernels
from piper_kernels.fusions.convrot_int8_sage_qk import key as key_projection

from .test_query import _compile_rdna4_launch, _operands


@pytest.mark.skipif(sys.platform != "linux", reason="offline ROCm compilation requires Linux")
@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("operation", ["key", "value_causal", "value_centered"])
def test_context_launches_compile_for_rdna4(monkeypatch, arch, head_dim, operation):
    function = (
        key_kernels._project_key_kernel if operation == "key" else _kernels._project_value_kernel
    )
    kernel = MagicMock()
    monkeypatch.setattr(key_kernels if operation == "key" else _kernels, function.__name__, kernel)
    monkeypatch.setattr(projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(key_projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda _: AcceleratorTarget("hip", arch))
    operands = _operands("meta", sequence=193, head_dim=head_dim)
    bias = torch.empty(3 * head_dim, device="meta", dtype=torch.bfloat16)
    if operation == "key":
        mean_function = key_projection.centered_projection._kernels._mean_finalize_kernel
        mean_kernel = MagicMock()
        monkeypatch.setattr(
            key_projection.centered_projection._kernels, "_mean_finalize_kernel", mean_kernel
        )
        monkeypatch.setattr(
            key_projection.qk_quantization,
            "prepare_key",
            lambda *args, **kwargs: key._new_outputs(
                operands[0], (2, 193, 3, head_dim), "per_warp"
            ),
        )
        backend = _backend.select_projection_backend(operands[0], head_dim=head_dim)
        backend.project_key(
            *operands,
            1e-6,
            bias,
            out=key._new_outputs(operands[0], (2, 193, 3, head_dim), "per_warp"),
        )
        mean_kernel.__getitem__.return_value.assert_called_once()
        mean_call = mean_kernel.__getitem__.return_value.call_args
        compiled_mean = _compile_rdna4_launch(mean_function, mean_call, arch)
        assert compiled_mean.metadata.shared <= 65536
        assert "arith.truncf" not in compiled_mean.asm["ttgir"]
    else:
        monkeypatch.setattr(projection, "project_prepared_input_mean_kernel", MagicMock())
        backend = _backend.select_projection_backend(operands[0], head_dim=head_dim)
        backend.project_value(
            *operands[:4],
            bias,
            is_causal=operation == "value_causal",
            out=value._new_outputs(operands[0], (2, 193, 3, head_dim)),
        )
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    for call in calls:
        compiled = _compile_rdna4_launch(function, call, arch)
        assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]
        assert compiled.metadata.shared <= 65536
        assert ("arith.truncf" in compiled.asm["ttgir"]) is (operation == "key")
