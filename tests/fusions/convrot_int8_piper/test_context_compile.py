"""Compile the actual K/V producer launch configurations for RDNA4."""

import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from _accelerators import rocm_fences_only_lds

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import _backend, _kernels, key, value
from piper_kernels.fusions.convrot_int8_piper import triton as projection
from piper_kernels.fusions.convrot_int8_projection import triton as shared_projection

from .test_query import _compile_rdna4_launch, _operands


@pytest.mark.skipif(sys.platform != "linux", reason="offline ROCm compilation requires Linux")
@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("operation", ["key", "value"])
def test_context_launches_compile_for_rdna4(monkeypatch, arch, head_dim, operation):
    function = (
        _kernels._project_key_kernel if operation == "key" else _kernels._project_value_kernel
    )
    kernel = MagicMock()
    monkeypatch.setattr(_kernels, function.__name__, kernel)
    monkeypatch.setattr(projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(shared_projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda _: AcceleratorTarget("hip", arch))
    operands = _operands("meta", sequence=193, head_dim=head_dim)
    bias = torch.empty(3 * head_dim, device="meta", dtype=torch.bfloat16)
    if operation == "key":
        monkeypatch.setattr(
            projection._quantization,
            "prepare_chunk_centered_key",
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
    else:
        chunk_function = shared_projection._chunk_dequantized_input_mean_kernel
        mean_function = shared_projection.project_prepared_chunk_means_kernel
        chunk_mean_kernel = MagicMock()
        project_mean_kernel = MagicMock()
        monkeypatch.setattr(
            shared_projection, "_chunk_dequantized_input_mean_kernel", chunk_mean_kernel
        )
        monkeypatch.setattr(
            shared_projection, "project_prepared_chunk_means_kernel", project_mean_kernel
        )
        backend = _backend.select_projection_backend(operands[0], head_dim=head_dim)
        backend.project_value(
            *operands[:4],
            bias,
            is_causal=False,
            out=value._new_outputs(operands[0], (2, 193, 3, head_dim)),
        )
        chunk_call = chunk_mean_kernel.__getitem__.return_value.call_args
        compiled_chunk = _compile_rdna4_launch(chunk_function, chunk_call, arch)
        assert compiled_chunk.metadata.shared <= 65536
        assert "arith.truncf" not in compiled_chunk.asm["ttgir"]
        mean_call = project_mean_kernel.__getitem__.return_value.call_args
        compiled_mean = _compile_rdna4_launch(mean_function, mean_call, arch)
        assert compiled_mean.metadata.shared <= 65536
        assert "arith.truncf" not in compiled_mean.asm["ttgir"]
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    for call in calls:
        compiled = _compile_rdna4_launch(function, call, arch)
        assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]
        assert compiled.metadata.shared <= 65536
        assert ("arith.truncf" in compiled.asm["ttgir"]) is (operation == "key")
        if rocm_fences_only_lds():
            assert "global_inv" not in compiled.asm["amdgcn"]
