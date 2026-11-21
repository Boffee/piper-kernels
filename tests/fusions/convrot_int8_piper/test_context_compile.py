"""Compile the actual K/V producer launch configurations for RDNA4."""

import re
import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
import triton
from _accelerators import rocm_fences_only_lds
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import _backend, _kernels, key, value
from piper_kernels.fusions.convrot_int8_piper import triton as projection
from piper_kernels.fusions.convrot_int8_projection import triton as shared_projection

from .test_query import _compile_rdna4_launch, _operands


@pytest.mark.skipif(sys.platform != "linux", reason="offline ROCm compilation requires Linux")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("operation", ["key", "value"])
def test_context_launches_compile_for_rdna4(monkeypatch, head_dim, operation):
    function = (
        _kernels._project_key_kernel if operation == "key" else _kernels._project_value_kernel
    )
    kernel = MagicMock()
    monkeypatch.setattr(_kernels, function.__name__, kernel)
    monkeypatch.setattr(projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(shared_projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda _: AcceleratorTarget("hip", "gfx1201")
    )
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
        compiled_chunk = _compile_rdna4_launch(chunk_function, chunk_call, "gfx1201")
        assert compiled_chunk.metadata.shared <= 65536
        assert "arith.truncf" not in compiled_chunk.asm["ttgir"]
        mean_call = project_mean_kernel.__getitem__.return_value.call_args
        compiled_mean = _compile_rdna4_launch(mean_function, mean_call, "gfx1201")
        assert compiled_mean.metadata.shared <= 65536
        assert "arith.truncf" not in compiled_mean.asm["ttgir"]
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    for call in calls:
        compiled = _compile_rdna4_launch(function, call, "gfx1201")
        assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]
        assert compiled.metadata.shared <= 65536
        assert ("arith.truncf" in compiled.asm["ttgir"]) is (operation == "key")
        if rocm_fences_only_lds():
            assert "global_inv" not in compiled.asm["amdgcn"]


def test_amd_dense_policy_and_gluon_eligibility_contracts():
    amd_gluon = _backend._amd_projection
    with FakeTensorMode():
        q_in = torch.empty((1, 256, 256), device="cuda:1", dtype=torch.int8)
        w_d64_even = torch.empty((4 * 64, 256), device="cuda:1", dtype=torch.int8)
        w_d64_odd = torch.empty((3 * 64, 256), device="cuda:1", dtype=torch.int8)
        w_d128 = torch.empty((3 * 128, 256), device="cuda:1", dtype=torch.int8)

        d64_plan = amd_gluon.default_execution_plan(
            q_in, w_d64_even, operation="query", head_dim=64, rotary_dim=48
        )
        assert d64_plan.kernel == "gluon"
        assert d64_plan.heads_per_program * 64 == 128

        d128_plan = amd_gluon.default_execution_plan(
            q_in, w_d128, operation="key", head_dim=128, rotary_dim=96
        )
        assert d128_plan.kernel == "gluon"
        assert d128_plan.heads_per_program * 128 == 128

        assert (
            amd_gluon.default_execution_plan(
                q_in, w_d128, operation="query", head_dim=128, rotary_dim=96, per_thread_scales=True
            ).kernel
            == "triton"
        )
        assert (
            amd_gluon.default_execution_plan(
                q_in, w_d128, operation="key", head_dim=128, rotary_dim=96, per_thread_scales=True
            ).kernel
            == "gluon"
        )
        assert (
            amd_gluon.default_execution_plan(q_in, w_d64_odd, operation="value", head_dim=64).kernel
            == "triton"
        )
        assert (
            amd_gluon.default_execution_plan(
                q_in[:, ::2], w_d64_even, operation="query", head_dim=64, rotary_dim=48
            ).kernel
            == "triton"
        )
        assert (
            amd_gluon.default_execution_plan(
                q_in, w_d64_even[:, ::2], operation="value", head_dim=64
            ).kernel
            == "triton"
        )
        # Signed 32-bit byte-offset check for BF16 key storage: 8_388_608 * 128 * 2 == 2**31,
        # while FP32 rotary indexing (8_388_608 * 32 + 64) * 4 stays under 2**31 for query.
        q_large = torch.empty((1, 8_388_608, 64), device="cuda:1", dtype=torch.int8)
        w_small = torch.empty((2 * 64, 64), device="cuda:1", dtype=torch.int8)
        assert (
            amd_gluon.default_execution_plan(
                q_large,
                w_small,
                operation="query",
                head_dim=64,
                rotary_dim=32,
                storage_sequence_length=8_388_608,
            ).kernel
            == "gluon"
        )
        assert (
            amd_gluon.default_execution_plan(
                q_large,
                w_small,
                operation="key",
                head_dim=64,
                rotary_dim=32,
                storage_sequence_length=8_388_608,
            ).kernel
            == "triton"
        )


_AMD_GLUON_POINTER_TYPES = {
    "input_ptr": "*i8",
    "weight_ptr": "*i8",
    "query_ptr": "*i8",
    "value_ptr": "*i8",
    "stored_ptr": "*bf16",
    "norm_weight_ptr": "*bf16",
    "bias_ptr": "*bf16",
}

_AMD_GLUON_COMPILE_CASES = [
    ("query", "full_tiles", True, False, False, False),
    ("query", "ragged_tail", True, False, True, False),
    ("query", "bias_without_norm", False, True, True, False),
    ("query", "full_rotary", True, True, False, True),
    ("key", "full_tiles", True, False, False, False),
    ("key", "ragged_tail", True, False, True, False),
    ("key", "bias_without_norm", False, True, True, False),
    ("key", "full_rotary", True, True, False, True),
    ("value", "full_tiles", False, False, False, False),
    ("value", "ragged_tail", False, False, True, False),
    ("value", "bias_tail", False, True, True, False),
]


def _compile_amd_gluon_dense_kernel(
    architecture: str,
    operation: str,
    *,
    head_dim: int,
    rotary_dim: int,
    affine: bool,
    with_bias: bool,
    mask_rows: bool,
):
    amd_policy = _backend.amd_policy
    amd_gluon = _backend._amd_projection
    function = getattr(amd_gluon, f"_{operation}_kernel")
    input_features = 2048 if head_dim == 64 else 5376
    heads = 32 if head_dim == 64 else 56
    constants = {
        "input_features": input_features,
        "heads": heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "norm_epsilon": 1e-6,
        "softmax_scale": head_dim**-0.5,
        "mask_rows": mask_rows,
        "group_m": amd_policy.GLUON_GROUP_M,
    }
    constants = {name: item for name, item in constants.items() if name in function.arg_names}
    signature, attributes = {}, {}
    for index, name in enumerate(function.arg_names):
        if name in constants:
            continue
        if (name == "bias_ptr" and not with_bias) or (name == "norm_weight_ptr" and not affine):
            constants[name], signature[name] = None, "constexpr"
        elif name.endswith("_ptr"):
            signature[name] = _AMD_GLUON_POINTER_TYPES.get(name, "*fp32")
            attributes[(index,)] = [["tt.divisibility", 16]]
        else:
            signature[name] = "i32"
    extra = amd_gluon._kernel_options(operation)
    compiled = triton.compile(
        GluonASTSource(function, signature, constexprs=constants, attrs=attributes),
        target=GPUTarget("hip", architecture, 32),
        options={"num_warps": amd_policy.GLUON_NUM_WARPS, **extra},
    )
    amdgcn = compiled.asm["amdgcn"]
    assert "v_wmma_i32_16x16x16_iu8" in amdgcn
    if operation == "query" or (operation == "key" and head_dim == 64 and rotary_dim == 48):
        assert "v_permlanex16_b32" in amdgcn
    assert "buffer_load_b64" in amdgcn
    expected_store = "buffer_store_b128" if operation == "key" else "buffer_store_b64"
    assert expected_store in amdgcn
    assert compiled.metadata.shared <= 16384
    spill_match = re.search(r"\.vgpr_spill_count:\s*(\d+)", amdgcn)
    assert spill_match is not None
    assert int(spill_match.group(1)) == 0
    if rocm_fences_only_lds():
        assert "global_inv" not in amdgcn
    truncations = [line for line in compiled.asm["ttgir"].splitlines() if "arith.truncf" in line]
    assert bool(truncations) is (operation == "key")
    assert all("f32" in line and "bf16" in line for line in truncations)
    return compiled


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    ("operation", "variant", "affine", "with_bias", "mask_rows", "full_rotary"),
    _AMD_GLUON_COMPILE_CASES,
)
def test_amd_gluon_dense_kernels_compile_offline_without_spills(
    head_dim,
    operation,
    variant,
    affine,
    with_bias,
    mask_rows,
    full_rotary,
):
    del variant
    if sys.platform != "linux":
        pytest.skip("ROCm support is Linux-only")
    fallback_rotary = 48 if head_dim == 64 else 96
    rotary_dim = 0 if operation == "value" else (head_dim if full_rotary else fallback_rotary)
    _compile_amd_gluon_dense_kernel(
        "gfx1201",
        operation,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        affine=affine,
        with_bias=with_bias,
        mask_rows=mask_rows,
    )
