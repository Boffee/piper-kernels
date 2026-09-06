"""Legacy names resolve directly to their owners without backend re-export chains."""

import pytest

from piper_kernels.linear.convrot import triton as convrot_backend
from piper_kernels.linear.convrot.int8 import _ops
from piper_kernels.linear.convrot.int8 import triton as legacy
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._generic import triton as generic
from piper_kernels.linear.convrot.int8._kernels import triton as kernels
from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia


@pytest.mark.parametrize(
    ("name", "owner", "canonical_name"),
    [
        ("_convert_gguf_out", generic, "convert_gguf_out"),
        ("_requantize_update_", generic, "_requantize_update_"),
        ("_int8_matmul_kernel", kernels, "int8_matmul_kernel"),
        ("_normalize_for_int8", kernels, "normalize_for_int8"),
        ("_requantize_update_rows_kernel", kernels, "requantize_update_rows_kernel"),
        ("_load_weight_chunk", kernels, "_load_weight_chunk"),
        ("_quantize_int8", kernels, "_quantize_int8"),
        ("_store_quantized_chunk", kernels, "_store_quantized_chunk"),
        ("rotate_quantize_rows_kernel", kernels, "rotate_quantize_rows_kernel"),
        ("quantize_rows_kernel", kernels, "quantize_rows_kernel"),
        ("scaled_int8_matmul", kernels, "scaled_int8_matmul"),
        ("_execute_prepared_linear", nvidia, "execute_prepared_linear"),
        ("_prepare_input", nvidia, "prepare_input_with_plan"),
        ("dtype_code", convrot_backend, "logical_dtype_code"),
        ("rotate_input", convrot_backend, "rotate_input"),
        ("rotate_groups_kernel", convrot_backend, "rotate_groups_kernel"),
        *[(name, _ops, name) for name in _ops.__all__],
    ],
)
def test_legacy_exports_reference_the_canonical_implementation(name, owner, canonical_name):
    assert name in legacy.__all__
    assert getattr(legacy, name) is getattr(owner, canonical_name)


@pytest.mark.parametrize("backend", [amd, nvidia])
def test_accelerator_modules_do_not_reexport_unrelated_operations(backend):
    for name in ("add_", "addmm_", "_convert_gguf_out", "_requantize_update_"):
        assert not hasattr(backend, name)
