"""Compatibility exports for the former combined ConvRot INT8 implementation.

Internal consumers import the owning modules directly. Legacy names, including
private helper names, are preserved only at this boundary.
"""

from piper_kernels.linear.convrot import triton as convrot_backend

from . import _policy
from ._generic import triton as generic_backend
from ._kernels import triton as kernels
from ._nvidia import triton as nvidia_backend
from ._ops import (
    add_,
    addmm_,
    dequantized_input_mean,
    linear,
    linear_prepared,
    prepare_input,
)

_convert_gguf_out = generic_backend.convert_gguf_out
_requantize_update_ = generic_backend._requantize_update_
_int8_matmul_kernel = kernels.int8_matmul_kernel
_load_weight_chunk = kernels._load_weight_chunk
_normalize_for_int8 = kernels.normalize_for_int8
_quantize_int8 = kernels._quantize_int8
_requantize_update_rows_kernel = kernels.requantize_update_rows_kernel
_store_quantized_chunk = kernels._store_quantized_chunk
quantize_rows_kernel = kernels.quantize_rows_kernel
rotate_quantize_rows_kernel = kernels.rotate_quantize_rows_kernel
scaled_int8_matmul = kernels.scaled_int8_matmul

_dequantized_input_mean_partial_kernel = nvidia_backend._dequantized_input_mean_partial_kernel
_dequantized_input_mean_reduce_kernel = nvidia_backend._dequantized_input_mean_reduce_kernel
_execute_prepared_linear = nvidia_backend.execute_prepared_linear
_prepare_input = nvidia_backend.prepare_input_with_plan
_prepare_input_with_production_plan = nvidia_backend._prepare_input_with_production_plan
default_execution_plan = nvidia_backend.default_execution_plan
fused_rotate_quantize_input = nvidia_backend.fused_rotate_quantize_input
quantize_input = nvidia_backend.quantize_input
run_linear = nvidia_backend.run_linear

dtype_code = convrot_backend.logical_dtype_code
rotate_groups_kernel = convrot_backend.rotate_groups_kernel
rotate_input = convrot_backend.rotate_input

__all__ = [
    "_convert_gguf_out",
    "_dequantized_input_mean_partial_kernel",
    "_dequantized_input_mean_reduce_kernel",
    "_execute_prepared_linear",
    "_int8_matmul_kernel",
    "_load_weight_chunk",
    "_normalize_for_int8",
    "_policy",
    "_prepare_input",
    "_prepare_input_with_production_plan",
    "_quantize_int8",
    "_requantize_update_",
    "_requantize_update_rows_kernel",
    "_store_quantized_chunk",
    "add_",
    "addmm_",
    "default_execution_plan",
    "dequantized_input_mean",
    "dtype_code",
    "fused_rotate_quantize_input",
    "linear",
    "linear_prepared",
    "prepare_input",
    "quantize_input",
    "quantize_rows_kernel",
    "rotate_groups_kernel",
    "rotate_input",
    "rotate_quantize_rows_kernel",
    "run_linear",
    "scaled_int8_matmul",
]
