"""ConvRot INT8 custom ops and NVIDIA kernel utilities.

Custom ops dispatch through the shared implementation boundary. Kernel-level
utilities remain available here for existing fusion and benchmark consumers.
"""

from . import _policy
from ._nvidia.triton import (
    _convert_gguf_out,
    _dequantized_input_mean_partial_kernel,
    _dequantized_input_mean_reduce_kernel,
    _execute_prepared_linear,
    _int8_matmul_kernel,
    _load_weight_chunk,
    _normalize_for_int8,
    _prepare_input,
    _prepare_input_with_production_plan,
    _quantize_int8,
    _requantize_update_,
    _requantize_update_rows_kernel,
    _store_quantized_chunk,
    default_execution_plan,
    dtype_code,
    fused_rotate_quantize_input,
    quantize_input,
    quantize_rows_kernel,
    rotate_groups_kernel,
    rotate_input,
    rotate_quantize_rows_kernel,
    run_linear,
    scaled_int8_matmul,
)
from ._ops import (
    add_,
    addmm_,
    dequantized_input_mean,
    linear,
    linear_prepared,
    prepare_input,
)

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
