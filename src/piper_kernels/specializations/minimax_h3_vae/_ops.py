"""Plan-selecting ConvRot operators for the MiniMax-H3 video VAE."""

from dataclasses import replace

import torch

from piper_kernels.linear.convrot.int8 import triton as convrot_int8_backend


def _execution_plan(
    weight_qdata: torch.Tensor,
    schedule: list[int],
) -> convrot_int8_backend._policy.LinearExecutionPlan:
    if len(schedule) != 5:
        raise ValueError("H3 VAE ConvRot schedule must contain exactly five integers")
    block_m, block_n, block_k, num_warps, num_stages = schedule
    return replace(
        convrot_int8_backend.default_execution_plan(weight_qdata),
        matmul_block_m=block_m,
        matmul_block_n=block_n,
        matmul_block_k=block_k,
        matmul_num_warps=num_warps,
        matmul_num_stages=num_stages,
    )


@torch.library.custom_op(
    "piper_kernels::minimax_h3_vae_convrot_int8_linear_prepared",
    mutates_args=(),
)
def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    schedule: list[int],
) -> torch.Tensor:
    """Apply one H3 VAE weight to an input prepared by ordinary ConvRot."""
    return convrot_int8_backend._execute_prepared_linear(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        logical_dtype,
        _execution_plan(weight_qdata, schedule),
    )


@linear_prepared.register_fake
def _linear_prepared_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    _schedule: list[int],
) -> torch.Tensor:
    return input_qdata.new_empty(
        (*input_qdata.shape[:-1], weight_qdata.shape[0]),
        dtype=logical_dtype,
    )


__all__ = ["linear_prepared"]
