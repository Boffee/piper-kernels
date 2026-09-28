"""Shared RMSNorm/RoPE K producer, centered K64 encoding, and optional routing."""

# Triton's launch options and constexpr arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

import torch
import triton

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_centered_projection import triton as centered_projection
from piper_kernels.fusions.convrot_int8_projection import _plan as projection_plan
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from . import _key_kernels


def source_files() -> tuple[str, ...]:
    """Track the shared K producer, mean reduction, and quantization sources."""
    return tuple(
        path
        for path in (
            __file__,
            projection_plan.__file__,
            _key_kernels.__file__,
            *centered_projection.source_files(),
            qk_quantization.__file__,
        )
        if path is not None
    )


def project_key(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    bias: torch.Tensor | None = None,
    *,
    execution_plan: ProjectionExecutionPlan,
    out: tuple[torch.Tensor, torch.Tensor],
    routing_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    routing_mode: int | None = None,
    block_lengths: torch.Tensor | None = None,
) -> None:
    """Project once, reduce the represented K mean, then encode centered K64."""
    batch, sequence_length, _ = input_qdata.shape
    if batch == 0:
        return
    _, heads, storage_length, head_dim = out[0].shape
    transformed, partials, mean = centered_projection.allocate_workspace(
        input_qdata, (batch, heads, storage_length, head_dim), tile_rows=64
    )
    summary, auxiliary = (None, None) if routing_out is None else routing_out
    with device_context(input_qdata.device):
        for count, offset, aligned_rows in (
            (sequence_length // execution_plan.block_m, 0, True),
            (
                int(sequence_length % execution_plan.block_m != 0),
                sequence_length // execution_plan.block_m,
                False,
            ),
        ):
            if not count:
                continue
            _key_kernels._project_key_kernel[
                (count, triton.cdiv(heads, execution_plan.heads_per_program), batch)
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                transformed,
                partials,
                summary,
                auxiliary,
                block_lengths,
                batch * sequence_length,
                sequence_length,
                storage_length,
                offset,
                input_features=input_qdata.shape[2],
                heads=heads,
                heads_per_program=execution_plan.heads_per_program,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=block_lengths is not None,
                aligned_projection=(
                    aligned_rows
                    and input_qdata.shape[2] % execution_plan.block_k == 0
                    and heads % execution_plan.heads_per_program == 0
                ),
                mask_ragged_tail=not aligned_rows,
                block_m=execution_plan.block_m,
                block_n=head_dim * execution_plan.heads_per_program,
                block_k=execution_plan.block_k,
                round_rsqrt_to_nearest=execution_plan.round_rsqrt_to_nearest,
                group_m=execution_plan.group_m,
                bias_ptr=bias,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
            )
        # Internal sparse padding represents zero K rows and remains in the
        # logical denominator, matching ordinary sparse Piper's global mean.
        centered_projection.finalize_mean(partials, sequence_length, out=mean)
        qk_quantization.prepare_key(
            transformed.narrow(2, 0, sequence_length),
            mean,
            grouped=True,
            storage_key_length=storage_length,
            out=out,
        )
