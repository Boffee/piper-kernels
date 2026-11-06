"""Dense Piper attention residual adds that hold one head group's K and V at a time."""

import torch

from . import key, output, value
from .output import DEFAULT_QUERY_CHUNK_ROWS


def validate_head_groups(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    weight_qdata: torch.Tensor,
    group_size: int,
    *,
    head_dim: int,
    head_groups: int,
) -> tuple[int, int]:
    """Return query and key/value rows per group after checking the split from metadata."""
    if head_groups < 1:
        raise ValueError("dense Piper head groups must be positive")
    if key_weight.shape != value_weight.shape:
        raise ValueError("dense Piper head groups require matching K and V projections")
    query_heads, query_remainder = divmod(query_weight.shape[0], head_dim)
    key_heads, key_remainder = divmod(key_weight.shape[0], head_dim)
    if query_remainder or key_remainder:
        raise ValueError("dense Piper head groups require whole-head projections")
    if query_heads % head_groups or key_heads % head_groups:
        raise ValueError(
            f"dense Piper head groups must divide {query_heads} query and {key_heads} key heads"
        )
    query_rows = query_heads // head_groups * head_dim
    if weight_qdata.shape[1] != query_heads * head_dim:
        raise ValueError("dense Piper head groups require an output projection over every head")
    if query_rows % group_size:
        # Each group's output projection rotates its own slice of the attention output.
        raise ValueError(
            f"dense Piper head groups need {query_rows} features per group to be a multiple "
            f"of the output projection's ConvRot group size {group_size}"
        )
    return query_rows, key_heads // head_groups * head_dim


def _rows(tensor: torch.Tensor | None, rows: slice) -> torch.Tensor | None:
    return None if tensor is None else tensor[rows]


@torch.library.custom_op(
    "piper_kernels::convrot_int8_piper_head_grouped_attention_residual_add_",
    mutates_args=("base",),
)
def _head_grouped_attention_residual_add_op(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    query_bias: torch.Tensor | None,
    key_input_qdata: torch.Tensor,
    key_input_scale: torch.Tensor,
    key_weight: torch.Tensor,
    key_weight_scale: torch.Tensor,
    key_norm_weight: torch.Tensor | None,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    key_norm_epsilon: float,
    key_bias: torch.Tensor | None,
    value_input_qdata: torch.Tensor,
    value_input_scale: torch.Tensor,
    value_weight: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    is_causal: bool,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_input_scale: torch.Tensor | None = None,
    *,
    head_dim: int,
    head_groups: int,
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> None:
    """Add dense attention into ``base`` one head group at a time.

    Equivalent to projecting K and V with ``convrot_int8_piper_project_key`` and
    ``convrot_int8_piper_project_value`` and then calling
    ``convrot_int8_piper_projected_query_attention_residual_add_``, except that each
    group projects its own K and V rows and frees them before the next group. Each group
    adds its share of the output projection, over its columns of ``weight_qdata``, into
    ``base``; ``bias`` is added once. Per-token output-activation scales cover each
    group's features separately, and ``base`` rounds once per group.

    ``query_chunk_rows`` caps an ungrouped window's rows. A group's windows may hold
    ``head_groups`` times as many rows, so each attention launch keeps the ungrouped
    rows times heads, and the query and attention-output buffers keep their size; only the
    output-projection scratch, which spans every output feature, grows with the rows.
    """
    query_rows, key_rows = validate_head_groups(
        query_weight,
        key_weight,
        value_weight,
        weight_qdata,
        group_size,
        head_dim=head_dim,
        head_groups=head_groups,
    )
    for group in range(head_groups):
        query = slice(group * query_rows, (group + 1) * query_rows)
        context = slice(group * key_rows, (group + 1) * key_rows)
        group_key = key._project_key_op(
            key_input_qdata,
            key_input_scale,
            key_weight[context],
            key_weight_scale[context],
            key_norm_weight,
            key_cos,
            key_sin,
            key_norm_epsilon,
            _rows(key_bias, context),
            head_dim=head_dim,
        )
        group_value = value._project_value_op(
            value_input_qdata,
            value_input_scale,
            value_weight[context],
            value_weight_scale[context],
            _rows(value_bias, context),
            head_dim=head_dim,
            is_causal=is_causal,
        )
        output._projected_query_attention_residual_add_op(
            base,
            gate,
            gate_indices,
            python_indexing,
            input_qdata,
            input_scale,
            query_weight[query],
            query_weight_scale[query],
            norm_weight,
            cos,
            sin,
            norm_epsilon,
            softmax_scale,
            _rows(query_bias, query),
            *group_key,
            *group_value,
            key_input_qdata.shape[1],
            is_causal,
            # The output GEMM needs each group's input columns contiguous.
            weight_qdata[:, query].contiguous(),
            weight_scale,
            bias if group == 0 else None,
            group_size,
            output_input_scale,
            head_dim=head_dim,
            query_chunk_rows=query_chunk_rows * head_groups,
        )
        # Release this group's K and V before the next group allocates its own.
        del group_key, group_value


@_head_grouped_attention_residual_add_op.register_fake
def _head_grouped_attention_residual_add_op_fake(  # noqa: PLR0913, PLR0917
    base: torch.Tensor,
    gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    query_weight: torch.Tensor,
    query_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    query_bias: torch.Tensor | None,
    key_input_qdata: torch.Tensor,
    key_input_scale: torch.Tensor,
    key_weight: torch.Tensor,
    key_weight_scale: torch.Tensor,
    key_norm_weight: torch.Tensor | None,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
    key_norm_epsilon: float,
    key_bias: torch.Tensor | None,
    value_input_qdata: torch.Tensor,
    value_input_scale: torch.Tensor,
    value_weight: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_bias: torch.Tensor | None,
    is_causal: bool,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_input_scale: torch.Tensor | None = None,
    *,
    head_dim: int,
    head_groups: int,
    query_chunk_rows: int = DEFAULT_QUERY_CHUNK_ROWS,
) -> None:
    query_rows, key_rows = validate_head_groups(
        query_weight,
        key_weight,
        value_weight,
        weight_qdata,
        group_size,
        head_dim=head_dim,
        head_groups=head_groups,
    )
    # Every group has the same shapes, so the first one validates them all.
    query = slice(0, query_rows)
    context = slice(0, key_rows)
    group_key = key._project_key_op_fake(
        key_input_qdata,
        key_input_scale,
        key_weight[context],
        key_weight_scale[context],
        key_norm_weight,
        key_cos,
        key_sin,
        key_norm_epsilon,
        _rows(key_bias, context),
        head_dim=head_dim,
    )
    group_value = value._project_value_op_fake(
        value_input_qdata,
        value_input_scale,
        value_weight[context],
        value_weight_scale[context],
        _rows(value_bias, context),
        head_dim=head_dim,
        is_causal=is_causal,
    )
    output._projected_query_attention_residual_add_op_fake(
        base,
        gate,
        gate_indices,
        python_indexing,
        input_qdata,
        input_scale,
        query_weight[query],
        query_weight_scale[query],
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        _rows(query_bias, query),
        *group_key,
        *group_value,
        key_input_qdata.shape[1],
        is_causal,
        weight_qdata[:, query].contiguous(),
        weight_scale,
        bias,
        group_size,
        output_input_scale,
        head_dim=head_dim,
        query_chunk_rows=query_chunk_rows * head_groups,
    )
