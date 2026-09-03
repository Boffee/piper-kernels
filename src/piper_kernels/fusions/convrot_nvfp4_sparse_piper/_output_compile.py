"""Fold sparse attention followed by a static ConvRot NVFP4 projection."""

from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from piper_kernels.fusions.nvfp4_sparse_piper import (
    _output_compile as nvfp4_output_compile,
)
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear.convrot._rotation import validate_group_size

from . import output


def _attention_output_pattern(
    *,
    with_block_lengths: bool,
    with_coarse: bool,
) -> CallFunction:
    return CallFunction(
        torch.ops.piper_kernels.convrot_nvfp4_linear.default,
        sparse_piper_pattern.reshaped_quantized_attention_pattern(
            with_block_lengths=with_block_lengths,
            with_coarse=with_coarse,
        ),
        KeywordArg("output_weight_qdata"),
        KeywordArg("output_weight_scale"),
        KeywordArg("output_weight_per_tensor_scale"),
        KeywordArg("output_activation_scale"),
        KeywordArg("output_bias"),
        False,
        KeywordArg("output_group_size"),
    )


def _valid_attention_output(match: Match) -> bool:
    if not nvfp4_output_compile._valid_attention_output(match):
        return False
    group_size = match.kwargs["output_group_size"]
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        return False
    try:
        validate_group_size(group_size)
    except ValueError:
        return False
    weight_node = match.kwargs["output_weight_qdata"]
    assert isinstance(weight_node, torch.fx.Node)
    weight = weight_node.meta.get("val")
    return isinstance(weight, torch.Tensor) and 2 * weight.shape[1] % group_size == 0


def _replace_attention_output(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default,
            args=(
                *sparse_piper_pattern.quantized_attention_arguments(match),
                match.kwargs["output_weight_qdata"],
                match.kwargs["output_weight_scale"],
                match.kwargs["output_weight_per_tensor_scale"],
                match.kwargs["output_activation_scale"],
                match.kwargs["output_bias"],
                match.kwargs["output_group_size"],
                output._DEFAULT_QUERY_CHUNK_ROWS,
                *sparse_piper_pattern.bounded_attention_arguments(match),
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


_patterns = PatternMatcherPass("convrot_nvfp4_sparse_piper_attention_output")
for _with_block_lengths in (False, True):
    for _with_coarse in (False, True):
        register_graph_pattern(
            _attention_output_pattern(
                with_block_lengths=_with_block_lengths,
                with_coarse=_with_coarse,
            ),
            extra_check=_valid_attention_output,
            pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
        )(_replace_attention_output)


def _fold_attention_output(graph: torch.fx.Graph) -> bool:
    """Replace one compatible materialized attention-to-output region."""
    changed = _patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__: list[str] = []
