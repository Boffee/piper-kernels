"""Compiler folding for a bounded-workspace ConvRot INT8 GELU FFN."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)
from torch._inductor.pattern_matcher import (
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from piper_kernels.fusions.convrot_int8_ffn import _compile as ffn_compile_common
from piper_kernels.fusions.convrot_int8_ffn import _core as ffn_core
from piper_kernels.fusions.ffn import _compile as ffn_compile
from piper_kernels.fusions.ffn import _pattern as ffn_pattern
from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.fusions.gelu_ffn import _pattern as gelu_ffn_pattern
from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views
from piper_kernels.linear.convrot.int8 import _compile as convrot_int8_compile
from piper_kernels.linear.convrot.int8 import _compile_fx

from . import triton as gelu_backend

_COMPILE_PASS_VERSION = "convrot-int8-gelu-ffn-compile-v1"


def _valid_semantic_ffn(match: Match) -> bool:
    if not ffn_compile_common.valid_semantic_ffn(match, ("up",)):
        return False
    input_value = ffn_compile_common.argument_metadata(match, "ffn_input")
    assert input_value is not None
    return match.kwargs["logical_dtype"] is input_value.dtype


def _chunk_rows(match: Match, *, gated_updates: bool) -> int:
    input_value = ffn_compile_common.argument_metadata(match, "ffn_input")
    up_weight = ffn_compile_common.argument_metadata(match, "up_weight_qdata")
    down_weight = ffn_compile_common.argument_metadata(match, "down_weight_qdata")
    assert input_value is not None
    assert up_weight is not None
    assert down_weight is not None
    return gelu_backend._default_chunk_rows(
        input_value,
        up_weight,
        down_weight,
        gated_updates=gated_updates,
    )


def _replace_semantic_ffn(  # noqa: PLR0913, PLR0917
    match: Match,
    ffn_input: torch.fx.Node,
    up_weight_qdata: torch.fx.Node,
    up_weight_scale: torch.fx.Node,
    up_bias: torch.fx.Node | None,
    up_group_size: int,
    down_weight_qdata: torch.fx.Node,
    down_weight_scale: torch.fx.Node,
    down_bias: torch.fx.Node | None,
    down_group_size: int,
    up_input_scale: torch.fx.Node | None,
    down_input_scale: torch.fx.Node | None,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_gelu_ffn.default,
            args=(
                ffn_input,
                up_weight_qdata,
                up_weight_scale,
                up_bias,
                up_group_size,
                down_weight_qdata,
                down_weight_scale,
                down_bias,
                down_group_size,
                _chunk_rows(match, gated_updates=False),
                up_input_scale,
                down_input_scale,
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


def _replace_semantic_ffn_gated_updates(  # noqa: PLR0913, PLR0917
    match: Match,
    ffn_input: torch.fx.Node,
    up_weight_qdata: torch.fx.Node,
    up_weight_scale: torch.fx.Node,
    up_bias: torch.fx.Node | None,
    up_group_size: int,
    down_weight_qdata: torch.fx.Node,
    down_weight_scale: torch.fx.Node,
    down_bias: torch.fx.Node | None,
    down_group_size: int,
    base: torch.fx.Node,
    reusable_update: torch.fx.Node,
    update_gate: torch.fx.Node,
    gate_indices: torch.fx.Node,
    ffn_gate: torch.fx.Node,
    up_input_scale: torch.fx.Node | None,
    down_input_scale: torch.fx.Node | None,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    python_indexing = ffn_compile.uses_python_indexing(match)
    with graph.inserting_before(original):
        mutation = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_gelu_ffn_gated_updates_.default,
            args=(
                ffn_input,
                up_weight_qdata,
                up_weight_scale,
                up_bias,
                up_group_size,
                down_weight_qdata,
                down_weight_scale,
                down_bias,
                down_group_size,
                base,
                reusable_update,
                update_gate,
                ffn_gate,
                gate_indices,
                python_indexing,
                _chunk_rows(match, gated_updates=True),
                up_input_scale,
                down_input_scale,
            ),
        )
    mutation.meta["val"] = None
    original.replace_all_uses_with(reusable_update)
    match.erase_nodes()


_gated_updates_patterns = PatternMatcherPass("convrot_int8_gelu_ffn_gated_updates")
_patterns = PatternMatcherPass("convrot_int8_gelu_ffn")
_semantic_ffn_pattern = gelu_ffn_pattern.semantic_ffn_pattern(
    ffn_compile_common.semantic_linear_pattern,
    ffn_compile_common.semantic_linear_pattern,
    promote_input=True,
)
for _use_aten_index in (False, True):
    register_graph_pattern(
        ffn_pattern.indexed_gated_updates_pattern(
            _semantic_ffn_pattern,
            use_aten_index=_use_aten_index,
        ),
        extra_check=lambda match: ffn_compile.valid_indexed_gated_updates(
            match,
            _valid_semantic_ffn,
        ),
        pass_dict=_gated_updates_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_semantic_ffn_gated_updates)
register_graph_pattern(
    _semantic_ffn_pattern,
    extra_check=_valid_semantic_ffn,
    pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
)(_replace_semantic_ffn)


def _fold_chunked_ffn(graph: torch.fx.Graph) -> bool:
    with _compile_fx.canonical_linear_calls(graph):
        changes = _gated_updates_patterns.apply(graph)
        changes += _patterns.apply(graph)
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold semantic GELU FFNs before ordinary ConvRot INT8 normalization."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            projection_views.normalize_projection_views(graph)
            _fold_chunked_ffn(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            tuple(
                file_name
                for file_name in (
                    __file__,
                    projection_views.__file__,
                    *ffn_compile_common.source_files(),
                    ffn_core.__file__,
                    gelu_backend.__file__,
                    ffn_compile.__file__,
                    ffn_pattern.__file__,
                    indexed_updates.__file__,
                    gelu_ffn_pattern.__file__,
                    input_activation_compile.__file__,
                )
                if file_name is not None
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_int8_gelu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install semantic GELU FFN folding before ordinary ConvRot INT8 folding."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (compile_pass, convrot_int8_compile.compile_pass),
    )


__all__ = ["convrot_int8_gelu_ffn_compile_options"]
