"""Compiler folding for a bounded-workspace standard NVFP4 GELU FFN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch._inductor.pattern_matcher import Match, PatternMatcherPass, register_graph_pattern
from torch.fx.node import Argument

from piper_kernels import _inductor_options as inductor_options
from piper_kernels.fusions.ffn import _compile as ffn_compile
from piper_kernels.fusions.ffn import _pattern as ffn_pattern
from piper_kernels.fusions.gelu_ffn import _pattern as gelu_ffn_pattern
from piper_kernels.fusions.nvfp4_ffn import _compile as ffn_compile_common
from piper_kernels.fusions.residual_updates import _gates as residual_gates
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx

from . import triton as gelu_backend


@dataclass(frozen=True, slots=True)
class _MatchedFfn:
    up: nvfp4_compile_fx.SemanticLinearNodes
    down: nvfp4_compile_fx.SemanticLinearNodes

    @classmethod
    def from_match(cls, match: Match) -> _MatchedFfn | None:
        projections = ffn_compile_common.matched_projections(match, ("up", "down"))
        return None if projections is None else cls(*projections)

    def arguments(self) -> tuple[Argument, ...]:
        """Return custom-op operands in semantic up/down order."""
        return (
            self.up.input,
            self.up.weight_qdata,
            self.up.weight_scale,
            self.up.weight_per_tensor_scale,
            self.up.activation_per_tensor_scale,
            self.up.bias,
            self.up.dynamic_activation_scale,
            self.up.high_first,
            self.down.weight_qdata,
            self.down.weight_scale,
            self.down.weight_per_tensor_scale,
            self.down.activation_per_tensor_scale,
            self.down.bias,
            self.down.dynamic_activation_scale,
            self.down.high_first,
        )


def _valid_semantic_ffn(match: Match) -> bool:
    operands = _MatchedFfn.from_match(match)
    if operands is None:
        return False
    input_value = preparation_sharing.tensor_metadata(operands.up.input)
    return bool(
        input_value is not None
        and ffn_compile_common.valid_semantic_ffn(match, (operands.up,), operands.down)
        and match.kwargs["logical_dtype"] is input_value.dtype
    )


def _chunk_rows(match: Match, *, adds_residual: bool) -> int:
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    input_value = preparation_sharing.tensor_metadata(operands.up.input)
    up_weight = preparation_sharing.tensor_metadata(operands.up.weight_qdata)
    down_weight = preparation_sharing.tensor_metadata(operands.down.weight_qdata)
    assert input_value is not None
    assert up_weight is not None
    assert down_weight is not None
    return gelu_backend._default_chunk_rows(
        input_value,
        up_weight,
        down_weight,
        adds_residual=adds_residual,
    )


def _replace_semantic_ffn(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.nvfp4_gelu_ffn.default,
            args=(*operands.arguments(), _chunk_rows(match, adds_residual=False)),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


def _replace_semantic_ffn_gated_updates(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    python_indexing = ffn_compile.uses_python_indexing(match)
    with graph.inserting_before(original):
        mutation = graph.call_function(
            torch.ops.piper_kernels.nvfp4_gelu_ffn_gated_updates_.default,
            args=(
                *operands.arguments(),
                match.kwargs["base"],
                match.kwargs["reusable_update"],
                match.kwargs["update_gate"],
                match.kwargs["ffn_gate"],
                match.kwargs["gate_indices"],
                python_indexing,
                _chunk_rows(match, adds_residual=True),
            ),
        )
    mutation.meta["val"] = None
    reusable_update = match.kwargs["reusable_update"]
    assert isinstance(reusable_update, torch.fx.Node)
    original.replace_all_uses_with(reusable_update)
    match.erase_nodes()


_gated_updates_patterns = PatternMatcherPass("nvfp4_gelu_ffn_gated_updates")
_patterns = PatternMatcherPass("nvfp4_gelu_ffn")
for _with_source_high_first in (False, True):
    for _with_down_high_first in (False, True):
        _source_projection_pattern = partial(
            ffn_compile_common.semantic_linear_pattern,
            target=torch.ops.piper_kernels.nvfp4_linear.default,
            with_group_size=False,
            with_high_first=_with_source_high_first,
        )
        _down_projection_pattern = partial(
            ffn_compile_common.semantic_linear_pattern,
            target=torch.ops.piper_kernels.nvfp4_linear.default,
            with_group_size=False,
            with_high_first=_with_down_high_first,
        )
        _semantic_ffn_pattern = gelu_ffn_pattern.semantic_ffn_pattern(
            _source_projection_pattern,
            _down_projection_pattern,
            promote_input=True,
        )
        for _selection in residual_gates.GATE_SELECTIONS:
            register_graph_pattern(
                ffn_pattern.indexed_gated_updates_pattern(
                    _semantic_ffn_pattern,
                    selection=_selection,
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
    changes = _gated_updates_patterns.apply(graph)
    changes += _patterns.apply(graph)
    changes += ffn_compile.fold_residual_adds(
        graph,
        torch.ops.piper_kernels.nvfp4_gelu_ffn.default,
        torch.ops.piper_kernels.nvfp4_gelu_ffn_residual_add_.default,
        gelu_backend._default_chunk_rows,
    )
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold semantic GELU FFNs before ordinary NVFP4 normalization."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            projection_views.normalize_projection_views(graph)
            _fold_chunked_ffn(graph)

    def uuid(self) -> bytes:
        return inductor_options.compute_compile_pass_uuid(self)


compile_pass = _CompilePass()


def nvfp4_gelu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install semantic GELU FFN folding before ordinary NVFP4 folding."""
    return inductor_options.add_ordered_post_grad_passes(
        options,
        (compile_pass, nvfp4_compile.compile_pass),
    )


__all__ = ["nvfp4_gelu_ffn_compile_options"]
