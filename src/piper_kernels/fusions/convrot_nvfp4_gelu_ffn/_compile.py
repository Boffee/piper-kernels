"""Compiler folding for bounded standard/ConvRot NVFP4 GELU FFNs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)
from torch._inductor.pattern_matcher import Match, PatternMatcherPass, register_graph_pattern
from torch.fx.node import Argument

from piper_kernels.fusions.convrot_nvfp4_ffn import _compile as ffn_compile_common
from piper_kernels.fusions.convrot_nvfp4_ffn import _preparation as convrot_source_preparation
from piper_kernels.fusions.ffn import _compile as ffn_compile
from piper_kernels.fusions.ffn import _pattern as ffn_pattern
from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.fusions.gelu_ffn import _pattern as gelu_ffn_pattern
from piper_kernels.fusions.nvfp4_ffn import _core as ffn_core
from piper_kernels.fusions.nvfp4_ffn import _preparation as source_preparation
from piper_kernels.fusions.nvfp4_gelu_ffn import _operands, _preparation
from piper_kernels.fusions.nvfp4_gelu_ffn import triton as standard_gelu_backend
from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views
from piper_kernels.linear.convrot.nvfp4 import _compile as convrot_nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile

from . import _preparation as convrot_preparation
from . import triton as gelu_backend

_COMPILE_PASS_VERSION = "convrot-nvfp4-gelu-ffn-compile-v2"


@dataclass(frozen=True, slots=True)
class _MatchedFfn:
    up: ffn_compile_common.MatchedProjection
    down: ffn_compile_common.MatchedProjection

    @classmethod
    def from_match(cls, match: Match) -> _MatchedFfn | None:
        projections = ffn_compile_common.matched_projections(match, ("up", "down"))
        return None if projections is None else cls(*projections)

    def arguments(self) -> tuple[Argument, ...]:
        """Return custom-op operands in semantic up/down order."""
        return (
            self.up.linear.input,
            *self.up.arguments(),
            *self.down.arguments(),
        )


def _valid_semantic_ffn(match: Match) -> bool:
    operands = _MatchedFfn.from_match(match)
    if operands is None:
        return False
    input_value = preparation_sharing.tensor_metadata(operands.up.linear.input)
    return bool(
        input_value is not None
        and ffn_compile_common.valid_semantic_ffn(match, (operands.up,), operands.down)
        and match.kwargs["logical_dtype"] is input_value.dtype
    )


def _chunk_rows(match: Match, *, gated_updates: bool) -> int:
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    input_value = preparation_sharing.tensor_metadata(operands.up.linear.input)
    up_weight = preparation_sharing.tensor_metadata(operands.up.linear.weight_qdata)
    down_weight = preparation_sharing.tensor_metadata(operands.down.linear.weight_qdata)
    assert input_value is not None
    assert up_weight is not None
    assert down_weight is not None
    return standard_gelu_backend._default_chunk_rows(
        input_value,
        up_weight,
        down_weight,
        gated_updates=gated_updates,
    )


def _replace_semantic_ffn(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.convrot_nvfp4_gelu_ffn.default,
            args=(*operands.arguments(), _chunk_rows(match, gated_updates=False)),
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
            torch.ops.piper_kernels.convrot_nvfp4_gelu_ffn_gated_updates_.default,
            args=(
                *operands.arguments(),
                match.kwargs["base"],
                match.kwargs["reusable_update"],
                match.kwargs["update_gate"],
                match.kwargs["ffn_gate"],
                match.kwargs["gate_indices"],
                python_indexing,
                _chunk_rows(match, gated_updates=True),
            ),
        )
    mutation.meta["val"] = None
    reusable_update = match.kwargs["reusable_update"]
    assert isinstance(reusable_update, torch.fx.Node)
    original.replace_all_uses_with(reusable_update)
    match.erase_nodes()


_gated_updates_patterns = PatternMatcherPass("convrot_nvfp4_gelu_ffn_gated_updates")
_patterns = PatternMatcherPass("convrot_nvfp4_gelu_ffn")
for _source_convrot, _down_convrot in ((True, True), (True, False), (False, True)):
    for _with_source_high_first in (False, True):
        for _with_down_high_first in (False, True):
            _source_projection_pattern = partial(
                ffn_compile_common.semantic_linear_pattern,
                convrot=_source_convrot,
                with_high_first=_with_source_high_first,
            )
            _down_projection_pattern = partial(
                ffn_compile_common.semantic_linear_pattern,
                convrot=_down_convrot,
                with_high_first=_with_down_high_first,
            )
            _semantic_ffn_pattern = gelu_ffn_pattern.semantic_ffn_pattern(
                _source_projection_pattern,
                _down_projection_pattern,
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
    changes = _gated_updates_patterns.apply(graph)
    changes += _patterns.apply(graph)
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold GELU FFNs containing ConvRot NVFP4 before linear normalization."""

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
                    source_preparation.__file__,
                    convrot_source_preparation.__file__,
                    _operands.__file__,
                    _preparation.__file__,
                    convrot_preparation.__file__,
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


def convrot_nvfp4_gelu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install GELU FFN folding before standard and ConvRot NVFP4 normalization."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (
            compile_pass,
            nvfp4_compile.compile_pass,
            convrot_nvfp4_compile.compile_pass,
        ),
    )


__all__ = ["convrot_nvfp4_gelu_ffn_compile_options"]
