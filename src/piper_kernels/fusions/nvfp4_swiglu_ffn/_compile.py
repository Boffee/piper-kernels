"""Compiler folding for a bounded-workspace NVFP4 SwiGLU FFN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.fx.node import Argument

from piper_kernels.fusions.swiglu_ffn import _compile as swiglu_ffn_compile
from piper_kernels.fusions.swiglu_ffn import _pattern as swiglu_ffn_pattern
from piper_kernels.fusions.swiglu_ffn import triton as swiglu_ffn_triton
from piper_kernels.linear import _bias
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation

from . import _compile_validation, _core, _preparation
from . import triton as ffn_backend

_COMPILE_PASS_VERSION = "nvfp4-swiglu-ffn-compile-v3"


@dataclass(frozen=True, slots=True)
class _MatchedFfn:
    gate: nvfp4_compile_fx.SemanticLinearNodes
    value: nvfp4_compile_fx.SemanticLinearNodes
    down: nvfp4_compile_fx.SemanticLinearNodes

    @classmethod
    def from_match(cls, match: Match) -> _MatchedFfn | None:
        calls = [
            node
            for node in match.nodes
            if node.op == "call_function"
            and node.target == torch.ops.piper_kernels.nvfp4_linear.default
        ]
        if len(calls) != 3:
            return None
        parsed: list[nvfp4_compile_fx.SemanticLinearNodes] = []
        for prefix in ("gate", "value", "down"):
            call = next(
                (
                    node
                    for node in calls
                    if _compile_validation.projection_call_matches(node, match, prefix)
                ),
                None,
            )
            if call is None:
                return None
            operands = nvfp4_compile_fx.SemanticLinearNodes.from_call(call)
            if operands is None:
                return None
            parsed.append(operands)
        return cls(*parsed)

    def arguments(self) -> tuple[Argument, ...]:
        """Return custom-op operands in semantic gate/value/down order."""
        return (
            self.gate.input,
            self.gate.weight_qdata,
            self.gate.weight_scale,
            self.gate.weight_per_tensor_scale,
            self.gate.activation_per_tensor_scale,
            self.gate.bias,
            self.gate.dynamic_activation_scale,
            self.gate.high_first,
            self.value.weight_qdata,
            self.value.weight_scale,
            self.value.weight_per_tensor_scale,
            self.value.activation_per_tensor_scale,
            self.value.bias,
            self.value.dynamic_activation_scale,
            self.value.high_first,
            self.down.weight_qdata,
            self.down.weight_scale,
            self.down.weight_per_tensor_scale,
            self.down.activation_per_tensor_scale,
            self.down.bias,
            self.down.dynamic_activation_scale,
            self.down.high_first,
        )


def _semantic_linear_pattern(
    input_pattern: object,
    prefix: str,
    users: int | None,
    *,
    with_high_first: bool,
) -> CallFunction:
    arguments = (
        input_pattern,
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        KeywordArg(f"{prefix}_weight_per_tensor_scale"),
        KeywordArg(f"{prefix}_activation_per_tensor_scale"),
        KeywordArg(f"{prefix}_bias"),
        KeywordArg(f"{prefix}_dynamic_activation_scale"),
        *((KeywordArg(f"{prefix}_high_first"),) if with_high_first else ()),
    )
    if users is None:
        return CallFunction(torch.ops.piper_kernels.nvfp4_linear.default, *arguments)
    return CallFunction(
        torch.ops.piper_kernels.nvfp4_linear.default,
        *arguments,
        _users=users,
    )


def _valid_semantic_ffn(match: Match, *, promote_gate: bool | None) -> bool:
    operands = _MatchedFfn.from_match(match)
    if operands is None:
        return False
    return _compile_validation.valid_semantic_ffn(
        match,
        operands.gate,
        operands.value,
        operands.down,
        promote_gate=promote_gate,
    )


def _valid_semantic_gated_updates(
    match: Match,
    *,
    promote_gate: bool | None,
) -> bool:
    return swiglu_ffn_compile.valid_gated_updates(
        match,
        partial(_valid_semantic_ffn, promote_gate=promote_gate),
    )


def _replace_semantic_ffn(match: Match, **_unused: object) -> None:
    original = match.output_node()
    graph = match.graph
    operands = _MatchedFfn.from_match(match)
    assert operands is not None
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.nvfp4_swiglu_ffn.default,
            args=(*operands.arguments(), ffn_backend._DEFAULT_CHUNK_ROWS),
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
    python_indexing = swiglu_ffn_compile.uses_python_indexing(match)
    with graph.inserting_before(original):
        mutation = graph.call_function(
            torch.ops.piper_kernels.nvfp4_swiglu_ffn_gated_updates_.default,
            args=(
                *operands.arguments(),
                match.kwargs["base"],
                match.kwargs["reusable_update"],
                match.kwargs["update_gate"],
                match.kwargs["ffn_gate"],
                match.kwargs["gate_indices"],
                python_indexing,
                ffn_backend._DEFAULT_CHUNK_ROWS,
            ),
        )
    mutation.meta["val"] = None
    reusable_update = match.kwargs["reusable_update"]
    assert isinstance(reusable_update, torch.fx.Node)
    original.replace_all_uses_with(reusable_update)
    match.erase_nodes()


_gated_updates_patterns = PatternMatcherPass("nvfp4_swiglu_ffn_gated_updates")
_patterns = PatternMatcherPass("nvfp4_swiglu_ffn")
for _with_source_high_first in (False, True):
    for _with_down_high_first in (False, True):
        _source_projection_pattern = partial(
            _semantic_linear_pattern,
            with_high_first=_with_source_high_first,
        )
        _down_projection_pattern = partial(
            _semantic_linear_pattern,
            with_high_first=_with_down_high_first,
        )
        for _promote_gate in (None, False, True):
            for _reverse_multiply in (False, True):
                _semantic_ffn_pattern = swiglu_ffn_pattern.semantic_ffn_pattern(
                    _source_projection_pattern,
                    _source_projection_pattern,
                    _down_projection_pattern,
                    promote_gate=_promote_gate,
                    reverse_multiply=_reverse_multiply,
                )
                register_graph_pattern(
                    _semantic_ffn_pattern,
                    extra_check=lambda match, promote_gate=_promote_gate: _valid_semantic_ffn(
                        match, promote_gate=promote_gate
                    ),
                    pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
                )(_replace_semantic_ffn)
                for _use_aten_index in (False, True):
                    register_graph_pattern(
                        swiglu_ffn_pattern.gated_updates_pattern(
                            _semantic_ffn_pattern,
                            use_aten_index=_use_aten_index,
                        ),
                        extra_check=lambda match, promote_gate=_promote_gate: (
                            _valid_semantic_gated_updates(
                                match,
                                promote_gate=promote_gate,
                            )
                        ),
                        pass_dict=_gated_updates_patterns,  # pyright: ignore[reportArgumentType]
                    )(_replace_semantic_ffn_gated_updates)


def _fold_chunked_ffn(graph: torch.fx.Graph) -> bool:
    changes = _gated_updates_patterns.apply(graph)
    changes += _patterns.apply(graph)
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold semantic gate/value FFNs before ordinary NVFP4 normalization."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            _fold_chunked_ffn(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            tuple(
                file_name
                for file_name in (
                    __file__,
                    _bias.__file__,
                    _core.__file__,
                    _preparation.__file__,
                    _compile_validation.__file__,
                    ffn_backend.__file__,
                    nvfp4_compile_fx.__file__,
                    nvfp4_validation.__file__,
                    swiglu_ffn_compile.__file__,
                    swiglu_ffn_pattern.__file__,
                    swiglu_ffn_triton.__file__,
                )
                if file_name is not None
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def nvfp4_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install semantic FFN folding before ordinary NVFP4 folding."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (compile_pass, nvfp4_compile.compile_pass),
    )


__all__ = ["nvfp4_swiglu_ffn_compile_options"]
