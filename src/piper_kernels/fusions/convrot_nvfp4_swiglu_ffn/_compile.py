"""Compiler folding for bounded standard/ConvRot NVFP4 SwiGLU FFNs."""

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

from piper_kernels.fusions.nvfp4_swiglu_ffn import _compile_validation, _core, _preparation
from piper_kernels.fusions.swiglu_ffn import _compile as swiglu_ffn_compile
from piper_kernels.fusions.swiglu_ffn import _pattern as swiglu_ffn_pattern
from piper_kernels.fusions.swiglu_ffn import triton as swiglu_ffn_triton
from piper_kernels.linear import _bias
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.nvfp4 import _compile as convrot_nvfp4_compile
from piper_kernels.linear.convrot.nvfp4 import _compile_fx as convrot_nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _compile as nvfp4_compile
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx

from . import triton as ffn_backend

_COMPILE_PASS_VERSION = "convrot-nvfp4-swiglu-ffn-compile-v4"


@dataclass(frozen=True, slots=True)
class _MatchedProjection:
    linear: nvfp4_compile_fx.SemanticLinearNodes
    group_size: int | None

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> _MatchedProjection | None:
        if node.target == torch.ops.piper_kernels.nvfp4_linear.default:
            linear = nvfp4_compile_fx.SemanticLinearNodes.from_call(node)
            return None if linear is None else cls(linear, None)
        if node.target == torch.ops.piper_kernels.convrot_nvfp4_linear.default:
            convrot = convrot_nvfp4_compile_fx.SemanticLinearNodes.from_call(node)
            return None if convrot is None else cls(convrot.linear, convrot.group_size)
        return None

    def arguments(self) -> tuple[Argument, ...]:
        return (
            self.linear.weight_qdata,
            self.linear.weight_scale,
            self.linear.weight_per_tensor_scale,
            self.linear.activation_per_tensor_scale,
            self.linear.bias,
            self.linear.dynamic_activation_scale,
            self.group_size,
            self.linear.high_first,
        )


@dataclass(frozen=True, slots=True)
class _MatchedFfn:
    gate: _MatchedProjection
    value: _MatchedProjection
    down: _MatchedProjection

    @classmethod
    def from_match(cls, match: Match) -> _MatchedFfn | None:
        targets = {
            torch.ops.piper_kernels.nvfp4_linear.default,
            torch.ops.piper_kernels.convrot_nvfp4_linear.default,
        }
        calls = [
            node for node in match.nodes if node.op == "call_function" and node.target in targets
        ]
        if len(calls) != 3:
            return None
        parsed: list[_MatchedProjection] = []
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
            operands = _MatchedProjection.from_call(call)
            if operands is None:
                return None
            parsed.append(operands)
        return cls(*parsed)

    def arguments(self) -> tuple[Argument, ...]:
        """Return custom-op operands in semantic gate/value/down order."""
        return (
            self.gate.linear.input,
            *self.gate.arguments(),
            *self.value.arguments(),
            *self.down.arguments(),
        )


def _semantic_linear_pattern(
    input_pattern: object,
    prefix: str,
    users: int | None,
    *,
    convrot: bool,
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
        *((KeywordArg(f"{prefix}_group_size"),) if convrot else ()),
        *((KeywordArg(f"{prefix}_high_first"),) if with_high_first else ()),
    )
    target = (
        torch.ops.piper_kernels.convrot_nvfp4_linear.default
        if convrot
        else torch.ops.piper_kernels.nvfp4_linear.default
    )
    if users is None:
        return CallFunction(target, *arguments)
    return CallFunction(target, *arguments, _users=users)


def _valid_semantic_ffn(match: Match, *, promote_gate: bool | None) -> bool:
    operands = _MatchedFfn.from_match(match)
    if operands is None:
        return False
    for name, projection in (
        ("gate", operands.gate),
        ("value", operands.value),
        ("down", operands.down),
    ):
        if projection.group_size is not None and (
            convrot_nvfp4_compile_fx.validated_semantic_linear(
                convrot_nvfp4_compile_fx.SemanticLinearNodes(
                    projection.linear,
                    projection.group_size,
                ),
                f"ConvRot NVFP4 FFN compiler {name} projection",
            )
            is None
        ):
            return False
    return bool(
        operands.gate.group_size == operands.value.group_size
        and _compile_validation.valid_semantic_ffn(
            match,
            operands.gate.linear,
            operands.value.linear,
            operands.down.linear,
            promote_gate=promote_gate,
        )
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
            torch.ops.piper_kernels.convrot_nvfp4_swiglu_ffn.default,
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
            torch.ops.piper_kernels.convrot_nvfp4_swiglu_ffn_gated_updates_.default,
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


_gated_updates_patterns = PatternMatcherPass("convrot_nvfp4_swiglu_ffn_gated_updates")
_patterns = PatternMatcherPass("convrot_nvfp4_swiglu_ffn")
for _source_convrot, _down_convrot in ((True, True), (True, False), (False, True)):
    for _with_source_high_first in (False, True):
        for _with_down_high_first in (False, True):
            _source_projection_pattern = partial(
                _semantic_linear_pattern,
                convrot=_source_convrot,
                with_high_first=_with_source_high_first,
            )
            _down_projection_pattern = partial(
                _semantic_linear_pattern,
                convrot=_down_convrot,
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
                    register_graph_pattern(
                        _semantic_ffn_pattern,
                        extra_check=lambda match, promote_gate=_promote_gate: _valid_semantic_ffn(
                            match, promote_gate=promote_gate
                        ),
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
    """Fold FFNs containing ConvRot NVFP4 before ordinary linear normalization."""

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
                    convrot_nvfp4_compile_fx.__file__,
                    nvfp4_compile_fx.__file__,
                    swiglu_ffn_compile.__file__,
                    swiglu_ffn_pattern.__file__,
                    swiglu_ffn_triton.__file__,
                )
                if file_name is not None
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_nvfp4_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install semantic FFN folding before standard and ConvRot NVFP4 normalization."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (
            compile_pass,
            nvfp4_compile.compile_pass,
            convrot_nvfp4_compile.compile_pass,
        ),
    )


__all__ = ["convrot_nvfp4_swiglu_ffn_compile_options"]
