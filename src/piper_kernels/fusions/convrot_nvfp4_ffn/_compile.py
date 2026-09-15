"""Compiler matching shared by bounded standard/ConvRot NVFP4 FFNs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch._inductor.pattern_matcher import CallFunction, Match
from torch.fx.node import Argument

from piper_kernels.fusions.nvfp4_ffn import _compile as nvfp4_ffn_compile
from piper_kernels.linear.convrot.nvfp4 import _compile_fx as convrot_nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx


def source_files() -> tuple[str, ...]:
    """Return source files whose changes invalidate mixed-format matching."""
    return tuple(
        file_name
        for file_name in (
            __file__,
            *nvfp4_ffn_compile.source_files(),
            convrot_nvfp4_compile_fx.__file__,
        )
        if file_name is not None
    )


@dataclass(frozen=True, slots=True)
class MatchedProjection:
    """One standard or ConvRot semantic NVFP4 projection."""

    linear: nvfp4_compile_fx.SemanticLinearNodes
    group_size: int | None

    @classmethod
    def from_call(cls, node: torch.fx.Node) -> MatchedProjection | None:
        """Parse a supported semantic NVFP4 projection call."""
        if node.target == torch.ops.piper_kernels.nvfp4_linear.default:
            linear = nvfp4_compile_fx.SemanticLinearNodes.from_call(node)
            return None if linear is None else cls(linear, None)
        if node.target == torch.ops.piper_kernels.convrot_nvfp4_linear.default:
            convrot = convrot_nvfp4_compile_fx.SemanticLinearNodes.from_call(node)
            return None if convrot is None else cls(convrot.linear, convrot.group_size)
        return None

    def arguments(self) -> tuple[Argument, ...]:
        """Return custom-op projection arguments including the optional rotation group."""
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


def matched_projections(
    match: Match,
    prefixes: tuple[str, ...],
) -> tuple[MatchedProjection, ...] | None:
    """Parse a topology's named standard or ConvRot projection calls."""
    targets = {
        torch.ops.piper_kernels.nvfp4_linear.default,
        torch.ops.piper_kernels.convrot_nvfp4_linear.default,
    }
    calls = [node for node in match.nodes if node.op == "call_function" and node.target in targets]
    if len(calls) != len(prefixes):
        return None
    parsed = []
    for prefix in prefixes:
        call = next(
            (
                node
                for node in calls
                if nvfp4_ffn_compile.projection_call_matches(node, match, prefix)
            ),
            None,
        )
        if call is None:
            return None
        projection = MatchedProjection.from_call(call)
        if projection is None:
            return None
        parsed.append(projection)
    return tuple(parsed)


def semantic_linear_pattern(
    input_pattern: object,
    prefix: str,
    users: int | None,
    *,
    convrot: bool,
    with_high_first: bool,
) -> CallFunction:
    """Match one standard or ConvRot semantic NVFP4 projection."""
    return nvfp4_ffn_compile.semantic_linear_pattern(
        input_pattern,
        prefix,
        users,
        target=(
            torch.ops.piper_kernels.convrot_nvfp4_linear.default
            if convrot
            else torch.ops.piper_kernels.nvfp4_linear.default
        ),
        with_group_size=convrot,
        with_high_first=with_high_first,
    )


def valid_semantic_ffn(
    match: Match,
    sources: tuple[MatchedProjection, ...],
    down: MatchedProjection,
) -> bool:
    """Validate mixed-format metadata and compatible source rotations."""
    if len(sources) not in (1, 2):
        return False
    projections = (*sources, down)
    for index, projection in enumerate(projections):
        if projection.group_size is not None and (
            convrot_nvfp4_compile_fx.validated_semantic_linear(
                convrot_nvfp4_compile_fx.SemanticLinearNodes(
                    projection.linear,
                    projection.group_size,
                ),
                f"ConvRot NVFP4 FFN compiler projection {index}",
            )
            is None
        ):
            return False
    return bool(
        all(source.group_size == sources[0].group_size for source in sources[1:])
        and nvfp4_ffn_compile.valid_semantic_ffn(
            match,
            tuple(source.linear for source in sources),
            down.linear,
        )
    )


__all__ = [
    "MatchedProjection",
    "matched_projections",
    "semantic_linear_pattern",
    "source_files",
    "valid_semantic_ffn",
]
