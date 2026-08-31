"""Inductor graph machinery for sharing input preparation across linear calls."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol

import torch

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"
type PreparationMatchKey = tuple[Hashable, Hashable]


class PreparationSharingRule[PreparedNodes](Protocol):
    """Backend rule consumed by :func:`share_preparation`."""

    @property
    def linear_target(self) -> object:
        """Return the semantic operator that consumes an unprepared input."""
        ...

    def match_key(self, node: torch.fx.Node) -> PreparationMatchKey | None:
        """Return repeated-family and exact-preparation keys for an eligible node."""
        ...

    def prepare(self, graph: torch.fx.Graph, first: torch.fx.Node) -> PreparedNodes:
        """Create preparation nodes using the first grouped linear."""
        ...

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: PreparedNodes,
    ) -> torch.fx.Node:
        """Create the prepared equivalent of one original linear."""
        ...


def tensor_metadata(node: torch.fx.Node) -> torch.Tensor | None:
    """Read post-AOT tensor metadata, failing closed when it is unavailable."""
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def dimension_key(dimension: int | torch.SymInt) -> tuple[str, int | str]:
    """Make static and symbolic metadata dimensions safe dictionary keys."""
    if isinstance(dimension, int):
        return ("static", dimension)
    return ("symbolic", str(dimension))


def _rewrite_rule[PreparedNodes](
    graph: torch.fx.Graph,
    rule: PreparationSharingRule[PreparedNodes],
) -> bool:
    groups: dict[Hashable, dict[Hashable, list[torch.fx.Node]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for node in graph.nodes:
        if node.op != "call_function" or node.target != rule.linear_target:
            continue
        match = rule.match_key(node)
        if match is not None:
            family_key, preparation_key = match
            groups[family_key][preparation_key].append(node)

    changed = False
    for preparation_groups in groups.values():
        if sum(len(nodes) for nodes in preparation_groups.values()) < 2:
            continue
        for nodes in preparation_groups.values():
            with graph.inserting_before(nodes[0]):
                prepared = rule.prepare(graph, nodes[0])
            for node in nodes:
                with graph.inserting_before(node):
                    replacement = rule.replace(graph, node, prepared)
                replacement.meta = node.meta.copy()
                replacement.meta.pop("eager_input_vals", None)
                node.replace_all_uses_with(replacement)
                graph.erase_node(node)
        changed = True
    return changed


def share_preparation(
    graph: torch.fx.Graph,
    rules: Sequence[PreparationSharingRule[Any]],
) -> bool:
    """Apply backend preparation-sharing rules and return whether the graph changed."""
    changed = False
    for rule in rules:
        changed = _rewrite_rule(graph, rule) or changed
    if changed:
        graph.lint()
    return changed


def add_post_grad_pass(
    options: Mapping[str, object] | None,
    compiler_pass: object,
) -> dict[str, object]:
    """Copy options and append one post-grad pass without mutating or duplicating."""
    combined = dict(options) if options is not None else {}
    existing = combined.get(_POST_GRAD_PRE_PASS)
    if existing is None:
        combined[_POST_GRAD_PRE_PASS] = compiler_pass
    elif existing is compiler_pass:
        pass
    elif isinstance(existing, (list, tuple)):
        if not any(item is compiler_pass for item in existing):
            combined[_POST_GRAD_PRE_PASS] = (*existing, compiler_pass)
    else:
        combined[_POST_GRAD_PRE_PASS] = (existing, compiler_pass)
    return combined


def add_ordered_post_grad_passes(
    options: Mapping[str, object] | None,
    compiler_passes: Sequence[object],
) -> dict[str, object]:
    """Install an ordered pass group where its first existing member was located."""
    if not compiler_passes:
        return dict(options) if options is not None else {}
    combined = dict(options) if options is not None else {}
    existing = combined.get(_POST_GRAD_PRE_PASS)
    if existing is None:
        passes: tuple[object, ...] = ()
    elif isinstance(existing, (list, tuple)):
        passes = tuple(existing)
    else:
        passes = (existing,)

    anchor_index = next(
        (
            index
            for index, compiler_pass in enumerate(passes)
            if any(compiler_pass is grouped for grouped in compiler_passes)
        ),
        len(passes),
    )
    unrelated = tuple(
        compiler_pass
        for compiler_pass in passes
        if not any(compiler_pass is grouped for grouped in compiler_passes)
    )
    insertion_index = sum(
        not any(compiler_pass is grouped for grouped in compiler_passes)
        for compiler_pass in passes[:anchor_index]
    )
    combined[_POST_GRAD_PRE_PASS] = (
        *unrelated[:insertion_index],
        *compiler_passes,
        *unrelated[insertion_index:],
    )
    return combined


__all__ = [
    "PreparationMatchKey",
    "PreparationSharingRule",
    "add_ordered_post_grad_passes",
    "add_post_grad_pass",
    "dimension_key",
    "share_preparation",
    "tensor_metadata",
]
