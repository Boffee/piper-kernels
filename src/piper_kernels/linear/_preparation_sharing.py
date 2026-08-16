"""Inductor graph machinery for sharing input preparation across linear calls."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


class PreparationSharingRule[PreparedNodes](Protocol):
    """Backend rule consumed by :class:`PreparationSharingPass`."""

    @property
    def linear_target(self) -> object:
        """Return the semantic operator that consumes an unprepared input."""
        ...

    def match_key(self, node: torch.fx.Node) -> Hashable | None:
        """Return a shared-preparation key, or ``None`` for an ineligible node."""
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
    groups: dict[Hashable, list[torch.fx.Node]] = defaultdict(list)
    for node in graph.nodes:
        if node.op != "call_function" or node.target != rule.linear_target:
            continue
        key = rule.match_key(node)
        if key is not None:
            groups[key].append(node)

    changed = False
    for nodes in groups.values():
        if len(nodes) < 2:
            continue
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


class PreparationSharingPass(CustomInferenceAwareGraphPass):
    """Apply one or more backend preparation-sharing rules during inference."""

    def __init__(
        self,
        rules: Sequence[PreparationSharingRule[Any]],
        *,
        version: str,
        source_files: Sequence[str],
    ) -> None:
        self._rules = tuple(rules)
        self._version = version
        self._source_files = (__file__, *source_files)

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        changed = False
        for rule in self._rules:
            changed = _rewrite_rule(graph, rule) or changed
        if changed:
            graph.lint()

    def uuid(self) -> bytes:
        """Invalidate Inductor caches when the engine or a backend rule changes."""
        return get_hash_for_files(self._source_files, extra=self._version)


def add_post_grad_pass(
    options: Mapping[str, object] | None,
    compiler_pass: CustomInferenceAwareGraphPass,
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


__all__ = [
    "PreparationSharingPass",
    "PreparationSharingRule",
    "add_post_grad_pass",
    "dimension_key",
    "tensor_metadata",
]
