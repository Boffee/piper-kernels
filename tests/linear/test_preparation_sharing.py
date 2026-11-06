"""Tests for the backend-neutral preparation-sharing graph pass."""

import operator

import torch

from piper_kernels.linear._preparation_sharing import (
    PreparationMatchKey,
    add_ordered_post_grad_passes,
    add_post_grad_pass,
    share_preparation,
)


class _ToyRule:
    """Replace repeated binary calls with calls fed by one prepared source."""

    def __init__(self, linear_target: object) -> None:
        self.linear_target = linear_target

    def match_key(self, node: torch.fx.Node) -> PreparationMatchKey | None:
        source = node.args[0]
        return (source, source) if isinstance(source, torch.fx.Node) else None

    def prepare(self, graph: torch.fx.Graph, first: torch.fx.Node) -> torch.fx.Node:
        source = first.args[0]
        assert isinstance(source, torch.fx.Node)
        return graph.call_function(operator.neg, args=(source,))

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: torch.fx.Node,
    ) -> torch.fx.Node:
        return graph.call_function(operator.sub, args=(prepared, node.args[1]))


def _call(
    graph: torch.fx.Graph,
    target: object,
    source: torch.fx.Node,
    value: int,
) -> torch.fx.Node:
    node = graph.call_function(target, args=(source, value))  # type: ignore[arg-type]
    node.meta = {"val": torch.empty(4), "eager_input_vals": (torch.empty(4), value)}
    return node


def test_pass_shares_groups_and_keeps_replacements_at_original_positions() -> None:
    graph = torch.fx.Graph()
    shared = graph.placeholder("shared")
    singleton = graph.placeholder("singleton")
    first = _call(graph, operator.add, shared, 1)
    middle = graph.call_function(torch.relu, args=(first,))
    second = _call(graph, operator.add, shared, 2)
    untouched = _call(graph, operator.add, singleton, 3)
    graph.output((first, middle, second, untouched))

    share_preparation(graph, (_ToyRule(operator.add),))

    nodes = list(graph.nodes)
    replacements = [node for node in nodes if node.target is operator.sub]
    assert len(replacements) == 2
    assert nodes.index(replacements[0]) < nodes.index(middle) < nodes.index(replacements[1])
    assert sum(node.target is operator.neg for node in nodes) == 1
    assert sum(node.target is operator.add for node in nodes) == 1
    assert all(node.meta["val"].shape == (4,) for node in replacements)
    assert all("eager_input_vals" not in node.meta for node in replacements)
    graph.lint()


def test_rewrite_runs_every_rule() -> None:
    graph = torch.fx.Graph()
    shared = graph.placeholder("shared")
    first_add = _call(graph, operator.add, shared, 1)
    second_add = _call(graph, operator.add, shared, 2)
    first_mul = _call(graph, operator.mul, shared, 3)
    second_mul = _call(graph, operator.mul, shared, 4)
    graph.output((first_add, second_add, first_mul, second_mul))

    share_preparation(graph, (_ToyRule(operator.add), _ToyRule(operator.mul)))

    nodes = list(graph.nodes)
    assert sum(node.target is operator.sub for node in nodes) == 4
    assert sum(node.target is operator.neg for node in nodes) == 2
    assert all(node.target not in (operator.add, operator.mul) for node in nodes)
    graph.lint()


def test_options_compose_without_mutating_or_duplicating() -> None:
    def existing_pass(_graph: torch.fx.Graph) -> None:
        pass

    def compiler_pass(_graph: torch.fx.Graph) -> None:
        pass

    original_options: dict[str, object] = {
        "max_autotune": True,
        "post_grad_custom_pre_pass": [existing_pass],
    }

    combined = add_post_grad_pass(original_options, compiler_pass)
    repeated = add_post_grad_pass(combined, compiler_pass)

    assert original_options["post_grad_custom_pre_pass"] == [existing_pass]
    assert combined["max_autotune"] is True
    assert combined["post_grad_custom_pre_pass"] == (existing_pass, compiler_pass)
    assert repeated == combined


def test_ordered_pass_group_replaces_anchor_without_moving_unrelated_passes() -> None:
    unrelated_before = object()
    fusion = object()
    backend = object()
    unrelated_after = object()
    original = {
        "post_grad_custom_pre_pass": (
            unrelated_before,
            backend,
            unrelated_after,
            fusion,
        )
    }

    combined = add_ordered_post_grad_passes(original, (fusion, backend))
    repeated = add_ordered_post_grad_passes(combined, (fusion, backend))

    assert combined["post_grad_custom_pre_pass"] == (
        unrelated_before,
        fusion,
        backend,
        unrelated_after,
    )
    assert repeated == combined


def test_ordered_pass_group_anchors_on_first_installed_member() -> None:
    unrelated_before = object()
    first = object()
    second = object()
    unrelated_after = object()

    combined = add_ordered_post_grad_passes(
        {"post_grad_custom_pre_pass": (unrelated_before, first, unrelated_after)},
        (first, second),
    )

    assert combined["post_grad_custom_pre_pass"] == (
        unrelated_before,
        first,
        second,
        unrelated_after,
    )
