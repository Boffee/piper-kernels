"""Compiler passes join Inductor options without mutating or duplicating them."""

import torch

from piper_kernels._inductor_options import (
    add_first_post_grad_pass,
    add_ordered_post_grad_passes,
    add_post_grad_pass,
)


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


def test_first_pass_moves_ahead_of_every_other_without_duplicating() -> None:
    before = object()
    first = object()
    after = object()

    combined = add_first_post_grad_pass(
        {"post_grad_custom_pre_pass": (before, first, after)}, first
    )
    single = add_first_post_grad_pass({"post_grad_custom_pre_pass": before}, first)

    assert combined["post_grad_custom_pre_pass"] == (first, before, after)
    assert single["post_grad_custom_pre_pass"] == (first, before)
    assert add_first_post_grad_pass(None, first) == {"post_grad_custom_pre_pass": (first,)}
