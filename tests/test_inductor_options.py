"""Compiler passes join Inductor options and key Inductor's cache on package source."""

import importlib
import importlib.util
import pkgutil
import shutil
from pathlib import Path
from types import ModuleType

import torch
from torch._inductor.custom_graph_pass import CustomPassBase

import piper_kernels
from piper_kernels._inductor_options import (
    add_first_post_grad_pass,
    add_ordered_post_grad_passes,
    add_post_grad_pass,
    compute_compile_pass_uuid,
)
from piper_kernels.linear.convrot.int8._compile import compile_pass


def _collect_compile_passes() -> list[CustomPassBase]:
    """Import every package module and return the compiler passes they define."""
    passes: dict[int, CustomPassBase] = {}
    for module_info in pkgutil.walk_packages(piper_kernels.__path__, "piper_kernels."):
        for value in vars(importlib.import_module(module_info.name)).values():
            if isinstance(value, CustomPassBase):
                passes[id(value)] = value
    return list(passes.values())


def _load_inductor_options(package: Path) -> ModuleType:
    """Load a fresh copy of the options module, which hashes the package around it."""
    spec = importlib.util.spec_from_file_location(
        "copied_inductor_options", package / "_inductor_options.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_compile_pass_has_its_own_package_source_key() -> None:
    passes = _collect_compile_passes()
    uuids = [compiler_pass.uuid() for compiler_pass in passes]

    assert passes
    assert uuids == [compute_compile_pass_uuid(compiler_pass) for compiler_pass in passes]
    assert len(set(uuids)) == len(passes)


def test_compile_pass_uuid_changes_with_any_package_source(tmp_path: Path) -> None:
    package = tmp_path / "piper_kernels"
    shutil.copytree(
        Path(piper_kernels.__file__).parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    copied = _load_inductor_options(package).compute_compile_pass_uuid(compile_pass)
    assert copied == compile_pass.uuid()

    # Accelerator targets decide at compile time whether a pass fuses.
    targets = package / "_triton" / "targets.py"
    targets.write_text(targets.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert _load_inductor_options(package).compute_compile_pass_uuid(compile_pass) != copied


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
