"""Unified Inductor compiler pipeline composition and source/config cache keys."""

import importlib
import importlib.util
import itertools
import pkgutil
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomPassBase

import piper_kernels
from piper_kernels import compile_options
from piper_kernels._inductor_options import (
    GraphPassHook,
    _PiperPostGradPass,
    _PiperPreGradPass,
    add_pre_grad_hook,
    compute_compile_pass_uuid,
)
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_conv3d_compile_options,
)


def _collect_compile_pass_classes() -> list[type[CustomPassBase]]:
    """Import every package module and return all CustomPassBase subclasses they define."""
    classes: dict[int, type[CustomPassBase]] = {}
    for module_info in pkgutil.walk_packages(piper_kernels.__path__, "piper_kernels."):
        module = importlib.import_module(module_info.name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, CustomPassBase)
                and value.__module__.startswith("piper_kernels")
            ):
                classes[id(value)] = value
    return list(classes.values())


def _load_inductor_options(package: Path) -> ModuleType:
    """Load a fresh copy of the options module, which hashes the package around it."""
    module_name = "copied_inductor_options"
    spec = importlib.util.spec_from_file_location(module_name, package / "_inductor_options.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_only_unified_compile_pass_classes_exist_and_key_on_package_source() -> None:
    classes = _collect_compile_pass_classes()
    assert set(classes) == {_PiperPostGradPass, _PiperPreGradPass}

    instances = [_PiperPostGradPass(), _PiperPreGradPass(conv3d=True)]
    uuids = [compiler_pass.uuid() for compiler_pass in instances]
    assert uuids == [compute_compile_pass_uuid(compiler_pass) for compiler_pass in instances]
    assert len(set(uuids)) == len(instances)


def test_compile_pass_uuid_changes_with_any_package_source(tmp_path: Path) -> None:
    package = tmp_path / "piper_kernels"
    shutil.copytree(
        Path(piper_kernels.__file__).parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    compiler_pass = _PiperPostGradPass(linear=True, ffn=True, attention=True)
    copied = _load_inductor_options(package).compute_compile_pass_uuid(compiler_pass)
    assert copied == compiler_pass.uuid()

    # Accelerator targets decide at compile time whether a pass fuses.
    targets = package / "_triton" / "targets.py"
    targets.write_text(targets.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert _load_inductor_options(package).compute_compile_pass_uuid(compiler_pass) != copied


def test_distinct_feature_combinations_produce_distinct_uuids() -> None:
    seen: dict[tuple[bool, bool, bool, bool, bool], bytes] = {}
    for linear, ffn, attention, attention_output, residual_updates in itertools.product(
        (False, True), repeat=5
    ):
        opts = compile_options(
            linear=linear,
            ffn=ffn,
            attention=attention,
            attention_output=attention_output,
            residual_updates=residual_updates,
        )
        passes = opts.get("post_grad_custom_pre_pass")
        effective = (
            linear,
            ffn,
            attention or attention_output,
            attention_output,
            residual_updates,
        )
        if not any(effective):
            assert passes is None
            continue
        assert isinstance(passes, tuple)
        assert len(passes) == 1
        pass_uuid = passes[0].uuid()
        if effective in seen:
            assert pass_uuid == seen[effective]
        else:
            seen[effective] = pass_uuid

    # Every distinct effective post-grad configuration has a unique FX cache key.
    assert len(set(seen.values())) == len(seen)

    pre_configs = [
        compile_options(linear=False, conv3d=True),
        minimax_h3_vae_convrot_int8_conv3d_compile_options(),
        minimax_h3_vae_convrot_int8_conv3d_compile_options(
            compile_options(linear=False, conv3d=True)
        ),
    ]
    padding, h3, combined = (opts["pre_grad_custom_pass"][0].uuid() for opts in pre_configs)  # type: ignore[index]
    assert h3 == combined
    assert padding != h3
    assert {padding, h3}.isdisjoint(seen.values())


def test_uuid_and_options_are_invariant_to_composition_call_order() -> None:
    direct = compile_options(
        linear=True,
        ffn=True,
        attention=True,
        attention_output=True,
        residual_updates=True,
        conv3d=True,
    )
    order_a = compile_options(
        compile_options(
            compile_options(linear=False, conv3d=True),
            residual_updates=True,
        ),
        ffn=True,
        attention_output=True,
    )
    order_b = compile_options(
        compile_options(
            compile_options(attention_output=True),
            ffn=True,
            conv3d=True,
        ),
        residual_updates=True,
    )

    assert order_a == direct == order_b
    assert direct["allow_buffer_reuse"] is False
    for key in ("post_grad_custom_pre_pass", "pre_grad_custom_pass"):
        passes = [opts[key] for opts in (direct, order_a, order_b)]
        for entry in passes:
            assert isinstance(entry, tuple)
            assert len(entry) == 1
        assert passes[1][0].uuid() == passes[0][0].uuid() == passes[2][0].uuid()  # type: ignore[index]


def test_compile_options_compose_without_mutating_or_duplicating() -> None:
    def existing_before(_graph: torch.fx.Graph) -> None:
        pass

    def existing_after(_graph: torch.fx.Graph) -> None:
        pass

    original_options: dict[str, object] = {
        "max_autotune": True,
        "post_grad_custom_pre_pass": [existing_before],
        "pre_grad_custom_pass": existing_before,
    }

    combined = compile_options(original_options, ffn=True, conv3d=True)
    repeated = compile_options(combined, ffn=True, conv3d=True)

    assert original_options["post_grad_custom_pre_pass"] == [existing_before]
    assert original_options["pre_grad_custom_pass"] is existing_before
    assert combined["max_autotune"] is True
    assert "allow_buffer_reuse" not in combined

    post_passes = combined["post_grad_custom_pre_pass"]
    assert isinstance(post_passes, tuple)
    assert len(post_passes) == 2
    assert post_passes[0] is existing_before
    assert isinstance(post_passes[1], _PiperPostGradPass)
    assert repeated == combined

    pre_passes = combined["pre_grad_custom_pass"]
    assert isinstance(pre_passes, tuple)
    assert len(pre_passes) == 2
    assert pre_passes[0] is existing_before
    assert isinstance(pre_passes[1], _PiperPreGradPass)

    # Merging into an options dict with trailing unrelated passes updates the Piper pass in place.
    with_trailing = dict(combined)
    with_trailing["post_grad_custom_pre_pass"] = (*post_passes, existing_after)
    merged = compile_options(with_trailing, attention=True, residual_updates=True)
    merged_passes = merged["post_grad_custom_pre_pass"]
    assert isinstance(merged_passes, tuple)
    assert len(merged_passes) == 3
    assert merged_passes[0] is existing_before
    assert merged_passes[2] is existing_after
    assert isinstance(merged_passes[1], _PiperPostGradPass)
    assert merged_passes[1].ffn is True
    assert merged_passes[1].attention is True
    assert merged_passes[1].residual_updates is True
    assert merged["allow_buffer_reuse"] is False


def test_hooks_deduplicate_and_order_deterministically() -> None:
    hook_a = GraphPassHook("a", lambda _g: None)
    hook_b = GraphPassHook("b", lambda _g: None)

    ab_pre = add_pre_grad_hook(add_pre_grad_hook(None, hook_a), hook_b)
    ba_pre = add_pre_grad_hook(add_pre_grad_hook(None, hook_b), hook_a)
    assert ab_pre == ba_pre
    assert ab_pre["pre_grad_custom_pass"][0].uuid() == (  # type: ignore[index]
        ba_pre["pre_grad_custom_pass"][0].uuid()  # type: ignore[index]
    )


@pytest.mark.parametrize(
    "flag",
    ["linear", "ffn", "attention", "attention_output", "residual_updates", "conv3d"],
)
def test_compile_options_rejects_non_boolean_flags(flag: str) -> None:
    with pytest.raises(TypeError, match="boolean"):
        compile_options(**{flag: 1})  # type: ignore[arg-type]
