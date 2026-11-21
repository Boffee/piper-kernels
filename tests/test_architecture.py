"""Repository architecture and boundary ratchets.

Keep these non-increasing ceilings strict: lower counts and remove allowlist
entries whenever cleanup reduces legacy surface area or private imports, and
never raise a ceiling without explicit justification and review.
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "piper_kernels"
_TESTS_ROOT = _REPO_ROOT / "tests"
_BENCHMARKS_ROOT = _REPO_ROOT / "benchmarks"

# Legacy surface-area ceilings in src/piper_kernels (lower as refactors land).
_MAX_FUSION_PACKAGES = 7
_MAX_CUSTOM_OPS = 52
_MAX_COMPILER_PASSES = 2
_MAX_PATTERN_MATCHER_PASSES = 0

# Legacy tests that inspect src/ ASTs (empty: zero AST tests outside this file).
_TEST_AST_ALLOWLIST: frozenset[str] = frozenset()

# Only the ptxas spill-sidecar helper may force a single recovery compile on sidecar miss.
_TRITON_ALWAYS_COMPILE_ALLOWLIST = frozenset({"tests/_nvidia_resources.py"})

# Legacy private piper_kernels imports in benchmarks/ (lower/delete on cleanup).
_BENCHMARK_PRIVATE_IMPORT_MAX: dict[str, int] = {
    "benchmarks/lib/suite_attention.py": 9,
    "benchmarks/lib/suite_pipeline.py": 11,
    "benchmarks/lib/suite_weights.py": 6,
    "benchmarks/tune_convrot_int8_linear.py": 2,
    "benchmarks/tune_piper_attention.py": 4,
    "benchmarks/tune_sage_attention_2pp.py": 2,
}

# Legacy private piper_kernels import totals in tests/ (lower as tests decouple).
_MAX_TEST_PRIVATE_IMPORT_FILES = 158
_MAX_TEST_PRIVATE_IMPORTS = 673

# Legacy private-global monkeypatch totals in tests/ (lower as tests decouple).
_MAX_TEST_PRIVATE_MONKEYPATCH_FILES = 55
_MAX_TEST_PRIVATE_MONKEYPATCHES = 313

_PASS_BASE_NAMES = frozenset(
    {
        "CustomGraphPass",
        "CustomInferenceAwareGraphPass",
        "CustomPassBase",
    }
)


def _iter_python_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)


def _callable_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_piper_custom_op_call(node: ast.Call) -> bool:
    if _callable_name(node.func) != "custom_op":
        return False
    name_arg: ast.expr | None = node.args[0] if node.args else None
    if name_arg is None:
        name_arg = next(
            (kw.value for kw in node.keywords if kw.arg == "name"),
            None,
        )
    return (
        isinstance(name_arg, ast.Constant)
        and isinstance(name_arg.value, str)
        and "::" in name_arg.value
    )


def _count_private_piper_imports(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "piper_kernels" and any(part.startswith("_") for part in parts[1:]):
                    count += 1
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] != "piper_kernels":
                continue
            module_is_private = any(part.startswith("_") for part in parts[1:])
            count += sum(
                1 for alias in node.names if module_is_private or alias.name.startswith("_")
            )
    return count


def _private_piper_import_names(tree: ast.AST) -> set[str]:
    """Return local names bound to private piper_kernels modules or symbols."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[0] == "piper_kernels"
                    and any(part.startswith("_") for part in parts[1:])
                    and alias.asname
                ):
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] != "piper_kernels":
                continue
            module_is_private = any(part.startswith("_") for part in parts[1:])
            for alias in node.names:
                if module_is_private or alias.name.startswith("_"):
                    names.add(alias.asname or alias.name)
    return names


def _is_private_string(expr: ast.expr | None) -> bool:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value.startswith("_") or "._" in expr.value
    if isinstance(expr, ast.JoinedStr):
        first = expr.values[0] if expr.values else None
        if (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.startswith("_")
        ):
            return True
        return any(
            isinstance(item, ast.Constant) and isinstance(item.value, str) and "._" in item.value
            for item in expr.values
        )
    return False


def _expr_is_private_target(expr: ast.expr | None, private_imports: set[str]) -> bool:
    current = expr
    while isinstance(current, ast.Attribute):
        if current.attr.startswith("_"):
            return True
        current = current.value
    if isinstance(current, ast.Name):
        return current.id.startswith("_") or current.id in private_imports
    if isinstance(current, ast.Call) and _callable_name(current.func) == "getattr" and current.args:
        return _expr_is_private_target(current.args[0], private_imports) or (
            len(current.args) >= 2 and _is_private_string(current.args[1])
        )
    return False


def _count_private_monkeypatches_in_source(source: str, filename: str = "<string>") -> int:
    """Count private-global monkeypatch/mock.patch calls in one Python source string."""
    tree = ast.parse(source, filename=filename)
    private_imports = _private_piper_import_names(tree)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind: str | None = None
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("setattr", "delattr"):
            kind = "monkeypatch"
        elif _callable_name(node.func) == "patch":
            kind = "patch"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "object"
            and _callable_name(node.func.value) == "patch"
        ):
            kind = "patch_object"
        if kind is None:
            continue

        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        target_expr = node.args[0] if node.args else keywords.get("target")
        if kind == "patch":
            attr_expr = None
        elif kind == "patch_object":
            attr_expr = node.args[1] if len(node.args) >= 2 else keywords.get("attribute")
        else:
            is_string_target = (
                isinstance(target_expr, ast.Constant) and isinstance(target_expr.value, str)
            ) or isinstance(target_expr, ast.JoinedStr)
            attr_expr = (
                None
                if is_string_target
                else (
                    node.args[1]
                    if len(node.args) >= 2
                    else (keywords.get("name") or keywords.get("attribute"))
                )
            )

        if (
            _is_private_string(target_expr)
            or _is_private_string(attr_expr)
            or _expr_is_private_target(target_expr, private_imports)
        ):
            count += 1
    return count


def _count_private_monkeypatches(path: Path) -> int:
    return _count_private_monkeypatches_in_source(path.read_text(encoding="utf-8"), str(path))


def _imports_ast(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "ast" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "ast":
            return True
    return False


def _uses_always_compile(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "always_compile":
            return True
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in {"always_compile", "TRITON_ALWAYS_COMPILE"}
        ):
            return True
    return False


def _count_src_metrics() -> tuple[int, int, int]:
    custom_ops = 0
    compiler_passes = 0
    pattern_matcher_passes = 0
    for path in _iter_python_files(_SRC_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if _is_piper_custom_op_call(node):
                    custom_ops += 1
                elif _callable_name(node.func) == "PatternMatcherPass":
                    pattern_matcher_passes += 1
            elif isinstance(node, ast.ClassDef) and any(
                _callable_name(base) in _PASS_BASE_NAMES for base in node.bases
            ):
                compiler_passes += 1
    return custom_ops, compiler_passes, pattern_matcher_passes


def _fusion_subpackages() -> list[str]:
    fusions_dir = _SRC_ROOT / "fusions"
    return sorted(
        entry.name
        for entry in fusions_dir.iterdir()
        if entry.is_dir() and _iter_python_files(entry)
    )


def test_src_surface_area_ratchets() -> None:
    """Block Cartesian fusion package, custom_op, compiler-pass, and pattern growth."""
    fusion_packages = _fusion_subpackages()
    custom_ops, compiler_passes, pattern_matcher_passes = _count_src_metrics()

    assert len(fusion_packages) <= _MAX_FUSION_PACKAGES, (
        f"Fusion subpackage count grew to {len(fusion_packages)} "
        f"(max {_MAX_FUSION_PACKAGES}); parameterize existing packages instead "
        f"(see AGENTS.md). Found: {fusion_packages}"
    )
    assert custom_ops <= _MAX_CUSTOM_OPS, (
        f"Custom op count grew to {custom_ops} (max {_MAX_CUSTOM_OPS}); "
        "parameterize existing ops unless mutates_args forces a split (see AGENTS.md)."
    )
    assert compiler_passes <= _MAX_COMPILER_PASSES, (
        f"Compiler pass class count grew to {compiler_passes} "
        f"(max {_MAX_COMPILER_PASSES}); unify into a phase-ordered pipeline (see AGENTS.md)."
    )
    assert pattern_matcher_passes <= _MAX_PATTERN_MATCHER_PASSES, (
        f"PatternMatcherPass count grew to {pattern_matcher_passes} "
        f"(max {_MAX_PATTERN_MATCHER_PASSES}); prefer imperative anchor matching "
        "(see AGENTS.md)."
    )


def test_no_new_ast_source_inspection_tests() -> None:
    """Prevent new tests from AST-parsing internal source modules."""
    actual = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _iter_python_files(_TESTS_ROOT)
        if path.relative_to(_REPO_ROOT).as_posix() != "tests/test_architecture.py"
        and _imports_ast(path)
    }

    unexpected = sorted(actual - _TEST_AST_ALLOWLIST)
    assert not unexpected, (
        "New AST-based source-inspection tests are disallowed (see AGENTS.md and "
        f"docs/development.md#writing-tests): {unexpected}"
    )
    stale = sorted(_TEST_AST_ALLOWLIST - actual)
    assert not stale, (
        f"Remove cleaned-up files from _TEST_AST_ALLOWLIST in tests/test_architecture.py: {stale}"
    )


def test_no_always_compile_in_tests() -> None:
    """Prevent tests from bypassing Triton's persistent compiler cache."""
    actual = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _iter_python_files(_TESTS_ROOT)
        if path.relative_to(_REPO_ROOT).as_posix() != "tests/test_architecture.py"
        and _uses_always_compile(path)
    }

    unexpected = sorted(actual - _TRITON_ALWAYS_COMPILE_ALLOWLIST)
    assert not unexpected, (
        "Do not disable Triton's compiler cache with always_compile or TRITON_ALWAYS_COMPILE "
        f"in tests (see docs/development.md#writing-tests): {unexpected}"
    )


def test_benchmark_private_import_ratchet() -> None:
    """Keep benchmarks decoupled from private piper_kernels internals."""
    actual_by_file: dict[str, int] = {}
    for path in _iter_python_files(_BENCHMARKS_ROOT):
        count = _count_private_piper_imports(path)
        if count > 0:
            actual_by_file[path.relative_to(_REPO_ROOT).as_posix()] = count

    unexpected = sorted(set(actual_by_file) - set(_BENCHMARK_PRIVATE_IMPORT_MAX))
    assert not unexpected, (
        "Benchmarks must use public piper_kernels APIs rather than private '_' internals "
        f"(see AGENTS.md): {unexpected}"
    )

    exceeded = {
        rel: (count, _BENCHMARK_PRIVATE_IMPORT_MAX[rel])
        for rel, count in sorted(actual_by_file.items())
        if count > _BENCHMARK_PRIVATE_IMPORT_MAX[rel]
    }
    assert not exceeded, f"Benchmark private import counts exceeded ratchet ceilings: {exceeded}"

    stale = sorted(set(_BENCHMARK_PRIVATE_IMPORT_MAX) - set(actual_by_file))
    assert not stale, (
        "Tighten _BENCHMARK_PRIVATE_IMPORT_MAX in tests/test_architecture.py by removing "
        f"cleaned-up files: {stale}"
    )


def test_test_private_import_ratchet() -> None:
    """Bound whitebox coupling to private piper_kernels internals across tests/."""
    actual_by_file: dict[str, int] = {}
    for path in _iter_python_files(_TESTS_ROOT):
        count = _count_private_piper_imports(path)
        if count > 0:
            actual_by_file[path.relative_to(_REPO_ROOT).as_posix()] = count

    total_files = len(actual_by_file)
    total_imports = sum(actual_by_file.values())

    assert total_files <= _MAX_TEST_PRIVATE_IMPORT_FILES, (
        f"Files importing private piper_kernels symbols grew to {total_files} "
        f"(max {_MAX_TEST_PRIVATE_IMPORT_FILES}); test public contracts and compiler "
        "invariants instead (see AGENTS.md and docs/development.md#writing-tests)."
    )
    assert total_imports <= _MAX_TEST_PRIVATE_IMPORTS, (
        f"Private piper_kernels imports in tests/ grew to {total_imports} "
        f"(max {_MAX_TEST_PRIVATE_IMPORTS}); test public contracts and compiler "
        "invariants instead (see AGENTS.md and docs/development.md#writing-tests)."
    )


def test_private_monkeypatch_ratchet() -> None:
    """Block new private-global monkeypatches across tests/ (see AGENTS.md Rule 8)."""
    actual_by_file: dict[str, int] = {}
    for path in _iter_python_files(_TESTS_ROOT):
        if path.relative_to(_REPO_ROOT).as_posix() == "tests/test_architecture.py":
            continue
        count = _count_private_monkeypatches(path)
        if count > 0:
            actual_by_file[path.relative_to(_REPO_ROOT).as_posix()] = count

    total_files = len(actual_by_file)
    total_patches = sum(actual_by_file.values())

    assert total_files <= _MAX_TEST_PRIVATE_MONKEYPATCH_FILES, (
        f"Files with private-global monkeypatches grew to {total_files} "
        f"(max {_MAX_TEST_PRIVATE_MONKEYPATCH_FILES}); test observable contracts or use "
        "explicit dependency injection instead (see AGENTS.md Rule 8)."
    )
    assert total_patches <= _MAX_TEST_PRIVATE_MONKEYPATCHES, (
        f"Private-global monkeypatches in tests/ grew to {total_patches} "
        f"(max {_MAX_TEST_PRIVATE_MONKEYPATCHES}); test observable contracts or use "
        "explicit dependency injection instead (see AGENTS.md Rule 8)."
    )


def test_private_monkeypatch_detector_catches_all_patterns() -> None:
    """Verify the AST ratchet catches string, attribute, and private-import patches."""
    assert (
        _count_private_monkeypatches_in_source(
            'monkeypatch.setattr("pkg._mod._fn", fake)\n'
            'monkeypatch.setattr(mod, "_helper", fake)\n'
            'monkeypatch.setattr(_backend, "select_linear_backend", fake)\n'
            'monkeypatch.delattr(mod._private, "flag")\n'
            'patch("piper_kernels._backend.select")\n'
            'patch.object(mod, "_internal")\n'
            "from piper_kernels.linear.convrot.int8 import _backend as linear_backend\n"
            'monkeypatch.setattr(linear_backend, "select_linear_backend", fake)\n'
            "from piper_kernels._triton.targets import AcceleratorTarget\n"
            'monkeypatch.setattr(AcceleratorTarget, "from_device", fake)\n'
        )
        == 8
    )
    assert (
        _count_private_monkeypatches_in_source(
            'monkeypatch.setattr(torch.cuda, "get_device_capability", fake)\n'
            'monkeypatch.setattr(sys, "platform", "linux")\n'
            'monkeypatch.setattr("torch.cuda.is_available", lambda: False)\n'
            'monkeypatch.setattr("pkg.mod.public", "_replacement_value")\n'
        )
        == 0
    )


def test_every_native_launch_and_descriptor_has_an_explicit_device_context() -> None:
    """Audit native launch ownership across src/ without importing optional dependencies."""
    unguarded: list[str] = []
    native_calls: list[str] = []

    class Audit(ast.NodeVisitor):
        depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list):
                return
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            guarded = any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func) in ("device_context", "torch.cuda.device")
                for item in node.items
            )
            self.depth += guarded
            self.generic_visit(node)
            self.depth -= guarded

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Subscript) or (
                isinstance(node.func, ast.Name)
                and node.func.id in ("TensorDescriptor", "install_uint8_int8_dot_hook")
            ):
                location = f"{path.relative_to(_SRC_ROOT)}:{node.lineno}"
                native_calls.append(location)
                if not self.depth:
                    unguarded.append(location)
            self.generic_visit(node)

    for path in _iter_python_files(_SRC_ROOT):
        Audit().visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    assert native_calls
    assert not unguarded, unguarded


_MATH_FLOAT_PREDICATES = frozenset({"isfinite", "isnan", "isinf"})
_RAW_MATH_FLOAT_PREDICATE_ALLOWLIST = frozenset({"src/piper_kernels/_validation.py"})


def _raw_math_float_predicates(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    math_aliases: set[str] = set()
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "math":
                    math_aliases.add(alias.asname or "math")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "math":
            for alias in node.names:
                if alias.name in _MATH_FLOAT_PREDICATES:
                    hits.append((node.lineno, f"math.{alias.name}"))
    if math_aliases:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in math_aliases
                and node.attr in _MATH_FLOAT_PREDICATES
            ):
                hits.append((node.lineno, f"math.{node.attr}"))
    return sorted(hits)


def test_no_raw_math_float_predicates_in_src() -> None:
    """Require scalar finiteness checks in src/ to route through _validation.py."""
    actual: dict[str, list[tuple[int, str]]] = {}
    for path in _iter_python_files(_SRC_ROOT):
        hits = _raw_math_float_predicates(path)
        if hits:
            actual[path.relative_to(_REPO_ROOT).as_posix()] = hits

    unexpected = {
        rel: hits
        for rel, hits in sorted(actual.items())
        if rel not in _RAW_MATH_FLOAT_PREDICATE_ALLOWLIST
    }
    assert not unexpected, (
        "Use piper_kernels._validation helpers instead of raw math.isfinite/isnan/isinf "
        f"in src/piper_kernels/ (SymFloat-safe under torch.compile): {unexpected}"
    )
    stale = sorted(_RAW_MATH_FLOAT_PREDICATE_ALLOWLIST - set(actual))
    assert not stale, (
        "Remove unused entries from _RAW_MATH_FLOAT_PREDICATE_ALLOWLIST in "
        f"tests/test_architecture.py: {stale}"
    )
