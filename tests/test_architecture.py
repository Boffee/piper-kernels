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
_MAX_FUSION_PACKAGES = 21
_MAX_CUSTOM_OPS = 68
_MAX_COMPILER_PASSES = 18
_MAX_PATTERN_MATCHER_PASSES = 20

# Legacy tests that inspect src/ ASTs (shrink and remove entries over time).
_TEST_AST_ALLOWLIST = frozenset(
    {
        "tests/attention/sparse_piper_attention/test_backend.py",
        "tests/fusions/convrot_int8_sparse_piper/test_backend.py",
        "tests/fusions/test_convrot_int8_backend.py",
        "tests/triton/test_device_launches.py",
    }
)

# Legacy private piper_kernels imports in benchmarks/ (lower/delete on cleanup).
_BENCHMARK_PRIVATE_IMPORT_MAX: dict[str, int] = {
    "benchmarks/benchmark_attention.py": 2,
    "benchmarks/benchmark_convrot_int8.py": 2,
    "benchmarks/benchmark_convrot_int8_conv3d.py": 7,
    "benchmarks/benchmark_convrot_int8_preparation.py": 6,
    "benchmarks/benchmark_integer_pv_dot.py": 2,
    "benchmarks/benchmark_sparse_piper.py": 8,
    "benchmarks/benchmark_sparse_piper_projection.py": 5,
    "benchmarks/benchmark_sparse_piper_scores.py": 5,
    "benchmarks/lib/attention_providers.py": 8,
    "benchmarks/lib/convrot.py": 3,
    "benchmarks/lib/convrot_int8_providers.py": 4,
    "benchmarks/lib/environment.py": 1,
    "benchmarks/lib/sparse_piper.py": 1,
    "benchmarks/lib/suite_attention.py": 10,
    "benchmarks/lib/suite_pipeline.py": 12,
    "benchmarks/lib/suite_weights.py": 8,
    "benchmarks/tune_convrot_int8_linear.py": 3,
    "benchmarks/tune_piper_attention.py": 4,
    "benchmarks/tune_sage_attention_2pp.py": 2,
}

# Legacy private piper_kernels import totals in tests/ (lower as tests decouple).
_MAX_TEST_PRIVATE_IMPORT_FILES = 181
_MAX_TEST_PRIVATE_IMPORTS = 786

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


def _imports_ast(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "ast" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "ast":
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
