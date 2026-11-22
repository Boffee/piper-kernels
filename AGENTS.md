# piper-kernels

Instructions for coding agents working in this repository. Start with
[README.md](README.md) for entry points and [docs/development.md](docs/development.md)
for architecture, validation, and tests.

## Rules

1. **Strict core scope:** Core `piper_kernels` owns reusable runtime operators and
   kernels only—no new single-model shape tables, hardcoded calibration constants, or
   offline encoders; external tools and specializations use public APIs only.
2. **Parameterize orthogonal axes:** Never add a Cartesian `<format>_<variant>_<op>`
   package, compiler pass, or test suite; parameterize format, basis, activation, and
   mutation, splitting a `custom_op` only when PyTorch's schema forces it (distinct
   `mutates_args`).
3. **Rule of Two (`<= 3` hops):** Do not add `_interfaces.py`, `_plan.py`, `_policy.py`,
   `_dispatch.py`, or vendor subpackages (`_nvidia/`, `_amd/`) without `>= 2` genuinely
   distinct implementations; keep `<= 3` module hops from public entry to kernel launch
   and delete unused layers.
4. **One compiler pipeline; anchor-match FX:** Put post-grad rewrites in one
   phase-ordered pipeline (hash config flags into `uuid()`), and match FX graphs
   imperatively from anchor ops via schema argument binding—never `2^k`
   `PatternMatcherPass` loops or positional arg-count checks.
5. **Specialize only the GPU loop:** Hardware divergence belongs inside
   `@triton.jit`/`@gluon.jit` (parameterizing shared stages and peeled tails with
   `constexpr`); write host orchestration, chunk/window drivers, peer exchange, and
   tensor-subclass dispatch once.
6. **Validate and dispatch once:** Validate metadata and resolve backend/plan once at
   the boundary; follow the [validation contract](docs/development.md#validation-contract)
   (no implicit content checks or readbacks, validate before mutating, never retry
   partial mutations, restore device state on failure).
7. **Remove what a refactor supersedes.** Prefer fewer lines and files. Removing
   duplicated logic within one file counts as cleanup. Justify any net growth in
   lines or files. When adding a variant, generalize the shared path and remove
   the superseded implementation in the same PR.
8. **Test contracts, not private plumbing:** Assert observable behavior and compiler
   invariants in shared parametrized suites—never hardcoded tile tuples, private-global
   monkeypatches, or `src/` AST checks outside
   [`tests/test_architecture.py`](tests/test_architecture.py). Never raise its ceilings
   without explicit approval; lower them as cleanup lands.

## Fixing kernel bugs

1. Reproduce the failure through the public operator with the affected inputs and
   target. Confirm which backend ran. A passing fallback can hide a broken GPU kernel.
2. For a numerical error, compare with the portable reference for the same
   quantized operation. For a mutation bug, check what changed in caller-owned
   storage and what should have stayed untouched.
3. Trace the failure to the code that owns the behavior and check sibling variants
   for the same cause. Verify the regression test fails for that reason before
   the fix and passes afterward. Report any target you could not exercise.

## Contracts & checks

- Target Python 3.14, PyTorch 2.15, and **Triton 3.9** during development and
  validation (verify `triton.__version__`; see [environments](docs/environments.md)).
- Check [attention](docs/attention.md) and [weights](docs/weights.md) before changing
  operators or rewrites; follow [development guide](docs/development.md) for
  architecture, backend/compiler contracts, tests, checks, commits/PRs, and
  [VERSIONING.md](VERSIONING.md) for compatibility.
- **Reviews:** Identify the failure, impact, and location, and flag Cartesian cloning,
  premature indirection, net surface-area growth on refactors, and whitebox test
  coupling; distinguish contract bugs from documented numerical preconditions.
- Use strong types (`pyright`), measured tuning
  ([performance tuning](docs/development.md#performance-tuning)), and in-place doc edits
  with one authoritative owner.
