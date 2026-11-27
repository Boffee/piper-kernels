# piper-kernels

Instructions for coding agents working in this repository. Start with
[README.md](README.md) for entry points and [docs/development.md](docs/development.md)
for architecture, validation, and tests.

## Rules

1. **Strict core scope:** Core `piper_kernels` owns reusable runtime operators,
   kernels, and generic format-level weight encoders (`from_hp`)—no new single-model
   shape tables, hardcoded calibration constants, calibration/dataset capture, or
   model-level quantization planners; external tools and specializations use public
   APIs only.
2. **Parameterize orthogonal axes:** Never add a Cartesian `<format>_<variant>_<op>`
   package, compiler pass, or test suite; parameterize format, basis, activation, and
   mutation, splitting a `custom_op` only when PyTorch's schema forces it (distinct
   `mutates_args`).
3. **Shallow boundaries (`<= 3` hops):** Keep `<= 3` module hops from public entry to
   kernel launch and delete unused layers. Do not add abstract wrapper layers
   (`_interfaces.py`, `_plan.py`, `_dispatch.py`) without `>= 2` genuinely distinct
   implementations. Introduce vendor subpackages (`_nvidia/`, `_amd/`) once a vendor
   has real specialization (multi-regime schedules, inline PTX/GCN primitives, or
   divergent kernels), grouped by role (`policy.py`, `isa.py`, divergent kernels)—not
   for trivial one-line constant tweaks (see
   [accelerator boundaries](docs/development.md#accelerator-boundaries-and-schedules)).
4. **One compiler pipeline; anchor-match FX:** Put post-grad rewrites in one
   phase-ordered pipeline (hash config flags into `uuid()`), and match FX graphs
   imperatively from anchor ops via schema argument binding—never `2^k`
   `PatternMatcherPass` loops or positional arg-count checks.
5. **Share orchestration and identical Triton loops; isolate hardware specialization:**
   Write host orchestration, chunk/window drivers, peer exchange, and tensor-subclass
   dispatch once, and keep identical Triton algorithm loops in one shared kernel
   parameterized with `constexpr` (never copy-paste a Triton kernel just to change
   tiles). Isolate vendor schedules (`policy.py`), inline PTX/GCN primitives
   (`isa.py`), and genuinely divergent GPU kernels (such as Gluon TMA vs. WMMA) in
   `_nvidia/` and `_amd/`.
6. **Validate and dispatch once:** Validate metadata and resolve backend/plan once at
   the boundary; follow the [validation contract](docs/development.md#validation-contract)
   (no implicit content checks or readbacks, validate before mutating, never retry
   partial mutations, restore device state on failure).
7. **Remove what a refactor supersedes:** Eliminate dead code, duplicated logic, and
   unnecessary call-chain indirection—do not treat splitting a large multi-concern
   kernel module (e.g., separating offline compile helpers, vendor policies/ISA, or
   distinct format kernels while staying within `<= 3` runtime hops) as bloat. Justify
   any net line growth on cleanup/architecture PRs; when adding a variant, generalize
   the shared path and delete the superseded implementation in the same PR.
8. **Test contracts, not private plumbing:** Assert observable behavior and compiler
   invariants in shared parametrized suites—never hardcoded tile tuples, private-global
   monkeypatches, or `src/` AST checks outside
   [`tests/test_architecture.py`](tests/test_architecture.py). Never raise its ceilings
   without explicit approval; lower them as cleanup lands.

## Fixing kernel bugs

1. For a runtime bug, call the public operator with the failing inputs on the
   affected hardware and confirm which backend ran. A passing fallback can hide
   a broken GPU kernel. For a compilation failure, reproduce it with the affected
   compiler target and report whether you also tested the fix on hardware.
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
  duplicated kernel loops, premature abstraction layers (`> 3` hops), public
  surface-area growth on refactors, and whitebox test coupling; distinguish contract
  bugs from documented numerical preconditions.
- Use strong types (`pyright`), measured tuning
  ([performance tuning](docs/development.md#performance-tuning)), and in-place doc edits
  with one authoritative owner.
