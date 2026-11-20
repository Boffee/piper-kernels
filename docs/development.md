# Developing Piper Kernels

[README](../README.md) · [Environments](environments.md) ·
[Attention](attention.md) · [Weights](weights.md) ·
[Benchmarks](../benchmarks/README.md)

Use this guide when changing the library: it explains package ownership,
conventions, validation, and testing. Piper Kernels owns operator semantics,
quantized formats, portable references, and optimized kernels; applications and
Piper Offload depend on it, not vice versa.

## Architecture

All package paths below are under [`src/piper_kernels`](../src/piper_kernels).

| Area | Responsibility |
|---|---|
| `attention/` | Dense, sparse, and SageAttention2++ APIs, references, and backends |
| `weights/` | Weight formats, quantization arithmetic, packed storage, tensor subclasses, updates, and sharding |
| `linear/`, `conv3d/` | Operators consuming weight formats, references, and backend dispatch |
| `fusions/` | Composed operations and compiler rewrites sharing preparation or intermediates |
| `specializations/` | Explicitly enabled model integrations built from reusable kernels |
| `_distributed/`, `_residual/`, `_input_activations/`, `_inductor_options.py` | Private contracts and compiler options shared across areas |
| `_triton/`, `attention/kernels/` | Triton runtime/target facts, compiler workarounds, and shared GPU primitives |
| `gguf/`, `trellis/`, `stochastic_quantization/` | Packed GGUF decoding, trellis encoding/decoding, and rounding arithmetic |

Code lives where its concept belongs. Concepts shared by several areas without
a single owner (such as residual adds) live in private root packages whose
`__init__.py` imports without Triton and whose `triton.py` holds Triton code.
`_triton/` depends on nothing else in the package.

`weights/` owns each format and its storage lifecycle, separating Triton-free
modules (`_layout.py`, `_rotation.py`, `_quantization.py`) from `triton.py`.
`linear/`, `conv3d/`, and `fusions/` reuse that arithmetic and defer target
launch policy to backend modules. Source-local READMEs own internals for
[dense attention](../src/piper_kernels/attention/piper_attention/README.md),
[dense projection fusion](../src/piper_kernels/fusions/convrot_int8_piper/README.md),
[NVIDIA INT8 linear](../src/piper_kernels/linear/convrot/int8/_nvidia/README.md),
[Conv3D](../src/piper_kernels/conv3d/convrot/int8/README.md),
[trellis-coded weights](../src/piper_kernels/trellis/README.md), and
[MiniMax-H3 encoder Conv3D](../src/piper_kernels/specializations/minimax_h3_vae/conv3d/README.md).
Tests mirror [`tests/`](../tests) structure.

## Code conventions

Target Python 3.13 and the Ruff and Pyright settings in
[`pyproject.toml`](../pyproject.toml). Use typed module boundaries that express
invariants; contain `Any` at external/dynamic boundaries and document why.

Keep one name for each concept across prose, Python orchestration, Triton/Gluon
kernels, PyTorch references, benchmarks, and tests. Preserve public names and
serialized fields, translating external terminology at integration boundaries:

- Start functions with a verb: `build_` assembles a value, `resolve_` settles a
  specification, `compute_` derives from data, `validate_` checks a contract,
  `select_` chooses a plan/implementation, and `require_` raises when support is
  absent. Match framework callbacks and alternate constructors to their conventions.
- Use plural nouns for collections, boolean predicates such as `has_padding`,
  `num_<things>` for scalar counts, `<thing>_index` for indices, and
  `start_<unit>` / exclusive `end_<unit>` for slice bounds.
- Restrict compact notation (`q`, `k`, `v`, `M`, `N`, `K`, `BLOCK_M`) to standard
  matrix roles. Name tensor pointers, derived intermediates, and layouts after
  their canonical concept (for example `key_center`, `key_center_shared`,
  `rotated_mean`, `feature_offsets`), not prototype shorthand (`kc`, `mu_h`, `q0`,
  `d_offsets`). Always make units explicit when ambiguous (bytes, elements,
  tokens, or blocks).
- Keep shared helpers in their owning private module, hide helper-internal
  batching/caching state behind clear function boundaries rather than duplicating
  loop/modulo logic at callers, and collapse repeated multi-line device/reference
  blocks or variant `*_with_*` adapters into parameterized helpers. Prefix
  module/class-private names with `_` and export public surfaces via `__all__`.
  See [VERSIONING.md](../VERSIONING.md) for compatibility boundaries.

## Documentation ownership

Write for the reader's task: lead with behavior and constraints, keep paragraphs
short, and keep one authoritative owner per explanation while linking from other
places:

- **Topic guides (`docs/`)**: Caller contracts, invariants, and end-to-end usage.
  Note caller-visible behavior differences concisely (such as causal vs
  non-causal semantics), and link source-local READMEs for backend internals,
  threshold numbers, or compiler/tiling mechanics.
- **Public API docstrings**: Brief summary followed by signature-ordered inputs,
  options, result shapes/dtypes, mutation, numerical behavior, and backend
  support, distinguishing caller preconditions under the [validation contract](#validation-contract).
- **Module docstrings & source-local READMEs**: Algorithms, packed layouts,
  internal invariants, and schedule rationale.
- **Benchmark & environment guides**: Methodology/diagnostics in [benchmarks](../benchmarks/README.md);
  CUDA, ROCm, and cache setup in [environments](environments.md).
- **This guide & CHANGELOG.md**: Cross-subsystem conventions live here; release
  history belongs in `CHANGELOG.md`.

To stop guides from growing unwieldy over successive PRs:

- **Edit in place instead of appending**: rework or tighten existing prose when
  behavior changes, and delete obsolete qualifiers rather than stacking trailing
  caveats.
- **State library-wide caveats once**: do not repeat generic reminders (quantized
  rounding differences, metadata-only validation, compile coverage vs on-device
  speed) in every section or README; link the owning contract only when a section
  breaks an assumption.
- **Keep one guide per audience**: split out standalone operational setups (as in
  [environments](environments.md)) before mixed concerns clutter a guide.

## Validation contract

Applies across inference, dispatch, compiler/fake paths, and weight wrapping
(including views and device moves):

- Validate metadata (`shape`, `dtype`, `device`, strides, flags, Python config)
  once at the boundary that owns it. Internal stages rely on upstream checks;
  independent entry points still check their own contract, sharing helpers.
- Numerical tensor contents are caller preconditions on all devices (for instance,
  finite positive static scales). Do not add host readbacks (`.item()`, `.cpu()`,
  `bool(tensor)`), synchronizations, scans, validation kernels, or extra
  allocations to check contents in inference, compilation, or weight wrapping;
  validation must remain traceable and CUDA-graph-safe.
- Exporters/loaders run explicit content validation outside these paths. Offline
  quantizers/converters (such as `from_hp(..., compute_per_tensor_scale=True)`)
  may validate derived scales as documented. Algorithmic guards for valid inputs
  (such as all-zero dynamic scaling) are still required.
- Validate metadata before mutating in-place or caller-owned storage (`out=`,
  in-place weight updates, fused residual targets). Never catch and retry a
  failure after a kernel launch may have partially mutated output storage, and
  always restore active device state on exceptions.

## Execution plans and schedules

An **execution plan** resolves one invocation (implementation, preparation,
memory access, launch settings) and may launch multiple kernels. A **schedule**
covers one stage's tiling, warps, pipeline depth, or chunks when tuned or reused
independently. Keep immutable `<Operation>ExecutionPlan` / `<Stage>Schedule`
types and config validation in `_plan.py`, and target/shape selection in
`policy.py` or `_policy.py`.

`select_execution_plan(...)` picks from host metadata; `default_execution_plan(...)`
resolves operand metadata beforehand. Use `execution_plan` (`plan` locally) and
`schedule` / `<stage>_schedule` in runtime args, fields, and benchmark metadata.
Type and policy modules must import without Triton. Every compiler pass's `uuid()`
returns `compute_compile_pass_uuid(self)` from [`_inductor_options.py`](../src/piper_kernels/_inductor_options.py)
and avoids storing configuration on instance state.

## Performance tuning

Use real model shapes as anchors, but base general dispatch decisions on hardware
or workload properties (alignment, useful tiles, parallelism) verified across
nearby shapes. Put model-specific assumptions in opt-in `specializations/`.
Compile or dispatch coverage alone does not prove on-device correctness or
speedup; measure with the [benchmark guide](../benchmarks/README.md).

During kernel prototyping and schedule exploration, batch multiple candidate
variants into one script run on a single anchor workload instead of testing one
change per round trip. Inspect generated IR and ISA (vector loads/stores such as
`buffer_load_b64`/`buffer_store_b64`, VGPR counts, scratch spills, and shared
memory) on a few canary configurations before sweeping the full variant space or
running the full benchmark catalog.

## Checks

In the default CUDA environment:

```shell
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

GPU tests carry the `gpu` marker; pre-commit hides CUDA and runs portable tests,
and [CI](../.github/workflows/ci.yml) checks portable behavior without validating
GPU kernels. See [environments](environments.md) for worker/thread variables,
compiler caches, and [ROCm regressions](environments.md#rocm-regressions).

### Writing tests

Build only what the test exercises, beside the subsystem it covers, using local
`conftest.py` fixtures, temporary directories for filesystem state, and lazy
backend imports. Test shared validation at its owning layer plus representative
caller integration cases, and verify storage ownership and untouched guard
regions for in-place and caller-owned outputs.

Keep expected results independent of the implementation under test, and assert
the intended capability or path (selected kernel, fusion, instruction emission,
or absence of fallback) so silently losing an optimization fails a test. Guard
against compilation, dispatch, and resource regressions with deterministic
counts and compiler invariants (compile/launch counts, spill/stack budgets,
shared-memory limits) rather than wall-clock timings, which belong in
`benchmarks/`.

Keep helpers narrow and important inputs/assertions visible; split helpers that
accumulate flags, format branches, or conditional expectations. Always select
backends through `AcceleratorTarget` (ROCm RX 9070 XT reports CUDA capability
`(12, 0)`); use `exact_sm120_available()` from [`tests/_accelerators.py`](../tests/_accelerators.py)
when targeting NVIDIA SM120 specifically.

Parametrize meaningful format/layout/tail/device boundaries without
multiplying unrelated dimensions. Preserve distinct scenarios, failure checks,
and cold-import isolation when consolidating — matching line or branch coverage
alone does not prove test equivalence. When trimming a matrix, state which
retained tests cover the dropped cases and compare `--durations=20` across
repeated runs with matching hardware, dependency versions, and cache state.

For offline `GluonASTSource` compile tests, include
`[["tt.divisibility", 16]]` in pointer argument `attrs` to match contiguous
PyTorch allocations so compiler vectorization and spill checks reflect live
launches. Gate broad ad-hoc compile sweeps on a small canary subset and
parallelize them across CPU workers (`ProcessPoolExecutor`); in permanent test
suites, avoid multiplying identical-ISA targets (such as `gfx1200` and
`gfx1201`) across every option flag when one target already covers variant
codegen.

## Commits and pull requests

Use standard Git commits: an imperative subject `<= 72` chars with no type
prefix (`feat:` etc.), a blank line, and a 72-column body explaining why. Never
commit generated outputs, benchmark artifacts, caches, checkpoints, or secrets.

Split large cross-cutting refactors cleanly along contract or subsystem boundaries
(for example, updating an operator's contract before downstream fusions or
sibling operators) into a reviewable stack, removing superseded code in its
owning change instead of adding throwaway intermediate states. Skip hooks on an
intermediate stacked commit only when downstream updates intentionally follow in
the next commit, and verify full repo checks at the stack tip.

Lead PR descriptions with the problem, then cover **Change** (design and what is
removed), **Compatibility** (API, metadata, rounding/accumulation, storage changes),
and **Validation** (commands, results, hardware/workloads, portable/compile vs
on-device coverage, and what remains unvalidated such as untested GPU targets or
OS paths; follow [benchmarks](../benchmarks/README.md) for performance claims).
Link related issues and PRs.

Add a **`CHANGELOG.md`** entry under `## [Unreleased]` (`Added`/`Changed`/`Fixed`,
with `Breaking:` prefix when breaking) whenever callers can observe API, behavior,
numerical, performance, mutation, or storage changes. Keep entries to 1–2 short
sentences on caller impact and hardware/measurement scope; update an existing
unreleased entry in place when that feature changes again rather than stacking
another bullet, and leave kernel mechanics and benchmark sweeps in PRs or READMEs.

## Releases

See [VERSIONING.md](../VERSIONING.md) for compatibility policy and the release checklist.
