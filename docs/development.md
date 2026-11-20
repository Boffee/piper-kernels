# Developing Piper Kernels

[README](../README.md) · [Environments](environments.md) ·
[Attention](attention.md) · [Weights](weights.md) ·
[Benchmarks](../benchmarks/README.md)

Piper Kernels owns operator semantics, quantized formats, portable references, and
optimized kernels; applications and Piper Offload depend on it, not vice versa. Follow
[AGENTS.md](../AGENTS.md) for core scope, orthogonal parameterization, and layering
guardrails.

## Architecture

All package paths below are under [`src/piper_kernels`](../src/piper_kernels):

| Area | Responsibility |
|---|---|
| `attention/` | Dense, sparse, and SageAttention2++ APIs, references, and backends |
| `weights/` | Weight formats, quantization math, packed storage, tensor subclasses, updates, sharding |
| `linear/`, `conv3d/` | Operators consuming weight formats, references, and backend dispatch |
| `fusions/` | Composed operations and compiler rewrites sharing preparation or intermediates |
| `_distributed/`, `_residual/`, `_input_activations/`, `_inductor_options.py` | Shared private contracts and compiler options |
| `_triton/`, `attention/kernels/` | Triton target/runtime facts, compiler workarounds, shared GPU primitives |
| `gguf/`, `trellis/`, `stochastic_quantization/` | Packed GGUF decoding, trellis encoding/decoding, rounding arithmetic |
| `specializations/` | Legacy model-specific integrations (must not expand; keep core generic) |

`_triton/` has zero internal dependencies. Shared private root packages import
`__init__.py` without Triton (Triton code lives in `triton.py`), and `weights/`
separates Triton-free math from `triton.py`. Existing `specializations/` and offline
`trellis/` encoding are legacy and must not expand. `tests/` mirrors this tree.

## Code conventions

Target Python 3.13 and the Ruff/Pyright settings in [`pyproject.toml`](../pyproject.toml).
Contain `Any` at dynamic boundaries and explain why. Use canonical names with explicit
units (`bytes`, `elements`, `tokens`, `blocks`) in kernels rather than prototype
shorthand, prefix internal names with `_`, and export public surfaces via `__all__`
(see [VERSIONING.md](../VERSIONING.md)).

## Documentation ownership

- **`docs/*.md`**: Caller contracts and end-to-end usage only—no internal schedules,
  compiler/tiling mechanics, or threshold lists.
- **Public docstrings**: Inputs, outputs, mutation, backend support, and caller
  preconditions.
- **Module docstrings & comments**: Non-obvious hardware or compiler *why* right next
  to code. Reserve source-local `README.md` files for cross-file binary/math specs
  (such as [`trellis/README.md`](../src/piper_kernels/trellis/README.md)), not
  per-backend summaries.
- Edit prose in place instead of appending caveats, state library-wide caveats once, and
  put release history in `CHANGELOG.md`.

## Validation contract

Applies across inference, dispatch, compiler/fake paths, and weight wrapping (views and
device moves included):

- Validate metadata (`shape`, `dtype`, `device`, strides, flags, Python config) once at
  the owning boundary; internal stages rely on upstream contracts.
- Numerical tensor contents (such as finite positive static scales) are caller
  preconditions: never add host readbacks (`.item()`, `.cpu()`, `bool(tensor)`),
  synchronizations, scans, validation kernels, or extra allocations in inference,
  compilation, or weight wrapping.
- Offline quantizers/converters (such as `from_hp(..., compute_per_tensor_scale=True)`)
  may validate derived scales as documented, and valid-input algorithmic guards (such as
  all-zero dynamic scaling) are still required.
- Validate metadata before mutating in-place or caller-owned storage (`out=`, in-place
  updates, fused residuals), never retry after a launch may have mutated output, and
  restore active device state on failure.

## Execution plans and schedules

Apply the Rule of Two before creating `_plan.py`, `policy.py`, or `_interfaces.py`; keep
single-schedule or constant-tile operators without plan/policy layers. When they exist,
keep plan/policy modules importable without Triton (`select_execution_plan(...)`,
`default_execution_plan(...)`, `execution_plan` / `schedule` naming).

Put Inductor post-grad rewrites into a unified phase-ordered pipeline using anchor-based
FX matching with schema argument binding rather than standalone `*_compile_options`
passes, `2^k` `PatternMatcherPass` loops, or positional arg-count checks. Combine
`compute_compile_pass_uuid(self)` from
[`_inductor_options.py`](../src/piper_kernels/_inductor_options.py) with any active
configuration flags in `uuid()` so unified passes remain cache-safe.

## Performance tuning

Use real workload shapes as anchors, base dispatch on general hardware and workload
properties verified across nearby shapes, and validate on-device speed and quality with
the [benchmark guide](../benchmarks/README.md) rather than compile coverage alone.

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

GPU tests carry the `gpu` marker; pre-commit hides CUDA and runs portable tests, and
[CI](../.github/workflows/ci.yml) validates portable behavior without GPU kernels. See
[environments](environments.md) for worker/thread variables, compiler caches, and
[ROCm regressions](environments.md#rocm-regressions).

### Writing tests

Keep tests narrow, local, and lazily importing backends; share one parametrized suite
across orthogonal formats and variants instead of cloned test directories. Verify storage
ownership and untouched guard regions for in-place and caller-owned outputs.

Assert observable invariants (fused op in compiled graph, compile/launch counts,
spill/stack/shared-memory budgets, numerical accuracy vs an independent reference)
rather than private plumbing (no hardcoded tile tuples, private `_` monkeypatches, or
`src/` AST inspection outside `tests/test_architecture.py`; keep timings in
`benchmarks/`). Preserve distinct scenarios and cold-import isolation when consolidating.

Select backends via `AcceleratorTarget` (ROCm RX 9070 XT reports CUDA `(12, 0)`), and
use `exact_sm120_available()` from [`tests/_accelerators.py`](../tests/_accelerators.py)
for NVIDIA SM120. [`tests/test_architecture.py`](../tests/test_architecture.py) enforces
non-increasing ratchets on surface area and private imports in `tests/` and
`benchmarks/`.

## Commits and pull requests

Use standard Git commits (imperative subject `<= 72` chars without `feat:` prefixes,
72-column body explaining why; never commit artifacts, caches, or checkpoints).

PRs lead with the problem, then cover **Change** (design, what was removed, and
surface-area impact on `src/`/`tests/`/`benchmarks/`: net files and LOC, any new custom
ops/passes with justification, and removed/lowered ratchet entries; cleanup PRs must
reduce net LOC and files), **Compatibility**, and **Validation** (commands, hardware,
unvalidated targets/OS paths).

Add 1–2-sentence caller-impact entries to `CHANGELOG.md` under `## [Unreleased]`
(`Added`/`Changed`/`Fixed`, `Breaking:` when breaking) for observable API, behavior,
numerical, performance, mutation, or storage changes; update unreleased bullets in place
and keep internal op names and benchmark sweeps out of the changelog.

## Releases

See [VERSIONING.md](../VERSIONING.md) for compatibility policy and the release checklist.
