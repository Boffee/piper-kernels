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
| `linear/`, `conv3d/`, `moe/` | Operators consuming weight formats, references, and backend dispatch |
| `fusions/` | Composed operations and compiler rewrites sharing preparation or intermediates |
| `_distributed/`, `_residual/`, `_input_activations/`, `_inductor_options.py` | Shared private contracts and compiler options |
| `_triton/`, `attention/kernels/` | Triton target/runtime facts, compiler workarounds, shared GPU primitives |
| `gguf/`, `trellis/`, `stochastic_quantization/` | Packed GGUF decoding, trellis encoding/decoding, rounding arithmetic |
| `specializations/` | Legacy model-specific integrations (must not expand; keep core generic) |

`_triton/` has zero internal dependencies. Shared private root packages import
`__init__.py` without Triton (Triton code lives in `triton.py`), and `weights/`
separates Triton-free math from `triton.py`. Existing `specializations/` is legacy
and must not expand; `trellis/` keeps generic format-level GPU encoding (`from_hp`
and caller-supplied second-moment inputs) alongside its decode kernels, while
calibration capture and model-level bit allocation live in external tooling. `tests/`
mirrors this tree.

## Code conventions

Target Python 3.14 and the Ruff/Pyright settings in [`pyproject.toml`](../pyproject.toml).
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

## Accelerator boundaries and schedules

Keep host orchestration, validation, and identical Triton algorithm loops single-source so
numerical and edge-case fixes land once across accelerators. Introduce `_nvidia/` or
`_amd/` subpackages once a vendor has real specialization—a multi-regime launch schedule,
inline PTX/GCN ISA primitives, or a genuinely divergent GPU kernel (such as Gluon TMA vs.
WMMA)—while keeping trivial one-line constant overrides inline and keeping abstract wrapper
layers (`_interfaces.py`, `_plan.py`, `_dispatch.py`) gated by the Rule of Two (`>= 2`
distinct implementations) and `<= 3` runtime module hops from public entry to kernel launch.

Model multi-accelerator operator packages on
[`linear/convrot/int8/`](../src/piper_kernels/linear/convrot/int8/) by grouping vendor
files by role rather than GPU generation:

- **Shared operator modules:** public dispatch, `_ops.py`, `reference.py`, shared
  `triton.py` (parameterized by `tl.constexpr`), and sibling modules when a file covers
  multiple distinct concerns (such as offline compile harnesses, W4A4 vs. W4A16 paths, or
  separate format kernels in `moe/`).
- **`_nvidia/` and `_amd/`:**
  - `policy.py` — per-architecture schedules (e.g., SM75/SM8x/SM120 or CDNA3/RDNA4),
    importable without Triton (`select_execution_plan(...)`, `default_execution_plan(...)`,
    `execution_plan` / `schedule` naming).
  - `isa.py` (when needed) — vendor inline-assembly or hardware-intrinsic decode/pack
    primitives (such as SM120 `e2m1x2` PTX) isolated from portable Triton code.
  - Divergent kernel modules (such as `gluon_tma.py`, `gluon_async_copy.py`, or `gluon.py`)
    only when the GPU loop genuinely diverges at the programming-model or ISA level.

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

During kernel prototyping, batch multiple candidate variants into one script on a
single anchor workload, and inspect generated IR/ISA (vector loads/stores, VGPRs,
scratch spills, LDS) on a small canary set before sweeping.

## Checks

- **Documentation only.** Check edited links, heading anchors, and named symbols.
  Run examples whose executable steps changed.
- **Operator or kernel.** Exercise the affected path on its target. For a regression,
  follow [Fixing kernel bugs](../AGENTS.md#fixing-kernel-bugs).
- **Compiler rewrite.** Inspect the compiled graph to confirm which operations
  fused or stayed separate. Compare the result with the equivalent eager operation.
  The [FFN rewrite tests](../tests/fusions/ffn/test_compile.py) show both checks.
- **Performance claim.** Measure the claimed metric before and after the change
  on the same workload and hardware. Verify output quality using the
  [benchmark guide](../benchmarks/README.md#measurements-and-outcomes).

For code changes, use the [CUDA](environments.md#cuda) or [ROCm](environments.md#rocm)
setup for your target. With PyTorch 2.15, verify that `triton.__version__` is
3.9.x before testing or benchmarking. In the default CUDA development environment,
run the repository checks below.

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

GPU tests carry the `gpu` marker. Pre-commit and [CI](../.github/workflows/ci.yml)
run portable checks with CUDA hidden. For RDNA4 hardware validation, use the
[ROCm regression runner](environments.md#rocm-regressions).

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

Include `[["tt.divisibility", 16]]` on pointer `attrs` in offline `GluonASTSource`
compile tests to match contiguous PyTorch tensors. Never disable compiler caches
(`always_compile = True` or throwaway cache directories) in permanent tests; cache
derived compiler metrics alongside `TRITON_CACHE_DIR` keyed by the compiled kernel hash
instead. Gate broad compile sweeps on a small canary subset parallelized across CPU
workers, and keep offline compile matrices to minimal distinct codegen paths rather than
Cartesian target/flag products (avoiding multiplying identical-ISA targets such as
`gfx1200` and `gfx1201` across every option flag when one covers variant codegen).

## Commits and pull requests

Use standard Git commits (imperative subject `<= 72` chars without `feat:` prefixes,
72-column body explaining why; never commit artifacts, caches, or checkpoints).

Lead with the problem, the design, and the caller impact. Report the checks and
hardware used and any affected targets or OS paths left unvalidated.

For cleanup or architecture PRs, say what was removed and report the net change
in files and lines under `src/`, `tests/`, and `benchmarks/`. In any PR that
changes an architecture ceiling or allowlist entry, identify the change. Explain
why any new custom operator or compiler pass is needed.

Add 1–2-sentence caller-impact entries to `CHANGELOG.md` under `## [Unreleased]`
(`Added`/`Changed`/`Fixed`, `Breaking:` when breaking) for observable API, behavior,
numerical, performance, mutation, or storage changes; update unreleased bullets in place
and keep internal op names and benchmark sweeps out of the changelog.

## Releases

See [VERSIONING.md](../VERSIONING.md) for compatibility policy and the release checklist.
