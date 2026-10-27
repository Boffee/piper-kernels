# Developing Piper Kernels

[README](../README.md) · [Attention](attention.md) · [Weights](weights.md) ·
[Benchmarks](../benchmarks/README.md)

Use this guide when changing the library: it explains ownership, implementation
contracts, and how to validate a change.

Piper Kernels owns operator semantics, quantized representations, portable
references, and optimized implementations. Applications and Piper Offload depend
on it; it does not depend on either or own model loading and offloading policy.

## Architecture

All package paths below are under [`src/piper_kernels`](../src/piper_kernels).

| Area | Responsibility |
|---|---|
| `attention/` | Dense, sparse, and SageAttention2++ APIs, references, and backends |
| `weights/` | Packed storage, tensor subclasses, quantization, updates, and sharding |
| `linear/`, `conv3d/` | Operators that consume weight formats, references, and backend dispatch |
| `fusions/` | Composed operations and compiler rewrites that share preparation or intermediates |
| `specializations/` | Explicitly enabled model/workload integrations built from reusable kernels |
| `_triton/`, `attention/kernels/` | Shared GPU primitives and target descriptions |
| `gguf/`, `stochastic_quantization/` | Packed GGUF decoding and reusable rounding arithmetic |

Weight representation and execution are separate concerns: `weights/` owns
storage and its lifecycle; linear and convolution operators own execution.
The package consumes supplied GGUF blocks and metadata rather than parsing model
repositories. Backend-specific modules own launch policy; public operators and
compiler rewrites use those choices rather than duplicating them.

The source-local READMEs explain implementation details for
[dense attention](../src/piper_kernels/attention/piper_attention/README.md),
[dense projection fusion](../src/piper_kernels/fusions/convrot_int8_piper/README.md),
[NVIDIA INT8 linear](../src/piper_kernels/linear/convrot/int8/_nvidia/README.md), and
[Conv3D](../src/piper_kernels/conv3d/convrot/int8/README.md).
Tests follow the operator and format structure in [`tests/`](../tests).

## Code conventions

Target Python 3.13 and the Ruff and Pyright settings in
[`pyproject.toml`](../pyproject.toml). Use typed interfaces between modules;
structured values shared across stages should have types that express their
fields and invariants. Contain `Any` at external or dynamic boundaries and
explain why it is needed there.

Keep one name for a concept in code and prose. Preserve documented public names,
serialized fields, and external names required for compatibility; translate
external terminology at its integration boundary. For new interfaces:

- Functions start with a verb describing their operation. Use `build_` to
  assemble a value, `resolve_` to settle a specification, `compute_` to derive a
  value from data, and `validate_` to check a contract. `select_` chooses an
  implementation or plan; `require_` raises when the requested support is
  unavailable. Keep framework callbacks and alternate constructors consistent
  with their established conventions.
- Variables name the values they hold. Use plural nouns for collections and
  predicates such as `has_padding` for booleans. Scalar counts use `num_<things>`,
  individual indices use `<thing>_index`, and slice boundaries use `start_<unit>`
  and exclusive `end_<unit>`.
- Mathematical and kernel notation such as `q`, `k`, `v`, `M`, `N`, `K`, and
  `BLOCK_M` is appropriate in numerical code. Make units explicit when a value
  could mean bytes, elements, tokens, or blocks; ordinary orchestration uses
  descriptive names.
- Put shared internal helpers in the private module that owns them, with names
  that callers can import without extending a public module's API. Prefer a
  leading underscore on symbols used only within their module or class. A
  module's `__all__`, when present, lists the names it intends callers to use.
  Package compatibility follows [VERSIONING.md](../VERSIONING.md), including
  its distinction between public APIs and private modules.

## Documentation ownership

Write for the reader's task. Lead with the behavior or decision they need, then
explain the relevant constraints. Use plain language, concrete terms, short
paragraphs, and small examples. Keep explanations concise while preserving the
contract and important numerical distinctions. Remove repetition and details
that do not help the reader use or maintain the code.

Topic guides explain subsystem concepts, ownership, invariants, and how APIs
work together. Public API docstrings state the behavior and constraints needed
to make a valid call and interpret its result, including relevant tensor layouts,
numerical behavior, mutation, and backend support. Reuse an existing contract
when it already covers the requirement.

Module docstrings and source-local READMEs explain algorithms, packed layouts,
internal assumptions, and execution-plan rationale. Benchmark tools and their
help own specialized experiments and measurement controls; the benchmark guide
owns shared methodology. Preserve useful information with its owner and link
to it, rather than maintaining the same explanation in several places.
Implementation details belong in caller-facing documentation when they affect
valid usage or observable behavior.

## Validation contract

This is a library-wide API contract, including inference, dispatch, compiler
rewrites, fake/meta implementations, and weight wrappers constructed or
reconstructed from supplied storage, including views and device moves.

Assign each metadata or configuration check to the boundary that owns the
contract. Internal stages can rely on guarantees already established along
their call path; independent entry points still need their required checks.
Share validation when several entry points own the same contract. Do not repeat
checks throughout a pipeline solely because values pass through another helper.

Validation may inspect host metadata: shapes, dtypes, devices, layouts/strides,
gradient flags, and Python configuration. Keep checks for supported storage and
operations. Numerical tensor contents are caller preconditions on every device.
For example, a ConvRot INT8 static input scale must be finite and positive;
the API does not promise runtime rejection of zero, negative, NaN, or infinite
scales.

These paths must not inspect tensor contents solely for validation. That rules
out host readbacks (`.item()`, `bool(tensor)`, `.cpu()`), synchronization,
tensor scans/reductions, device assertions, validation kernel launches, and
temporary device allocations for that purpose. Validation must work without
tensor contents during tracing and fake/meta execution and must not introduce
barriers to CUDA graph capture.

Exporters, checkpoint loaders, and other callers own required numerical
validation at ingestion. A dedicated tensor-content validation API must be
explicitly invoked outside inference, compilation, and weight wrapping, never
run implicitly in those paths.

Offline quantization/conversion is distinct from wrapping supplied storage.
Content checks are allowed as documented parts of those conversions: for
example, NVFP4 `from_hp(..., compute_per_tensor_scale=True)` derives a global
scale from the weight values and rejects a non-finite maximum. This does not
permit implicit content validation in inference, compiler, or wrapper paths.

This boundary avoids synchronization and extra inference work. It does not
remove algorithmic guards: deriving a usable dynamic scale for an all-zero
input is required handling of valid input.

## Execution plans and schedules

An **execution plan** contains the resolved choices for one invocation:
implementation, preparation and fusion, memory access, and launch settings.
It can execute one kernel or several. A **schedule** describes how work is
organized, such as tiles, warps, pipeline stages, or chunk sizes. Give a stage
its own schedule when it is independently selected, reused, or tuned; a flat
plan is otherwise sufficient.

Immutable `<Operation>ExecutionPlan` and `<Stage>Schedule` values live in
`_plan.py`, with their configuration validation and reporting. Target/shape
selection belongs in `policy.py` or `_policy.py`. Vendor-specific plans may
extend a shared plan. The [INT8 plans](../src/piper_kernels/linear/convrot/int8/_plan.py)
and [NVIDIA policy](../src/piper_kernels/linear/convrot/int8/_nvidia/policy.py)
show this division.

`select_execution_plan(...)` selects from host metadata;
`default_execution_plan(...)` resolves operand metadata before selection.
Runtime parameters and stored fields use `execution_plan` (a local variable may
be `plan`); independently selected stages use `schedule` or `<stage>_schedule`.
Benchmark metadata uses the same distinction. Document changes to serialized
fields so saved measurements remain interpretable.

Type and policy modules must import without Triton. Compiler passes include
implementation source files in their cache keys; keep that tracking complete
when definitions move, including plan and policy modules.

Sparse Piper's backend specifies the final V storage through
[`ValueFormat`](../src/piper_kernels/attention/kernels/sparse_piper/layout.py).
Standalone preparation and fused projections write that format directly. The
projection custom op carries its format as a static argument, shared by runtime
allocation, fake tensors, and compiler metadata. Q/K storage, scale groups,
logical K64 route indices, and prequantization routing/coarse summaries retain
their common contracts.

Every sparse-attention backend binds immutable sequence-global K/V once.
The bound launch owns those tensors and its backend state: AMD's folded scale
table or NVIDIA TMA's K/V descriptors. Each call supplies query-local Q, scales,
routes, output, and optional coarse output/gate. Rebind after changing context
tensors; there is no implicit tensor cache or second V representation. The
[`prepared-state contracts`](../src/piper_kernels/attention/sparse_piper_attention/_prepared.py)
define the ownership and query-window boundaries.

## Performance tuning

Real model shapes are useful benchmark anchors. General dispatch decisions
need a hardware or workload explanation, such as useful tile count, alignment,
or available parallelism, rather than matching a model's exact dimensions or
the endpoints of a benchmark sweep. Check representative nearby shapes before
generalizing a measured improvement. Explicit model assumptions belong in
opt-in `specializations/`, where their scope is visible to callers.

Distinguish compiler coverage from on-device correctness and performance
evidence. Successful dispatch or compilation on an architecture is not evidence
that its schedule was measured there. Use the [benchmark guide](../benchmarks/README.md)
for reproducible runs and backend notes for the rationale behind current policy.

## Checks

For the repository's default CUDA development environment:

```shell
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

Tests use `pytest-xdist` with work stealing: larger initial batches reuse compiled
kernels within each worker, and idle workers take pending tests from busier workers.
CPU runs use at most 16 workers and runs with a visible accelerator use at most 8.
Set `PYTEST_XDIST_AUTO_NUM_WORKERS` to adjust for device memory, or pass `-n0`
to debug serially (`--pdb` does so automatically).
Tests that allocate gigabytes of device memory or spawn extra GPU processes
use `@pytest.mark.usefixtures("large_device_memory")` to serialize that work.
CPU operations default to one thread per process to avoid competing thread pools
across workers. Set `OMP_NUM_THREADS` or `MKL_NUM_THREADS` before pytest to override;
child processes inherit these settings.

GPU tests use the `gpu` marker and require a compatible accelerator. The
pre-commit hook hides CUDA and runs portable tests; pytest exercises
installed GPU backends when CUDA is visible. [CI](../.github/workflows/ci.yml) checks the
portable suite, tooling, and distributions on Linux, plus portable attention and
Triton selection on Windows. A passing portable run does not validate GPU code.

Triton and PyTorch Inductor manage their own persistent compiler caches. Pytest
uses their default locations or honors `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR` from the environment. The suite does not create a
separate cache or clear compiled artifacts on exit, so later runs can reuse them.

Configure cache directories on a local filesystem that supports executable
mappings (DLL loading on Windows), with enough free space and access restricted
to the current user. These caches contain native code that the test process
loads. Concurrent compilation on network/distributed filesystems can fail even
when executable mappings work; see [the upstream Triton report](https://github.com/triton-lang/triton/issues/11512).

For a cold compilation run, point both cache variables at new, empty directories.
To reset existing caches or reclaim space, remove the selected directories only
when no processes are using them. The next run rebuilds missing artifacts.
Temporary-directory cleanup and reboot retention depend on the chosen filesystem
and operating-system configuration.

### Writing tests

Build only what a test exercises. Put tests beside the operator or subsystem they
cover, prefer fixtures in local `conftest.py` files, and isolate filesystem state
with temporary paths. Load optional backends and fixture dependencies when the
test needs them. Test shared validation at its owning layer, with representative
integration cases for each caller.

Extend existing cases while preserving numerical oracles, storage invariants, and
distinct failure checks. Keep helpers focused on shared mechanics. Parametrize
meaningful format, layout, and device boundaries without multiplying cases that
execute the same path. Related checks may share a process when their isolation
requirements still hold. Fresh-process import checks must preserve their cold
import order and keep forbidden imports blocked throughout the check.
When reducing a matrix, name the retained numerical or failure tests that cover
the omitted combinations. Use `--durations=20` to distinguish fixture cost from
test execution. Compare repeated pytest runs with matching workers, threads,
hardware, and caches; fewer cases alone do not establish a speedup.

## Accelerator environments

Package installation and platform prerequisites are in the
[README](../README.md#installation). The repository's `uv` sources select CUDA.
For ROCm development, use a separate environment with matching ROCm PyTorch and
Triton, the relevant package extras, and the `test` group dependencies from
[pyproject.toml](../pyproject.toml). Do not use the checkout's `uv sync` to
provision that environment.

For Linux RX 9070 XT development, use stable ROCm 10.1.0 with PyTorch 2.14
and the matching Triton selected by its wheel. From this checkout, create a
fresh environment and install the stack before the test dependencies:

```shell
uv --no-config venv --python 3.13 .venv-rocm
uv --no-config pip install --python .venv-rocm/bin/python \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1201]==2.14.0+rocm10.1.0"
uv --no-config pip install --python .venv-rocm/bin/python --group test \
  "torchao==0.18.0" "numpy>=2"
```

The device extra installs the matching ROCm libraries; torchvision and torchaudio
are not required. Keep nightly experiments in a separate pinned environment.
Run the regressions below and representative benchmarks before promoting a new
stack to the development or CI default.

AMD dispatch accepts Linux and Windows with the same architecture limits;
existing on-device validation is Linux-only. The documented INT8 hardware
validation is RX 9070 XT (`gfx1201`); compiler coverage for other accepted targets
does not establish hardware correctness or performance. Run the regressions
below before relying on Windows execution. Consult the relevant operator guide
and backend policy for supported targets.

### ROCm regressions

From the checkout, run the focused RDNA4 suite using that environment's Python:

```shell
/path/to/rocm-env/bin/python scripts/run_rocm_regressions.py --junitxml=artifacts/rocm-results.xml
```

On Windows, use its `Scripts/python.exe` instead. The runner imports this
checkout, reports environment and device versions, and requires native RDNA4
attention, INT8 linear/Conv3D, and sparse projection/output backends. Missing
hardware or backends fail the run instead of silently skipping it.

The [runner](../scripts/run_rocm_regressions.py) defines the regression selection,
covering numerical behavior, GQA, static/dynamic scales, fusions, dynamic
compilation, and graph capture. It runs serially to bound VRAM and isolate
kernel-cache assertions. Additional pytest arguments select subsets, for
example `-k sparse_gqa`.

The opt-in [ROCm workflow](../.github/workflows/rocm.yml) runs this script on a
provisioned RDNA4 runner, nightly or by manual dispatch.

## Releases

[VERSIONING.md](../VERSIONING.md) defines public compatibility and the release
process.
