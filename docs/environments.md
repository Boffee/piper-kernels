# Accelerator environments and caches

[README](../README.md) · [Development](development.md) ·
[Benchmarks](../benchmarks/README.md)

The repository's `uv` sources target the default CUDA development environment. Use
separate environments for CUDA and ROCm stacks, and keep persistent compiler
caches on a local filesystem.

## CUDA

```shell
uv sync --dev
```

Set `PYTEST_XDIST_AUTO_NUM_WORKERS` when adjusting parallel worker memory, or pass
`-n0` (`--pdb` does so automatically) to run pytest serially. CPU runs cap at
`16` workers and visible-accelerator runs cap at `8`. Tests marked with
`large_device_memory` serialize automatically. CPU execution defaults to one
thread per process (`OMP_NUM_THREADS` / `MKL_NUM_THREADS` override this before
pytest starts).

## ROCm

For ROCm development, create an isolated environment with matching ROCm PyTorch,
Triton, package extras, and the `test` group dependencies from
[`pyproject.toml`](../pyproject.toml). Do not use `uv sync` to provision it.

Validated Linux RX 9070 XT (`gfx1201`) setup with stable ROCm 10.1.0, PyTorch
2.14, and matching Triton:

```shell
uv --no-config venv --python 3.13 .venv-rocm
uv --no-config pip install --python .venv-rocm/bin/python \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1201]==2.14.0+rocm10.1.0"
uv --no-config pip install --python .venv-rocm/bin/python --group test \
  "torchao==0.18.0" "numpy>=2"
```

PyTorch 2.15 for ROCm 10.0 (`torch==2.15.0+rocm10.0` from PyTorch's `rocm10.0`
index) with Triton 3.9 also passes the regressions. Attention uses RDNA4 codegen
workarounds there ([dense implementation guide](../src/piper_kernels/attention/piper_attention/README.md#rdna4-behavior)),
and ConvRot INT8 linear/FFN kernels remain 2–5% slower with Triton 3.9 than 3.8.

AMD dispatch accepts Linux and Windows within the same architecture limits,
though on-device INT8 validation currently covers Linux RX 9070 XT (`gfx1201`)
only; offline compiler checks for other targets do not establish hardware
correctness or performance. Consult the operator guide and run regressions
before relying on unmeasured targets.

### ROCm regressions

Run the focused RDNA4 suite serially with the ROCm environment's Python (on
Windows, use `Scripts/python.exe`):

```shell
/path/to/rocm-env/bin/python scripts/run_rocm_regressions.py --junitxml=artifacts/rocm-results.xml
```

The [runner](../scripts/run_rocm_regressions.py) verifies environment metadata
and requires native RDNA4 attention, INT8 linear/Conv3D, and sparse
projection/output backends, failing rather than skipping when absent. Pass pytest
selectors such as `-k sparse_gqa` to narrow a run. The opt-in
[ROCm workflow](../.github/workflows/rocm.yml) runs this script on a provisioned
RDNA4 runner.

## Compiler caches

Triton and PyTorch Inductor manage persistent caches at their default locations
or via `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR`. Pytest retains caches
across runs; Linux CI persists them per platform and dependency lockfile and
prints `--durations=20` for slow cases.

Point both cache variables at a local filesystem supporting executable mappings
(DLL loading on Windows), with access restricted to the current user; network
and distributed filesystems can fail under concurrent compilation (see
[the upstream Triton issue](https://github.com/triton-lang/triton/issues/11512)).
To reset caches cleanly when no processes are running, delete the target
directories or point both variables at fresh empty paths.
