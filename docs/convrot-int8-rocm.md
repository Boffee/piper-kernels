# ROCm ConvRot INT8 validation

## Scope and structure

The NVIDIA-only refactor is commit `6484fd9` on `refactor/convrot-int8-backend`.
The AMD integration builds on that boundary, incorporating preparation and RDNA4
schedules from the older `feature/rocm-convrot-int8` branch at `c26dc83`.

Public validation, custom-op schemas, fake implementations, and compiler rewrites remain
accelerator-independent. Operation-specific dispatch selects `_amd` or `_nvidia`.
Each implementation owns preparation strategy, launch policy, and compiler options.
`_kernels/triton.py` shares portable quantization, update, and signed-INT8 GEMM arithmetic;
the AMD implementation does not import the NVIDIA implementation.

Enabled operations: linear; ordinary/GELU-tanh/SwiGLU preparation; prepared and paired
projections with mixed-precision bias and row-strided output; caller-owned preparation
buffers; dense and low-rank in-place updates including seeded stochastic rounding;
and base Inductor preparation sharing.

GGUF conversion, dequantized-input means, specialized FFN/attention fusions, attention,
and NVFP4 remain outside this ROCm integration. Their support gates were not widened.
The existing general tuning/phase-benchmark utilities still expose NVIDIA launch policy.

## Hardware and runtime

Validated on Linux, 2026-09-05, with an AMD Radeon RX 9070 XT (`gfx1201`):

- Python 3.13.13
- PyTorch `2.13.0+rocm10.0.0`
- `torch.version.hip`: `7.15.26333`
- Triton module version `3.8.0` (Torch requires its matching ROCm build)

These are the recorded local runtime versions, not a guarantee for arbitrary ROCm/Triton
combinations. Device zero was selected with `HIP_VISIBLE_DEVICES=0`; the host's `gfx1036`
integrated GPU is not enabled by this backend.

`gfx942`, `gfx1100`, `gfx1151`, and `gfx1200` have offline compiler coverage only.
The tests check activated preparation and paired GEMM, including generated
`v_mfma_i32` or `v_wmma_i32` instructions. They do not establish hardware correctness or
performance on those devices. Unknown architectures fall back to the PyTorch reference.

## Reproduce correctness checks

Install a ROCm PyTorch distribution and its matching Triton into a separate environment.
Do not use this repository's default `uv sync` for that environment: its development
sources select CUDA. Piper's Linux extra no longer pins a competing Triton version.

With `ROCM_PYTHON` pointing to that environment's Python, run from this worktree:

```sh
"$ROCM_PYTHON" -m pip install 'pytest~=9.0' 'numpy>=2' 'gguf>=0.18,<0.20'
HIP_VISIBLE_DEVICES=0 PYTHONPATH=src "$ROCM_PYTHON" -m pytest -q \
  tests/linear/convrot/int8 \
  --ignore=tests/linear/convrot/int8/test_gguf.py \
  --ignore=tests/linear/convrot/int8/test_nvidia_compile.py
```

The base INT8 run produced **534 passed, 75 skipped**. Skips are explicitly NVIDIA-only
low-level preparation/policy tests; GGUF and offline NVIDIA compilation are excluded
from this ROCm command. The full repository's GPU tests include out-of-scope NVIDIA
operators and are not a ROCm support claim.

Coverage includes all three logical dtypes and group sizes, ragged and power-of-two
widths, wide split preparation, zero/tiny scales, noncontiguous inputs and bias,
full/tail GEMM tiles, paired projections and storage identity, eager/compiled updates,
custom-op checks, and actual Inductor preparation sharing. Eight exact cache-group
regression cases cover M=1023, 1024, 1025, and 2049 with ragged N, paired projections,
and row-strided outputs.

Two numerical details were checked explicitly:

- HIP PyTorch's scalar division uses a rounded FP32 reciprocal. Direct division in
  Triton differed by one ULP for `3.25 / 127`, which could change a BF16 output after
  bias. AMD preparation/update scales now explicitly use reciprocal multiplication.
  CUDA retains its original division expression. A regression test covers full-row,
  chunked, and split AMD preparation.
- The factorized H4 transform and the reference matrix multiply have different FP32
  reduction orders. One code in 99,072 values crossed a half-bin boundary
  (`6.5000024` versus `6.4999943`). FP32 reference comparisons allow at most one code
  and bound the fraction of differing codes; fused/split unactivated preparation
  remains bitwise-equal in the tested cases.

Offline tests also pass with the CUDA environment's Triton 3.7.1. NVIDIA SM89/SM120
preparation and paired projection compile. SM75 preparation compiles; its INT8 projection
retains the pre-existing Triton 3.7.1 expected failure reproduced against upstream.

Combined benchmark-support, backend-selection, and AMD/NVIDIA offline compiler
coverage passes **230 tests with 1 expected failure** (SM75), including the ROCm
benchmark's 13 CPU tests. Run it in the repository's CUDA development environment:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/benchmark_support \
  tests/linear/convrot/int8/test_backend.py \
  tests/linear/convrot/int8/test_amd_compile.py \
  tests/linear/convrot/int8/test_nvidia_compile.py
```

Repository lint, formatting, and type checks also pass. These results are from the
2026-09-05 validation; earlier integration-wide checks are recorded in the
[tuning history](convrot-int8-rocm-tuning.md).

## Current production performance

The retained tuning change reduces the large-tile M cache group from 16 to 8;
the production implementation remains Triton. The **300 dense INT8 TOPS target
has not been reached**: for M=8192, K=6144, N=4096 it requires approximately
1.374 ms for prepared GEMM, excluding preparation. Dense TOPS counts the multiply
and add as two operations; full-linear TOPS below includes preparation.

### Reproduce the phase benchmark

```sh
HIP_VISIBLE_DEVICES=0 PYTHONPATH=src "$ROCM_PYTHON" \
  benchmarks/benchmark_convrot_int8_rocm.py \
  --shape 8192,6144,4096 --shape 129,512,300 \
  --shape 2048,9216,1024 --shape 8192,16384,512
```

The benchmark prepares normal BF16 inputs and weights through the actual AMD
ConvRot path, verifies prepared and full linear against an exact INT32 GEMM
reference before timing, and prints JSON lines. Preparation and prepared GEMM
use caller-owned buffers; full linear retains production allocation behavior.
It reports cache-flushed measurements and sustained graph replay separately.
Graphs remove host launch overhead but do not guarantee faster results: sustained
power limits can make the large compute-bound GEMM slower.

The 2026-09-05 production group-8 anchor run measured preparation 0.3622 ms, GEMM 1.6914 ms
(243.8 dense TOPS), and full linear 1.9926 ms (206.9 effective dense TOPS).
Graph timings were 0.2833, 1.7366, and 2.0198 ms respectively. Short isolated
tuning samples occasionally approached 260 TOPS; they are not substituted for
the production result.

See the [tuning history](convrot-int8-rocm-tuning.md) for the old-branch comparison,
interleaved group-size measurements, and native HIP/hipBLASLt experiments. Those
prototypes are not production paths. No clock, voltage, fan, or power settings were
changed.
