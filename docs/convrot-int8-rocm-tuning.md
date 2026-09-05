# ROCm ConvRot INT8 tuning history

These experiments and integration checks were recorded on 2026-09-05 on the RX 9070 XT
(`gfx1201`) with ROCm 10. They are historical measurements, not additional backend
support claims. See the [validation and reproduction guide](convrot-int8-rocm.md)
for the runtime environment, current correctness results, and production benchmark.

## Earlier integration validation

Before the eight cache-group regression cases were added, the base INT8 suite passed
526 tests with 75 skips. The earlier full repository suite in the CPU-only CUDA
environment produced 941 passed, 827 skipped, and 1 expected failure. Lint, formatting,
type checks, and source/wheel builds also passed at that stage. These counts do not
represent a fresh full-repository run after the later tuning and benchmark additions.

## Performance comparison

Compared with the old ROCm branch at `c26dc83`, using the same ROCm environment and GPU,
BF16 inputs, group size 256, no activation or bias. Timings use `triton.testing.do_bench`,
50 ms warmup, 150 ms measurement, median of three median measurements. The end-to-end
call is each backend's raw `run_linear`; allocation behavior is unchanged between runs.

| M × K × N | Old ROCm linear (ms) | Modular AMD linear (ms) |
| --- | ---: | ---: |
| 129 × 512 × 300 | 0.02820 | 0.02832 |
| 8192 × 6144 × 4096 | 1.99765 | 1.99945 |
| 2048 × 9216 × 1024 | 0.32204 | 0.32338 |
| 8192 × 16384 × 512 | 1.99243 | 1.98231 |

These spot checks stay within 0.6% end-to-end, consistent with retaining the previous
ROCm schedules. They are not an exhaustive tuning result. The 6144-width anchor measured
0.3630/0.3628 ms for old/new preparation and 1.7253/1.7237 ms for old/new prepared GEMM.
No NVIDIA hardware was available for a runtime performance comparison.

## Tuning toward 300 dense INT8 TOPS

The target for `M=8192, K=6144, N=4096` is approximately **1.374 ms for
prepared GEMM**. It is not an end-to-end target and has **not been reached**.
The multiply and add each count as one operation; sparse/INT4 marketing rates
are not used in this calculation.

The retained software change reduces the large-tile M cache group from 16 to 8.
Three interleaved comparisons, reversing order between rounds, measured median
prepared GEMM times of 1.7351 ms (group 16) and 1.6972 ms (group 8), a 2.2%
reduction. End-to-end medians were 2.0054 and 1.9935 ms (0.6% reduction).
The 2048x9216x1024 case improved from 0.2136 to 0.2052 ms prepared and
0.3247 to 0.3198 ms end-to-end. The other two representative shapes showed no
material regression. These are small improvements, not a 300-TOPS result.

Sweeps covered tile sizes, 4/8/16/32 warps, K depths, software-pipeline stages,
loop unrolling, workgroup ordering, cache policies, and LLVM schedulers. Explicit
Gluon WMMA layouts, buffer addressing, prefetching, and double-buffered shared
memory were also prototyped. None improved on the retained implementation;
many increased register spilling. Experimental kernels are not production paths.
The existing kernel already emits `v_wmma_i32_16x16x16_iu8`, uses 128-bit global
loads, and has no register spills at the default 128x256x64 / 8-warp schedule.

Sustained load materially affects these measurements. At the unchanged 304 W
board limit, a sustained production GEMM sample reported 298 W and approximately
1.97 GHz. A compute-only experiment (operands loaded once, repeated register-only
WMMA, no real K-streaming) reached 292.3 TOPS with full-range signed INT8, around
2.27 GHz and 298-309 W. A low-magnitude {-1,0,1} version reached 306.8 TOPS at
2.40 GHz. **Neither compute-only result is a GEMM benchmark or a proven hardware
upper bound.** They indicate power/data-activity sensitivity and make the
advertised boost-clock peak an optimistic reference for sustained workloads.
No clock, voltage, fan, or power settings were changed.

### Instruction-level and vendor-library follow-up

Additional experiments on the same device and ROCm 10 environment tested native
HIP wave32/wave64 WMMA, 128-bit shared-memory reads using the
[AMD wide-K technique](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/),
buffer-addressed global loads, single/double shared-memory buffers, and macro tiles
up to 256x512 and 512x256. Correct candidates were checked against exact INT32
matrix multiplication followed by FP32 row/column scaling and BF16 conversion.
None demonstrated an improvement worth replacing the production Triton path.
These native prototypes remain outside the package; no HIP build dependency was
added. Additional LLVM scheduling strategies (`max-ilp`, `max-memory-clause`,
and `iterative-maxocc`) also failed to improve the retained schedule.

A separate hipBLASLt search requested 100 algorithms with up to 256 MiB of
workspace for the anchor shape. All 100 returned INT32-output algorithms passed
exact integer comparison. The fastest used no workspace and measured 1.6316 ms
(252.7 dense TOPS) in a short cache-flushed run, versus 1.8197 ms (226.6 TOPS)
under graph replay. This was **unscaled INT32 output**, not the ConvRot prepared
projection. The INT8-input/BF16-output request with FP32 outer-vector scales
returned zero heuristic algorithms; this describes the tested configuration,
not every possible hipBLASLt configuration or architecture.

The follow-up query requested 1,000 algorithms and returned 780. Screening the
remaining 680 (plus a control) and retesting the eight best candidates did not
find a better library path. A separate 780-algorithm search with M and N swapped
was also slower; a transposed intermediate would additionally require a layout
conversion before satisfying the prepared-projection output contract.

Combining native wave64 WMMA with direct-to-register weight loads did yield a
small gain. The best tested configuration was a 128x256x128 macro tile, four
64-lane waves partitioned 1x4, 238 VGPRs, 16 KiB shared memory, and no spills.
Three interleaved production-data comparisons reversed order between rounds:

| Path | Prepared GEMM (ms / dense TOPS) | Full linear (ms / effective dense TOPS) | Graph GEMM (ms) |
| --- | ---: | ---: | ---: |
| Production Triton, group 8 | 1.6972 / 242.9 | 1.9922 / 207.0 | 1.7448 |
| Experimental native direct-weight load | 1.6674 / 247.3 | 1.9663 / 209.7 | 1.6862 |

Both used the same normal BF16 inputs, actual ConvRot preparation, FP32 scales,
BF16 output, and exact INT32-reference verification. Full linear included fresh
preparation and output allocation. The native path has only aligned/no-bias
prototype coverage and is **not integrated or a new backend support claim**.
Its 1.8% prepared and 1.3% full-linear gain does not yet justify a second compiled
implementation. Vectorized output staging, smaller streamed weight fragments,
additional unrolling, more waves, and Gluon register-order changes did not
improve this candidate. The 300-TOPS objective remains unmet.

Both the native and library experiments used full-range random signed INT8
inputs for initial screening; the interleaved native comparison above instead
used production-prepared normal BF16 data. Short screening timings are not
interchangeable with production-preparation measurements. A library switch does
not currently provide an evidenced route to the 300-TOPS target.
