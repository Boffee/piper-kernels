# Fixed-budget forced-tile experiment

[Benchmark guide](README.md) · [API contract](../docs/attention.md#caller-forced-tiles-experimental)

The measurements below used the prototype on base revision `04f976d`, before
rebasing onto `f2de9cc` (0.7.7). They are retained as experiment evidence;
they have not been rerun as a complete sweep on the rebased implementation.

Reproduce this custom-shape diagnostic with:

```shell
.venv-rocm/bin/python benchmarks/benchmark_sparse_piper_forced.py \
  --seconds 5 10 15 --tiling frame cube --ratios 0.25 --samples 7 --rep-ms 150 \
  --json artifacts/force-tiles/h3-25pct.json
.venv-rocm/bin/python benchmarks/benchmark_sparse_piper_forced.py \
  --seconds 5 10 15 --tiling frame cube --ratios 0.1 0.5 --samples 7 --rep-ms 150 \
  --phases routing public --json artifacts/force-tiles/h3-10-50pct.json
```

Use a provisioned GPU environment; `.venv-rocm` is the local environment used for this
experiment. Synthetic BF16 Q/K/V use B1/H56/D128 and H3's default mean routing. At
1280x768, spatial compression by 16 and 1x2x2 patchification give 960 tokens per latent
frame. At 24 fps, the nearest valid H3 output lengths are 124, 243, and 362 frames
(5.17, 10.13, and 15.08 seconds), corresponding to 37, 72, and 107 latent frames and
35,520, 69,120, and 102,720 video tokens. Each sequence also has 256 text tokens and
stereo audio at 40 latent rows/second/channel, treated as a dense suffix. There is no
conditioning/reference video. Block lengths mask unused rows in partial tiles.

Frame tiling forces all 15 K64 tiles in the query's latent frame. The 4x4x4 cube layout
forces all 60 tiles in its four-frame slab: this guarantees same-frame coverage for
every token in a query tile, while also including the other frames in that slab.
Forced IDs occupy the original per-head budget; ordinary scores fill the remaining
slots. In particular, the five-second cube case uses its entire 10% budget for the
forced slab. Inputs are shared across heads and batch elements. Caller list generation
is outside timing; membership construction is included in routing and public calls.

On the RX 9070 XT (gfx1201), Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, and Triton
`3.8.0+git4cff872c.rocm10.0.0`, seven shuffled paired panels at 25% keep gave these medians.
Routing starts from prepared summaries and includes scoring, membership construction,
and selection, measured with cache-flushed GPU events. Public calls include summary
preparation, quantization, allocation, routing, and attention, measured with synchronized
wall time. The independently timed phases are not additive.

| Tiling | Requested seconds | Routing baseline / forced (ms) | Added routing (ms) | Public baseline / forced (ms) |
|---|---:|---:|---:|---:|
| Frame | 5 | 1.666 / 1.716 | 0.050 | 87.890 / 87.944 |
| Frame | 10 | 4.851 / 4.997 | 0.146 | 293.657 / 293.470 |
| Frame | 15 | 9.739 / 10.012 | 0.273 | 611.936 / 611.425 |
| Cube | 5 | 1.834 / 1.896 | 0.062 | 99.899 / 99.865 |
| Cube | 10 | 4.872 / 5.020 | 0.148 | 294.366 / 293.507 |
| Cube | 15 | 9.827 / 10.119 | 0.292 | 623.155 / 622.928 |

Routing overhead is 2.8–3.4% in this sweep; the complete-call differences are within
0.3% and do not establish a speedup. The empty-force control runs the new path with
zero counts, retaining exactly the baseline routes, and differs from the baseline
public call by at most 0.22%. It separates bookkeeping overhead from the locality
changes caused by forcing different routes. Membership construction alone takes
0.010–0.014 ms. The largest membership bitset is 0.32 MiB; caller IDs/counts add
0.38 MiB in the largest cube case. No per-head membership copy is required.

The 10%/50% sensitivity sweep adds 0.057–0.258 ms to routing. Complete-call median
changes range from -0.51% to +0.12%, and the empty-force control stays within 0.19%.
A separate 15-second frame-layout repeat at 25% gives 9.716 / 9.975 ms for routing
and 608.191 / 608.071 ms for the complete call. Absolute isolated routing timings
vary across keep rates; compare forced and baseline within each paired configuration.
None of these complete-call results establishes a consistent latency penalty or gain.

These are standalone attention measurements, with no model weights, quality evaluation,
VAE, or complete generation. The forced-call prototype uses a separate custom operator
and does not yet participate in projection/output fusion rewrites. Its full fused H3
cost therefore remains unmeasured. Tests cover exact fixed budgets, forced membership,
stable ties, unchanged coarse scores, dense suffixes, GQA, full keep, and changing
IDs/counts under compilation and GPU graph replay without additional specializations.
