# Dense Piper Attention backends

The public `piper_attention` API and `piper_kernels::piper_attention` custom operator
are shared by NVIDIA and AMD. Dispatch inspects operand-device metadata only.
Unsupported targets retain the portable quantized reference. Numerical reductions
here are part of preparation or attention, not implicit input validation.

- `_nvidia/`: the existing SM8x/SM12x Triton kernels, the `cp.async` Gluon kernel
  (`gluon_async_copy.py`), and measured launch policies.
- `_amd/`: ROCm RDNA4 (`gfx1200`/`gfx1201`) Gluon attention and packed V preparation.
- `_quantization.py`: shared FP32 K/V statistics and per-token V quantization.
- `attention/kernels/piper/_amd/`: shared dense/sparse AMD matrix fragments.

## NVIDIA scheduling

The attention recurrence handles full query tiles and a ragged final tile in one
launch. Full tiles retain unmasked access, including Q descriptor loads when selected;
the tail uses masked Q pointer loads and output stores. Aligned grids compile without
the tail branch. Optimized causal traversal visits query tiles in reverse order.
Quantization and statistics preparation use separate launches.

Exact SM120 uses Q64 tiles for causal attention and Q128 for non-causal attention,
with K64 tiles and four warps. D128 splits PV into two FP32 D64 accumulators.
Non-causal attention derives its V log-scale bound. FP32 numerators remain in
probability-code units until the output epilogue.

- Causal D64 enables loop-invariant code motion only for ragged query grids.
- Causal D128 uses K/V descriptors from 1,024 query tokens and pointer loads below
  that boundary to avoid descriptor setup overhead in short eager calls. Both use
  three pipeline stages.
- Non-causal D128 uses K/V descriptors and two pipeline stages at every length.

Exact SM89 runs the Gluon kernel instead, staging Q, K, and V tiles through
`cp.async` commit groups with Q64/K64 tiles and four warps. It keeps per-thread Q/K
scales and derives the V log-scale bound. Register caps of 168 (D64) and 232 (D128)
let three and two CTAs share an SM. Only the final K64 tile carries masks, so ragged
lengths reuse one compiled kernel. Causal grids start with the longest query rows.

Selection uses host metadata only. See [_nvidia/policy.py](_nvidia/policy.py) for
launch choices and the [benchmark guide](../../../../benchmarks/README.md#attention-tuning-workload-anchors)
for performance measurements.

## RDNA4 behavior

The schedule is four wave32 warps, Q64/K64, with native signed INT8 QK
and unsigned UINT8-by-signed INT8 PV. D64 and D128 support FP16/BF16 outputs,
causal attention, non-causal rectangular attention, ragged tails, arbitrary outer
strides, and GQA/MQA without repeating K/V storage. Dynamic fullgraph compilation
and graph capture retain the same custom operator boundary.

Dense is not sparse attention with a full route list. Dense V has one FP32 scale
per token; sparse V has one per K64 tile. The dense recurrence therefore loads
per-token scale metadata and advances one K64 tile at a time. Q/K preparation
uses the existing Q32/K64 scale groups (`per_warp` in reference/benchmark settings).
FP32 online numerators remain in probability-code units until the final division
by 255 and the denominator. AMD packs probabilities with round-to-nearest-even;
cross-backend bitwise equivalence is not promised.

The softmax consumes 16-column fragments, fusing score scaling into FP32 maximum
and exponential arguments. Probability rounding still uses each logical K64
tile's maximum and each token's V scale; the denominator uses unrounded FP32
probabilities. Interior tiles skip elementwise masks, while key tails and causal
diagonals retain them. Zero QK scales are handled before applying masked infinities.
FP32 FMA and reduction reassociation can change output rounding slightly; bitwise
equivalence with the initial schedule is not promised.

Non-causal V is centered in FP32 and its mean restored in the output. Causal V
is not centered, and each query tile visits only its prefix, masking the diagonal.
The existing Q/K smoothing and group-quantization behavior is unchanged.
All-zero inputs retain the algorithmic scale guards. Value preparation writes
the packed WMMA layout directly, including initialized padding, without an extra
full-size V transpose/repacking allocation.

`gfx1201` has on-device correctness/performance coverage on an RX 9070 XT.
`gfx1200` has offline compiler coverage only. Other AMD architectures are not
enabled by this change. This is a measured schedule, not exhaustive tuning.

## Validation and benchmarking

Use an existing ROCm environment; this repository's default `uv` sources select CUDA.

```shell
PYTHONPATH=src /path/to/rocm-env/bin/python -m pytest -o addopts='' \
  tests/attention/piper_attention tests/attention/test_gqa.py \
  tests/attention/sparse_piper_attention/test_amd_fragments.py

PYTHONPATH=src /path/to/rocm-env/bin/python benchmarks/benchmark_attention.py \
  --sequence 1024 4096 8192 --heads 16 --head-dim 128 --dtype float16 \
  --warmup-ms 30 --measurement-time-ms 150 --json artifacts/rocm-dense.json
```

Repeat with `--causal`, `--head-dim 64`, or `--dtype bfloat16`. The benchmark now
selects native Piper and standard PyTorch SDPA by default on RDNA4. It reports
prepared device execution separately from complete operator wall time, including
preparation, allocation, host launches, and synchronization. The SDPA baseline uses
PyTorch's default backend selection, not the portable quantized Piper reference.
Quality is measured against SDPA on the original floating-point inputs.
Separate profiling of the 1K FP16 D64/D128 causal and non-causal baselines confirmed
`aten::_scaled_dot_product_flash_attention` rather than the unfused math fallback.

Initial, pre-tuning synthetic non-causal FP16 measurements on RX 9070 XT,
batch 1, 16 heads, D128,
PyTorch `2.14.0+rocm10.1.0a20260908`, HIP `7.16.26354`, Triton 3.8.0:

| Tokens | Piper complete wall ms | SDPA complete wall ms | SDPA / Piper |
|---:|---:|---:|---:|
| 1,024 | 0.197 | 0.160 | 0.81x |
| 4,096 | 1.577 | 2.087 | 1.32x |
| 8,192 | 5.632 | 8.101 | 1.44x |

These are warmed p50 operator measurements (30 ms warmup, 150 ms measurement),
using the benchmark's default seed 0, not graph-replay or model-throughput results.
Quantization/preparation overhead
can outweigh the faster recurrence at short lengths. This sweep measured roughly
36 dB SQNR against FP16 SDPA; it does not establish checkpoint-level quality.
No GPU clock or power settings were changed. Results vary with workload and ROCm stack.

### H3-shaped long-sequence comparison

The [MiniMax-H3 configuration](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/transformer/config.json)
uses 56 attention heads of width 128 (attention width 7,168, not the 5,376 hidden
width). The diagnostic sweep below uses batch 1, BF16, non-causal full attention,
and all 56 heads on the same RX 9070 XT. Inputs are seeded Gaussian tensors in
projection-style BNHD storage, with per-head unit-RMS Q/K normalization. They are
not captured H3 activations and do not include learned normalization gains or RoPE.

| Tokens | Initial dense ms | Tuned dense ms | Sparse 100% keep ms | ROCm Flash SDPA ms |
|---:|---:|---:|---:|---:|
| 16,384 | 74.1 | 58.9 | 54.0 | 111.8 |
| 32,768 | 287.8 | 226.7 | 207.1 | 446.9 |
| 65,536 | 1,144.6 | 884.1 | 803.8 | 1,812.3 |
| 100,032 | 2,656.8 | 2,074.9 | 1,878.1 | 4,242.1 |

These are complete operator synchronized-wall medians of five calls following
two warmup calls, including quantization, routing where applicable, allocation,
and launch overhead. Providers run separately without retaining other prepared
states. The initial dense column comes from the preceding baseline run; the other
three columns are measured together after tuning. The SDPA comparison explicitly
selects PyTorch FlashAttention, with no quadratic math fallback. No timing is
extrapolated from fewer heads or shorter sequences.

The retained tuning reduces complete dense latency by about 21-23%, but remains
roughly 9-11% slower than full-keep sparse on these shapes. At 100,032 tokens, prepared
dense execution is 2,046.9 ms versus 1,846.6 ms for sparse: preparation is not the
remaining bottleneck. Peak allocated attention-call memory, including Q/K/V, is
7.42 GiB for dense and 8.30 GiB for sparse; these figures exclude model weights.
Sampled original-input FP32 attention checks (nine queries in three heads, each
attending all keys) measured about 1.5% relative L2 error for dense and 1.7% for
sparse. This is not checkpoint-level quality validation. Dense and sparse retain
different V quantization granularity.
