# ConvRot follow-on experiments

This report starts where the standalone ConvRot linear optimization ends. It
contains benchmark-only experiments that fuse adjacent model operations into
ConvRot preparation or epilogues. None changes production dispatch or public
APIs on this branch.

Read [the incremental adoption roadmap](convrot-adoption.md) first. The stable
ConvRot operator design and core results remain in
[the core optimization report](convrot-optimization.md).

## Decision summary

In this report, **retain** means preserve the benchmark and investigate through
the adoption gates. It does not mean that a production operator or model adapter
exists.

| Experiment | Classification | Decision | Next possible stage |
|:---|:---|:---|:---|
| Packed weights, TMA, and persistence | Kernel-local GEMM | Reject for current shapes | Evidence only |
| `triton_op` composition | Compiler integration control | Useful traceability; reject as automatic fusion | Inform operator registration |
| Paired FC1 output-SwiGLU | Generic operator candidate | Retain prototype; modest gain | Explicit operator API |
| QKV RMSNorm/RoPE | MiniMax-shaped attention boundary | Retain prototype | Explicit API, then attention adapter |
| Gated residual | Generic epilogue plus MiniMax row mapping | Retain selectively | Explicit mutation API, then adapter |
| RMSNorm/AdaLN preparation | MiniMax-shaped input boundary | Retain; eager numerical contract unresolved | Scale/shift first; full RMS only after quality validation |

All latency results are isolated synthetic boundaries. The branch has no real
50-block MiniMax quality, hook, offload, compile, or end-to-end integration
result for these candidates, and their speedups must not be added together.

After the implemented core checkpoint, the GPU was verified at its stock 575 W
power limit and the following exact-boundary ideas were tested as benchmark-only
kernels. None of the files in this section changes production dispatch or public
APIs. They preserve the research needed to implement selected boundaries cleanly
on a new branch.

## Packed weights, descriptors, and persistence: reject

[`benchmark_convrot_gemm_experiments.py`](../benchmarks/benchmark_convrot_gemm_experiments.py)
isolates the prequantized INT8 GEMM with preallocated outputs. It compares the
core checkpoint pointer kernel with:

- a true physical `[K,N]` copy, viewed with strides `(1,N)`;
- host-created Triton tensor descriptors over canonical `[N,K]` storage;
- persistent tile scheduling;
- Blackwell warp specialization; and
- `BN` of 128/256 with `BK` of 64/128.

One reproducible short-window screening run at `M=37,710` gave:

| Projection | Pointer baseline | Physical KN / baseline | Best TMA / baseline |
|:---|---:|---:|---:|
| QKV | 733 TOP/s | 0.733x | 0.956x |
| Attention output | 729 TOP/s | 0.715x | 0.955x |
| MLP FC1 | 725 TOP/s | 0.737x | 0.962x |
| MLP FC2 | 728 TOP/s | 0.718x | 0.953x |

The descriptor kernels compile to real TMA loads and INT8 tensor-core MMA, but
none crosses the 750 TOP/s go/no-go target or beats the pointer kernel. Short
screening medians vary by a few percent, so the table is directional rather
than a claim of sub-percent precision. Two persistent/warp-specialized
configurations also exceed the RTX 5090's 101,376-byte per-block shared-memory
limit. A physical KN copy is decisively worse and
would additionally duplicate 367.5 MiB of H3 weights per layer. Keep canonical
`[N,K]` storage and the existing pointer GEMM.

Reproduce the sweep with:

```shell
uv run python benchmarks/benchmark_convrot_gemm_experiments.py \
  --cases qkv attention-out mlp-fc1 mlp-fc2 \
  --warmup-ms 50 --measurement-time-ms 100
```

## QKV projection plus per-head RMSNorm and partial RoPE: retain

H3 has an unusually clean local boundary:

```text
Q/K/V segment width = 7168 = 28 * 256
head dimension = 128
GEMM BN = 256 = exactly two complete heads
```

[`benchmark_convrot_qkv_epilogue.py`](../benchmarks/benchmark_convrot_qkv_epilogue.py)
rounds each dequantized Q/K result to BF16, performs per-head RMSNorm, rounds
the weighted RMSNorm output to BF16, applies split-half RoPE to the first 96 of
128 channels, and stores the final Q/K/V buffer. Those two BF16 boundaries
match the materialized H3 graph. Fusion removes the separate Q/K BF16
read-write pass while keeping the attention boundary intact.

| M | Core GEMM + Comfy RMS/RoPE | Fused pure Triton | Speedup | Avoided traffic |
|---:|---:|---:|---:|---:|
| 37,710 | 13.4134 ms | 12.4375 ms | 1.078x | 2.162 GB |
| 131,072 | 47.1091 ms | 43.2937 ms | 1.088x | 7.516 GB |

The standalone pure-Triton comparator measured 13.2932 and 46.7261 ms, so the
fused path also wins without relying on Comfy Kitchen. Comfy Kitchen 0.2.28's
[Triton implementation](https://github.com/Comfy-Org/comfy-kitchen/blob/75aa2ab6f9f45575205489b9593cf9fe01a57028/comfy_kitchen/backends/triton/rms_rope.py#L232-L272)
falls back when `rot_dim != head_dim`; H3 uses 96 and 128, respectively. Its
usual NVIDIA comparator here is the native CUDA path. The benchmark checks 257
stratified rows against an independent eager reference and the Comfy path; at
128K it additionally checks the three rows around the first signed-int32 output
offset boundary. Versus Comfy, the winning fused layouts differed in 10 of
3,684,352 sampled Q/K values at 37,710 rows and 3 of 3,727,360 at 128K, with
maximum absolute error 0.00390625 in both runs. Any integration needs an
explicit API such as `linear_qkv_rms_rope(...)`; ordinary linear semantics must
not change. A corrected-boundary row sweep was near parity around
`M=2,048--4,096`, clearly positive at 8,192, and negative at 512, so this is a
guarded specialization rather than a universal path. Default and group-64 tile
orders traded sub-percent leads across repeated large-M runs, so that choice
needs a longer autotuning bucket rather than a hard-coded conclusion.

```shell
uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot_qkv_epilogue.py \
  --rows 37710 --warmup-ms 200 --measurement-time-ms 500
```

## Gated residual output epilogue: retain selectively

H3 materializes attention-out and FC2 projections, then applies a segmented
feature-wise gate in place:

```text
residual += gate[row_ids, :] * round_bf16(projected)
```

[`benchmark_convrot_gated_residual.py`](../benchmarks/benchmark_convrot_gated_residual.py)
uses a small gate table plus `row_ids[M]` and preserves the BF16 projection
boundary exactly. Fusion removes `4*M*N` bytes: the fresh projection write and
its read by the residual kernel. The benchmark prebuilds a synthetic random
`row_ids`; model integration must construct the map once per `mod_segments`
signature/forward and reuse it across blocks, or express H3's contiguous
segment lookup directly. That setup cost is excluded from these timings.
Its materialized comparator is deliberately optimistic: one custom Triton
`addcmul` pass over the dense row map, rather than H3's current launch per
contiguous segment.

| M | Projection | Materialized composite | Fused | Speedup |
|---:|:---|---:|---:|---:|
| 37,710 | Attention output | 4.5903 ms | 4.2906 ms | 1.070x |
| 37,710 | MLP FC2 | 8.4980 ms | 8.4111 ms | 1.010x |
| 131,072 | Attention output | 16.0000 ms | 14.2387 ms | 1.124x |
| 131,072 | MLP FC2 | 29.4912 ms | 27.8871 ms | 1.058x |

Avoided traffic is 0.811 GB per site at 37,710 rows and 2.819 GB at 128K.
The ordinary epilogue is best at 37,710 rows, while consuming the accumulator in
two 128-column halves is best at 65K and 128K. The crossover was not monotonic in
shorter row sweeps, so dispatch needs measured shape buckets rather than one
assumed threshold. The exact feature-wise full-tile kernel reaches 255
registers/thread and 80 spills; the split epilogue reduces that to 10 spills at
the same register cap. This pressure explains why FC2's short-sequence gain is
small and why the split tradeoff can reverse with M. Validation against the
materialized BF16 operation is bit-exact. Repeated forward/reversed provider
orders retained roughly a 1.06--1.08x attention-output win and a smaller
1.01--1.02x FC2 win at 37,710 rows, so the latter remains real but marginal.

```shell
uv run python benchmarks/benchmark_convrot_gated_residual.py \
  --rows 37710 --warmup-ms 200 --measurement-time-ms 500
uv run python benchmarks/benchmark_convrot_gated_residual.py \
  --rows 131072 --warmup-ms 100 --measurement-time-ms 300
```

Repeat either command with `--reverse-provider-order` to audit timing-order
sensitivity.

The clean API is a mutating explicit boundary such as
`linear_output_addcmul_(activation, weight, residual, gate, row_ids)`, with a
portable materialized fallback.

## RMSNorm and segmented AdaLN inside ConvRot preparation: retain

Before QKV and FC1, H3 computes BF16 RMSNorm and then feature-wise scale/shift.
The existing one-row ConvRot preparation program already owns the complete K
row, so it can compute the RMS reduction and apply the modulation before the H256
butterflies.

[`benchmark_convrot_rms_adaln_preparation.py`](../benchmarks/benchmark_convrot_rms_adaln_preparation.py)
compares against a conservative best-case baseline: one custom Triton
RMSNorm+AdaLN kernel that writes BF16, followed by core preparation. The
actual eager graph can require more passes. As in the gated-residual experiment,
the synthetic random segment `row_ids` map is prebuilt and its construction is
outside the timed region. A real integration should construct it once per
`mod_segments` signature/forward and reuse it across blocks.

| M | Best staged path | Fused w4 | Speedup | Avoided BF16 write+read |
|---:|---:|---:|---:|---:|
| 37,710 | 0.9244 ms | 0.4844 ms | 1.908x | 0.811 GB |
| 131,072 | 3.1925 ms | 1.7163 ms | 1.860x | 2.819 GB |

The fused kernel uses 158 registers/thread, no spills, and 32 KiB of
compiler-reported dynamic shared memory. Sampled qdata and scales are
bit-identical to the custom staged kernel. PyTorch RMSNorm uses a different
valid reduction tree: against that eager reference, sampled qdata differed by
at most two codes and row scales by at most 0.77% relative.
Any implementation must therefore choose and quality-test the numerical contract
rather than labeling the prototype byte-identical to eager PyTorch.

```shell
uv run python benchmarks/benchmark_convrot_rms_adaln_preparation.py \
  --rows 37710 131072 --validate-rows 128 \
  --warmup-ms 100 --measurement-time-ms 300
```

Use an explicit boundary such as
`linear_input_rmsnorm_scale_shift(...)`; a lower-risk first integration can fuse
only scale/shift and retain the existing RMSNorm output.

## `torch.library.triton_op` composition control

[`benchmark_convrot_triton_op.py`](../benchmarks/benchmark_convrot_triton_op.py)
wraps the core preparation kernel with `triton_op`/`wrap_triton`, then places
compiled PyTorch RMSNorm/AdaLN immediately before it. This is a useful integration
improvement over an opaque `custom_op`: full-graph compilation succeeds and the
wrapped preparation produces the expected outputs. It does not, however, rewrite
the user-authored Triton kernel to consume the preceding expression.

At `M=37,710`, `K=5,376`, the traceable composition took 0.9357 ms versus
0.4793 ms for the explicit one-pass kernel, a 1.952x difference. CUDA profiling
showed two launches:

1. one Inductor RMSNorm/index/scale/shift kernel;
2. `_rotate_quantize_rows_kernel`.

The BF16 producer output was therefore still materialized. `triton_op` remains
valuable for tracing, export, tensor-subclass dispatch, and making the internal
launches visible to compiler tooling, but this Torch 2.12.1 test did not recover
the cross-boundary bandwidth saving automatically.

```shell
uv run python benchmarks/benchmark_convrot_triton_op.py \
  --rows 37710 --warmup-ms 100 --measurement-time-ms 500
```

## Paired FC1 output SwiGLU: retain

Current FC1 writes `[gate | up]` with width 28,672, and FC2 preparation reads
both halves while applying SwiGLU. A paired FC1 tile computes matching 128-wide
gate/up columns with two accumulators, applies the established BF16 SwiGLU
boundaries, and writes only the 14,336-wide activated tensor.

[`benchmark_convrot_fc1_output_swiglu.py`](../benchmarks/benchmark_convrot_fc1_output_swiglu.py)
compares the real current boundary, not an eager activation strawman:

| Path | M=37,710 |
|:---|---:|
| Current FC1 + fused SwiGLU/rotation/quantization | 17.3885 ms |
| Paired FC1 output-SwiGLU + ordinary preparation | 16.8141 ms |
| Speedup | 1.034x |

The actual chain saves 2.162 GB. Exact validation at 257 rows with the full H3
`F=14,336`, `K=5,376` widths found both the BF16 output and downstream prepared
INT8 tensor/scales bit-identical to the current path. The winning
`128 x (128+128) x 128` tile uses 237 registers/thread, no spills, 98,304 bytes
of compiler-reported dynamic shared memory, and the same one-CTA/SM residency
ceiling as the core FC1 GEMM.
Forward and reversed provider-order runs kept the gain at about 1.03x; the
displayed 1.034x run was additionally monitored at the 575 W limit.

```shell
uv run python benchmarks/benchmark_convrot_fc1_output_swiglu.py \
  --rows 37710 --warmup-ms 100 --measurement-time-ms 300
```

Use `--reverse-provider-order` for the paired/current-chain order check.

The explicit API boundary is naturally `linear_output_act(..., "swiglu")`.
Do not start with a monolithic FC1-to-FC2 kernel: exact FC2 rowwise scaling still
requires the maximum across all 14,336 intermediate features.

## Model integration notes: design only

No ComfyUI or Diffusers adapter was implemented on this branch. The current
Diffusers MiniMax-H3 source places QKV projection, per-head Q/K RMSNorm, and
partial RoPE together inside
[`MiniMaxH3AttnProcessor`](https://github.com/huggingface/diffusers/blob/50e7158093710f9c1b4ea9ff100137a91c9228f3/src/diffusers/models/transformers/transformer_minimax_h3.py#L158-L207).
That processor is the smallest integration seam for the QKV epilogue candidate:
install a packed QKV module and replace the processor while retaining the
selected attention backend and output projection.

The earlier RMSNorm/AdaLN and later gated residual are performed by the parent
[`MiniMaxH3TransformerBlock`](https://github.com/huggingface/diffusers/blob/50e7158093710f9c1b4ea9ff100137a91c9228f3/src/diffusers/models/transformers/transformer_minimax_h3.py#L345-L371).
Fusing either boundary therefore requires a block wrapper or replacement. Such
a wrapper should preserve the original forward signature and transfer
unaffected child-module objects under their existing names. Hooks on reused
children remain attached, but a forward hook cannot run for an operation that
the fused kernel no longer invokes as a standalone module; expose the fused
operation itself as a hookable child.

Perform model conversion before installing group/CPU offload hooks or calling
`torch.compile`. Treat state-dict conversion, LoRA merging, batch and sequence
layouts, context parallelism, and fallback behavior as adapter acceptance
requirements rather than kernel details.

## Candidate graph-boundary evaluation order

The matrix-only implementation is already near its local optimum. Further work
should be reviewed as explicit graph boundaries, in this order:

1. RMSNorm/scale/shift into QKV and FC1 preparation, with a declared numerical
   contract and model-quality validation.
2. QKV RMSNorm/partial-RoPE epilogue, because H3 head and tile widths align
   exactly and both 37K and 128K measurements win.
3. Paired FC1 output-SwiGLU, independently from the existing FC2 input fusion.
4. Gated residual epilogues, with full versus split epilogues selected only for
   measured row-count buckets.

These boundaries compose, but their isolated savings must not simply be added
and reported as a model-level speedup. Validate the combined implementation in
the real 50-block H3 graph, including weight offload, attention backend hooks,
segment maps, power state, and quality.
