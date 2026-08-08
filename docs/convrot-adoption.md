# ConvRot incremental adoption roadmap

This is the entry point for the ConvRot optimization work. The branch contains
one implemented operator checkpoint and a separate set of benchmark-only
experiments. They should not be reviewed or adopted as one feature.

## Status at a glance

| Scope | State | Runtime or public API on this branch? |
|:---|:---|:---|
| ConvRot H4 factorization, one-pass preparation, GEMM scheduling, and 64-bit indexing | Implemented core checkpoint | Yes |
| Raw FC2 input-SwiGLU preparation | Implemented optional ConvRot API | Yes: `linear_input_act(..., "swiglu")` |
| Packed/TMA/persistent GEMM variants | Rejected benchmark result | No |
| `triton_op` as automatic graph fusion | Rejected mechanism; traceability still useful | No |
| RMSNorm/AdaLN, QKV RMSNorm/RoPE, paired FC1 SwiGLU, and gated residual fusion | Promising isolated prototypes | No |
| ComfyUI or Diffusers model adapters | Design discussion only | No |
| Whole-block fusion | Not implemented or benchmarked end to end | No |

The implemented core is documented in
[`convrot-optimization.md`](convrot-optimization.md). Detailed measurements for
everything after that checkpoint are in
[`convrot-follow-on-experiments.md`](convrot-follow-on-experiments.md).

## Dependency ladder

```text
Benchmark baseline
├─ H4 factorization + 64-bit addressing
│  └─ exact fused rotate/max/quantize preparation
│     ├─ raw FC2 input-SwiGLU API                 [implemented core extension]
│     └─ RMS/AdaLN input preparation              [prototype]
└─ GEMM tiling/grid scheduling
   ├─ QKV RMSNorm/RoPE epilogue                   [prototype]
   ├─ paired FC1 output-SwiGLU                    [prototype]
   └─ gated residual epilogue                     [prototype]
```

The prototype branches are independent. Adopting QKV fusion does not require
paired FC1 SwiGLU or gated residual fusion, and none is required to adopt the
core ConvRot operator.

## Adoption stages

### Stage 0: freeze the research baseline

Use commit `b0817bd` as the implemented ConvRot checkpoint. Preserve benchmark
commands, environment details, quality checks, and the unfused fallback before
reimplementing anything.

Stopping here is valid: the checkpoint already beats the measured Comfy Kitchen
projection path at the principal MiniMax H3 shapes.

### Stage 1: adopt standalone ConvRot

Land the core as reviewable changes rather than copying the research diff:

1. benchmark presets and phase timing;
2. factorized H4 algebra and 64-bit row-base products;
3. exact one-pass rotate/max/quantize preparation behind a conservative guard;
4. GEMM scheduling as a separate performance change; and
5. raw input-SwiGLU as a separate, explicit optional API.

Acceptance gates:

- exact FP16/BF16 preparation checks against the split path;
- portable reference and unsupported-shape fallbacks;
- ordinary and raw-SwiGLU projection quality tests;
- 37K and 128K address-boundary coverage;
- isolated preparation, GEMM, and end-to-end timings; and
- measurements on a pre-Blackwell NVIDIA GPU before generalizing SM120 tuning.

No model module replacement is needed at this stage. Ordinary `F.linear` tensor
dispatch and the explicit input-activation helper are sufficient.

### Stage 2: promote one graph boundary to a generic operator

Choose one prototype and give it an explicit tensor-level API. Each operator
must land independently with:

- a declared numerical contract and logical dtype boundaries;
- a portable materialized reference/fallback;
- fake/meta and compile behavior;
- shape and architecture dispatch guards;
- direct correctness tests and an isolated benchmark; and
- no dependency on ComfyUI, Diffusers, or a particular model class.

The paired FC1 output-SwiGLU experiment is the cleanest generic API candidate,
for example `linear_output_act(..., "swiglu")`, but its measured H3 gain is only
about 3%. The gated residual operation is also generic in principle, but needs a
clear mutation/alias contract and measured row-count buckets.

`triton_op` may be used to make an operator traceable. The measured control
proved that it does not automatically fuse surrounding PyTorch expressions into
a user-written Triton kernel, so explicit fused kernels remain necessary.

### Stage 3: add an opt-in MiniMax adapter

Wire already-tested generic operators into a model without changing their math.
The smallest first integration is QKV projection plus Q/K RMSNorm and partial
RoPE through an attention-processor replacement. Pre-projection RMSNorm/AdaLN
and post-projection gated residuals cross the parent block boundary and should
use a block wrapper only when those operators are ready.

Adapter acceptance gates:

- preserve attention backend selection and Q/K/V layouts;
- support or explicitly reject batch, dynamic sequence, and context-parallel
  layouts;
- preserve state-dict conversion, device movement, and group/CPU offload;
- document which original module hooks still execute and expose hookable fused
  boundaries;
- handle LoRA by merging before packing or retaining a supported fallback;
- validate a real model block before replacing all 50 blocks; and
- measure full-model quality and latency without summing isolated speedups.

The current branch does not contain this adapter.

### Stage 4: evaluate combined block fusion

Only after individual graph-boundary operators and the adapter pass their own
gates should they be combined in a replacement block. Retain the original block
forward signature and reuse unaffected child modules so integration remains
incremental and inspectable.

This stage must be justified by a real MiniMax H3 run including weight offload,
attention integration, row-map construction, hooks, compilation, memory peak,
power state, and output quality. A collection of positive microbenchmarks is not
evidence of an additive model-level speedup.

## Research artifact index

| Artifact | Category | Decision | Earliest stage |
|:---|:---|:---|:---|
| `benchmark_convrot.py` | Core end-to-end validation | Retain | Stage 0 |
| `benchmark_convrot_preparation.py` | Core phase isolation | Retain | Stage 0 |
| `benchmark_convrot_gemm_experiments.py` | Kernel-local GEMM research | Preserve negative result; do not adopt | Stage 1 evidence only |
| `benchmark_convrot_triton_op.py` | Compiler integration control | Traceability only; not automatic fusion | Stage 2 evidence only |
| `benchmark_convrot_fc1_output_swiglu.py` | Generic operator candidate | Retain prototype | Stage 2 |
| `benchmark_convrot_gated_residual.py` | Generic epilogue plus MiniMax row mapping | Retain selectively | Stage 2, then Stage 3 |
| `benchmark_convrot_qkv_epilogue.py` | MiniMax-shaped attention boundary | Retain prototype | Stage 2 API, then Stage 3 |
| `benchmark_convrot_rms_adaln_preparation.py` | MiniMax-shaped input boundary | Retain with unresolved eager numerical contract | Stage 3; full RMS fusion only after quality acceptance |

All follow-on files are benchmarks, not production kernels. Their use of
private backend helpers is intentional for isolation and is not a proposed API.

## Recommended stopping points

- Need a faster generic ConvRot linear: stop after Stage 1.
- Need an adjacent activation or epilogue in multiple models: adopt one Stage 2
  operator and stop.
- Need MiniMax-specific QKV savings: add only the Stage 3 attention adapter.
- Need maximum MiniMax performance: consider Stage 4 only after the preceding
  stages have independent correctness and integration evidence.
