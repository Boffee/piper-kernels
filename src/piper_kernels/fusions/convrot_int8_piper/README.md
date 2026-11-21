# ConvRot INT8 projection fusion internals

For enabling the compiler integration, see
[attention](../../../../docs/attention.md). The opt-in pass fuses ConvRot INT8
projection, per-head RMSNorm, split-half RoPE, signed-Hadamard smoothing, and INT8
Q quantization. Compatible normalized RoPE K and plain projected V also fuse;
unsupported K/V transforms or escaping operands retain the Q-only path.
Projections sharing an input and static-scale setting reuse input preparation.

`compile_options(attention_output=True)` additionally matches Q projection,
attention, and an exclusive ConvRot INT8 output projection. The matched output
layout is BHSD transposed to BSHD with merged heads. Output fusion is disabled by
default. These are multi-kernel pipelines behind compiler boundaries. The
unified post-grad pass orders projection fusion before linear preparation sharing
and output fusion after it, while preserving caller-supplied options.

## Matching and numerical contract

Supported patterns cover FP16/BF16, D64/D128, MHA/GQA, ragged sequences, causal
attention, and rectangular non-causal attention. Projection backends target
SM120, SM89, and RDNA4. Unmatched patterns and unsupported targets keep the
original operations. RMSNorm can be affine or non-affine; RoPE can cover part or
all of a head; projections may have bias. Output preparation supports dynamic
per-token or supplied static scales.

Projection, RMSNorm, and RoPE use FP32 arithmetic. K uses BF16 temporary
storage, then invokes the shared `K1024` encoder (`prepare_chunk_centered_key`)
to compute rotated per-chunk K centers `[B, H, C, D]` and quantize INT8 K with
K64 group or per-key scales.
Q/V keep FP32 intermediates until quantization. These rounding boundaries differ
from materializing every operation in FP16/BF16 or retaining full FP32 K storage.
Transformed K must remain finite in BF16; execution does not scan for violations.

Both non-causal and causal V project per-1024-token-chunk represented input
means (`project_value_chunk_means`) before per-token quantization, with zero
sequence-wide reductions. Prepared storage has these contracts:

- Q: contiguous Q64-padded INT8 storage and FP32 scales incorporating the
  softmax scale and `log2(e)`.
- K: K64-padded INT8 storage, FP32 scales, and FP32 K1024 rotated chunk centers.
- V: K64-padded storage with FP32 per-token multipliers/log scales and FP32 V1024
  chunk means; codes are transposed on NVIDIA and packed WMMA tiles on RDNA4.
  NVIDIA log scales retain their FP16 rounding.

Q/K scales use the granularity the target's attention kernels read: Q32 and K64
group scales on SM120 and RDNA4, and on SM89, as in its native dense attention,
per-thread groups stored as one scale per Q row and K key. The compiler chooses
the granularity for the device; the producer operators take it as
`qk_quantization` (`per_warp` or `per_thread`), and the attention boundaries
read it from the scale shapes.

The producer operators are `convrot_int8_piper_project_query`,
`convrot_int8_piper_project_key`, and `convrot_int8_piper_project_value` in the
`piper_kernels` namespace. `piper_attention_from_quantized_query` consumes
prepared Q with floating K/V; `piper_attention_from_quantized` consumes all
prepared operands. Validation reads host metadata only. Fake execution allocates
outputs without hardware selection or tensor-content checks; see the
[validation contract](../../../../docs/development.md#validation-contract).

## Shared ownership and bounded storage

Dense and sparse paths share projection storage validation, RMSNorm/RoPE tiles, and
chunk-mean projection helpers (`convrot_int8_projection`), compiler tuple matching,
output projection, and the attention-to-output pipeline; bounded dense output windows
live in [`_schedule.py`](_schedule.py).
Dense K shares `piper_attention._quantization.prepare_chunk_centered_key` directly.
Dense V's per-token quantization and `V1024` chunk means remain distinct from
sparse V's `K64` tile scales and `K64` means.
Typed backend methods operate on caller-owned buffers; target launch choices
live in separate NVIDIA/RDNA4 modules. On NVIDIA, D128 projections run Gluon kernels
built from shared [fragments](../convrot_int8_projection/_nvidia/fragments.py) using
``cp.async`` copies on SM8x and TMA loads on SM120, with one shared
[policy](../convrot_int8_projection/_nvidia/policy.py). On AMD RDNA4, aligned
D64/D128 Q/K/V projections (and Q32-scale Q) run Gluon kernels built from shared
4x2 wave32 [fragments](../convrot_int8_projection/_amd/fragments.py) and one shared
[policy](../convrot_int8_projection/_amd/policy.py), falling back to shared Triton
launchers otherwise. [_backend.py](_backend.py) provides optional selection for
matching and required selection for validated execution.

K/V remain global. Q uses one reusable buffer with global RoPE positions and
causal coordinates. When output width is at least merged attention width,
chunks can occupy unwritten final-output rows. The ConvRot projector reads a
batch's entire attention chunk into separate INT8 preparation storage before
overwriting those rows. Any other projector must read the entire chunk before
writing into it. Narrower outputs retain up to two attention buffers.

Dense output fusion uses an 8192-row cap, shared with sparse Piper. Dense
[_schedule.py](_schedule.py) balances 128-aligned windows under the cap and
can reduce SM120 non-causal scheduling waves using target occupancy metadata.
Causal execution and other targets use balanced windows; sparse uses fixed
windows. The internal output operator's `query_chunk_rows` overrides the cap.
Scheduling does not dispatch on model identity or benchmark sequence ranges.

Output, prepared input, and K/V remain full-sized. Only Q and output-preparation
scratch are window-bounded; K preparation still requires a global BF16 temporary
and compact FP32 statistics. Output fusion has no automatic profitability guard.
Reduced scratch does not guarantee lower peak allocation or faster execution.
Measure the complete pipeline, including output storage and workspace, with the
[dense fusion benchmark](../../../../benchmarks/README.md#run-the-suite).

## In-place residual adds

When the [residual-update rewrite](../residual_updates/_compile.py) adds a fused
dense output into a copy-back block's input, the output fold then computes that
output inside the add ([`_residual_fold.py`](../attention/_residual_fold.py)): it calls
`convrot_int8_piper_projected_query_attention_residual_add_`, which takes the
output's operands, uses `base`'s dtype as the output dtype, and adds
`gate[gate_indices] * projection` into `base` without allocating a full-size output.
Sparse Piper's projected-Q output does the same with its counterpart,
`convrot_int8_sparse_piper_projected_query_attention_residual_add_`.

Residual rows hold live data, so attention windows never borrow them: attention
uses its separate window buffers (two when several windows overlap projection).
The linear backend owns the residual addition: a native GEMM epilogue adds the
projection directly into the residual, while the fallback first writes a bounded
projection scratch window. Each add rounds `base` once per window row;
see [`LinearBackend.linear_prepared_residual`](../../linear/convrot/int8/_backend.py)
and the [SM120 TMA residual epilogue](../../linear/convrot/int8/_nvidia/gluon_tma.py)
for numerical and backend details.

When its heads split into aligned halves, the fold runs this add in two head slices,
calling `convrot_int8_piper_project_key`, `convrot_int8_piper_project_value`, and the add
once per slice on head-sliced weights. `_HEAD_SLICING` in [`_compile.py`](_compile.py)
names the arguments that split by head; the mechanism is
[`attention/_head_slices.py`](../attention/_head_slices.py). Each slice's windows hold
twice the rows, so attention launches keep their unsliced size.

Sparse Piper's fold slices its add the same way, with its K and V projections and its
coarse gate's projection rows, as its own
[`_HEAD_SLICING`](../convrot_int8_sparse_piper/_compile.py) names. Keep budgets are
rounded across all heads, so each slice's add takes its position and covers its heads'
share of the whole profile's budget. A coarse gate the fusion receives whole holds every
head, so that add runs as one slice. Sparse windows keep their unsliced row cap: their Q,
attention, and routing-score buffers then halve with the heads, at a small speed cost.

## Sequence-parallel output collectives

When `attention_output=True`, the unified post-grad pass runs
[`_distributed_compile.py`](_distributed_compile.py) via the shared
[`_distributed_fold.py`](../attention/_distributed_fold.py) helper to rewrite
`convrot_int8_piper_projected_query_attention_output -> reduce_scatter_tensor` (and an
exclusive trailing in-place `residual_add_`) into
`convrot_int8_piper_projected_query_attention_reduce_scatter` or
`..._reduce_scatter_residual_add_`. Both custom operators live in
[`distributed_output.py`](distributed_output.py) and delegate bounded peer chunk
exchange behind own-rank attention to
[`_peer_exchange.py`](../../_distributed/_peer_exchange.py).

Rank-local sequences divisible by 128 tokens (matching dense query-tile alignment on SM89,
SM120, and RDNA4) use bounded ping-pong peer chunks; shorter or unaligned local
sequences fall back to whole-sequence reduce-scatter. On aligned sequences with a residual
target, peer and local rank updates add into `base` in two rounded steps rather than
materializing a full-sequence buffer; see
[sequence-parallel weights](../../../../docs/weights.md#shard-a-quantized-weight).
