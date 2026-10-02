# ConvRot INT8 projection fusion internals

For enabling the compiler integration, see
[attention](../../../../docs/attention.md). The opt-in pass fuses ConvRot INT8
projection, per-head RMSNorm, split-half RoPE, signed-Hadamard smoothing, and Q32
INT8 quantization. Compatible normalized RoPE K and plain projected V also fuse;
unsupported K/V transforms or escaping operands retain the Q-only path.
Projections sharing an input and static-scale setting reuse input preparation.

`fuse_output=True` additionally matches Q projection, attention, and an exclusive
ConvRot INT8 output projection. The matched output layout is BHSD transposed to
BSHD with merged heads. Output fusion is disabled by default. These are
multi-kernel pipelines behind compiler boundaries. The pass runs before ordinary
ConvRot optimizations and preserves caller-supplied options.

## Matching and numerical contract

Supported patterns cover FP16/BF16, D64/D128, MHA/GQA, ragged sequences, causal
attention, and rectangular non-causal attention. Projection backends target
SM120 and RDNA4. Unmatched patterns and unsupported targets keep the original
operations. RMSNorm can be affine or non-affine; RoPE can cover part or all of a
head; projections may have bias. Output preparation supports dynamic per-token
or supplied static scales.

Projection, RMSNorm, and RoPE use FP32 arithmetic. K then uses BF16 temporary
storage and FP32 tile sums of those represented values. Global mean reduction
and centering precede K64 quantization. Dense and sparse Piper share this K
producer; sparse routing summaries retain the uncentered FP32 transform values.
Q/V keep FP32 intermediates until quantization. These rounding boundaries differ
from materializing every operation in FP16/BF16 or retaining full FP32 K storage.
Transformed K must remain finite in BF16; execution does not scan for violations.

Non-causal V projects the represented input mean before per-token quantization;
causal V remains uncentered. Prepared storage has these contracts:

- Q: contiguous Q64-padded INT8 storage and FP32 Q32 scales incorporating the
  softmax scale and `log2(e)`.
- K: K64-padded storage and FP32 K64 scales.
- V: K64-padded storage with FP32 per-token multipliers/log scales; codes are
  transposed on NVIDIA and packed WMMA tiles on RDNA4. NVIDIA log scales retain
  their FP16 rounding.

The producer operators are `convrot_int8_piper_project_query`,
`convrot_int8_piper_project_key`, and `convrot_int8_piper_project_value` in the
`piper_kernels` namespace. `piper_attention_from_quantized_query` consumes
prepared Q with floating K/V; `piper_attention_from_quantized` consumes all
prepared operands. Validation reads host metadata only. Fake execution allocates
outputs without hardware selection or tensor-content checks; see the
[validation contract](../../../../docs/development.md#validation-contract).

## Shared ownership and bounded storage

Dense and sparse paths share projection storage validation, RMSNorm/RoPE tiles,
projected-mean reduction, compiler tuple matching, output projection, and the
attention-to-output pipeline. `convrot_int8_centered_projection` owns BF16
projection storage and FP32 statistics. `convrot_int8_sage_qk.key` owns K
transforms and centered K64 encoding, with optional sparse routing summaries.
Dense V's per-token quantization remains distinct from sparse V's tile scales.
Typed backend methods operate on caller-owned buffers; target launch choices
live in separate NVIDIA/RDNA4 modules. [_backend.py](_backend.py) provides
optional selection for matching and required selection for validated execution.

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
[dense fusion benchmark](../../../../benchmarks/README.md#attention-and-projection-fusion).
