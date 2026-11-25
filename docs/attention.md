# Attention

Piper's attention operators approximate attention through quantization. Choose the
operator for your model's semantics, then check quality on representative inputs.
The portable fallbacks preserve each operator's quantized algorithm; they are
slow correctness paths, not exact PyTorch SDPA replacements.

| Operator | Use when | Q/K/V layout | Input dtype |
|---|---|---|---|
| `piper_attention` | Every query attends all keys, or causal attention is needed | `[batch, heads, sequence, features]` | FP16, BF16 |
| `SparsePiperAttention` | The model supplies a block layout and per-head sparse budgets | `[batch, sequence, heads, features]` | FP16, BF16, FP32 |
| `sage_attention_2pp` | You specifically want the canonical SageAttention2++ 8+8 algorithm | `[batch, heads, sequence, features]` | FP16, BF16 |

All three require head dimensions of 64 or 128, matching Q/K/V devices and dtypes,
and a contiguous final feature dimension. Outer dimensions may be strided. They
preserve the query shape and dtype and are inference-only. Use detached inputs or
run under `torch.no_grad()` / `torch.inference_mode()`.

## Dense attention

This example runs on CPU or GPU. Unsupported accelerators use the portable path.

```python
import torch
from piper_kernels import piper_attention, sage_attention_2pp

device = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn(1, 4, 65, 64, device=device, dtype=torch.float16)
k = torch.randn(1, 2, 97, 64, device=device, dtype=torch.float16)
v = torch.randn_like(k)

with torch.inference_mode():
    output = piper_attention(q, k, v)  # [1, 4, 65, 64], grouped-query attention
    causal = piper_attention(q, q, q, is_causal=True)
    sage = sage_attention_2pp(q, q, q)  # Sage requires equal Q/K/V head counts
```

Non-causal calls allow different query and key lengths. K and V must have the same
shape, and all sequences must be nonempty. Causal calls require equal query/key
lengths; there is no offset-causal decoding mode. `scale` defaults to
`head_dim**-0.5` and, when supplied, must be finite and positive. These entry points
do not accept arbitrary attention masks or dropout.

Piper infers grouped-query attention from the shapes: `Hq` must be an integer
multiple of `Hkv`, and query head `h` reads KV head `h // (Hq // Hkv)`. This includes
multi-query attention with one KV head; do not repeat K/V yourself. Sage requires
equal head counts.

Dense Piper uses INT8 Q/K and per-token INT8 V, centering K and V independently per
1024-token chunk (`K1024 + V1024`) on both non-causal and causal calls. Because chunks
share no cross-chunk statistics, token-sharded chunks need no cross-chunk state and
future tokens in later chunks never affect earlier outputs (tokens sharing a causal
chunk share its local statistics). Sage uses INT8 Q/K and FP8 probabilities/V with
FP16 accumulation. Their numerical results are not interchangeable.

Dense Piper has native NVIDIA SM8x/SM12x and AMD RDNA4 backends. Sage's optimized
path requires NVIDIA FP8 tensor cores with FP16 accumulation. See
[`_quantization.py`](../src/piper_kernels/attention/piper_attention/_quantization.py),
[`_nvidia/policy.py`](../src/piper_kernels/attention/piper_attention/_nvidia/policy.py),
and [`_amd/gluon.py`](../src/piper_kernels/attention/piper_attention/_amd/gluon.py)
for arithmetic, architecture differences, and scheduling.

## Sparse attention

Sparse Piper is non-causal self-attention. Q/K/V must have the same sequence length,
at least 64 rows. It supports the same GQA head relationship as dense Piper, but
uses sequence-major tensors and one keep ratio per **query** head.

```python
import torch
from piper_kernels import SparsePiperAttention

device = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn(1, 193, 4, 64, device=device, dtype=torch.bfloat16)
k = torch.randn(1, 193, 2, 64, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
attention = SparsePiperAttention((0.5, 0.5, 1.0, 1.0), routing="minmax")

with torch.inference_mode():
    output = attention(q, k, v, sparse_key_blocks=2, sparse_query_blocks=2)
assert output.shape == q.shape
```

Here, the first 128 key rows form two routeable K64 blocks. Routed queries select
blocks from that prefix; they also attend **every** key in the remaining 65 rows,
within the same softmax. Only the first 128 query rows are routed; later queries
attend all keys densely. Omitting `sparse_query_blocks` routes every query block;
zero makes every query dense.

Keep ratios lie in `(0, 1]` and belong to immutable model configuration. Physical
budgets round to whole blocks across heads, with at least one retained block per
head. `attention.resolve_head_keep_blocks(sparse_key_blocks)` returns the exact
per-query-head budgets as a tuple of Python integers. Routes are recomputed from
the current Q/K values on every call. The default `"minmax"` policy scores
extrema summaries; `"mean"` scores block means. The application owns the ratio
profile and its quality validation. Even a full keep budget uses sparse Piper's
per-`K64` tile centering and quantization (`K64 + V64`), which differs numerically from
dense Piper's `V` quantization.

`sparse_key_blocks` counts complete physical blocks in the leading key prefix and
must be between 1 and `min(sequence_length // 64, 65536)`. A compact partial final
block therefore belongs to the dense suffix. `sparse_query_blocks` may range from
zero through `ceil(sequence_length / 64)`. `scale` has the same contract as dense
attention.

For a layout with padding **inside** blocks, pass `block_lengths`: a contiguous
INT32 tensor on the input device containing one length in `[1, 64]` per physical
K64 block. Sequence storage must then have exactly `64 * len(block_lengths)` rows;
valid tokens occupy the beginning of each block. Padding is excluded from
attention. Output keeps that physical layout, and padded query outputs are
unspecified: gather only valid rows. Supplying valid length values is the caller's
responsibility under the [validation contract](development.md#validation-contract).

Native sparse backends cover NVIDIA SM8x/SM120 and AMD RDNA4. Other devices run the
portable quantized reference. There is a separate exact sparse reference for
quality comparisons; it is not the fallback used by this API.

### Caller-forced tiles (experimental)

Pass contiguous device INT32 `force_tile_ids` (`[query_blocks, capacity]`) and
`force_tile_counts` (`[query_blocks]`) to pin active prefix IDs across batches and
heads, filling each head's remaining keep budget by score selection (dense suffixes
ignore routes). As metadata-only caller preconditions: supply one row per physical
query block, keep counts in `[0, min(capacity, min(budgets))]` where
`budgets = attention.resolve_head_keep_blocks(sparse_key_blocks)` (budgets couple
across heads, so recompute when the profile or `sparse_key_blocks` changes), and keep
active IDs unique in `[0, sparse_key_blocks)`.

IDs and counts can update in place at fixed shapes without recompiling (zeros disable
forcing), explicit CUDA graphs replay updated buffers, and compatible ConvRot/NVFP4
fusions preserve forced routing (coarse residual is unaffected). For dynamic sequence
lengths or force capacities in `torch.compile`, mark those dimensions with
`torch._dynamo.mark_dynamic` on inputs and force tensors (head counts/dims stay fixed).

### Coarse residual

`sparse_piper_coarse_residual` returns a separate branch residual: it derives
mean/minmax block scores, mean-pools `V`, expands to token rows, and multiplies
the supplied gate directly without applying an activation. Q/K/V and the gate
must match in shape (equal head counts) with matching query dtype/device on the
gate. Pass `routing="mean"` or `"minmax"` and a finite positive `coarse_scale`;
`coarse_key_blocks` defaults to all available blocks and supports compact tails and
`block_lengths`. See [coarse.py](../src/piper_kernels/attention/sparse_piper_attention/coarse.py).
Compatible compiler fusions accumulate fine and coarse outputs in FP32 before
casting, so outputs can differ slightly from standalone addition.

## Persistent chunk-major KV pool

Dense KV chunks of 1024 finalized tokens can be quantized and
stored persistently in a global layer-shared `PiperKVChunkPool` via
`freeze_kv_chunks_` and sized from `kv_chunk_storage_specs`:

```python
import torch
from piper_kernels import PiperKVChunkPool, freeze_kv_chunks_, kv_chunk_storage_specs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
specs = kv_chunk_storage_specs(
    num_slots=8, kv_heads=4, head_dim=128, device=device, kv_dtype=torch.uint8
)
pool = PiperKVChunkPool.from_tensors(
    **{
        name: torch.empty(shape, device=device, dtype=dtype)
        for name, (shape, dtype) in specs.items()
    }
)

new_k = torch.randn(2, 4, 1024, 128, device=device, dtype=torch.bfloat16)
new_v = torch.randn(2, 4, 1024, 128, device=device, dtype=torch.bfloat16)
slot_ids = torch.tensor([0, 3], device=device, dtype=torch.int32)

with torch.inference_mode():
    freeze_kv_chunks_(new_k, new_v, pool, slot_ids)
```

Pool and operator contract:

- **Buffer geometry:** Parameterized by `num_slots >= 0`, `kv_heads >= 1`, and
  `head_dim` in `{64, 128}`, with all 6 buffers on the same device: `key_int8`
  (`[num_slots, kv_heads, 1024, head_dim]`), `value_int8`
  (`[num_slots, kv_heads, 16, head_dim, 64]`), `key_scale` (`float32`,
  `[num_slots, kv_heads, 16]` on SM12x / RDNA4 / portable targets or
  `[num_slots, kv_heads, 1024]` on SM8x), `key_center` and `value_mean` (`float32`,
  `[num_slots, kv_heads, head_dim]`), and `value_scale` (`float32`,
  `[num_slots, kv_heads, 1024]`). `key_int8` and `value_int8` must match in dtype
  (`torch.int8` or `torch.uint8`).
- **Opaque target-family storage & host staging:** `value_int8`'s trailing 64-token axis
  uses target-specific ordering (WMMA-packed on RDNA4 `gfx1200`/`gfx1201`, unpermuted
  elsewhere), and `freeze_kv_chunks_` enforces the active GPU's `key_scale` width
  (`1024` on SM8x, `16` elsewhere). When allocating CPU staging buffers for GPU host-DMA
  swapping, pass the target GPU `device` (or `"sm80"` / `"sm120"` / `"gfx1201"`) to
  `kv_chunk_storage_specs` instead of `None`/`"cpu"`; frozen slab bytes are portable across
  devices in the same target family, not across differing target families.
- **Shared byte arenas and strided `stride(0)`:** `key_int8` and `value_int8` accept
  either `torch.int8` (default) or `torch.uint8` so callers managing dynamic shared byte
  arenas (such as interleaved `[num_slots, 2, kv_heads, 1024, head_dim]` views and
  `uint8` MoE buffers) can share one `uint8` backing allocation while keeping FP32
  metadata in separate `float32` buffers. Every pool tensor must be dense row-major
  within each slot, with 16-byte-aligned base pointers, storage offsets, and multi-slot
  `stride(0)` byte steps (`stride(0) >= slot_elements`), and pairwise disjoint per-slot
  byte spans.
- **Slot freezing:** `freeze_kv_chunks_` quantizes non-overlapping `key` / `value` inputs
  of shape `[num_new, kv_heads, 1024, head_dim]` sharing dtype (`float16` or `bfloat16`)
  and `pool.device` (`stride(-1) == 1`; outer dims may be strided) directly into
  contiguous `int32` `slot_ids` (`[num_new]`, non-overlapping with `pool`) using zero
  host readbacks. Entries `< 0` or `>= pool.num_slots`, `num_new == 0`, and
  `pool.num_slots == 0` are safe no-ops leaving untouched slots unchanged; valid
  `slot_ids` within a single call must be distinct. `freeze_kv_chunks_` is inference-only
  (enforcing `InferenceMode` if any pool buffer is an inference tensor) and supports CUDA
  Graph capture and `torch.compile(fullgraph=True, dynamic=True)`.

## Compile and fuse projections

Standalone attention works with `torch.compile(piper_attention, fullgraph=True)`.
Replacing `SparsePiperAttention` with another profile of the same head count and
routing policy reuses compiled graphs (including compatible fusions). Because
runtime profiles disable automatic CUDA graph capture around those ops, use
explicit CUDA graphs when a profile is fixed and recapture if it changes.

For quantized projection/output fusion, pass `piper_kernels.compile_options` to
`torch.compile(..., options=...)` following [compiler options](weights.md#compile-a-model):

```python
from piper_kernels import compile_options

# Dense or sparse projection fusions (ConvRot INT8, NVFP4, ConvRot NVFP4):
options = compile_options(attention=True)
# Add bounded dense ConvRot INT8 attention-to-output fusion (off by default):
options = compile_options(attention=True, attention_output=True)
```

`attention=True` folds eligible ConvRot INT8 dense/sparse and NVFP4 / ConvRot
NVFP4 sparse Piper projection patterns (including sparse bounded output fusion);
`attention_output=True` additionally enables bounded dense ConvRot INT8
attention-to-output fusion (and implies `attention=True`). ConvRot INT8 dense
fusion supports MHA/GQA and targets NVIDIA SM89/SM120 and AMD RDNA4; sparse
targets SM8x/SM120 and RDNA4 (NVFP4 / ConvRot NVFP4 target SM120). See the
[fusion guide](../src/piper_kernels/fusions/convrot_int8_piper/README.md) for
matchers, storage contracts, and bounded-window internals.

### Update a residual stream in place

When a compiled block writes gated attention and FFN outputs back into its
residual input via `hidden.copy_(...)`, `compile_options(residual_updates=True)`
computes eligible additions in place without allocating intermediate full-size
output buffers:

```python
from piper_kernels import compile_options

options = compile_options(
    ffn=True,
    attention=True,
    attention_output=True,
    residual_updates=True,
)


def update_block(hidden, *block_inputs):
    return hidden.copy_(block(hidden, *block_inputs))


compiled = torch.compile(update_block, fullgraph=True, options=options)
```

Returning `hidden.copy_(...)` declares that the caller will not read `hidden` (or
any view of it) after the call; if the block reads the original input afterward,
Inductor inserts a copy instead. Per-token indexed gates
(`base + gate[indices] * update`) and eligible ConvRot/NVFP4 FFN, Q-projected
ConvRot INT8 attention outputs, and Q-projected NVFP4 / ConvRot NVFP4 sparse
attention outputs fold into the add; row-broadcast gates stay unchanged. In-place
adds round separately per update (ConvRot INT8 uses fused 16-bit
[residual epilogues](../src/piper_kernels/linear/convrot/int8/_nvidia/gluon_tma.py)
on SM120 and RDNA4).

Eligible folded ConvRot INT8 attention runs in two sequential head slices when
head counts are even and output widths align, trading slightly lower active K/V
memory for small quantization rounding differences; see
[in-place residual adds](../src/piper_kernels/fusions/convrot_int8_piper/README.md#in-place-residual-adds)
for matcher, slicing, and buffer details.

Return to [README](../README.md), [weights](weights.md), or
[benchmarks](../benchmarks/README.md).
