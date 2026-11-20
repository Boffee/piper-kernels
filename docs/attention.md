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
path requires NVIDIA FP8 tensor cores with FP16 accumulation. See the
[dense implementation guide](../src/piper_kernels/attention/piper_attention/README.md)
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
`block_lengths`. See [residual.py](../src/piper_kernels/attention/sparse_piper_attention/residual.py)
and [coarse.py](../src/piper_kernels/attention/sparse_piper_attention/coarse.py).
Compatible compiler fusions accumulate fine and coarse outputs in FP32 before
casting, so outputs can differ slightly from standalone addition.

## Compile and fuse projections

Standalone attention works with `torch.compile(piper_attention, fullgraph=True)`.
Replacing `SparsePiperAttention` with another profile of the same head count and
routing policy reuses compiled graphs (including compatible fusions). Because
runtime profiles disable automatic CUDA graph capture around those ops, use
explicit CUDA graphs when a profile is fixed and recapture if it changes.

For quantized projection/output fusion, pass the matching helper to
`torch.compile(..., options=...)` following [compiler options](weights.md#compile-a-model):

| Projection format and attention | Helper package under `piper_kernels.fusions` |
|---|---|
| ConvRot INT8 + dense Piper | `convrot_int8_piper` |
| ConvRot INT8 + sparse Piper | `convrot_int8_sparse_piper` |
| NVFP4 / ConvRot NVFP4 + sparse Piper | `nvfp4_sparse_piper` |

Each exports `<package_name>_compile_options()`. ConvRot INT8 dense fusion
supports MHA/GQA and optional `fuse_output=True` (bounded attention-to-output
execution, off by default); dense targets NVIDIA SM89/SM120 and AMD RDNA4, sparse
targets SM8x/SM120 and RDNA4. See the
[fusion guide](../src/piper_kernels/fusions/convrot_int8_piper/README.md) for
matchers, storage contracts, and bounded-window internals.

### Update a residual stream in place

When a compiled block writes gated attention and FFN outputs back into its
residual input via `hidden.copy_(...)`, `residual_updates_compile_options`
computes eligible additions in place without allocating intermediate full-size
output buffers:

```python
from piper_kernels.fusions.convrot_int8_piper import convrot_int8_piper_compile_options
from piper_kernels.fusions.convrot_int8_swiglu_ffn import convrot_int8_swiglu_ffn_compile_options
from piper_kernels.fusions.residual_updates import residual_updates_compile_options

options = convrot_int8_swiglu_ffn_compile_options()
options = convrot_int8_piper_compile_options(options, fuse_output=True)
options = residual_updates_compile_options(options)


def update_block(hidden, *block_inputs):
    return hidden.copy_(block(hidden, *block_inputs))


compiled = torch.compile(update_block, fullgraph=True, options=options)
```

Returning `hidden.copy_(...)` declares that the caller will not read `hidden` (or
any view of it) after the call; if the block reads the original input afterward,
Inductor inserts a copy instead. Per-token indexed gates
(`base + gate[indices] * update`) and eligible ConvRot/NVFP4 FFN and
Q-projected ConvRot INT8 attention outputs fold into the add; row-broadcast gates
stay unchanged. In-place adds round separately per update (`_nvidia` ConvRot INT8
uses its [residual epilogues](../src/piper_kernels/linear/convrot/int8/_nvidia/README.md#residual-epilogues)).

Eligible folded attention runs in two sequential head slices when head counts are
even and output widths align, trading slightly lower active K/V memory for small
quantization rounding differences; see
[in-place residual adds](../src/piper_kernels/fusions/convrot_int8_piper/README.md#in-place-residual-adds)
for matcher, slicing, and buffer details.

Return to [README](../README.md), [weights](weights.md), or
[benchmarks](../benchmarks/README.md).
