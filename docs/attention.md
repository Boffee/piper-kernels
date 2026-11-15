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

Dense Piper uses INT8 Q/K and per-token INT8 V with unsigned INT8 probabilities.
Both non-causal and causal calls center K and V independently per 1024-token
chunk (`K1024 + V1024`), without sequence-wide mean reductions. K centering adds
the per-chunk `Q @ H(K_mean)` score bias and per-chunk V means are restored per
chunk, so token-sharded chunks need no cross-chunk state; future tokens in later
chunks cannot affect earlier outputs, while tokens sharing one causal chunk share
its local chunk statistics. Sage uses INT8 Q/K and FP8 probabilities/V with
FP16 tile accumulation. Their numerical results are not interchangeable.

Dense Piper has native NVIDIA SM8x/SM12x and AMD RDNA4 backends. Sage's optimized
path requires NVIDIA FP8 tensor cores with FP16 accumulation. See the
[dense implementation guide](../src/piper_kernels/attention/piper_attention/README.md)
for arithmetic, architecture differences, and scheduling.

## Sparse attention

Sparse Piper is non-causal self-attention. K and V must have the same shape, with at
least 64 sequence rows, and Q has their sequence length unless the call computes a
[query slice](#query-slices). It supports the same GQA head relationship as dense
Piper, but uses sequence-major tensors and one keep ratio per **query** head.

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
quantized arithmetic (`K64 + V64` per-tile centering: `K` centered per `K64` tile
by its rotated midpoint with 2-term INT8 centers restored in scores, and `V`
centered per `K64` tile by its BF16 mean restored in the numerator), which differs
from dense Piper's `V` quantization.

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

### Query slices

Each K64 block of query rows depends only on its own queries and every key, so a
call can compute just a slice of the query blocks, such as one device's share under
context parallelism. Pass the slice of Q, the whole sequence's K and V, and the
global index of the slice's first row as `global_row_offset`:

```python
start, end = 128, 193  # this caller's rows; start is a multiple of 64
with torch.inference_mode():
    part = attention(
        q[:, start:end],
        k,
        v,
        sparse_key_blocks=2,
        sparse_query_blocks=2,
        global_row_offset=start,
    )
assert torch.equal(part, output[:, start:end])
```

The slice must start on a K64 block and cover whole blocks of the sequence; only a
compact slice that ends the sequence may end in a partial block, so with
`block_lengths` every slice holds a multiple of 64 physical rows. Other placements
raise `ValueError`. Everything else still describes the whole sequence:
`block_lengths`, `sparse_query_blocks`, and forced tiles index global blocks, so every
slice's call receives the same tensors. The output holds the slice's rows, each
equal to that row of the whole-sequence call, so concatenating the slices' outputs
reproduces it. The operators see only local tensors; gathering K/V and outputs
across devices belongs to the caller. The projection fusions need Q and K/V
projected from the same rows, so a sliced call compiles without them.

### Caller-forced tiles (experimental)

This path additionally accepts contiguous device INT32 `force_tile_ids`
(`[query_blocks, capacity]`) and `force_tile_counts` (`[query_blocks]`). Every
active ID in a row's counted prefix is shared across batches and heads and stays
in the kept set, with remaining per-head slots filled by score selection. Dense
query suffixes already attend every key and ignore routes.

As caller preconditions with metadata-only validation: supply one row per
physical query block, keep counts in `[0, capacity]` and within every head's
physical keep budget, and keep active IDs unique in `[0, sparse_key_blocks)`.
Unused slots are ignored. Query budgets outside `torch.compile`:

```python
attention = SparsePiperAttention((0.5, 0.5))
budgets = attention.resolve_head_keep_blocks(sparse_key_blocks=3)  # (2, 1)
capacity = 2
max_force_count = min(capacity, min(budgets))  # 1 across heads
```

Because rounding is coupled across heads, independently rounding ratios can
overestimate a head's budget; recompute whenever the profile or `sparse_key_blocks`
changes. ID and count values can update in place at fixed shapes without
recompiling (zeros disable forcing), and explicit CUDA graphs replay updated
buffers. Compatible ConvRot INT8, NVFP4, and ConvRot NVFP4 fusions preserve forced
routing; the coarse residual is unaffected.

To share one dynamic Torch graph across sequence lengths or force capacities,
mark those dimensions dynamic on the model inputs and force tensors (and derive
Python ints like `sparse_key_blocks` from tensor shapes when they vary):

```python
torch._dynamo.mark_dynamic(hidden_states, 1)  # [batch, sequence, features]
torch._dynamo.mark_dynamic(force_tile_ids, (0, 1))
torch._dynamo.mark_dynamic(force_tile_counts, 0)
```

Keep head counts and head dimensions fixed; shape guards, backend policies, and
zero/singleton compiler specializations still apply.

### Coarse residual

`sparse_piper_coarse_residual` returns a separate branch residual: it derives
mean/minmax block scores, mean-pools `V`, expands to token rows, and multiplies
the supplied gate directly without applying an activation. Q/K/V and the gate
must match in shape (equal head counts) with matching query dtype/device on the
gate. Pass `routing="mean"` or `"minmax"` and a finite positive `coarse_scale`;
`coarse_key_blocks` defaults to all available blocks and supports compact tails,
`block_lengths`, and `global_row_offset` for a [query slice](#query-slices), whose Q and
gate then hold only the slice's rows. See [residual.py](../src/piper_kernels/attention/sparse_piper_attention/residual.py)
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
| NVFP4 + sparse Piper | `nvfp4_sparse_piper` |
| ConvRot NVFP4 + sparse Piper | `convrot_nvfp4_sparse_piper` |

Each exports `<package_name>_compile_options()`. ConvRot INT8 dense fusion
supports MHA/GQA and optional `fuse_output=True` (bounded attention-to-output
execution, off by default); dense targets NVIDIA SM89/SM120 and AMD RDNA4, sparse
targets SM8x/SM120 and RDNA4. On NVIDIA D128, Gluon projections keep the shared
FP32 arithmetic but can differ from Triton by one INT8 code at rounding ties. See
the [fusion guide](../src/piper_kernels/fusions/convrot_int8_piper/README.md) for
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

When query and K/V head counts are even, half the output projection's input
width is a multiple of the ConvRot group size, and a sparse coarse gate (if
any) is a ConvRot INT8 projection of the attention's input, folded attention
also runs in two sequential head halves; otherwise the whole attention runs as
one slice. Slicing halves active K/V memory while keeping per-head sparse block
selection intact, leaves dense attention speed unchanged (sparse runs slightly
slower), and slightly alters numerics because each slice quantizes its own
share of the output projection input. See
[in-place residual adds](../src/piper_kernels/fusions/convrot_int8_piper/README.md#in-place-residual-adds)
for matcher and buffer-allocation details.

Return to [README](../README.md), [weights](weights.md), or
[benchmarks](../benchmarks/README.md).
