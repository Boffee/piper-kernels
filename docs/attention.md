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
It centers V for non-causal calls and restores the mean in the output; causal calls
leave V uncentered so future V values cannot affect earlier outputs through
quantization. Sage uses INT8 Q/K and FP8 probabilities/V with FP16 tile
accumulation. Their numerical results are not interchangeable.

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
the current Q/K values on every call. The default `"minmax"`
policy scores extrema summaries; `"mean"` scores block means. The application owns
the ratio profile and its quality validation. Even a full keep budget uses sparse
Piper's quantized arithmetic, which differs from dense Piper's V quantization.

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

Native sparse backends cover NVIDIA SM89/SM120 and AMD RDNA4. Other devices run the
portable quantized reference. There is a separate exact sparse reference for
quality comparisons; it is not the fallback used by this API.

### Caller-forced tiles (experimental)

The experimental fixed-budget forced-routing path additionally accepts
`force_tile_ids` (contiguous device INT32 `[query_blocks, capacity]`) and
`force_tile_counts` (contiguous device INT32 `[query_blocks]`). Active IDs in each
row's counted prefix are shared across batches and heads. Every forced tile is
included, and the remaining per-head budget is filled by ordinary score selection.
Counts must lie in `[0, capacity]`; active IDs must be unique and lie in
`[0, sparse_key_blocks)`, and each count must fit every affected head's resolved
physical keep budget. These are caller preconditions, with metadata-only validation.
Supply rows for every physical query block; dense query suffixes already include
every key and ignore route selection. Unused ID slots are ignored.

Query the budgets outside the compiled forward call when preparing forced lists:

```python
attention = SparsePiperAttention((0.5, 0.5))
budgets = attention.resolve_head_keep_blocks(sparse_key_blocks=3)  # (2, 1)
capacity = 2  # padded ID slots per query block
max_force_count = min(capacity, min(budgets))  # 1
```

Rounding is shared across heads, so independently rounding each ratio can
overestimate a head's budget. Here, each row may force at most one tile even
though the first head keeps two. The query allocates no tensors and accesses no
device state; recompute the limit when the ratio profile or sparse prefix size changes.

ID and count contents can change at fixed tensor shapes without recompiling; zero
counts disable forcing without changing the compiled call. Explicit graph replay
with a fixed ratio profile uses updated contents in the captured input buffers.
Compatible ConvRot INT8, NVFP4, and ConvRot NVFP4 projection/output fusions
preserve forced routing. The coarse residual is unaffected.

Changing sequence length or force capacity can also reuse a Torch graph when
the caller marks the corresponding input dimensions dynamic before the compiled
call. Mark the model input's sequence dimension and the query-row dimension of
both force tensors; mark the IDs' capacity dimension if it varies. For example:

```python
torch._dynamo.mark_dynamic(hidden_states, 1)  # [batch, sequence, features]
torch._dynamo.mark_dynamic(force_tile_ids, (0, 1))
torch._dynamo.mark_dynamic(force_tile_counts, 0)
```

Other sequence-shaped model inputs, such as rotary tables, need matching dynamic
dimensions. Changing a Python integer argument such as `sparse_key_blocks` can
still recompile; derive it from dynamic tensor shapes inside the compiled
function when applicable. Keep model head count and head dimension fixed.
Shape guards and GPU kernel policies still apply: one Torch graph does not
promise one kernel variant across every layout, alignment, or backend. Zero and
singleton dimensions may also receive separate compiler specializations.

### Coarse residual

For models with a coarse branch, `sparse_piper_coarse_residual` returns a separate
residual to add to the fine output. It derives mean/minmax block scores, mean-pools
V, expands the coarse result to token rows, and multiplies the supplied gate
directly; it does not apply a sigmoid or another activation. Q/K/V and the gate
must have the same shape, including equal head counts; the gate shares the query
dtype and device. Supply `routing="mean"` or `"minmax"` and a finite positive
`coarse_scale`. `coarse_key_blocks` defaults to all available blocks and may include
a compact partial tail. It also supports `block_lengths`.

See the [coarse API](../src/piper_kernels/attention/sparse_piper_attention/residual.py)
and [score-based coarse operations](../src/piper_kernels/attention/sparse_piper_attention/coarse.py)
for composing this branch or supplying learned block scores. Compatible compiler
fusions accumulate fine output and gated coarse contribution in FP32 before the
final output cast, so fused and separately materialized results can differ.

## Compile and fuse projections

The public attention operators support `torch.compile`; no special options are
needed for standalone attention. For example, the dense setup above can use
`compiled = torch.compile(piper_attention, fullgraph=True)` and then
`compiled(q, k, v)` with its dense-layout inputs.

To change sparse keep ratios, replace the `SparsePiperAttention` module with one
containing the new profile. Profiles with the same head count and routing policy
reuse a compiled graph, including compatible sparse projection and output fusions.
Ratio values are runtime operator inputs; other model and shape changes can still
require compilation. Operations that consume runtime profiles are excluded from
automatic CUDA graph capture, since replay would retain the captured ratios.
Without graph partitioning, this disables capture for the containing compiled
graph. Explicit CUDA graphs remain supported for a fixed profile; capture a new
graph when it changes.

For quantized projections around attention, opt into the matching compiler helper:

| Projection format and attention | Helper package under `piper_kernels.fusions` |
|---|---|
| ConvRot INT8 + dense Piper | `convrot_int8_piper` |
| ConvRot INT8 + sparse Piper | `convrot_int8_sparse_piper` |
| NVFP4 + sparse Piper | `nvfp4_sparse_piper` |
| ConvRot NVFP4 + sparse Piper | `convrot_nvfp4_sparse_piper` |

Each package exports `<package_name>_compile_options()` and includes the necessary
linear passes. Follow the shared [compiler-helper usage](weights.md#compile-a-model)
for passing options to `torch.compile` and handling unmatched patterns.

Dense ConvRot INT8 fusion supports MHA/GQA. Its optional `fuse_output=True` also
enables bounded attention-to-output-projection execution; output fusion defaults
to off. Supported patterns and numerical/storage contracts live in the
[fusion guide](../src/piper_kernels/fusions/convrot_int8_piper/README.md).
ConvRot INT8 sparse fusion supports NVIDIA SM89/SM120 and AMD RDNA4. SM89 D128
projection kernels preserve the shared FP32 stages but can differ by one INT8
code at rounding boundaries.
Sparse integrations additionally recognize compatible coarse branches and padded
layouts; their exact matchers live beside the helper in each package. Do not infer
fusion eligibility from the standalone attention API alone.

### Update a residual stream in place

Transformer blocks often add gated attention and FFN outputs to a residual stream.
When a compiled block may overwrite that input, `residual_updates_compile_options`
from `piper_kernels.fusions.residual_updates` makes those additions write into it
directly instead of allocating new full-size outputs. Its rewrite runs ahead of the
fusion passes, so apply it before or after the attention and FFN helpers:

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

The copy back declares that the caller no longer needs the original input; do not
read it, or any view of its storage, after the call. Without it the block is
unchanged.

Each gated sum `base + gate[indices] * update` that builds the copied-back result
from the input becomes an add into the input, whatever produced its update, which
removes the block's new output buffers. The helper's docstring states which gates,
updates, and devices qualify; per-token gates selected by index do, while blocks whose
gates broadcast across rows are left unchanged. When an FFN fusion,
`{convrot_int8,nvfp4,convrot_nvfp4}_{swiglu,gelu}_ffn`, or a fused ConvRot INT8 Piper
attention output, dense or sparse with fused Q projection, produced the update, that
fusion computes its output inside the add, which also removes that full-size output.
The rewritten graph stays functional, and Inductor adds an update in place only when
the block reads none of the input's original values afterward, through any view;
otherwise it copies the input first, which keeps the result correct but forgoes that
update's saving.
Updates round the residual separately, so results can differ slightly. The helper
also disables Inductor buffer reuse, which could otherwise hold a dead full-size
buffer across attention. See the
[fusion guide](../src/piper_kernels/fusions/convrot_int8_piper/README.md#in-place-residual-adds)
for numerical and storage details.

For installation and next steps, return to the [README](../README.md). See
[weights](weights.md) for constructing projection weights and
[benchmarks](../benchmarks/README.md) for reproducible quality and performance checks.
