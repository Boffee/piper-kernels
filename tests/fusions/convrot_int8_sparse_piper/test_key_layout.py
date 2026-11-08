"""Regional routing retains output fusion, coarse execution, and head slicing."""

import pytest
import torch
from _compile_capture import FunctionalTargetCapturePass

from piper_kernels import (
    SparsePiperAttention,
    SparsePiperKeyLayout,
    SparsePiperKeyRegion,
    sparse_piper_coarse_residual,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
    output,
)
from piper_kernels.fusions.residual_updates import residual_updates_compile_options

from ._helpers import output_available
from .test_compile import (
    _DynamicSparseProjectionAttentionOutput,
    _make_output_projection,
    _run_explicit_attention_output,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not output_available(), reason="requires fused sparse output"),
]


class _RegionAttention(_DynamicSparseProjectionAttentionOutput):
    heads = 4
    output_features = 256

    def __init__(self, coarse, routing="minmax"):
        super().__init__()
        self.sparse_attention = SparsePiperAttention((0.3,) * self.heads, routing=routing)
        self.routing = routing
        self.gate = (
            _make_output_projection(self.input_features, self.heads * self.head_dim, bias=False)
            if coarse
            else None
        )

    def forward(self, hidden, cos, sin, key_layout, lengths, ids, counts, query_blocks):
        batch, sequence, _ = hidden.shape
        q = self._dynamic_norm_rope(self.query(hidden), self.query_norm, cos, sin)
        k = self._dynamic_norm_rope(self.key(hidden), self.key_norm, cos, sin)
        v = self.value(hidden).view(batch, sequence, self.heads, self.head_dim)
        attended = self.sparse_attention(
            q,
            k,
            v,
            key_layout=key_layout,
            block_lengths=lengths,
            sparse_query_blocks=query_blocks,
            force_tile_ids=ids,
            force_tile_counts=counts,
        )
        if self.gate is not None:
            gate = self.gate(hidden).reshape(batch, sequence, self.heads, self.head_dim)
            attended = attended + sparse_piper_coarse_residual(
                q,
                k,
                v,
                gate,
                routing=self.routing,
                coarse_scale=0.125,
                coarse_key_blocks=sequence // 64,
                block_lengths=lengths,
            )
        return self.output(attended.flatten(2))


class _Block(torch.nn.Module):
    def __init__(self, coarse, residual, routing="minmax"):
        super().__init__()
        self.attention = _RegionAttention(coarse, routing)
        self.residual = residual
        self.register_buffer("gate", torch.randn(2, 256, device="cuda", dtype=torch.bfloat16) * 0.1)

    def forward(self, hidden, indices, cos, sin, key_layout, lengths, ids, counts, query_blocks):
        attended = self.attention(hidden, cos, sin, key_layout, lengths, ids, counts, query_blocks)
        if self.residual:
            return hidden.copy_(hidden + self.gate[indices] * attended)
        return attended


def _inputs(layout, padded=False):
    sequence = layout.num_blocks * 64
    hidden = torch.randn(1, sequence, 256, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(sequence, device="cuda") % 2
    angles = torch.randn(sequence, 96, device="cuda")
    lengths = ids = counts = None
    if padded:
        lengths = torch.full((layout.num_blocks,), 59, device="cuda", dtype=torch.int32)
        ids = torch.full(
            (layout.num_blocks, 1), layout.num_blocks - 1, device="cuda", dtype=torch.int32
        )
        counts = torch.ones(layout.num_blocks, device="cuda", dtype=torch.int32)
    return hidden, indices, angles.cos(), angles.sin(), layout, lengths, ids, counts, None


def _reference(model, args):
    hidden, indices, cos, sin, layout, lengths, ids, counts, query_blocks = args
    gate = model.attention.gate
    coarse_gate = (
        None
        if gate is None
        else gate(hidden).reshape(
            1, hidden.shape[1], model.attention.heads, model.attention.head_dim
        )
    )
    projected, _ = _run_explicit_attention_output(
        model.attention,
        hidden,
        cos,
        sin,
        layout,
        coarse_gate=coarse_gate,
        coarse_scale=None if gate is None else 0.125,
        coarse_key_blocks=None if gate is None else layout.num_blocks,
        block_lengths=lengths,
        sparse_query_blocks=query_blocks,
        force_tile_ids=ids,
        force_tile_counts=counts,
    )
    return (
        (hidden.float() + model.gate[indices].float() * projected.float()).to(hidden.dtype)
        if model.residual
        else projected
    )


def _assert_close(actual, expected, lengths):
    if lengths is not None:
        valid = (torch.arange(64, device="cuda")[None, :] < lengths[:, None]).flatten()
        actual, expected = actual[:, valid], expected[:, valid]
    assert torch.isfinite(actual).all()
    # Head slices quantize independently and round the residual once per slice.
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.02


@pytest.mark.parametrize(
    ("coarse", "residual", "routing"),
    [
        (False, False, "minmax"),
        (True, False, "minmax"),
        (False, True, "minmax"),
        (True, True, "minmax"),
        (True, True, "mean"),
    ],
)
def test_regions_fuse_projected_queries_outputs_and_coarse(coarse, residual, routing, monkeypatch):
    torch.manual_seed(1907)
    torch._dynamo.reset()
    model = _Block(coarse, residual, routing)
    if routing == "mean":
        # Independently scaled Q/gate preparation must stay inside each window.
        model.attention.query.weight.act_per_tensor_scale = torch.tensor(0.04, device="cuda")
        model.attention.gate.weight.act_per_tensor_scale = torch.tensor(0.06, device="cuda")
    # Each eight-block sparse region rounds to (3, 3, 2, 2) over all four heads.
    layout = SparsePiperKeyLayout(
        (
            SparsePiperKeyRegion(8),
            SparsePiperKeyRegion(1, sparse=False),
            SparsePiperKeyRegion(8),
            SparsePiperKeyRegion(8),
        )
    )
    args = _inputs(layout, padded=True)
    args = (*args[:-1], layout.num_blocks - 2)  # Dense queries at the end.
    # Small windows exercise global row/forced-route offsets across several chunks.
    monkeypatch.setattr(output, "_DEFAULT_QUERY_CHUNK_ROWS", 128)
    capture = FunctionalTargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    if residual:
        options = residual_updates_compile_options(options)
    compiled = torch.compile(model, fullgraph=True, options=capture.wrap_options(options))
    with torch.inference_mode():
        expected = _reference(model, args)
        actual = compiled(*args)
    _assert_close(actual, expected, args[5])
    ops = torch.ops.piper_kernels
    target = (
        ops.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default
        if residual
        else ops.convrot_int8_sparse_piper_projected_query_attention_output.default
    )
    slices = 2 if residual else 1
    assert capture.targets.count(target) == slices, capture.target_names
    value_target = (
        ops.convrot_int8_sparse_piper_project_value_with_block_means.default
        if coarse
        else ops.convrot_int8_sparse_piper_project_value.default
    )
    assert capture.targets.count(value_target) == slices
    assert ops.sparse_piper_attention.default not in capture.targets
    if residual:
        assert actual.data_ptr() == args[0].data_ptr()


def test_regional_residual_reuses_graph_for_layouts_and_sequence_lengths():
    torch.manual_seed(1911)
    torch._dynamo.reset()
    model = _Block(coarse=True, residual=True)
    capture = FunctionalTargetCapturePass()
    compiled = torch.compile(
        model,
        fullgraph=True,
        options=capture.wrap_options(
            residual_updates_compile_options(convrot_int8_sparse_piper_compile_options())
        ),
    )
    layouts = (
        SparsePiperKeyLayout((SparsePiperKeyRegion(16),)),
        SparsePiperKeyLayout((SparsePiperKeyRegion(8), SparsePiperKeyRegion(8))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(8), SparsePiperKeyRegion(9))),
        SparsePiperKeyLayout((SparsePiperKeyRegion(2, sparse=False), SparsePiperKeyRegion(16))),
        SparsePiperKeyLayout.prefix(4, 16),
        # Compact scores skip both dense regions.
        SparsePiperKeyLayout(
            (
                SparsePiperKeyRegion(6),
                SparsePiperKeyRegion(2, sparse=False),
                SparsePiperKeyRegion(6),
                SparsePiperKeyRegion(2, sparse=False),
            )
        ),
    )
    with torch.inference_mode():
        for layout in layouts:
            args = _inputs(layout, padded=True)
            torch._dynamo.mark_dynamic(args[0], 1)
            for index in (1, 2, 3, 5, 6, 7):
                torch._dynamo.mark_dynamic(args[index], 0)
            expected = _reference(model, args)
            actual = compiled(*args)
            _assert_close(actual, expected, args[5])
        invalid = SparsePiperKeyLayout((SparsePiperKeyRegion(15),))
        with pytest.raises(ValueError, match="key_layout must cover exactly"):
            compiled(*args[:4], invalid, *args[5:])
    assert capture.calls == 1
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default
        )
        == 2
    )
