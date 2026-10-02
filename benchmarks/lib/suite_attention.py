"""Full-call attention adapters and bounded references for the common case catalog."""

from __future__ import annotations

from functools import partial

import torch

from piper_kernels import SparsePiperAttention, piper_attention, sage_attention_2pp
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention.dispatch import _supports_triton as supports_piper
from piper_kernels.attention.sage_attention_2pp.dispatch import _supports_triton as supports_sage
from piper_kernels.attention.sparse_piper_attention._backend import (
    require_attention_backend,
    select_attention_backend,
)
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._routes import PackedRoutes
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import routing_mode_from_name

from .attention import AttentionConfig, AttentionInputs, attention_dtype, run_sdpa
from .cases import AttentionCase
from .quality import measure_quality
from .sparse_piper import reference_prepared_query
from .suite_types import Implementation, Operation, QualityCheck, normal_tensor, sample_indices


def make_inputs(case: AttentionCase, device: torch.device) -> AttentionInputs:
    """Generate the same head-major Q/K/V values for every provider and device."""
    q_shape = (case.batch, case.heads, case.sequence, case.head_dim)
    kv_shape = (case.batch, case.kv_heads, case.sequence, case.head_dim)
    dtype = attention_dtype(case.dtype)
    return (
        normal_tensor(q_shape, dtype=dtype, device=device, seed=case.seed),
        normal_tensor(kv_shape, dtype=dtype, device=device, seed=case.seed + 1),
        normal_tensor(kv_shape, dtype=dtype, device=device, seed=case.seed + 2),
    )


def sampled_dense_reference(
    inputs: AttentionInputs, rows: torch.Tensor, *, causal: bool = False
) -> torch.Tensor:
    """Evaluate sampled queries against all keys in FP32, one head at a time.

    Sampling the sequence before attention would change the softmax and could
    hide errors in long-context cases. The causal mask uses original row indices.
    """
    query, key, value = inputs
    batch, heads, _, width = query.shape
    reference = torch.empty((batch, heads, rows.numel(), width), device=query.device)
    key_positions = torch.arange(key.shape[2], device=key.device)
    for b in range(batch):
        for head in range(heads):
            kv_head = head // (heads // key.shape[1])
            q = query[b, head].index_select(0, rows).float()
            scores = (q @ key[b, kv_head].float().T) * width**-0.5
            if causal:
                scores.masked_fill_(key_positions[None, :] > rows[:, None], -torch.inf)
            reference[b, head] = scores.softmax(dim=-1) @ value[b, kv_head].float()
    return reference


def _dense_operation(case: AttentionCase, device: torch.device, name: str) -> Operation:
    inputs = make_inputs(case, device)
    config = AttentionConfig(attention_dtype(case.dtype), is_causal=case.causal, seed=case.seed)
    if name == "pytorch-sdpa":
        run = partial(run_sdpa, inputs, config)
        limit = 0.01
    else:
        function = piper_attention if name == "piper_attention" else sage_attention_2pp
        run = partial(function, *inputs, is_causal=case.causal)
        # A synthetic-input approximation gate, not a model-quality guarantee.
        limit = 0.06

    def check(output: torch.Tensor) -> QualityCheck:
        rows = sample_indices(case.sequence, device=device)
        expected = sampled_dense_reference(inputs, rows, causal=case.causal)
        actual = output.index_select(2, rows).float()
        return QualityCheck(
            metrics=measure_quality(actual, expected),
            reference="fp32_attention_full_kv_sampled_queries",
            sample_count=actual.numel(),
            total_count=output.numel(),
            relative_l2_limit=limit,
        )

    return Operation(run=run, check=check, configuration={"layout": "BHSD"})


def _selected_key_rows(
    routes: PackedRoutes, *, head: int, batch: int, block: int, sequence: int
) -> torch.Tensor:
    """Resolve the actual sparse support, including a dense ragged suffix."""
    device = routes.indices.device
    if routes.indices.shape[-1]:
        start, stop = routes.route_head_offsets[head : head + 2].tolist()
        tiles = routes.indices[batch, block, start:stop].long()
    else:
        tiles = torch.arange(sequence // 64, device=device)
    prefix = (tiles[:, None] * 64 + torch.arange(64, device=device)).flatten()
    return torch.cat((prefix, torch.arange(sequence // 64 * 64, sequence, device=device)))


def _sparse_operation(case: AttentionCase, device: torch.device) -> Operation:
    assert case.keep_ratio is not None
    ratios = (case.keep_ratio,) * case.heads
    inputs = make_inputs(case, device)
    query, key, value = inputs
    sequence_major = tuple(tensor.transpose(1, 2) for tensor in inputs)
    attention = SparsePiperAttention(ratios, routing="minmax")
    blocks = case.sequence // 64

    def run() -> torch.Tensor:
        return attention(*sequence_major, sparse_key_blocks=blocks)

    def check(output: torch.Tensor) -> QualityCheck:
        backend = require_attention_backend(query)
        layout = _resolve_route_layout(_normalize_head_keep_ratios(ratios), blocks, device)
        routes = packed_routes_from_sequences(
            query,
            key[:, :, : blocks * 64],
            layout,
            routing_mode_from_name("minmax"),
            skip_dense_routing=backend.skip_dense_routing,
        )
        prepared = backend.prepare(
            query,
            case.head_dim**-0.5,
            sparse_key_blocks=blocks,
            combined_key=key,
            combined_value=value,
        ).with_routes(routes.indices, routes.head_keep_blocks, routes.route_head_offsets)
        actual_samples, quantized_samples, floating_samples = [], [], []
        for b in range(case.batch):
            for head in sorted({0, case.heads // 2, case.heads - 1}):
                kv_head = head // (case.heads // case.kv_heads)
                query_blocks = (case.sequence + 63) // 64
                for block in sorted({0, query_blocks // 2, query_blocks - 1}):
                    start, stop = block * 64, min((block + 1) * 64, case.sequence)
                    actual_samples.append(output[b, start:stop, head].float())
                    quantized_samples.append(
                        reference_prepared_query(
                            prepared, b, head, block, output_dtype=output.dtype
                        ).float()
                    )
                    indices = _selected_key_rows(
                        routes, head=head, batch=b, block=block, sequence=case.sequence
                    )
                    q = query[b, head, start:stop].float()
                    k = key[b, kv_head].index_select(0, indices).float()
                    v = value[b, kv_head].index_select(0, indices).float()
                    probabilities = ((q @ k.T) * case.head_dim**-0.5).softmax(dim=-1)
                    floating_samples.append(probabilities @ v)
        actual = torch.cat(actual_samples)
        return QualityCheck(
            metrics=measure_quality(actual, torch.cat(quantized_samples)),
            reference="fp64_quantized_sparse_full_selected_kv_sampled_query_blocks",
            sample_count=actual.numel(),
            total_count=output.numel(),
            relative_l2_limit=0.015,
            comparisons={
                "fp32_same_sparse_routes": measure_quality(actual, torch.cat(floating_samples))
            },
        )

    return Operation(
        run=run,
        check=check,
        configuration={"layout": "BSHD", "routing": "minmax", "sparse_key_blocks": blocks},
    )


def implementations(case: AttentionCase, device: torch.device) -> list[Implementation]:
    """Enumerate native providers explicitly; portable fallbacks are not performance entries."""
    if case.keep_ratio is not None:
        reason = None
        if case.causal:
            reason = "Sparse Piper supports non-causal self-attention only"
        elif case.sequence < 64:
            reason = "Sparse Piper requires at least one complete K64 block"
        elif select_attention_backend(torch.empty(0, device=device)) is None:
            reason = "No native sparse Piper backend for this device and installation"
        return [
            Implementation(
                "sparse_piper_attention", partial(_sparse_operation, case, device), reason
            )
        ]

    target = AcceleratorTarget.from_device(device)
    piper_reason = (
        None
        if supports_piper(target)
        else "No native Piper backend for this device and installation"
    )
    sage_reason = (
        None if supports_sage(target) else "Native Sage requires NVIDIA FP8 tensor cores and Triton"
    )
    if case.heads != case.kv_heads:
        sage_reason = "Sage requires equal query and key/value head counts"
    return [
        Implementation("pytorch-sdpa", partial(_dense_operation, case, device, "pytorch-sdpa")),
        Implementation(
            "piper_attention",
            partial(_dense_operation, case, device, "piper_attention"),
            piper_reason,
        ),
        Implementation(
            "sage_attention_2pp",
            partial(_dense_operation, case, device, "sage_attention_2pp"),
            sage_reason,
        ),
    ]
