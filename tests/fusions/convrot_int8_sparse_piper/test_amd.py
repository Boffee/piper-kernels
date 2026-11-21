"""RDNA4 fused projections against independent FP64 math at H3 dimensions."""

import sys
from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._amd import gluon as amd_attention
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
    _SparsePiperRatioProfile,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import _backend, key, output, query, value
from piper_kernels.fusions.convrot_int8_sparse_piper import triton as shared_triton
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import policy
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import triton as amd
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import (
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    padded_sequence_length,
)
from piper_kernels.linear.convrot.int8 import _backend as linear_backend

from ._reference import (
    assert_int8_codes_close,
    check_qk_sample_fp64,
    check_value_sample_fp64,
    reconstruct_key_center,
)


def _available():
    return torch.cuda.is_available() and policy.supports_target(
        AcceleratorTarget.from_device(torch.device("cuda"))
    )


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201", "gfx1100", "gfx942", "gfx9999"])
def test_amd_projection_support_is_limited_to_linux_and_windows_rdna4(monkeypatch, platform, arch):
    monkeypatch.setattr(sys, "platform", platform)
    assert policy.supports_target(AcceleratorTarget("hip", arch)) is (
        platform in ("linux", "win32") and arch in ("gfx1200", "gfx1201")
    )


def _assert_projection_matches(operation, actual, expected, *, allow_empty_nan=False):
    assert_int8_codes_close(actual[0], expected[0])
    if operation == "key":
        torch.testing.assert_close(actual[1], expected[1], rtol=3e-5, atol=2e-5)
        actual_center = reconstruct_key_center(actual[2], actual[3])
        expected_center = reconstruct_key_center(expected[2], expected[3])
        torch.testing.assert_close(actual_center, expected_center, rtol=2e-4, atol=2e-4)
        tail = zip(actual[4:], expected[4:], strict=True)
    else:
        tail = zip(actual[1:], expected[1:], strict=True)
    for actual_metadata, expected_metadata in tail:
        rtol = 1.6e-2 if expected_metadata.dtype is torch.bfloat16 else 3e-5
        if not allow_empty_nan and expected_metadata.numel() > 0:
            assert not bool(actual_metadata.isnan().any())
        torch.testing.assert_close(
            actual_metadata,
            expected_metadata,
            rtol=rtol,
            atol=2e-4,
            equal_nan=allow_empty_nan,
        )


def _filled_outputs(
    operation,
    device,
    *,
    batch,
    heads,
    storage,
    head_dim,
    routing_mode=_MINMAX_ROUTING,
    value_format=ValueFormat.AMD_WMMA,
):
    if operation == "query":
        return (
            torch.full((batch, heads, storage, head_dim), 99, device=device, dtype=torch.int8),
            torch.full(
                (batch, heads, storage // QUERY_SCALE_ROWS),
                -7.0,
                device=device,
                dtype=torch.float32,
            ),
            torch.full(
                (batch, heads, storage // TILE_ROWS, head_dim),
                -7.0,
                device=device,
                dtype=torch.float32,
            ),
        )
    tiles = storage // TILE_ROWS
    if operation == "key":
        return (
            torch.full((batch, heads, storage, head_dim), 99, device=device, dtype=torch.int8),
            torch.full((batch, heads, tiles), -7.0, device=device, dtype=torch.float32),
            torch.full((batch, heads, tiles, 2, head_dim), 99, device=device, dtype=torch.int8),
            torch.full((batch, heads, tiles, 2), -7.0, device=device, dtype=torch.float32),
            torch.full((batch, heads, tiles, head_dim), -7.0, device=device, dtype=torch.float32),
            torch.full(
                (batch, heads, 0 if routing_mode == _MEAN_ROUTING else tiles, head_dim),
                -7.0,
                device=device,
                dtype=torch.float32,
            ),
        )
    return (
        torch.full(
            value_format.shape(batch, heads, storage, head_dim),
            99,
            device=device,
            dtype=torch.int8,
        ),
        torch.full((batch, heads, tiles, 1), -7.0, device=device, dtype=torch.float32),
        torch.full((batch, heads, tiles, head_dim), -7.0, device=device, dtype=torch.bfloat16),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("width", "heads"), [(272, 3), (256, 4)])
def test_grouped_order_preserves_complete_outputs_with_partial_groups_and_query_window(
    monkeypatch, head_dim, width, heads
):
    torch.manual_seed(137)
    batch, sequence = 2, 1217
    qdata = torch.randint(-128, 128, (batch, sequence, width), device="cuda", dtype=torch.int8)
    scale = torch.rand((batch, sequence), device="cuda") * 0.01 + 0.001
    weight = torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5
    angles = torch.rand((sequence, head_dim * 3 // 4), device="cuda")
    cos, sin = angles.cos(), angles.sin()

    def run():
        return (
            query._launch_query_projection_range(
                qdata,
                scale,
                weight,
                weight_scale,
                norm,
                cos,
                sin,
                1e-5,
                head_dim**-0.5,
                0,
                chunk_start=128,
                chunk_rows=1089,
            ),
            key._project_key_op(qdata, scale, weight, weight_scale, norm, cos, sin, 1e-5, 0),
            value._project_value_op(
                qdata,
                scale,
                weight,
                weight_scale,
                head_dim=head_dim,
                value_format=int(ValueFormat.AMD_WMMA),
            ),
        )

    actual = run()
    select_plan = policy.select_execution_plan
    monkeypatch.setattr(
        policy,
        "select_execution_plan",
        lambda *args, **kwargs: replace(select_plan(*args, **kwargs), group_m=0),
    )
    expected = run()
    for operation, left, right in zip(("query", "key", "value"), actual, expected, strict=True):
        _assert_projection_matches(operation, left, right)


@pytest.mark.usefixtures("large_device_memory")
@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
@pytest.mark.parametrize("sequence", [193, 8192, 100_000])
@pytest.mark.parametrize(("width", "heads", "head_dim"), [(2048, 32, 64), (5376, 56, 128)])
@pytest.mark.parametrize("affine", [True, False])
def test_h3_projections_match_sampled_fp64_math_through_100k(
    sequence, width, heads, head_dim, affine
):
    # H3 VAE decoder and diffusion transformer projection dimensions.
    # Execute every row; verify complete first/middle/last K64 blocks and heads.
    torch.manual_seed(731)
    rotary = head_dim * 3 // 4
    qdata = torch.randint(-128, 128, (1, sequence, width), device="cuda", dtype=torch.int8)
    scale = torch.rand((1, sequence), device="cuda") * 0.01 + 0.001
    weights = [
        torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
        for _ in range(3)
    ]
    scales = [torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001 for _ in range(3)]
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5 if affine else None
    angles = torch.rand((sequence, rotary), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    # Exercise device guards while using a non-default current stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q = query._project_query_op(
            qdata,
            scale,
            weights[0],
            scales[0],
            norm,
            cos,
            sin,
            1e-5,
            head_dim**-0.5,
            0,
            head_dim=head_dim,
        )
        k = key._project_key_op(
            qdata, scale, weights[1], scales[1], norm, cos, sin, 1e-5, 0, head_dim=head_dim
        )
        v = value._project_value_op(
            qdata,
            scale,
            weights[2],
            scales[2],
            head_dim=head_dim,
            value_format=int(ValueFormat.AMD_WMMA),
        )
    stream.synchronize()
    assert _backend.select_output_backend(q[0]) is not None

    storage = (sequence + 63) // 64 * 64
    for head in (0, heads // 2, heads - 1):
        columns = slice(head * head_dim, (head + 1) * head_dim)
        for start in sorted({0, (sequence // 2) // 64 * 64, (sequence - 1) // 64 * 64}):
            rows = min(64, sequence - start)
            for index, projected_operands in enumerate((q, k, v)):
                projected = (
                    qdata[0, start : start + rows].double() @ weights[index][columns].double().T
                )
                projected *= scale[0, start : start + rows, None].double()
                projected *= scales[index][columns, 0].double()
                if index < 2:
                    check_qk_sample_fp64(
                        projected,
                        projected_operands,
                        head,
                        start,
                        rows,
                        norm,
                        cos,
                        sin,
                        is_query=index == 0,
                    )
                else:
                    check_value_sample_fp64(projected, projected_operands, head, start, rows)
    assert q[0].shape == (1, heads, storage, head_dim)
    assert k[0].shape == (1, heads, storage, head_dim)
    assert v[0].shape == ValueFormat.AMD_WMMA.shape(1, heads, storage, head_dim)


@pytest.mark.usefixtures("large_device_memory")
@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
@pytest.mark.parametrize("sequence", [193, 8193, 100_000])
@pytest.mark.parametrize(("width", "heads", "head_dim"), [(2048, 32, 64), (5376, 56, 128)])
@torch.no_grad()
def test_h3_chunked_output_matches_materialized_boundary_with_bounded_workspace(  # noqa: PLR0915
    monkeypatch, sequence, width, heads, head_dim
):
    torch.manual_seed(733)
    chunk_rows = 4096
    qdata = torch.randint(-128, 128, (1, sequence, width), device="cuda", dtype=torch.int8)
    scale = torch.rand((1, sequence), device="cuda") * 0.01 + 0.001
    weights = [
        torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
        for _ in range(3)
    ]
    scales = [torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001 for _ in weights]
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5
    angles = torch.rand((sequence, head_dim * 3 // 4), device="cuda") * (2 * torch.pi)
    cos, sin = angles.cos(), angles.sin()
    query_args = qdata, scale, weights[0], scales[0], norm, cos, sin, 1e-5, head_dim**-0.5
    q = query._project_query_op(*query_args, 0)
    k = key._project_key_op(qdata, scale, weights[1], scales[1], norm, cos, sin, 1e-5, 0)
    v = value._project_value_op(
        qdata,
        scale,
        weights[2],
        scales[2],
        head_dim=head_dim,
        value_format=int(ValueFormat.AMD_WMMA),
    )
    profile = _SparsePiperRatioProfile((250_000,) * heads)
    attention_tail = *k, *v, profile, sequence // 64, sequence, 0
    attention = _quantized_dispatch._sparse_piper_attention_from_quantized_op(*q, *attention_tail)
    out_weight = torch.randint(
        -128, 128, (width, heads * head_dim), device="cuda", dtype=torch.int8
    )
    out_scale = torch.rand((width, 1), device="cuda") * 0.01 + 0.001
    bias = torch.randn(width, device="cuda", dtype=torch.float32)
    projection_args = out_weight, out_scale, bias, 256
    expected = linear_backend.require_linear_backend(attention).linear(
        attention.flatten(2), *projection_args
    )
    del q, attention
    torch.cuda.synchronize()

    windows, parameter_calls = [], []
    project_query = query._launch_query_projection_range
    prepare_parameters = amd_attention.prepare_parameters

    def track_query(*args, **kwargs):
        windows.append((kwargs["chunk_start"], kwargs["chunk_rows"]))
        assert kwargs["chunk_rows"] <= chunk_rows
        return project_query(*args, **kwargs)

    def track_parameters(context):
        parameter_calls.append(True)
        return prepare_parameters(context)

    monkeypatch.setattr(query, "_launch_query_projection_range", track_query)
    monkeypatch.setattr(amd_attention, "prepare_parameters", track_parameters)
    # Reuse each ping-pong slot many times, starting from a caller's side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.stream(stream):
        actual = output._projected_query_attention_output_op(
            *query_args, *attention_tail, *projection_args, chunk_rows
        )
    stream.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert windows == [
        (start, min(chunk_rows, sequence - start)) for start in range(0, sequence, chunk_rows)
    ]
    assert len(parameter_calls) == 1
    assert actual.shape == expected.shape == (1, sequence, width)
    assert actual.is_contiguous()
    # Bound comparison workspace too; every output row participates in the check.
    error, energy = 0.0, 0.0
    for start in range(0, sequence, chunk_rows):
        left = actual[:, start : start + chunk_rows].float()
        right = expected[:, start : start + chunk_rows].float()
        assert bool(torch.isfinite(left).all())
        error += float((left - right).square().sum())
        energy += float(right.square().sum())
    assert (error / max(energy, 1e-30)) ** 0.5 < 0.015
    if sequence == 100_000:
        # Only the final output grows with S; V is already in its final format.
        assert peak < actual.numel() * actual.element_size() + 512 * 1024**2


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
@pytest.mark.parametrize(
    ("head_dim", "rotary_dim"),
    [(hd, rd) for hd, dims in policy.GLUON_ROTARY_DIMS.items() for rd in dims],
)
@pytest.mark.parametrize("variant", ["full_minmax", "window_mean_block_lengths"])
def test_amd_gluon_projections_match_triton_across_rotary_and_edge_modes(
    head_dim, rotary_dim, variant
):
    torch.manual_seed(907)
    batch, sequence = 2, 205
    width = 192 if head_dim == 64 else 256
    heads = 4 if head_dim == 64 else 3
    qdata = torch.randint(-128, 128, (batch, sequence, width), device="cuda", dtype=torch.int8)
    scale = torch.rand((batch, sequence), device="cuda") * 0.01 + 0.001
    weights = [
        torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
        for _ in range(3)
    ]
    scales = [torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001 for _ in range(3)]
    windowed = variant == "window_mean_block_lengths"
    affine = not windowed
    bias = (
        torch.randn((heads * head_dim,), device="cuda", dtype=torch.bfloat16) if windowed else None
    )
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5 if affine else None
    routing_mode = _MEAN_ROUTING if windowed else _MINMAX_ROUTING
    angles = torch.rand((sequence, rotary_dim), device="cuda") * (2 * torch.pi)
    cos, sin = angles.cos(), angles.sin()
    block_lengths = (
        torch.tensor([64, 0, 17, 13], device="cuda", dtype=torch.int32) if windowed else None
    )
    chunk_start, chunk_rows = (64, 141) if windowed else (0, sequence)
    storage_q = padded_sequence_length(chunk_rows)
    storage_kv = padded_sequence_length(sequence)

    triton_qk = policy.select_execution_plan("key", head_dim=head_dim)
    triton_v = policy.select_execution_plan("value", head_dim=head_dim)
    assert triton_qk.kernel == "triton"
    assert triton_v.kernel == "triton"
    assert (
        amd.default_execution_plan(
            qdata,
            weights[0],
            operation="query",
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            storage_sequence_length=storage_q,
        ).kernel
        == "gluon"
    )
    assert (
        amd.default_execution_plan(
            qdata,
            weights[1],
            operation="key",
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            storage_sequence_length=storage_kv,
        ).kernel
        == "gluon"
    )
    assert (
        amd.default_execution_plan(
            qdata,
            weights[2],
            operation="value",
            head_dim=head_dim,
            storage_sequence_length=storage_kv,
            value_format=ValueFormat.AMD_WMMA,
        ).kernel
        == "gluon"
    )

    q_actual = _filled_outputs(
        "query", "cuda", batch=batch, heads=heads, storage=storage_q, head_dim=head_dim
    )
    q_expected = _filled_outputs(
        "query", "cuda", batch=batch, heads=heads, storage=storage_q, head_dim=head_dim
    )
    amd.project_query(
        qdata,
        scale,
        weights[0],
        scales[0],
        norm,
        cos,
        sin,
        1e-5,
        head_dim**-0.5,
        routing_mode,
        block_lengths,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=q_actual,
        bias=bias,
    )
    shared_triton.project_query(
        qdata,
        scale,
        weights[0],
        scales[0],
        norm,
        cos,
        sin,
        1e-5,
        head_dim**-0.5,
        routing_mode,
        block_lengths,
        execution_plan=triton_qk,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=q_expected,
        bias=bias,
    )
    _assert_projection_matches("query", q_actual, q_expected, allow_empty_nan=windowed)

    k_actual = _filled_outputs(
        "key",
        "cuda",
        batch=batch,
        heads=heads,
        storage=storage_kv,
        head_dim=head_dim,
        routing_mode=routing_mode,
    )
    k_expected = _filled_outputs(
        "key",
        "cuda",
        batch=batch,
        heads=heads,
        storage=storage_kv,
        head_dim=head_dim,
        routing_mode=routing_mode,
    )
    amd.project_key(
        qdata,
        scale,
        weights[1],
        scales[1],
        norm,
        cos,
        sin,
        1e-5,
        routing_mode,
        block_lengths,
        out=k_actual,
        bias=bias,
    )
    shared_triton.project_key(
        qdata,
        scale,
        weights[1],
        scales[1],
        norm,
        cos,
        sin,
        1e-5,
        routing_mode,
        block_lengths,
        execution_plan=triton_qk,
        out=k_expected,
        bias=bias,
    )
    _assert_projection_matches("key", k_actual, k_expected, allow_empty_nan=windowed)

    if rotary_dim == policy.GLUON_ROTARY_DIMS[head_dim][0]:
        v_actual = _filled_outputs(
            "value", "cuda", batch=batch, heads=heads, storage=storage_kv, head_dim=head_dim
        )
        v_expected = _filled_outputs(
            "value", "cuda", batch=batch, heads=heads, storage=storage_kv, head_dim=head_dim
        )
        amd.project_value(
            qdata,
            scale,
            weights[2],
            scales[2],
            block_lengths,
            value_format=ValueFormat.AMD_WMMA,
            out=v_actual,
            bias=bias,
        )
        shared_triton.project_value(
            qdata,
            scale,
            weights[2],
            scales[2],
            block_lengths,
            execution_plan=triton_v,
            value_format=ValueFormat.AMD_WMMA,
            out=v_expected,
            bias=bias,
        )
        _assert_projection_matches("value", v_actual, v_expected)


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
@pytest.mark.parametrize(("head_dim", "heads", "rotary_dim"), [(64, 4, 48), (128, 3, 96)])
def test_amd_gluon_public_window_bias_and_partial_kv_contracts(head_dim, heads, rotary_dim):
    torch.manual_seed(911)
    batch, sequence, width = 2, 256, 256
    qdata = torch.randint(-128, 128, (batch, sequence, width), device="cuda", dtype=torch.int8)
    scale = torch.rand((batch, sequence), device="cuda") * 0.01 + 0.001
    weights = [
        torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
        for _ in range(3)
    ]
    scales = [torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001 for _ in range(3)]
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5
    angles = torch.rand((sequence, rotary_dim), device="cuda") * (2 * torch.pi)
    cos, sin = angles.cos(), angles.sin()
    bias_f32 = torch.randn((heads * head_dim,), device="cuda", dtype=torch.float32)

    # Empty-tile padding equivalence: chunk_rows=77 vs chunk_rows=128 with shortened block_lengths.
    q_tail = query._launch_query_projection_range(
        qdata,
        scale,
        weights[0],
        scales[0],
        norm,
        cos,
        sin,
        1e-5,
        head_dim**-0.5,
        _MINMAX_ROUTING,
        chunk_start=64,
        chunk_rows=77,
        bias=bias_f32,
    )
    short_lengths = torch.tensor([64, 64, 13, 0], device="cuda", dtype=torch.int32)
    q_masked = query._launch_query_projection_range(
        qdata,
        scale,
        weights[0],
        scales[0],
        norm,
        cos,
        sin,
        1e-5,
        head_dim**-0.5,
        _MINMAX_ROUTING,
        short_lengths,
        chunk_start=64,
        chunk_rows=128,
        bias=bias_f32,
    )
    for left, right in zip(q_tail, q_masked, strict=True):
        assert torch.equal(left, right)

    # Partial K/V launches preserve block_m=128 overlap and match full launches bitwise.
    seq_kv = 384
    qk_data = torch.randint(-128, 128, (1, seq_kv, width), device="cuda", dtype=torch.int8)
    kv_scale = torch.rand((1, seq_kv), device="cuda") * 0.01 + 0.001
    kv_angles = torch.rand((seq_kv, rotary_dim), device="cuda") * (2 * torch.pi)
    k_cos, k_sin = kv_angles.cos(), kv_angles.sin()
    assert (
        amd.partial_row_block_size(
            qk_data, weights[1], operation="key", head_dim=head_dim, rotary_dim=rotary_dim
        )
        == 128
    )
    assert (
        amd.partial_row_block_size(
            qk_data,
            weights[2],
            operation="value",
            head_dim=head_dim,
            value_format=ValueFormat.AMD_WMMA,
        )
        == 128
    )

    k_full = _filled_outputs("key", "cuda", batch=1, heads=heads, storage=seq_kv, head_dim=head_dim)
    k_split = _filled_outputs(
        "key", "cuda", batch=1, heads=heads, storage=seq_kv, head_dim=head_dim
    )
    amd.project_key(
        qk_data, kv_scale, weights[1], scales[1], norm, k_cos, k_sin, 1e-5, 0, None, out=k_full
    )
    amd.project_key(
        qk_data,
        kv_scale,
        weights[1],
        scales[1],
        norm,
        k_cos,
        k_sin,
        1e-5,
        0,
        None,
        out=k_split,
        start_row_block=0,
        num_row_blocks=1,
    )
    amd.project_key(
        qk_data,
        kv_scale,
        weights[1],
        scales[1],
        norm,
        k_cos,
        k_sin,
        1e-5,
        0,
        None,
        out=k_split,
        start_row_block=1,
        num_row_blocks=None,
    )
    for left, right in zip(k_full, k_split, strict=True):
        assert torch.equal(left, right)

    v_full = _filled_outputs(
        "value", "cuda", batch=1, heads=heads, storage=seq_kv, head_dim=head_dim
    )
    v_split = _filled_outputs(
        "value", "cuda", batch=1, heads=heads, storage=seq_kv, head_dim=head_dim
    )
    amd.project_value(
        qk_data,
        kv_scale,
        weights[2],
        scales[2],
        None,
        value_format=ValueFormat.AMD_WMMA,
        out=v_full,
        bias=bias_f32,
    )
    amd.project_value(
        qk_data,
        kv_scale,
        weights[2],
        scales[2],
        None,
        value_format=ValueFormat.AMD_WMMA,
        out=v_split,
        bias=bias_f32,
        start_row_block=1,
        num_row_blocks=2,
    )
    amd.project_value(
        qk_data,
        kv_scale,
        weights[2],
        scales[2],
        None,
        value_format=ValueFormat.AMD_WMMA,
        out=v_split,
        bias=bias_f32,
        start_row_block=0,
        num_row_blocks=1,
    )
    for left, right in zip(v_full, v_split, strict=True):
        assert torch.equal(left, right)

    seq_oversized = 192
    oversized_storage = seq_oversized + 128
    active_tiles = seq_oversized // TILE_ROWS
    v_in = qk_data[:, :seq_oversized].contiguous()
    v_in_scale = kv_scale[:, :seq_oversized].contiguous()
    v_exact = _filled_outputs(
        "value", "cuda", batch=1, heads=heads, storage=seq_oversized, head_dim=head_dim
    )
    v_oversized = _filled_outputs(
        "value", "cuda", batch=1, heads=heads, storage=oversized_storage, head_dim=head_dim
    )
    amd.project_value(
        v_in,
        v_in_scale,
        weights[2],
        scales[2],
        None,
        value_format=ValueFormat.AMD_WMMA,
        out=v_exact,
        bias=bias_f32,
    )
    amd.project_value(
        v_in,
        v_in_scale,
        weights[2],
        scales[2],
        None,
        value_format=ValueFormat.AMD_WMMA,
        out=v_oversized,
        bias=bias_f32,
    )
    for actual_tensor, expected_tensor in zip(v_oversized, v_exact, strict=True):
        assert torch.equal(actual_tensor[:, :, :active_tiles], expected_tensor)
        sentinel = 99 if actual_tensor.dtype is torch.int8 else -7.0
        assert torch.all(actual_tensor[:, :, active_tiles:] == sentinel)


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
def test_amd_triton_fallbacks_cover_odd_d64_heads_unaligned_width_rotary_and_transposed_v():
    torch.manual_seed(919)
    cases = [
        # Sequence length, width, head count, head width, rotary width, and value storage layout.
        (137, 256, 3, 64, 48, ValueFormat.AMD_WMMA),
        (137, 272, 4, 64, 48, ValueFormat.AMD_WMMA),
        (137, 256, 4, 64, 16, ValueFormat.AMD_WMMA),
        (137, 256, 3, 128, 96, ValueFormat.TRANSPOSED),
    ]
    for sequence, width, heads, head_dim, rotary_dim, value_format in cases:
        qdata = torch.randint(-128, 128, (1, sequence, width), device="cuda", dtype=torch.int8)
        scale = torch.rand((1, sequence), device="cuda") * 0.01 + 0.001
        w = torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
        ws = torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001
        norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5
        angles = torch.rand((sequence, rotary_dim), device="cuda")
        cos, sin = angles.cos(), angles.sin()
        expect_v_gluon = (
            value_format is ValueFormat.AMD_WMMA
            and width % 64 == 0
            and (head_dim == 128 or heads % 2 == 0)
        )
        expect_k_gluon = (
            width % 64 == 0
            and (head_dim == 128 or heads % 2 == 0)
            and rotary_dim in policy.GLUON_ROTARY_DIMS[head_dim]
        )
        assert amd.default_execution_plan(
            qdata, w, operation="key", head_dim=head_dim, rotary_dim=rotary_dim
        ).kernel == ("gluon" if expect_k_gluon else "triton")
        assert amd.default_execution_plan(
            qdata, w, operation="value", head_dim=head_dim, value_format=value_format
        ).kernel == ("gluon" if expect_v_gluon else "triton")
        q_out = query._project_query_op(
            qdata, scale, w, ws, norm, cos, sin, 1e-5, head_dim**-0.5, 0
        )
        k_out = key._project_key_op(qdata, scale, w, ws, norm, cos, sin, 1e-5, 0)
        v_out = value._project_value_op(
            qdata, scale, w, ws, head_dim=head_dim, value_format=int(value_format)
        )
        assert torch.isfinite(q_out[1]).all()
        assert torch.isfinite(k_out[1]).all()
        assert torch.isfinite(v_out[1]).all()
