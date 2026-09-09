"""RDNA4 fused projections against independent FP64 math at H3 dimensions."""

import sys
from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._amd import gluon as amd_attention
from piper_kernels.fusions.convrot_int8_sparse_piper import _backend, key, output, query, value
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import policy
from piper_kernels.fusions.convrot_int8_sparse_piper._amd import triton as amd
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8 import _ops

from ._reference import assert_int8_codes_close, check_qk_sample_fp64, check_value_sample_fp64


def _available():
    return torch.cuda.is_available() and policy.supports_target(
        AcceleratorTarget.from_device(torch.device("cuda"))
    )


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201", "gfx1100", "gfx942", "gfx9999"])
def test_amd_projection_support_is_limited_to_linux_rdna4(monkeypatch, platform, arch):
    monkeypatch.setattr(sys, "platform", platform)
    assert policy.supports_target(AcceleratorTarget("hip", arch)) is (
        platform == "linux" and arch in ("gfx1200", "gfx1201")
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires Linux RDNA4 ROCm")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_grouped_order_preserves_complete_outputs_with_partial_groups_and_query_window(
    monkeypatch, head_dim
):
    torch.manual_seed(137)
    batch, sequence, width, heads = 2, 1217, 272, 3
    qdata = torch.randint(-128, 128, (batch, sequence, width), device="cuda", dtype=torch.int8)
    scale = torch.rand((batch, sequence), device="cuda") * 0.01 + 0.001
    weight = torch.randint(-128, 128, (heads * head_dim, width), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001
    norm = torch.rand(head_dim, dtype=torch.bfloat16, device="cuda") + 0.5
    angles = torch.rand((sequence, head_dim * 3 // 4), device="cuda")
    cos, sin = angles.cos(), angles.sin()
    mean = _ops.dequantized_input_mean(qdata, scale)

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
            value._project_value_with_block_means_op(
                qdata, scale, mean, weight, weight_scale, head_dim=head_dim
            ),
        )

    actual = run()
    for launch in (amd.project_query, amd.project_key, amd.project_value):
        monkeypatch.setitem(
            launch.keywords, "config", replace(launch.keywords["config"], group_m=0)
        )
    expected = run()
    for left, right in zip(actual, expected, strict=True):
        assert_int8_codes_close(left[0], right[0])
        for actual_metadata, expected_metadata in zip(left[1:], right[1:], strict=True):
            torch.testing.assert_close(actual_metadata, expected_metadata, rtol=3e-5, atol=2e-5)


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
        mean = _ops.dequantized_input_mean(qdata, scale)
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
        v = value._project_value_with_block_means_op(
            qdata, scale, mean, weights[2], scales[2], head_dim=head_dim
        )
    stream.synchronize()
    assert _backend.select_output_backend(q[0]) is not None

    # Independently check all represented-input feature means with bounded workspace.
    expected_mean = torch.zeros((1, width), device="cuda", dtype=torch.float64)
    for start in range(0, sequence, 4096):
        expected_mean += (
            qdata[:, start : start + 4096].double() * scale[:, start : start + 4096, None].double()
        ).sum(1)
    expected_mean /= sequence
    torch.testing.assert_close(mean.double(), expected_mean, rtol=2e-5, atol=2e-6)
    storage = (sequence + 63) // 64 * 64
    for head in (0, heads // 2, heads - 1):
        columns = slice(head * head_dim, (head + 1) * head_dim)
        mean_v = (expected_mean @ weights[2][columns].double().T) * scales[2][columns, 0].double()
        torch.testing.assert_close(v[2][:, head].double(), mean_v, rtol=2e-5, atol=2e-5)
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
                    check_value_sample_fp64(
                        projected, projected_operands, head, start, rows, mean_v
                    )
    assert q[0].shape == (1, heads, storage, head_dim)
    assert k[0].shape == (1, heads, storage, head_dim)
    assert v[0].shape == (1, heads, head_dim, storage)


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
    mean = _ops.dequantized_input_mean(qdata, scale)
    v = value._project_value_op(qdata, scale, mean, weights[2], scales[2], head_dim=head_dim)
    attention_tail = *k, *v, [250_000] * heads, sequence // 64, sequence, 0
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

    windows, packing_calls = [], []
    project_query = query._launch_query_projection_range
    pack_context = amd_attention.pack_context

    def track_query(*args, **kwargs):
        windows.append((kwargs["chunk_start"], kwargs["chunk_rows"]))
        assert kwargs["chunk_rows"] <= chunk_rows
        return project_query(*args, **kwargs)

    def track_packing(context):
        packing_calls.append(True)
        return pack_context(context)

    monkeypatch.setattr(query, "_launch_query_projection_range", track_query)
    monkeypatch.setattr(amd_attention, "pack_context", track_packing)
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
    assert len(packing_calls) == 1
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
        # Final output and existing AMD packed V scale with S. All additional
        # working storage fits comfortably below even one full INT8 Q tensor.
        assert peak < actual.numel() * actual.element_size() + v[0].numel() + 512 * 1024**2
