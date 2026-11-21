"""RDNA4 Gluon dense Piper Q/K/V and end-to-end attention output contracts."""

import math

import pytest
import torch
from _accelerators import rdna4_available

from piper_kernels import piper_attention
from piper_kernels.fusions.convrot_int8_piper import key, output, query, value  # noqa: F401
from piper_kernels.linear.convrot import convrot_int8_linear
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_SIGNED_HADAMARD_MASK = (0xF62B39F3, 0x802576E6, 0x910D4CFD, 0x4F877B5A)
_PROJECT_QUERY_OP = torch.ops.piper_kernels.convrot_int8_piper_project_query.default
_PROJECT_KEY_OP = torch.ops.piper_kernels.convrot_int8_piper_project_key.default
_PROJECT_VALUE_OP = torch.ops.piper_kernels.convrot_int8_piper_project_value.default
_PROJECTED_ATTENTION_OUTPUT_OP = (
    torch.ops.piper_kernels.convrot_int8_piper_projected_query_attention_output.default
)

_RDNA4 = pytest.mark.skipif(not rdna4_available(), reason="requires Linux/Windows RDNA4 ROCm")


def _operands(
    *,
    batch: int = 2,
    sequence: int = 193,
    heads: int = 4,
    head_dim: int = 64,
    features: int = 256,
    rotary_dim: int | None = None,
    affine: bool = True,
    with_bias: bool = False,
):
    resolved_rotary = head_dim * 3 // 4 if rotary_dim is None else rotary_dim
    x = torch.randint(-127, 128, (batch, sequence, features), device="cuda", dtype=torch.int8)
    xs = torch.rand((batch, sequence), device="cuda") * 0.01 + 0.001
    w = torch.randint(-127, 128, (heads * head_dim, features), device="cuda", dtype=torch.int8)
    ws = torch.rand((heads * head_dim, 1), device="cuda") * 0.01 + 0.001
    norm = torch.rand(head_dim, device="cuda", dtype=torch.bfloat16) + 0.5 if affine else None
    angles = torch.rand((sequence, resolved_rotary), device="cuda") * (2 * torch.pi)
    bias = torch.randn(heads * head_dim, device="cuda", dtype=torch.bfloat16) if with_bias else None
    return x, xs, w, ws, norm, angles.cos(), angles.sin(), bias


def _hadamard_matrix(head_dim: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(head_dim, device=device)
    masks = torch.tensor(_SIGNED_HADAMARD_MASK, device=device, dtype=torch.int64)
    signs = 2 * ((masks[indices // 32] >> (indices % 32)) & 1) - 1
    bit_products = indices[:, None] & indices[None, :]
    parity = torch.zeros_like(bit_products)
    for bit in range(head_dim.bit_length() - 1):
        parity ^= (bit_products >> bit) & 1
    return ((1 - 2 * parity).double() * signs[:, None]) / math.sqrt(head_dim)


def _reference_qk_transform(
    x: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    ws: torch.Tensor,
    norm: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    bias: torch.Tensor | None,
    head_dim: int,
    norm_epsilon: float = 1e-6,
) -> torch.Tensor:
    batch, sequence, _ = x.shape
    heads = w.shape[0] // head_dim
    projected = (x.double() @ w.double().T) * xs.double()[..., None] * ws.double()[:, 0]
    if bias is not None:
        projected += bias.double()
    projected = projected.reshape(batch, sequence, heads, head_dim)
    normalized = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + norm_epsilon)
    if norm is not None:
        normalized *= norm.double()
    rotary_dim = cos.shape[1]
    first, second = normalized[..., :rotary_dim].chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    rotary = (
        normalized[..., :rotary_dim] * cos.double()[None, :, None]
        + rotated * sin.double()[None, :, None]
    )
    return torch.cat((rotary, normalized[..., rotary_dim:]), -1).transpose(1, 2)


def _quantize_qk_groups(
    rotated: torch.Tensor,
    sequence: int,
    head_dim: int,
    granularity: str,
    group_rows: int,
    scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads = rotated.shape[:2]
    storage = (sequence + 63) // 64 * 64
    padded = rotated.new_zeros((batch, heads, storage, head_dim))
    padded[:, :, :sequence] = rotated
    if granularity == "per_thread":
        if group_rows == 32:
            groups = padded.reshape(batch, heads, storage // 32, 4, 8, head_dim)
            scale = groups.abs().amax((-1, -3), keepdim=True) / 127 + 1e-7
            codes = torch.trunc(groups / scale + 0.5 * (groups / scale).sign()).clamp(-127, 127)
            row_scale = scale.expand(-1, -1, -1, 4, -1, 1).reshape(batch, heads, storage).clone()
            row_scale[:, :, sequence:] = 0
        else:
            groups = padded.reshape(batch, heads, storage // 64, 8, 4, 2, head_dim)
            scale = groups.abs().amax((-1, -2, -4), keepdim=True) / 127 + 1e-7
            codes = torch.trunc(groups / scale + 0.5 * (groups / scale).sign()).clamp(-127, 127)
            row_scale = scale.expand(-1, -1, -1, 8, -1, 2, 1).reshape(batch, heads, storage).clone()
        return codes.to(torch.int8).reshape_as(padded), (row_scale * scale_factor).float()
    groups = padded.reshape(batch, heads, storage // group_rows, group_rows, head_dim)
    scale = groups.abs().amax((-1, -2)) / 127 + 1e-7
    normalized = groups / scale[..., None, None]
    codes = torch.trunc(normalized + 0.5 * normalized.sign()).clamp(-127, 127).to(torch.int8)
    valid_groups = (sequence + group_rows - 1) // group_rows
    if group_rows == 32:
        scale[:, :, valid_groups:] = 0
    return codes.reshape_as(padded), (scale * scale_factor).float()


def _unpack_amd_value(codes: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    batch = codes.shape[0]
    storage = codes.shape[3]
    packed = codes.view(batch, heads, storage // 64, head_dim, 64)
    token = torch.arange(64, device=codes.device)
    permutation = (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
    return packed[..., permutation].permute(0, 1, 2, 4, 3).reshape(batch, heads, storage, head_dim)


def _assert_query_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor],
    operands,
    head_dim: int,
    granularity: str = "per_warp",
) -> None:
    x, xs, w, ws, norm, cos, sin, bias = operands
    sequence = x.shape[1]
    transformed = _reference_qk_transform(x, xs, w, ws, norm, cos, sin, bias, head_dim)
    rotated = transformed @ _hadamard_matrix(head_dim, x.device)
    expected_codes, expected_scales = _quantize_qk_groups(
        rotated, sequence, head_dim, granularity, 32, head_dim**-0.5 * math.log2(math.e)
    )
    codes, scales = actual
    assert (codes.short() - expected_codes.short()).abs().max() <= 1
    assert float((codes != expected_codes).float().mean()) < 1e-4
    torch.testing.assert_close(scales, expected_scales, rtol=3e-5, atol=1e-7)
    assert torch.count_nonzero(codes[:, :, sequence:]) == 0
    if granularity == "per_warp":
        assert torch.count_nonzero(scales[:, :, (sequence + 31) // 32 :]) == 0


def _assert_key_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    operands,
    head_dim: int,
    granularity: str = "per_warp",
) -> None:
    x, xs, w, ws, norm, cos, sin, bias = operands
    sequence = x.shape[1]
    # K stores BF16 intermediate rows before K1024 chunk centering and signed Hadamard.
    transformed = _reference_qk_transform(x, xs, w, ws, norm, cos, sin, bias, head_dim)
    stored_bf16 = transformed.bfloat16().double()
    hadamard = _hadamard_matrix(head_dim, x.device)
    chunk_count = (sequence + 1023) // 1024
    expected_centers = []
    centered = torch.zeros_like(stored_bf16)
    for chunk in range(chunk_count):
        start, end = chunk * 1024, min((chunk + 1) * 1024, sequence)
        mean = stored_bf16[:, :, start:end].mean(dim=2)
        expected_centers.append(mean @ hadamard)
        centered[:, :, start:end] = stored_bf16[:, :, start:end] - mean[:, :, None, :]
    rotated = centered @ hadamard
    expected_codes, expected_scales = _quantize_qk_groups(
        rotated, sequence, head_dim, granularity, 64, 1.0
    )
    expected_center = torch.stack(expected_centers, dim=2).float()
    codes, scales, centers = actual
    assert (codes.short() - expected_codes.short()).abs().max() <= 1
    assert float((codes != expected_codes).float().mean()) < 2e-4
    torch.testing.assert_close(scales, expected_scales, rtol=3e-3, atol=2e-6)
    torch.testing.assert_close(centers, expected_center, rtol=2e-3, atol=2e-4)
    assert torch.count_nonzero(codes[:, :, sequence:]) == 0


def _assert_value_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    operands,
    head_dim: int,
) -> None:
    x, xs, w, ws, _norm, _cos, _sin, bias = operands
    batch, sequence, _ = x.shape
    heads = w.shape[0] // head_dim
    projected = (x.double() @ w.double().T) * xs.double()[..., None] * ws.double()[:, 0]
    if bias is not None:
        projected += bias.double()
    reference = projected.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    codes, multipliers, logs, means = actual
    chunk_count = (sequence + 1023) // 1024
    for chunk in range(chunk_count):
        start, end = chunk * 1024, min((chunk + 1) * 1024, sequence)
        expected_mean = reference[:, :, start:end].mean(2).float()
        torch.testing.assert_close(means[:, :, chunk], expected_mean, rtol=3e-5, atol=3e-5)
        reference[:, :, start:end] -= expected_mean.double()[:, :, None]
    scales = reference.abs().amax(-1) / 127 + 1e-7
    normalized = reference / scales[..., None]
    expected_codes = (
        torch.trunc(normalized + 0.5 * normalized.sign()).clamp(-127, 127).to(torch.int8)
    )
    unpacked_all = _unpack_amd_value(codes, heads, head_dim)
    unpacked = unpacked_all[:, :, :sequence]
    decoded = unpacked.double() * (multipliers[:, :, :sequence].double() / 255.0)[..., None]
    assert ((decoded - reference).abs() <= scales[..., None] * 0.51 + 2e-5).all()
    if sequence > 1 and sequence % 1024 != 1:
        assert (unpacked.short() - expected_codes.short()).abs().max() <= 1
    torch.testing.assert_close(
        multipliers[:, :, :sequence].double(), scales * 255.0, rtol=3e-5, atol=2e-5
    )
    torch.testing.assert_close(
        logs[:, :, :sequence], (multipliers[:, :, :sequence] / 255.0).log2(), rtol=1e-5, atol=1e-5
    )
    assert torch.count_nonzero(unpacked_all[:, :, sequence:]) == 0


@pytest.mark.gpu
@_RDNA4
@pytest.mark.parametrize(
    ("head_dim", "rotary_dim"),
    [(64, rd) for rd in (32, 48, 64)] + [(128, rd) for rd in (32, 64, 96, 128)],
)
@pytest.mark.parametrize("affine", [True, False])
def test_rdna4_gluon_qkv_match_fp64_across_rotary_widths(head_dim, rotary_dim, affine):
    torch.manual_seed(2027 + head_dim + rotary_dim)
    heads = 4 if head_dim == 64 else 3
    operands = _operands(
        batch=2,
        sequence=205,
        heads=heads,
        head_dim=head_dim,
        features=256,
        rotary_dim=rotary_dim,
        affine=affine,
        with_bias=not affine,
    )
    x, xs, w, ws, norm, cos, sin, bias = operands
    q = _PROJECT_QUERY_OP(
        x,
        xs,
        w,
        ws,
        norm,
        cos,
        sin,
        1e-6,
        head_dim**-0.5,
        bias,
        head_dim=head_dim,
        qk_quantization="per_warp",
    )
    k = _PROJECT_KEY_OP(
        x, xs, w, ws, norm, cos, sin, 1e-6, bias, head_dim=head_dim, qk_quantization="per_warp"
    )
    v = _PROJECT_VALUE_OP(x, xs, w, ws, bias, head_dim=head_dim, is_causal=False)
    _assert_query_matches_reference(q, operands, head_dim)
    _assert_key_matches_reference(k, operands, head_dim)
    _assert_value_matches_reference(v, operands, head_dim)


@pytest.mark.gpu
@_RDNA4
@pytest.mark.parametrize("sequence", [1, 64, 65, 192, 1025])
@pytest.mark.parametrize(("head_dim", "heads"), [(64, 4), (128, 3)])
@pytest.mark.parametrize("causal", [False, True])
def test_rdna4_gluon_qkv_handle_tile_boundaries_and_k1024_chunks(sequence, head_dim, heads, causal):
    torch.manual_seed(2039 + sequence)
    operands = _operands(
        batch=1,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        features=256,
        with_bias=True,
    )
    x, xs, w, ws, norm, cos, sin, bias = operands
    q = _PROJECT_QUERY_OP(
        x,
        xs,
        w,
        ws,
        norm,
        cos,
        sin,
        1e-6,
        head_dim**-0.5,
        bias,
        head_dim=head_dim,
        qk_quantization="per_warp",
    )
    k = _PROJECT_KEY_OP(
        x, xs, w, ws, norm, cos, sin, 1e-6, bias, head_dim=head_dim, qk_quantization="per_warp"
    )
    k_pt = _PROJECT_KEY_OP(
        x, xs, w, ws, norm, cos, sin, 1e-6, bias, head_dim=head_dim, qk_quantization="per_thread"
    )
    v = _PROJECT_VALUE_OP(x, xs, w, ws, bias, head_dim=head_dim, is_causal=causal)
    _assert_query_matches_reference(q, operands, head_dim)
    _assert_key_matches_reference(k, operands, head_dim, granularity="per_warp")
    _assert_key_matches_reference(k_pt, operands, head_dim, granularity="per_thread")
    _assert_value_matches_reference(v, operands, head_dim)


@pytest.mark.gpu
@_RDNA4
@pytest.mark.parametrize(("width", "heads", "head_dim"), [(2048, 32, 64), (5376, 56, 128)])
def test_rdna4_gluon_h3_production_shapes_match_fp64_sample_heads(width, heads, head_dim):
    torch.manual_seed(2053 + width)
    sequence = 8192
    operands = _operands(
        batch=1,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        features=width,
        with_bias=True,
    )
    x, xs, w, ws, norm, cos, sin, bias = operands
    q = _PROJECT_QUERY_OP(
        x,
        xs,
        w,
        ws,
        norm,
        cos,
        sin,
        1e-6,
        head_dim**-0.5,
        bias,
        head_dim=head_dim,
        qk_quantization="per_warp",
    )
    k = _PROJECT_KEY_OP(
        x, xs, w, ws, norm, cos, sin, 1e-6, bias, head_dim=head_dim, qk_quantization="per_warp"
    )
    v = _PROJECT_VALUE_OP(x, xs, w, ws, bias, head_dim=head_dim, is_causal=False)

    for head in (0, heads // 2, heads - 1):
        cols = slice(head * head_dim, (head + 1) * head_dim)
        sub_ops = (
            x,
            xs,
            w[cols].contiguous(),
            ws[cols].contiguous(),
            norm,
            cos,
            sin,
            bias[cols].contiguous(),
        )
        _assert_query_matches_reference(
            (q[0][:, head : head + 1], q[1][:, head : head + 1]), sub_ops, head_dim
        )
        _assert_key_matches_reference(
            (k[0][:, head : head + 1], k[1][:, head : head + 1], k[2][:, head : head + 1]),
            sub_ops,
            head_dim,
        )
        _assert_value_matches_reference(
            (
                v[0][:, head : head + 1],
                v[1][:, head : head + 1],
                v[2][:, head : head + 1],
                v[3][:, head : head + 1],
            ),
            sub_ops,
            head_dim,
        )


@pytest.mark.gpu
@_RDNA4
@pytest.mark.parametrize(("head_dim", "q_heads", "kv_heads"), [(64, 12, 4), (128, 8, 2)])
@pytest.mark.parametrize("sequence", [193, 1057])
@pytest.mark.parametrize("causal", [False, True])
def test_rdna4_gluon_windowed_attention_output_gqa_matches_piper_attention(
    head_dim, q_heads, kv_heads, sequence, causal
):
    torch.manual_seed(2063 + sequence + q_heads)
    features = 256
    q_ops = _operands(
        batch=1,
        sequence=sequence,
        heads=q_heads,
        head_dim=head_dim,
        features=features,
        with_bias=True,
    )
    k_ops = _operands(
        batch=1,
        sequence=sequence,
        heads=kv_heads,
        head_dim=head_dim,
        features=features,
        with_bias=True,
    )
    kv_x, kv_xs, kw, kws, knorm, kcos, ksin, kbias = k_ops
    vw = torch.randint(-127, 128, (kv_heads * head_dim, features), device="cuda", dtype=torch.int8)
    vws = torch.rand((kv_heads * head_dim, 1), device="cuda") * 0.01 + 0.001
    vbias = torch.randn(kv_heads * head_dim, device="cuda", dtype=torch.bfloat16)

    k_state = _PROJECT_KEY_OP(
        kv_x,
        kv_xs,
        kw,
        kws,
        knorm,
        kcos,
        ksin,
        1e-6,
        kbias,
        head_dim=head_dim,
        qk_quantization="per_warp",
    )
    v_state = _PROJECT_VALUE_OP(kv_x, kv_xs, vw, vws, vbias, head_dim=head_dim, is_causal=causal)

    out_weight = torch.randint(
        -100, 100, (192, q_heads * head_dim), device="cuda", dtype=torch.int8
    )
    out_scale = torch.full((192, 1), 0.001, device="cuda")
    out_bias = torch.randn(192, device="cuda", dtype=torch.bfloat16)

    actual = _PROJECTED_ATTENTION_OUTPUT_OP(
        q_ops[0],
        q_ops[1],
        q_ops[2],
        q_ops[3],
        q_ops[4],
        q_ops[5],
        q_ops[6],
        1e-6,
        head_dim**-0.5,
        q_ops[7],
        *k_state,
        *v_state,
        sequence,
        causal,
        torch.bfloat16,
        out_weight,
        out_scale,
        out_bias,
        64,
        None,
        head_dim=head_dim,
        query_chunk_rows=128,
    )

    q_ref = _reference_qk_transform(*q_ops, head_dim).bfloat16()
    k_ref = _reference_qk_transform(*k_ops, head_dim).bfloat16()
    v_proj = (kv_x.double() @ vw.double().T) * kv_xs.double()[..., None] * vws.double()[:, 0]
    v_proj += vbias.double()
    v_ref = v_proj.reshape(1, sequence, kv_heads, head_dim).transpose(1, 2).bfloat16()
    attended = (
        piper_attention(q_ref, k_ref, v_ref, is_causal=causal)
        .transpose(1, 2)
        .reshape(1, sequence, -1)
    )
    expected = convrot_int8_linear(
        attended,
        ConvRotInt8Tensor.from_quantized(
            out_weight, out_scale, group_size=64, logical_dtype=torch.bfloat16
        ),
        bias=out_bias,
    )
    error_norm = torch.linalg.vector_norm(actual.float() - expected.float())
    reference_norm = torch.linalg.vector_norm(expected.float())
    assert error_norm <= 0.025 * reference_norm


@pytest.mark.gpu
@_RDNA4
def test_rdna4_triton_fallbacks_cover_odd_d64_width_rotary_and_per_thread_q():
    torch.manual_seed(2081)
    cases = [
        (137, 256, 3, 64, 48, "per_warp"),
        (137, 272, 4, 64, 48, "per_warp"),
        (137, 256, 4, 64, 16, "per_warp"),
        (137, 256, 3, 128, 48, "per_warp"),
        (137, 256, 4, 64, 48, "per_thread"),
    ]
    for sequence, features, heads, head_dim, rotary_dim, granularity in cases:
        operands = _operands(
            batch=1,
            sequence=sequence,
            heads=heads,
            head_dim=head_dim,
            features=features,
            rotary_dim=rotary_dim,
            with_bias=True,
        )
        x, xs, w, ws, norm, cos, sin, bias = operands
        q = _PROJECT_QUERY_OP(
            x,
            xs,
            w,
            ws,
            norm,
            cos,
            sin,
            1e-6,
            head_dim**-0.5,
            bias,
            head_dim=head_dim,
            qk_quantization=granularity,
        )
        k = _PROJECT_KEY_OP(
            x, xs, w, ws, norm, cos, sin, 1e-6, bias, head_dim=head_dim, qk_quantization=granularity
        )
        v = _PROJECT_VALUE_OP(x, xs, w, ws, bias, head_dim=head_dim, is_causal=False)
        _assert_query_matches_reference(q, operands, head_dim, granularity=granularity)
        _assert_key_matches_reference(k, operands, head_dim, granularity=granularity)
        _assert_value_matches_reference(v, operands, head_dim)


@pytest.mark.gpu
@_RDNA4
@pytest.mark.parametrize("head_dim", [64, 128])
def test_rdna4_gluon_qkv_opcheck_and_graph_replay(head_dim):
    torch.manual_seed(2089 + head_dim)
    operands = _operands(
        batch=2, sequence=193, heads=4, head_dim=head_dim, features=256, with_bias=True
    )
    x, xs, w, ws, norm, cos, sin, bias = operands

    for op, args, kwargs in (
        (
            _PROJECT_QUERY_OP,
            (x, xs, w, ws, norm, cos, sin, 1e-6, head_dim**-0.5, bias),
            {"head_dim": head_dim, "qk_quantization": "per_warp"},
        ),
        (
            _PROJECT_KEY_OP,
            (x, xs, w, ws, norm, cos, sin, 1e-6, bias),
            {"head_dim": head_dim, "qk_quantization": "per_warp"},
        ),
        (
            _PROJECT_VALUE_OP,
            (x, xs, w, ws, bias),
            {"head_dim": head_dim, "is_causal": False},
        ),
    ):
        checks = torch.library.opcheck(op, args, kwargs=kwargs)
        assert set(checks.values()) == {"SUCCESS"}
        expected = op(*args, **kwargs)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = op(*args, **kwargs)
        x.zero_()
        updated = op(*args, **kwargs)
        graph.replay()
        for left, right in zip(captured, updated, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        x.random_(-127, 128)
        assert expected[0].dtype is torch.int8


@pytest.mark.gpu
@_RDNA4
@pytest.mark.parametrize("head_dim", [64, 128])
def test_rdna4_gluon_zero_input_without_bias_keeps_finite_scales_and_zero_codes(head_dim):
    torch.manual_seed(2099 + head_dim)
    sequence = 137
    x, xs, w, ws, norm, cos, sin, bias = _operands(
        batch=1,
        sequence=sequence,
        heads=4,
        head_dim=head_dim,
        features=256,
        with_bias=False,
    )
    x.zero_()
    q_codes, q_scales = _PROJECT_QUERY_OP(
        x,
        xs,
        w,
        ws,
        norm,
        cos,
        sin,
        1e-6,
        head_dim**-0.5,
        bias,
        head_dim=head_dim,
        qk_quantization="per_warp",
    )
    k_codes, k_scales, k_centers = _PROJECT_KEY_OP(
        x, xs, w, ws, norm, cos, sin, 1e-6, bias, head_dim=head_dim, qk_quantization="per_warp"
    )
    v_codes, v_multipliers, v_logs, v_means = _PROJECT_VALUE_OP(
        x, xs, w, ws, bias, head_dim=head_dim, is_causal=False
    )
    assert torch.count_nonzero(q_codes) == 0
    assert torch.count_nonzero(k_codes) == 0
    assert torch.count_nonzero(v_codes) == 0
    for tensor in (q_scales, k_scales, k_centers, v_multipliers, v_logs, v_means):
        assert torch.isfinite(tensor).all()
    valid_q_groups = (sequence + 31) // 32
    expected_q_scale = 1e-7 * head_dim**-0.5 * math.log2(math.e)
    torch.testing.assert_close(
        q_scales[:, :, :valid_q_groups],
        torch.full_like(q_scales[:, :, :valid_q_groups], expected_q_scale),
    )
    assert torch.count_nonzero(q_scales[:, :, valid_q_groups:]) == 0
    torch.testing.assert_close(
        v_multipliers[:, :, :sequence],
        torch.full_like(v_multipliers[:, :, :sequence], 1e-7 * 255.0),
    )
