"""Global K centering and dense per-token V remain intact under projection fusion."""

from dataclasses import replace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as quantization
from piper_kernels.attention.piper_attention import _quantized_dispatch as dispatch
from piper_kernels.attention.piper_attention._nvidia import policy
from piper_kernels.attention.piper_attention._nvidia import triton as nvidia
from piper_kernels.fusions.convrot_int8_piper import _backend, key, value
from piper_kernels.fusions.convrot_int8_piper import triton as projection

from .test_query import _NATIVE, _operands


def _projected(operands, bias, head_dim):
    x, xs, w, ws = operands[:4]
    batch, sequence, _ = x.shape
    projected = (x.double() @ w.double().T) * xs.double()[..., None] * ws.double()[:, 0]
    if bias is not None:
        projected += bias.double()
    return projected.reshape(batch, sequence, -1, head_dim)


def _key_reference(operands, bias, head_dim):
    projected = _projected(operands, bias, head_dim)
    norm, cos, sin = operands[4:]
    normalized = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-6)
    if norm is not None:
        normalized *= norm.double()
    rotary = normalized[..., : cos.shape[1]]
    first, second = rotary.chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    rotary = rotary * cos.double()[None, :, None] + rotated * sin.double()[None, :, None]
    return torch.cat((rotary, normalized[..., cos.shape[1] :]), -1).transpose(1, 2).float()


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sequence", [1, 65, 193, 1025])
@pytest.mark.parametrize("affine", [False, True])
def test_key_uses_global_post_transform_mean(head_dim, sequence, affine):
    operands = _operands("cuda", sequence=sequence, head_dim=head_dim, affine=affine)
    bias = torch.randn(3 * head_dim, device="cuda")
    actual, scales = key._project_key_op(*operands, 1e-6, bias, head_dim=head_dim)
    reference = _key_reference(operands, bias, head_dim).bfloat16()
    expected, expected_scale = quantization.prepare_key(
        reference,
        reference.float().mean(2),
        grouped=True,
        storage_key_length=actual.shape[2],
    )
    # Projection/reduction ordering can straddle BF16 and INT8 rounding boundaries.
    assert (actual.int() - expected.int()).abs().max() <= 1
    torch.testing.assert_close(scales, expected_scale, atol=2e-7, rtol=3e-5)
    assert torch.count_nonzero(actual[:, :, sequence:]) == 0


@pytest.mark.gpu
@_NATIVE
def test_key_storage_preserves_values_above_fp16_range():
    operands = _operands("cuda", sequence=65, head_dim=64)
    operands[4].fill_(131072)
    reference = _key_reference(operands, None, 64).bfloat16()
    assert reference.abs().max() > torch.finfo(torch.float16).max
    actual, scales = key._project_key_op(*operands, 1e-6, head_dim=64)
    expected, expected_scales = quantization.prepare_key(
        reference, reference.float().mean(2), grouped=True, storage_key_length=actual.shape[2]
    )
    assert torch.isfinite(scales).all()
    assert (actual.short() - expected.short()).abs().max() <= 1
    torch.testing.assert_close(scales, expected_scales, atol=2e-7, rtol=3e-5)


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sequence", [1, 65, 193])
@pytest.mark.parametrize("causal", [False, True])
def test_value_uses_per_token_scales_and_correct_centering(head_dim, sequence, causal):
    operands = _operands("cuda", sequence=sequence, head_dim=head_dim)
    bias = torch.randn(3 * head_dim, device="cuda")
    codes, multipliers, logs, mean = value._project_value_op(
        *operands[:4],
        bias,
        head_dim=head_dim,
        is_causal=causal,
    )
    reference = _projected(operands, bias, head_dim).transpose(1, 2).float()
    if not causal:
        expected_mean = reference.mean(2)
        torch.testing.assert_close(mean, expected_mean, atol=3e-5, rtol=3e-5)
        reference -= expected_mean[:, :, None]
    scales = reference.abs().amax(-1) / 127 + 1e-7
    normalized = reference / scales[..., None]
    expected_codes = (
        torch.trunc(normalized + 0.5 * normalized.sign()).clamp(-127, 127).to(torch.int8)
    )
    if AcceleratorTarget.from_device(codes.device).is_amd_hip:
        # Unpack the RDNA4 tile permutation for the independent reference.
        packed = codes.view(2, 3, -1, head_dim, 64)
        token = torch.arange(64, device=codes.device)
        permutation = (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
        codes = (
            packed[..., permutation]
            .permute(0, 1, 2, 4, 3)
            .reshape(2, 3, -1, head_dim)
            .transpose(2, 3)
        )
    decoded = (
        codes[:, :, :, :sequence].transpose(2, 3).float()
        * (multipliers[:, :, :sequence] / 255)[..., None]
    )
    assert ((decoded - reference).abs() <= scales[..., None] * 0.51 + 2e-5).all()
    if sequence > 1 or causal:
        assert (
            codes[:, :, :, :sequence].transpose(2, 3).int() - expected_codes.int()
        ).abs().max() <= 1
    torch.testing.assert_close(multipliers[:, :, :sequence], scales * 255, atol=2e-5, rtol=3e-5)
    expected_logs = (multipliers[:, :, :sequence] / 255).log2()
    if AcceleratorTarget.from_device(codes.device).is_nvidia_cuda:
        expected_logs = expected_logs.half().float()
    torch.testing.assert_close(logs[:, :, :sequence], expected_logs, atol=0.002, rtol=1e-5)
    assert torch.count_nonzero(codes[:, :, :, sequence:]) == 0


@pytest.mark.gpu
@_NATIVE
def test_causal_value_does_not_reduce_or_depend_on_future_rows(monkeypatch):

    operands = _operands("cuda", sequence=193)

    def forbidden(*args, **kwargs):
        raise AssertionError("causal V must not compute a global mean")

    monkeypatch.setattr(projection._ops, "dequantized_input_mean", forbidden)
    first = value._project_value_op(*operands[:4], head_dim=64, is_causal=True)
    operands[0][:, 65:].zero_()
    second = value._project_value_op(*operands[:4], head_dim=64, is_causal=True)
    for index in (1, 2):
        torch.testing.assert_close(
            first[index][:, :, :65], second[index][:, :, :65], atol=0, rtol=0
        )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA native comparison",
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sequence", [65, 1025])
def test_padded_quantized_boundary_matches_native_dense(head_dim, causal, dtype, sequence):
    torch.manual_seed(333)
    q = torch.randn(2, 4, sequence, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(
        2, 2, sequence if causal else sequence + 17, head_dim, device="cuda", dtype=dtype
    )
    v = torch.randn_like(k)
    target = AcceleratorTarget.from_device(q.device)
    if not target.is_cuda_capability(12):
        pytest.skip("requires grouped Q32")
    plan = policy.select_execution_plan(
        target, head_dim=head_dim, is_causal=causal, query_length=sequence
    )
    context = nvidia._prepare_piper_context(
        k,
        v,
        is_causal=causal,
        execution_plan=replace(plan, use_tensor_descriptors=False, derive_value_log_bound=False),
    )
    qdata, qs = quantization.prepare_query(
        q, head_dim**-0.5, grouped=True, storage_query_length=(sequence + 63) // 64 * 64
    )
    padding = (-k.shape[2]) % 64
    prepared = (
        qdata,
        qs,
        F.pad(context.key, (0, 0, 0, padding)),
        context.key_scale,
        F.pad(context.value, (0, padding)),
        F.pad(context.value_scale_multiplier, (0, padding)),
        F.pad(context.value_log_scale.float(), (0, padding)),
        torch.empty((2, 2, head_dim), device="cuda") if causal else context.value_mean,
        sequence,
        k.shape[2],
        causal,
        dtype,
    )
    actual = dispatch._piper_attention_from_quantized_op(*prepared)
    expected = piper_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [0, 2])
def test_context_fake_paths_do_not_probe_hardware(monkeypatch, batch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fake execution must not probe hardware")

    monkeypatch.setattr(_backend, "select_projection_backend", forbidden)
    monkeypatch.setattr(AcceleratorTarget, "from_device", forbidden)
    with FakeTensorMode():
        operands = list(_operands("cpu"))
        operands[0] = torch.empty((batch, 65, 272), dtype=torch.int8)
        operands[1] = torch.empty((batch, 65))
        k, ks = key._project_key_op(*operands, 1e-6, head_dim=64)
        v, mult, logs, mean = value._project_value_op(*operands[:4], head_dim=64, is_causal=True)
        q = torch.empty((batch, 6, 128, 64), dtype=torch.int8)
        qs = torch.empty((batch, 6, 4))
        out = dispatch._piper_attention_from_quantized_op(
            q, qs, k, ks, v, mult, logs, mean, 65, 65, True, torch.bfloat16
        )
        assert out.shape == (batch, 6, 65, 64)
        assert out.dtype is torch.bfloat16


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("causal", [False, True])
def test_zero_projection_retains_usable_scales_and_opcheck(causal):
    operands = _operands("cuda", sequence=65)
    operands[0].zero_()
    k, ks = key._project_key_op(*operands, 1e-6, head_dim=64)
    v, mult, logs, mean = value._project_value_op(*operands[:4], head_dim=64, is_causal=causal)
    assert torch.count_nonzero(k) == 0
    assert torch.count_nonzero(v) == 0
    assert torch.count_nonzero(mean) == 0
    assert (ks > 0).all()
    assert (mult > 0).all()
    assert torch.isfinite(logs).all()
    torch.library.opcheck(key._project_key_op, (*operands, 1e-6), kwargs={"head_dim": 64})
    torch.library.opcheck(
        value._project_value_op, tuple(operands[:4]), kwargs={"head_dim": 64, "is_causal": causal}
    )
