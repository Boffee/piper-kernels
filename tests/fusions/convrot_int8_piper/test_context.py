"""K1024 and V1024 centering and dense per-token V remain intact under projection fusion."""

from dataclasses import replace

import pytest
import torch
from _accelerators import forbid_device_probes
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as quantization
from piper_kernels.attention.piper_attention import _quantization as piper_quantization
from piper_kernels.attention.piper_attention import _quantized_dispatch as dispatch
from piper_kernels.attention.piper_attention._nvidia import policy
from piper_kernels.attention.piper_attention._nvidia import triton as nvidia
from piper_kernels.fusions.convrot_int8_piper import key, value
from piper_kernels.linear.convrot.int8 import _ops as linear_ops

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
@pytest.mark.parametrize("granularity", ["per_warp", "per_thread"])
def test_key_uses_chunked_post_transform_mean(head_dim, sequence, affine, granularity):
    operands = _operands("cuda", sequence=sequence, head_dim=head_dim, affine=affine)
    bias = torch.randn(3 * head_dim, device="cuda")
    actual, scales, centers = key._project_key_op(
        *operands, 1e-6, bias, head_dim=head_dim, qk_quantization=granularity
    )
    reference = _key_reference(operands, bias, head_dim).bfloat16()
    expected, expected_scale, expected_center = piper_quantization.prepare_chunk_centered_key(
        reference,
        grouped=granularity == "per_warp",
        storage_key_length=actual.shape[2],
    )
    assert (actual.int() - expected.int()).abs().max() <= 1
    torch.testing.assert_close(
        scales[:, :, : expected_scale.shape[2]],
        expected_scale,
        atol=2e-7,
        rtol=1e-3 if granularity == "per_thread" else 3e-5,
    )
    torch.testing.assert_close(centers, expected_center, atol=2e-4, rtol=1e-3)
    assert torch.count_nonzero(actual[:, :, sequence:]) == 0


@pytest.mark.gpu
@_NATIVE
def test_key_storage_preserves_values_above_fp16_range():
    operands = _operands("cuda", sequence=65, head_dim=64)
    operands[4].fill_(131072)
    reference = _key_reference(operands, None, 64).bfloat16()
    assert reference.abs().max() > torch.finfo(torch.float16).max
    actual, scales, centers = key._project_key_op(
        *operands, 1e-6, head_dim=64, qk_quantization="per_warp"
    )
    expected, expected_scales, expected_centers = piper_quantization.prepare_chunk_centered_key(
        reference, grouped=True, storage_key_length=actual.shape[2]
    )
    assert torch.isfinite(scales).all()
    assert torch.isfinite(centers).all()
    assert (actual.short() - expected.short()).abs().max() <= 1
    torch.testing.assert_close(scales, expected_scales, atol=2e-7, rtol=3e-5)
    torch.testing.assert_close(centers, expected_centers, atol=1e-2, rtol=1e-3)


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sequence", [1, 65, 193, 1025])
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
    chunk_count = (sequence + 1023) // 1024
    for chunk in range(chunk_count):
        start = chunk * 1024
        end = min(start + 1024, sequence)
        expected_mean = reference[:, :, start:end].mean(2)
        torch.testing.assert_close(mean[:, :, chunk], expected_mean, atol=3e-5, rtol=3e-5)
        reference[:, :, start:end] -= expected_mean[:, :, None]
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
    if sequence > 1 and sequence % 1024 != 1:
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
    operands = _operands("cuda", sequence=1057)

    def forbidden(*args, **kwargs):
        raise AssertionError("V1024 must not compute a global mean")

    monkeypatch.setattr(linear_ops, "dequantized_input_mean", forbidden)
    first = value._project_value_op(*operands[:4], head_dim=64, is_causal=True)
    operands[0][:, 1024:].zero_()
    second = value._project_value_op(*operands[:4], head_dim=64, is_causal=True)
    for index in (1, 2):
        torch.testing.assert_close(
            first[index][:, :, :1024], second[index][:, :, :1024], atol=0, rtol=0
        )
    torch.testing.assert_close(first[3][:, :, :1], second[3][:, :, :1], atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not AcceleratorTarget.from_device(torch.device("cuda")).is_nvidia_cuda,
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
    if not dispatch.supports_quantized_query(target):
        pytest.skip("requires native quantized-Q Piper")
    grouped = dispatch.select_qk_quantization(target) == "per_warp"
    plan = policy.select_execution_plan(
        target, head_dim=head_dim, is_causal=causal, query_length=sequence
    )
    if plan.attention_kernel == "triton":
        # Pointer loads and stored V logs match the native contract's storage.
        plan = replace(plan, use_tensor_descriptors=False, derive_value_log_bound=False)
    context = nvidia._prepare_piper_context(k, v, is_causal=causal, execution_plan=plan)
    assert isinstance(context.key, torch.Tensor)
    assert isinstance(context.value, torch.Tensor)
    qdata, qs = quantization.prepare_query(
        q, head_dim**-0.5, grouped=grouped, storage_query_length=(sequence + 63) // 64 * 64
    )
    padding = (-k.shape[2]) % 64
    storage = k.shape[2] + padding
    logs = (
        torch.zeros((2, 2, storage), device="cuda")
        if plan.derive_value_log_bound
        else F.pad(context.value_log_scale.float(), (0, padding))
    )
    prepared = (
        qdata,
        qs,
        F.pad(context.key, (0, 0, 0, storage - context.key.shape[2])),
        context.key_scale if grouped else F.pad(context.key_scale, (0, padding)),
        context.key_center,
        F.pad(context.value, (0, storage - context.value.shape[3])),
        F.pad(context.value_scale_multiplier, (0, padding)),
        logs,
        context.value_mean,
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
    forbid_device_probes(monkeypatch, "fake execution must not probe hardware")
    with FakeTensorMode():
        operands = list(_operands("cuda:0"))
        operands[0] = torch.empty((batch, 65, 272), device="cuda:0", dtype=torch.int8)
        operands[1] = torch.empty((batch, 65), device="cuda:0")
        k, ks, kc = key._project_key_op(*operands, 1e-6, head_dim=64, qk_quantization="per_warp")
        v, mult, logs, mean = value._project_value_op(*operands[:4], head_dim=64, is_causal=True)
        q = torch.empty((batch, 6, 128, 64), device="cuda:0", dtype=torch.int8)
        qs = torch.empty((batch, 6, 4), device="cuda:0")
        out = dispatch._piper_attention_from_quantized_op(
            q, qs, k, ks, kc, v, mult, logs, mean, 65, 65, True, torch.bfloat16
        )
        assert out.shape == (batch, 6, 65, 64)
        assert out.dtype is torch.bfloat16


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("causal", [False, True])
def test_zero_projection_retains_usable_scales_and_opcheck(causal):
    operands = _operands("cuda", sequence=65)
    operands[0].zero_()
    k, ks, kc = key._project_key_op(*operands, 1e-6, head_dim=64, qk_quantization="per_warp")
    v, mult, logs, mean = value._project_value_op(*operands[:4], head_dim=64, is_causal=causal)
    assert torch.count_nonzero(k) == 0
    assert torch.count_nonzero(kc) == 0
    assert torch.count_nonzero(v) == 0
    assert torch.count_nonzero(mean) == 0
    assert (ks > 0).all()
    assert (mult > 0).all()
    assert torch.isfinite(logs).all()
    torch.library.opcheck(
        key._project_key_op,
        (*operands, 1e-6),
        kwargs={"head_dim": 64, "qk_quantization": "per_warp"},
    )
    torch.library.opcheck(
        value._project_value_op, tuple(operands[:4]), kwargs={"head_dim": 64, "is_causal": causal}
    )


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("granularity", ["per_warp", "per_thread"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_key_chunk_does_not_depend_on_future_chunks(granularity, head_dim):
    operands = _operands("cuda", sequence=1057, head_dim=head_dim)
    first = key._project_key_op(*operands, 1e-6, head_dim=head_dim, qk_quantization=granularity)
    operands[0][:, 1024:].zero_()
    second = key._project_key_op(*operands, 1e-6, head_dim=head_dim, qk_quantization=granularity)
    scale_count = 16 if granularity == "per_warp" else 1024
    torch.testing.assert_close(first[0][:, :, :1024], second[0][:, :, :1024], atol=0, rtol=0)
    torch.testing.assert_close(
        first[1][:, :, :scale_count], second[1][:, :, :scale_count], atol=0, rtol=0
    )
    torch.testing.assert_close(first[2][:, :, :1], second[2][:, :, :1], atol=0, rtol=0)


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("input_features", [256, 272])
@pytest.mark.parametrize("constant_value", [2.0**100, -(2.0**100)])
def test_value_bounds_large_constant_chunk_scale(input_features, constant_value):
    input_qdata = torch.zeros((1, 1088, input_features), device="cuda", dtype=torch.int8)
    input_scale = torch.ones((1, 1088), device="cuda", dtype=torch.float32)
    weight_qdata = torch.zeros((2 * 128, input_features), device="cuda", dtype=torch.int8)
    weight_qdata[:, 0] = 1
    input_qdata[:, 1024:, 0] = 1
    weight_scale = torch.ones((2 * 128, 1), device="cuda", dtype=torch.float32)
    bias = torch.full((2 * 128,), constant_value, device="cuda", dtype=torch.bfloat16)

    v, multipliers, logs, mean = value._project_value_op(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias=bias,
        head_dim=128,
        is_causal=False,
    )

    assert not bool(v.any())
    torch.testing.assert_close(mean, torch.full_like(mean, constant_value), atol=0, rtol=0)
    assert bool((multipliers >= abs(constant_value) * (2.0**-80) * 255.0).all())
    assert torch.isfinite(logs).all()
    target = AcceleratorTarget.from_device(v.device)
    if target.is_nvidia_cuda and dispatch.supports_quantized_query(target):
        granularity = dispatch.select_qk_quantization(target)
        q = torch.zeros((1, 2, 1088, 128), device="cuda", dtype=torch.bfloat16)
        qdata, qs = quantization.prepare_query(
            q, 128**-0.5, grouped=granularity == "per_warp", storage_query_length=1088
        )
        k_int8 = torch.zeros((1, 2, 1088, 128), device="cuda", dtype=torch.int8)
        k_scales = torch.ones(
            (1, 2, 1088 // 64 if granularity == "per_warp" else 1088),
            device="cuda",
            dtype=torch.float32,
        )
        k_center = torch.zeros((1, 2, 2, 128), device="cuda", dtype=torch.float32)
        out = dispatch._piper_attention_from_quantized_op(
            qdata,
            qs,
            k_int8,
            k_scales,
            k_center,
            v,
            multipliers,
            logs,
            mean,
            1088,
            1088,
            False,
            torch.bfloat16,
        )
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, torch.full_like(out, constant_value), atol=0, rtol=0)
