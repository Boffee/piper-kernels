"""Dense fused Q emits only Q32 data/scales with the shared FP32 transform contract."""

import math
import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage._rotation import SIGNED_HADAMARD_MASK
from piper_kernels.fusions.convrot_int8_piper import _backend, query
from piper_kernels.fusions.convrot_int8_piper import triton as projection


def _available():
    return (
        torch.cuda.is_available()
        and _backend.select_projection_backend(
            torch.empty(0, device="cuda"),
            head_dim=128,
        )
        is not None
    )


_NATIVE = pytest.mark.skipif(not _available(), reason="requires exact SM120 or RDNA4")


def _operands(
    device,
    *,
    sequence=65,
    heads=3,
    head_dim=64,
    features=272,
    dtype=torch.bfloat16,
    affine=True,
):
    torch.manual_seed(881)
    input_qdata = torch.randint(-127, 128, (2, sequence, features), device=device, dtype=torch.int8)
    input_scale = torch.rand((2, sequence), device=device) * 0.01 + 0.001
    weight_qdata = torch.randint(
        -127, 128, (heads * head_dim, features), device=device, dtype=torch.int8
    )
    weight_scale = torch.rand((heads * head_dim, 1), device=device) * 0.01 + 0.001
    norm = (torch.rand(head_dim, device=device) + 0.5).to(dtype) if affine else None
    angles = torch.rand((sequence, head_dim * 3 // 4), device=device) * (2 * torch.pi)
    return input_qdata, input_scale, weight_qdata, weight_scale, norm, angles.cos(), angles.sin()


def _reference(operands, bias, head_dim):
    x, xs, weight, ws, norm, cos, sin = operands
    batch, sequence, _ = x.shape
    heads = weight.shape[0] // head_dim
    projected = (x.double() @ weight.double().T) * xs.double()[..., None] * ws.double()[:, 0]
    if bias is not None:
        projected += bias.double()
    projected = projected.reshape(batch, sequence, heads, head_dim)
    projected *= torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-6)
    if norm is not None:
        projected *= norm.double()
    rotary_dim = cos.shape[1]
    first, second = projected[..., :rotary_dim].chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    rotary = (
        projected[..., :rotary_dim] * cos.double()[None, :, None]
        + rotated * sin.double()[None, :, None]
    )
    transformed = torch.cat((rotary, projected[..., rotary_dim:]), -1)
    indices = torch.arange(head_dim, device=x.device)
    masks = torch.tensor(SIGNED_HADAMARD_MASK, device=x.device, dtype=torch.int64)
    signs = 2 * ((masks[indices // 32] >> (indices % 32)) & 1) - 1
    bit_products = indices[:, None] & indices[None, :]
    parity = torch.zeros_like(bit_products)
    for bit in range(head_dim.bit_length() - 1):
        parity ^= (bit_products >> bit) & 1
    hadamard = ((1 - 2 * parity).double() * signs[:, None]) / math.sqrt(head_dim)
    rotated_query = (transformed @ hadamard).transpose(1, 2)
    storage = (sequence + 63) // 64 * 64
    padded = rotated_query.new_zeros((batch, heads, storage, head_dim))
    padded[:, :, :sequence] = rotated_query
    groups = padded.reshape(batch, heads, storage // 32, 32, head_dim)
    scale = groups.abs().amax((-1, -2)) / 127 + 1e-7
    normalized = groups / scale[..., None, None]
    codes = torch.trunc(normalized + 0.5 * normalized.sign()).clamp(-127, 127).to(torch.int8)
    scale *= head_dim**-0.5 * math.log2(math.e)
    scale[:, :, (sequence + 31) // 32 :] = 0
    return codes.reshape_as(padded), scale


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize(
    ("sequence", "heads", "features"), [(1, 1, 64), (65, 3, 272), (193, 6, 256)]
)
def test_query_matches_independent_projection_norm_rope_and_q32_reference(
    head_dim,
    dtype,
    affine,
    sequence,
    heads,
    features,
):
    operands = _operands(
        "cuda",
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        features=features,
        dtype=dtype,
        affine=affine,
    )
    bias = torch.randn(heads * head_dim, device="cuda", dtype=dtype) if affine else None
    codes, scales = query._project_query_op(
        *operands, 1e-6, head_dim**-0.5, bias, head_dim=head_dim
    )
    expected_codes, expected_scales = _reference(operands, bias, head_dim)
    assert codes.shape == expected_codes.shape
    assert codes.dtype is torch.int8
    assert scales.dtype is torch.float32
    assert (codes.short() - expected_codes.short()).abs().max() <= 1
    torch.testing.assert_close(scales.double(), expected_scales, rtol=3e-5, atol=1e-7)
    assert torch.count_nonzero(codes[:, :, sequence:]) == 0
    assert torch.count_nonzero(scales[:, :, (sequence + 31) // 32 :]) == 0


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
def test_zero_input_keeps_usable_real_group_scales_and_neutral_padding(head_dim):
    operands = _operands("cuda", head_dim=head_dim)
    operands[0].zero_()
    codes, scales = query._project_query_op(*operands, 1e-6, head_dim**-0.5)
    assert torch.count_nonzero(codes) == 0
    torch.testing.assert_close(
        scales[:, :, :3],
        torch.full_like(scales[:, :, :3], 1e-7 * head_dim**-0.5 * math.log2(math.e)),
    )
    assert torch.count_nonzero(scales[:, :, 3:]) == 0


@pytest.mark.gpu
@_NATIVE
def test_query_boundary_opcheck_compile_and_cuda_graph_replay():
    operands = _operands("cuda")
    arguments = (*operands, 1e-6, 64**-0.5)
    assert set(torch.library.opcheck(query._project_query_op, arguments).values()) == {"SUCCESS"}
    compiled = torch.compile(query._project_query_op, backend="eager", fullgraph=True)
    expected = query._project_query_op(*arguments)
    actual = compiled(*arguments)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = query._project_query_op(*arguments)
    operands[0].zero_()
    updated = query._project_query_op(*arguments)
    graph.replay()
    for left, right in zip(captured, updated, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("affine", [False, True])
def test_fake_query_validates_metadata_without_target_probes(monkeypatch, head_dim, affine):
    def forbidden(*args, **kwargs):
        pytest.fail("fake query projection inspected hardware")

    monkeypatch.setattr(_backend, "select_projection_backend", forbidden)
    operands = _operands("meta", head_dim=head_dim, affine=affine)
    codes, scales = query._project_query_op(*operands, 1e-6, head_dim**-0.5, head_dim=head_dim)
    assert codes.shape == (2, 3, 128, head_dim)
    assert scales.shape == (2, 3, 4)
    assert codes.dtype is torch.int8
    assert scales.dtype is torch.float32
    assert codes.device.type == "meta"
    assert scales.device.type == "meta"


def test_empty_batch_skips_target_probe_and_accepts_learned_norm_under_no_grad(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("empty query projection inspected hardware")

    monkeypatch.setattr(_backend, "select_projection_backend", forbidden)
    operands = list(_operands("cpu"))
    operands[0] = operands[0][:0]
    operands[1] = operands[1][:0]
    operands[4].requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        query._launch_query_projection(*operands, 1e-6, 64**-0.5)
    with torch.no_grad():
        codes, scales = query._launch_query_projection(*operands, 1e-6, 64**-0.5)
    assert codes.shape == (0, 3, 128, 64)
    assert scales.shape == (0, 3, 4)
    assert codes.dtype is torch.int8
    assert scales.dtype is torch.float32


@pytest.mark.parametrize(
    "invalid",
    [
        "input_scale",
        "weight_scale",
        "rope_stride",
        "rope_dtype",
        "head_dim",
        "epsilon",
        "softmax",
    ],
)
def test_fake_query_rejects_invalid_metadata(invalid):
    operands = list(_operands("meta"))
    epsilon, softmax, head_dim = 1e-6, 64**-0.5, 64
    if invalid == "input_scale":
        operands[1] = torch.empty((2, 66), device="meta")
    elif invalid == "weight_scale":
        operands[3] = operands[3].to(torch.bfloat16)
    elif invalid == "rope_stride":
        operands[5] = torch.empty((65, 96), device="meta")[:, ::2]
    elif invalid == "rope_dtype":
        operands[6] = operands[6].to(torch.bfloat16)
    elif invalid == "head_dim":
        head_dim = 128
    elif invalid == "epsilon":
        epsilon = 0.0
    elif invalid == "softmax":
        softmax = float("inf")
    with pytest.raises(ValueError, match=r"projection|head_dim|RMSNorm"):
        query._project_query_op(*operands, epsilon, softmax, head_dim=head_dim)


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cuda", "sm121"), False),
        (AcceleratorTarget("cuda", "sm89"), False),
        (AcceleratorTarget("hip", "gfx1201"), True),
        (AcceleratorTarget("hip", "gfx1100"), False),
        (AcceleratorTarget("cpu"), False),
    ],
)
def test_projection_selector_uses_validated_targets(monkeypatch, target, supported):
    monkeypatch.setattr(_backend.amd_policy.sys, "platform", "linux")
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda _: target)
    value = torch.empty(0)
    assert (_backend.select_projection_backend(value, head_dim=64) is not None) is supported
    assert _backend.select_projection_backend(value, head_dim=32) is None


@pytest.mark.skipif(sys.platform != "linux", reason="offline ROCm compilation requires Linux")
@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("affine", [False, True])
def test_production_query_launches_compile_for_rdna4(monkeypatch, arch, head_dim, affine):
    function = projection._project_qk_kernel
    kernel = MagicMock()
    monkeypatch.setattr(projection, "_project_qk_kernel", kernel)
    monkeypatch.setattr(projection, "device_context", lambda _: nullcontext())
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda _: AcceleratorTarget("hip", arch))
    operands = _operands("meta", sequence=193, head_dim=head_dim, affine=affine)
    bias = torch.empty(3 * head_dim, device="meta", dtype=torch.bfloat16) if affine else None
    query._launch_query_projection(*operands, 1e-6, head_dim**-0.5, bias, head_dim=head_dim)
    calls = kernel.__getitem__.return_value.call_args_list
    assert len(calls) == 2
    assert [call.args[0] for call in kernel.__getitem__.call_args_list] == [(3, 3, 2), (1, 3, 2)]
    for call in calls:
        compiled = _compile_rdna4_launch(function, call, arch)
        assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]
        assert compiled.metadata.shared <= 65536
        assert "arith.truncf" not in compiled.asm["ttgir"]


def _compile_rdna4_launch(function, call, arch):
    arguments = dict(zip(function.arg_names, call.args, strict=False))
    arguments.update(
        {name: item for name, item in call.kwargs.items() if name in function.arg_names}
    )
    constants, signature = {}, {}
    types = {torch.int8: "*i8", torch.float32: "*fp32", torch.bfloat16: "*bf16"}
    for parameter in function.params:
        argument = arguments[parameter.name]
        if parameter.is_constexpr:
            constants[parameter.name] = argument
        elif argument is None:
            constants[parameter.name] = None
            signature[parameter.name] = "constexpr"
        else:
            signature[parameter.name] = (
                types[argument.dtype] if isinstance(argument, torch.Tensor) else "i32"
            )
    compiled = triton.compile(
        ASTSource(function, signature, constexprs=constants),
        target=GPUTarget("hip", arch, 32),
        options={name: call.kwargs[name] for name in ("num_warps", "num_stages")},
    )
    return compiled
