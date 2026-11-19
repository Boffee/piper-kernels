"""SM89 and SM120 project dense D128 heads with Gluon kernels, and other shapes with Triton."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from _accelerators import exact_sm120_available
from _nvidia_resources import PROJECTION_SPILL_STORE_BUDGET_BYTES, compile_with_spill_stores
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import key, query, value
from piper_kernels.fusions.convrot_int8_piper import triton as projection
from piper_kernels.fusions.convrot_int8_piper._nvidia import dispatch, gluon, policy
from piper_kernels.fusions.convrot_int8_projection._nvidia import _tma

_SM89_TARGET = AcceleratorTarget("cuda", "sm89")
_SM120_TARGET = AcceleratorTarget("cuda", "sm120")
_NVIDIA_TARGETS = [AcceleratorTarget("cuda", architecture) for architecture in ("sm89", "sm120")]
# Only Q and K produce Q/K scales, so V runs once.
_OPERATION_GRANULARITIES = [
    ("query", "per_warp"),
    ("query", "per_thread"),
    ("key", "per_warp"),
    ("key", "per_thread"),
    ("value", "per_warp"),
]
# Each kernel varies only the constexprs it reads: Q its scale granularity.
_COMPILE_VARIANTS = [
    *(("query", variant) for variant in ("full_tiles", "masked_tail", "per_warp")),
    ("query", "bias_without_norm"),
    *(("key", variant) for variant in ("full_tiles", "masked_tail", "bias_without_norm")),
    *(("value", variant) for variant in ("full_tiles", "masked_tail", "bias_without_norm")),
]
_NVIDIA = torch.cuda.is_available() and AcceleratorTarget.from_device(
    torch.device("cuda")
).cuda_capability_at_least(8)


def _operands(
    device, *, batch=2, sequence=193, heads=3, input_features=256, head_dim=128, rotary_dim=96
):
    angles = torch.rand((sequence, rotary_dim), device=device).mul_(2 * torch.pi)
    return (
        torch.randint(
            -127, 128, (batch, sequence, input_features), device=device, dtype=torch.int8
        ),
        torch.rand((batch, sequence), device=device).mul_(0.01).add_(0.001),
        torch.randint(
            -127, 128, (heads * head_dim, input_features), device=device, dtype=torch.int8
        ),
        torch.rand((heads * head_dim, 1), device=device).mul_(0.01).add_(0.001),
        torch.rand(head_dim, device=device).add_(0.5).bfloat16(),
        angles.cos(),
        angles.sin(),
    )


def _outputs(operation, device, *, rows, batch=2, heads=3, head_dim=128, granularity="per_thread"):
    """Allocate sentinel-filled outputs, so unwritten elements compare equal."""
    shape = (batch, rows, heads, head_dim)
    with torch.device(device):
        if operation == "query":
            outputs = query._new_outputs(torch.empty(0, dtype=torch.int8), shape, granularity)
        elif operation == "key":
            outputs = key._new_outputs(torch.empty(0, dtype=torch.int8), shape, granularity)
        else:
            outputs = value._new_outputs(torch.empty(0, dtype=torch.int8), shape)
    for tensor in outputs:
        tensor.fill_(99 if tensor.dtype is torch.int8 else -7.0)
    return outputs


def _launch(
    module,
    operation,
    operands,
    out,
    *,
    bias=None,
    window=(64, 129),
    causal=False,
    target=_SM89_TARGET,
    **options,
):
    input_qdata, input_scale, weight_qdata, weight_scale, *_ = operands
    if module in (dispatch, gluon) and "execution_plan" not in options:
        metadata = {
            "operation": operation,
            "head_dim": out[0].shape[2 if operation == "value" else 3],
            "rotary_dim": 0 if operation == "value" else operands[5].shape[1],
        }
        options["execution_plan"] = (
            dispatch.default_execution_plan(input_qdata, weight_qdata, target=target, **metadata)
            if module is dispatch
            else policy.select_execution_plan(
                target, input_features=input_qdata.shape[2], operands_aligned=True, **metadata
            )
        )
    if operation == "query":
        module.project_query(
            *operands,
            1e-5,
            out[0].shape[3] ** -0.5,
            bias,
            chunk_start=window[0],
            chunk_rows=window[1],
            out=out,
            **options,
        )
    elif operation == "key":
        module.project_key(*operands, 1e-5, bias, out=out, **options)
    else:
        module.project_value(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            is_causal=causal,
            out=out,
            **options,
        )


@pytest.mark.parametrize(
    ("head_dim", "input_features", "rotary_dim", "supported"),
    [
        (128, 256, 96, True),
        (128, 320, 32, True),
        (128, 256, 128, True),
        # Feature-bit RoPE needs rotary widths in multiples of 32.
        (128, 256, 48, False),
        # cp.async copies whole K64 slices.
        (128, 272, 96, False),
        (64, 256, 32, False),
    ],
)
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("target", _NVIDIA_TARGETS, ids=str)
def test_every_target_uses_gluon_only_for_supported_projections(
    monkeypatch, target, operation, head_dim, input_features, rotary_dim, supported
):
    monkeypatch.setattr(dispatch.linear_gluon, "operands_aligned", lambda *operands: True)
    gluon_launch, triton_launch = Mock(), Mock()
    monkeypatch.setattr(gluon, f"project_{operation}", gluon_launch)
    monkeypatch.setattr(projection, f"project_{operation}", triton_launch)
    with FakeTensorMode():
        operands = _operands(
            "cuda:1", input_features=input_features, head_dim=head_dim, rotary_dim=rotary_dim
        )
        out = _outputs(operation, "cuda:1", rows=193, head_dim=head_dim)
        _launch(dispatch, operation, operands, out, target=target)
    # V has no RoPE, so only its head and input widths matter.
    uses_gluon = supported or (
        operation == "value" and head_dim == 128 and input_features % 64 == 0
    )
    launched, idle = (gluon_launch, triton_launch) if uses_gluon else (triton_launch, gluon_launch)
    launched.assert_called_once()
    idle.assert_not_called()
    assert launched.call_args.kwargs["out"] is out
    plan = launched.call_args.kwargs["execution_plan"]
    expected_kernel = (
        ("gluon_tma" if target.is_cuda_capability(12, 0) else "gluon_async_copy")
        if uses_gluon
        else "triton"
    )
    assert plan.kernel == expected_kernel
    if not uses_gluon:
        assert plan is policy.select_execution_plan(
            target,
            operation=operation,
            input_features=272,
            head_dim=128,
            operands_aligned=True,
        )


@pytest.mark.parametrize("target", _NVIDIA_TARGETS, ids=str)
@pytest.mark.parametrize("operand_index", [0, 2], ids=["input", "weight"])
def test_execution_plan_requires_contiguous_operands(target, operand_index):
    operands = list(_operands("cpu", batch=1, sequence=192, heads=2))
    operands[0] = operands[0][:, ::2, :] if operand_index == 0 else operands[0]
    operands[2] = operands[2][::2, :] if operand_index == 2 else operands[2]
    assert dispatch.linear_gluon.operands_aligned(operands[0], operands[2])
    assert not operands[operand_index].is_contiguous()
    plan = dispatch.default_execution_plan(
        operands[0],
        operands[2],
        target=target,
        operation="query",
        head_dim=128,
        rotary_dim=96,
    )
    assert plan.kernel == "triton"


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(
    ("architecture", "heads_per_program", "num_warps"),
    # SM120's two-head tiles exceed SM89's shared memory.
    [("sm89", 1, 4), ("sm120", 2, 8)],
)
def test_unaligned_operands_use_the_target_triton_schedule(
    operation, architecture, heads_per_program, num_warps
):
    plan = policy.select_execution_plan(
        AcceleratorTarget("cuda", architecture),
        operation=operation,
        input_features=5376,
        head_dim=128,
        rotary_dim=96,
        operands_aligned=False,
    )
    assert plan.kernel == "triton"
    assert (plan.heads_per_program, plan.num_warps, plan.block_k) == (
        heads_per_program,
        num_warps,
        128,
    )


@pytest.mark.parametrize("architecture", ["sm80", "sm86", "sm90", "sm121"])
def test_other_nvidia_targets_have_no_dense_projection_policy(architecture):
    target = AcceleratorTarget("cuda", architecture)
    assert not policy.supports_target(target)
    with pytest.raises(ValueError, match="no NVIDIA policy"):
        policy.select_execution_plan(
            target, operation="value", input_features=256, head_dim=128, operands_aligned=True
        )


@pytest.mark.parametrize(
    ("operation", "granularity"),
    _OPERATION_GRANULARITIES,
)
def test_gluon_launches_full_tiles_then_one_masked_tail(monkeypatch, operation, granularity):
    function = getattr(gluon, f"_{operation}_kernel")
    kernel = MagicMock()
    monkeypatch.setattr(gluon, function.__name__, kernel)
    project_value_chunk_means = Mock()
    monkeypatch.setattr(
        gluon.shared_projection,
        "project_value_chunk_means",
        project_value_chunk_means,
    )
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(gluon, "device_context", guard)
    encode_key = Mock()
    monkeypatch.setattr(gluon.piper_quantization, "prepare_chunk_centered_key", encode_key)
    sequence, window = 448, (128, 300)
    rows = window[1] if operation == "query" else sequence
    storage = (rows + 63) // 64 * 64
    with FakeTensorMode():
        operands = _operands("cuda:1", sequence=sequence)
        out = _outputs(operation, "cuda:1", rows=rows, granularity=granularity)
        _launch(gluon, operation, operands, out, window=window, causal=False)

    guard.assert_called_once_with(torch.device("cuda:1"))
    # Heads vary fastest, then 128-row blocks, then the batch.
    grids = [call.args[0] for call in kernel.__getitem__.call_args_list]
    assert grids == [(3, rows // 128, 2), (3, 1, 2)]
    calls = kernel.__getitem__.return_value.call_args_list
    for index, call in enumerate(calls):
        arguments = dict(zip(function.arg_names, call.args, strict=False)) | call.kwargs
        assert arguments["row_block_offset"] == (0 if index == 0 else rows // 128)
        assert arguments["storage_tiles"] == storage // 64
        assert arguments["logical_sequence_length"] == sequence
        assert arguments["mask_rows"] is (index == 1)
        assert arguments["num_warps"] == 4
        if operation == "query":
            assert arguments["chunk_start"] == window[0]
            assert arguments["query_sequence_end"] == sum(window)
            assert arguments["per_thread_scales"] is (granularity == "per_thread")
        if operation == "value":
            assert arguments["chunk_count"] == 1
    if operation == "value":
        expected_plan = policy.select_execution_plan(
            _SM89_TARGET,
            operation="value",
            input_features=operands[0].shape[2],
            head_dim=128,
            operands_aligned=True,
        )
        project_value_chunk_means.assert_called_once()
        assert (
            project_value_chunk_means.call_args.kwargs["block_n"]
            == expected_plan.value_mean_block_n
        )
        assert project_value_chunk_means.call_args.kwargs["out"] is out[3]
    else:
        project_value_chunk_means.assert_not_called()
    if operation == "key":
        stored = calls[0].args[8]
        assert stored.dtype is torch.bfloat16
        assert stored.shape == out[0].shape
        encode_key.assert_called_once()
        assert encode_key.call_args.kwargs["grouped"] is (granularity == "per_warp")
        assert encode_key.call_args.kwargs["storage_key_length"] == storage
        assert encode_key.call_args.kwargs["out"] == out
    else:
        encode_key.assert_not_called()


@pytest.mark.parametrize("operation", ["query", "key", "value"])
def test_sm120_tma_launches_use_descriptors_and_skip_empty_batches(monkeypatch, operation):
    function = getattr(gluon, f"_{operation}_kernel")
    kernel = MagicMock()
    monkeypatch.setattr(gluon, function.__name__, kernel)
    monkeypatch.setattr(gluon, "device_context", lambda device: nullcontext())
    monkeypatch.setattr(gluon.shared_projection, "project_value_chunk_means", Mock())
    monkeypatch.setattr(gluon.piper_quantization, "prepare_chunk_centered_key", Mock())
    descriptors = [object(), object()]
    descriptor_builder = Mock(side_effect=descriptors)
    monkeypatch.setattr(_tma, "build_matrix_descriptor", descriptor_builder)
    tma_plan = policy.select_execution_plan(
        _SM120_TARGET,
        operation=operation,
        input_features=256,
        head_dim=128,
        rotary_dim=96,
        operands_aligned=True,
    )
    assert tma_plan.kernel == "gluon_tma"
    with FakeTensorMode():
        operands = _operands("cuda:1", batch=1, sequence=128)
        out = _outputs(operation, "cuda:1", rows=128, batch=1)
        _launch(gluon, operation, operands, out, window=(0, 128), execution_plan=tma_plan)
    assert descriptor_builder.call_count == 2
    kernel.__getitem__.return_value.assert_called_once()
    call = kernel.__getitem__.return_value.call_args
    assert call.args[0] is descriptors[0]
    assert call.args[2] is descriptors[1]

    descriptor_builder.reset_mock()
    kernel.reset_mock()
    with FakeTensorMode():
        empty_operands = _operands("cuda:1", batch=0, sequence=128)
        empty_out = _outputs(operation, "cuda:1", rows=128, batch=0)
        _launch(
            gluon, operation, empty_operands, empty_out, window=(0, 128), execution_plan=tma_plan
        )
    descriptor_builder.assert_not_called()
    kernel.__getitem__.assert_not_called()


# Register allocation depends on the specialized input width, and small widths can
# hide spills that production widths show, so kernels compile at H3's 5376 features.
_INPUT_FEATURES = 5376

_POINTER_TYPES = {
    "input_ptr": "*i8",
    "weight_ptr": "*i8",
    "query_ptr": "*i8",
    "value_ptr": "*i8",
    "stored_ptr": "*bf16",
    "norm_weight_ptr": "*bf16",
    "bias_ptr": "*bf16",
}
_TMA_DESCRIPTOR_SIGNATURE = (
    f"tensordesc<i8[128,64],{gl.NVMMASharedLayout.get_default_for([128, 64], gl.int8)!r}>"
)


def _compile_dense_kernel(architecture, operation, variant, *, input_features, use_tma):
    function = getattr(gluon, f"_{operation}_kernel")
    constants = {
        "input_features": input_features,
        "heads": 3,
        "rotary_dim": 96,
        "norm_epsilon": 1e-5,
        "softmax_scale": 128**-0.5,
        "per_thread_scales": variant != "per_warp",
        "is_causal": variant == "causal",
        "mask_rows": variant != "full_tiles",
    }
    constants = {name: value for name, value in constants.items() if name in function.arg_names}
    signature, attributes = {}, {}
    for index, name in enumerate(function.arg_names):
        if name in constants:
            continue
        omitted_bias = name == "bias_ptr" and variant != "bias_without_norm"
        omitted_norm = name == "norm_weight_ptr" and variant == "bias_without_norm"
        if omitted_bias or omitted_norm:
            constants[name], signature[name] = None, "constexpr"
        elif use_tma and name in ("input_ptr", "weight_ptr"):
            signature[name] = _TMA_DESCRIPTOR_SIGNATURE
        elif name.endswith("_ptr"):
            signature[name] = _POINTER_TYPES.get(name, "*fp32")
            # Launches specialize PyTorch's 16-byte-aligned allocations; cp.async relies on it.
            attributes[(index,)] = [["tt.divisibility", 16]]
        else:
            signature[name] = "i32"
    compiled, spill_stores = compile_with_spill_stores(
        GluonASTSource(function, signature, constexprs=constants, attrs=attributes),
        GPUTarget("cuda", architecture, 32),
        {"num_warps": 4},
    )
    assert compiled.asm["cubin"]
    # Two programs share each SM's 99 KiB of shared memory.
    assert compiled.metadata.shared <= 99 * 1024 // 2
    assert spill_stores <= PROJECTION_SPILL_STORE_BUDGET_BYTES
    truncations = [line for line in compiled.asm["ttgir"].splitlines() if "arith.truncf" in line]
    # K rounds its transformed rows to BF16 for the centered encoder; V rounds log scales to FP16.
    assert bool(truncations) is (operation != "query")
    if operation == "key":
        assert all("f32" in line and "bf16" in line for line in truncations)


@pytest.mark.parametrize("architecture", [89, 120])
@pytest.mark.parametrize(("operation", "variant"), _COMPILE_VARIANTS)
def test_gluon_kernels_compile_for_two_programs_per_sm(architecture, operation, variant):
    _compile_dense_kernel(
        architecture, operation, variant, input_features=_INPUT_FEATURES, use_tma=False
    )


@pytest.mark.parametrize("input_features", [_INPUT_FEATURES, 4096])
@pytest.mark.parametrize(("operation", "variant"), _COMPILE_VARIANTS)
def test_sm120_tma_kernels_compile_for_two_programs_per_sm(input_features, operation, variant):
    _compile_dense_kernel(120, operation, variant, input_features=input_features, use_tma=True)


def _assert_matches(operation, actual, expected):
    if operation == "value":
        # V shares the Triton projection and per-token quantization arithmetic.
        for left, right in zip(actual, expected, strict=True):
            assert torch.equal(left, right)
        return
    codes = (actual[0].to(torch.int16) - expected[0].to(torch.int16)).abs()
    assert int(codes.max()) <= 1
    assert float((codes > 0).float().mean()) < 1e-4
    # K scales come from BF16 K storage, where 1-ulp FP32 differences can cross a rounding
    # boundary, as between dense and sparse K.
    torch.testing.assert_close(
        actual[1], expected[1], rtol=3e-3 if operation == "key" else 1e-5, atol=0
    )
    if operation == "key":
        torch.testing.assert_close(actual[2], expected[2], rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not _NVIDIA, reason="requires an NVIDIA GPU with cp.async and mma_v2")
@pytest.mark.parametrize(("operation", "granularity"), _OPERATION_GRANULARITIES)
@pytest.mark.parametrize(
    "case",
    [
        # Five K64 slices do not fill whole three-stage rounds; the Q window starts mid-sequence.
        {"batch": 2, "sequence": 1000, "input_features": 320, "window": (128, 700)},
        # The last 128-row tile ends 64 rows past the padded storage.
        {"batch": 1, "sequence": 4160, "rotary_dim": 64, "window": (0, 4160), "causal": True},
        {"batch": 2, "sequence": 1024, "rotary_dim": 128, "bias": True, "window": (896, 128)},
        {"batch": 1, "sequence": 200, "rotary_dim": 32, "affine": False, "window": (0, 77)},
    ],
)
def test_gluon_projections_match_the_triton_launchers(operation, granularity, case):
    torch.manual_seed(1113)
    batch, sequence = case["batch"], case["sequence"]
    operands = _operands(
        "cuda",
        batch=batch,
        sequence=sequence,
        input_features=case.get("input_features", 256),
        rotary_dim=case.get("rotary_dim", 96),
    )
    if not case.get("affine", True):
        operands = (*operands[:4], None, *operands[5:])
    options = {
        "bias": torch.randn(3 * 128, device="cuda").bfloat16() if case.get("bias") else None,
        "window": case["window"],
        "causal": case.get("causal", False),
    }
    rows = case["window"][1] if operation == "query" else sequence
    allocation = {"rows": rows, "batch": batch, "granularity": granularity}
    expected = _outputs(operation, "cuda", **allocation)
    actual = _outputs(operation, "cuda", **allocation)
    triton_options = {"packed_amd": False} if operation == "value" else {}

    _launch(
        projection,
        operation,
        operands,
        expected,
        execution_plan=policy.select_execution_plan(
            _SM89_TARGET,
            operation=operation,
            input_features=operands[0].shape[2],
            head_dim=128,
            operands_aligned=False,
        ),
        **options,
        **triton_options,
    )
    _launch(gluon, operation, operands, actual, **options)

    _assert_matches(operation, actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(("operation", "granularity"), _OPERATION_GRANULARITIES)
@pytest.mark.parametrize("input_features", [64, 128, 192, 320, 4096, 5376])
def test_sm120_tma_projections_match_async_copy(operation, granularity, input_features):
    torch.manual_seed(1913 + input_features)
    batch, sequence, heads = 2, 193, 3
    operands = _operands(
        "cuda", batch=batch, sequence=sequence, heads=heads, input_features=input_features
    )
    window = (64, 129)
    rows = window[1] if operation == "query" else sequence
    allocation = {"rows": rows, "batch": batch, "heads": heads, "granularity": granularity}
    plan = dispatch.default_execution_plan(
        operands[0], operands[2], operation=operation, head_dim=128, rotary_dim=96
    )
    assert plan.kernel == "gluon_tma"
    actual = _outputs(operation, "cuda", **allocation)
    expected = _outputs(operation, "cuda", **allocation)
    _launch(dispatch, operation, operands, actual, window=window, execution_plan=plan)
    _launch(
        dispatch,
        operation,
        operands,
        expected,
        window=window,
        execution_plan=replace(plan, kernel="gluon_async_copy"),
    )
    if operation == "query":
        assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        # Triton's default FP fusion can shift an epilogue FMA by 1 ULP at small K.
        assert int((actual[1].view(torch.int32) - expected[1].view(torch.int32)).abs().max()) <= 1
    else:
        for left, right in zip(actual, expected, strict=True):
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
