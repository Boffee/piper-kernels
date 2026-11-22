"""NVIDIA projects D128 heads with Gluon kernels and other shapes with the Triton launchers."""

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
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import triton as projection
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import padded_sequence_length
from piper_kernels.fusions.convrot_int8_sparse_piper._nvidia import dispatch, gluon, policy
from piper_kernels.linear.convrot.int8._nvidia import gluon_tma

from ._reference import reconstruct_key_center

_SM89_TARGET = AcceleratorTarget("cuda", "sm89")
_SM120_TARGET = AcceleratorTarget("cuda", "sm120")
_NVIDIA_TARGETS = [
    AcceleratorTarget("cuda", architecture) for architecture in ("sm80", "sm86", "sm89", "sm120")
]
_MMA_V2_AVAILABLE = torch.cuda.is_available() and AcceleratorTarget.from_device(
    torch.device("cuda")
).cuda_capability_at_least(8)


def _operands(
    device, *, batch=2, sequence=193, heads=3, input_features=256, head_dim=128, rotary_dim=96
):
    input_shape, weight_shape = (
        (batch, sequence, input_features),
        (heads * head_dim, input_features),
    )
    angles = torch.rand((sequence, rotary_dim), device=device).mul_(2 * torch.pi)
    return (
        torch.randint(-127, 128, input_shape, device=device, dtype=torch.int8),
        torch.rand((batch, sequence), device=device).mul_(0.01).add_(0.001),
        torch.randint(-127, 128, weight_shape, device=device, dtype=torch.int8),
        torch.rand((heads * head_dim, 1), device=device).mul_(0.01).add_(0.001),
        torch.rand(head_dim, device=device).add_(0.5).bfloat16(),
        angles.cos(),
        angles.sin(),
    )


def _outputs(
    operation,
    device,
    *,
    storage,
    batch=2,
    heads=3,
    head_dim=128,
    mean_routing=False,
):
    """Allocate sentinel-filled outputs, so unwritten elements compare equal."""

    def filled(shape, dtype=torch.float32):
        if dtype is torch.int8:
            fill = 99
        elif dtype is torch.bfloat16:
            fill = -7.0
        else:
            fill = -7.0
        return torch.full(shape, fill, device=device, dtype=dtype)

    storage_tiles = storage // 64
    if operation == "query":
        return (
            filled((batch, heads, storage, head_dim), torch.int8),
            filled((batch, heads, storage // 32)),
            filled((batch, heads, storage_tiles, head_dim)),
        )
    if operation == "key":
        return (
            filled((batch, heads, storage, head_dim), torch.int8),
            filled((batch, heads, storage_tiles)),
            filled((batch, heads, storage_tiles, 2, head_dim), torch.int8),
            filled((batch, heads, storage_tiles, 2)),
            filled((batch, heads, storage_tiles, head_dim)),
            filled((batch, heads, 0 if mean_routing else storage_tiles, head_dim)),
        )
    return (
        filled((batch, heads, head_dim, storage), torch.int8),
        filled((batch, heads, storage_tiles, 1)),
        filled((batch, heads, storage_tiles, head_dim), torch.bfloat16),
    )


def _launch(
    module,
    operation,
    operands,
    out,
    *,
    routing_mode=_MINMAX_ROUTING,
    block_lengths=None,
    bias=None,
    window=(64, 129),
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
            routing_mode,
            block_lengths,
            chunk_start=window[0],
            chunk_rows=window[1],
            out=out,
            bias=bias,
            **options,
        )
    elif operation == "key":
        module.project_key(
            *operands, 1e-5, routing_mode, block_lengths, out=out, bias=bias, **options
        )
    else:
        module.project_value(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            block_lengths,
            value_format=ValueFormat.TRANSPOSED,
            out=out,
            bias=bias,
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
        out = _outputs(operation, "cuda:1", storage=256, head_dim=head_dim)
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
    if uses_gluon:
        expected_kernel = "gluon_tma" if target.is_cuda_capability(12, 0) else "gluon_async_copy"
        assert plan.kernel == expected_kernel
    else:
        assert plan is policy.select_execution_plan(
            target,
            operation=operation,
            input_features=272,
            head_dim=128,
            operands_aligned=True,
        )


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("with_block_lengths", [False, True])
def test_gluon_launches_full_tiles_then_one_masked_tail(monkeypatch, operation, with_block_lengths):
    function = getattr(gluon, f"_{operation}_kernel")
    kernel = MagicMock()
    monkeypatch.setattr(gluon, function.__name__, kernel)
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", guard)
    sequence, window = 448, (64, 300)
    rows = window[1] if operation == "query" else sequence
    storage = padded_sequence_length(rows)
    with FakeTensorMode():
        operands = _operands("cuda:1", sequence=sequence)
        block_lengths = (
            torch.empty(sequence // 64, device="cuda:1", dtype=torch.int32)
            if with_block_lengths
            else None
        )
        out = _outputs(operation, "cuda:1", storage=storage)
        _launch(gluon, operation, operands, out, block_lengths=block_lengths, window=window)

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
        assert arguments["mask_block_lengths"] is with_block_lengths
        assert arguments["mask_rows"] is (with_block_lengths or index == 1)
        assert arguments["num_warps"] == 4
        if operation == "query":
            assert arguments["chunk_start"] == window[0]
            assert arguments["query_sequence_end"] == sum(window)


@pytest.mark.parametrize("operation", ["query", "key", "value"])
def test_sm120_tma_launches_use_descriptors_and_skip_empty_batches(monkeypatch, operation):
    function = getattr(gluon, f"_{operation}_kernel")
    kernel = MagicMock()
    monkeypatch.setattr(gluon, function.__name__, kernel)
    monkeypatch.setattr(torch.cuda, "device", lambda _device=None: nullcontext())
    descriptors = [object(), object()]
    descriptor_builder = Mock(side_effect=descriptors)
    monkeypatch.setattr(gluon_tma, "build_matrix_descriptor", descriptor_builder)
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
        out = _outputs(operation, "cuda:1", storage=128, batch=1)
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
        empty_out = _outputs(operation, "cuda:1", storage=128, batch=0)
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
    "key_ptr": "*i8",
    "key_center_ptr": "*i8",
    "value_ptr": "*i8",
    "value_mean_ptr": "*bf16",
    "block_lengths_ptr": "*i32",
    "norm_weight_ptr": "*bf16",
    "bias_ptr": "*bf16",
}


# Spill stores measured past the budget and accepted, keyed by architecture, operation,
# and variant. SM80's instruction order spills masked Q's K loop when a bias replaces the
# norm weight, a rare configuration that every tile reaches only with block lengths. K32
# slices spill SM89's full-tile K mean-routing epilogue after the K loop; K64 slices
# avoided it but spilled SM120's full tiles, and one slice width serves every target.
_ACCEPTED_SPILL_STORES = {
    (80, "query", "bias_without_norm"): 104,
}

_COMPILE_VARIANTS = [
    "full_tiles",
    "full_tiles_mean_routing",
    "masked_tail",
    "mean_routing",
    "block_lengths",
    "bias_without_norm",
]
_TMA_DESCRIPTOR_SIGNATURE = (
    f"tensordesc<i8[128,64],{gl.NVMMASharedLayout.get_default_for([128, 64], gl.int8)!r}>"
)


def _compile_sparse_kernel(architecture, operation, variant, *, input_features, use_tma):
    function = getattr(gluon, f"_{operation}_kernel")
    constants = {
        "input_features": input_features,
        "heads": 3,
        "rotary_dim": 96,
        "norm_epsilon": 1e-5,
        "softmax_scale": 128**-0.5,
        "mean_pool_summary": variant in ("full_tiles_mean_routing", "mean_routing"),
        "mask_block_lengths": variant == "block_lengths",
        "mask_rows": variant not in ("full_tiles", "full_tiles_mean_routing"),
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
    truncations = [line for line in compiled.asm["ttgir"].splitlines() if "arith.truncf" in line]
    # Only V rounds its per-tile means to BF16.
    assert bool(truncations) is (operation == "value")
    assert all("f32" in line and "bf16" in line for line in truncations)
    return spill_stores


# SM86 and SM87 build the same PTX as SM89; SM80's differs in instruction order.
@pytest.mark.parametrize("architecture", [80, 89, 120])
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("variant", _COMPILE_VARIANTS)
def test_gluon_kernels_compile_for_two_programs_per_sm(architecture, operation, variant):
    spill_stores = _compile_sparse_kernel(
        architecture, operation, variant, input_features=_INPUT_FEATURES, use_tma=False
    )
    assert spill_stores <= _ACCEPTED_SPILL_STORES.get(
        (architecture, operation, variant), PROJECTION_SPILL_STORE_BUDGET_BYTES
    )


@pytest.mark.parametrize("input_features", [_INPUT_FEATURES, 4096])
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("variant", _COMPILE_VARIANTS)
def test_sm120_tma_kernels_compile_for_two_programs_per_sm(input_features, operation, variant):
    spill_stores = _compile_sparse_kernel(
        120, operation, variant, input_features=input_features, use_tma=True
    )
    assert spill_stores <= PROJECTION_SPILL_STORE_BUDGET_BYTES


def _assert_matches(actual, expected, operation):
    codes = (actual[0].to(torch.int16) - expected[0].to(torch.int16)).abs()
    if operation == "key":
        expected_scale = expected[1].repeat_interleave(64, dim=2)[..., None]
        non_trivial_tiles = expected_scale > 1e-6
        assert int(codes.masked_fill(~non_trivial_tiles, 0).max()) <= 1
        assert float(((codes > 0) & non_trivial_tiles).float().mean()) < 1e-4
        dequant_actual = actual[0].float() * actual[1].repeat_interleave(64, dim=2)[..., None]
        dequant_expected = expected[0].float() * expected_scale
        assert torch.all(
            (dequant_actual - dequant_expected).abs()
            <= torch.clamp(expected_scale * 1.05, min=1e-5)
        )
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-8)
        actual_center = reconstruct_key_center(actual[2], actual[3])
        expected_center = reconstruct_key_center(expected[2], expected[3])
        scale = float(expected_center.abs().max()) if expected_center.numel() else 1.0
        torch.testing.assert_close(actual_center, expected_center, rtol=1e-4, atol=1e-4 * scale)
        tail = zip(actual[4:], expected[4:], strict=True)
    else:
        assert int(codes.max()) <= 1
        assert float((codes > 0).float().mean()) < 1e-4
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-8)
        tail = zip(actual[2:], expected[2:], strict=True)
    for left, right in tail:
        left_fp32, right_fp32 = left.float(), right.float()
        # Q summaries add a block's maximum and minimum, and V means are BF16.
        scale = float(right_fp32[torch.isfinite(right_fp32)].abs().max()) if right.numel() else 1.0
        rtol = 1.6e-2 if right.dtype is torch.bfloat16 else 1e-5
        torch.testing.assert_close(
            left_fp32, right_fp32, rtol=rtol, atol=1e-6 * scale, equal_nan=True
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _MMA_V2_AVAILABLE, reason="requires NVIDIA SM80 or newer")
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(
    "case",
    [
        # Five K64 slices do not fill whole three-stage rounds; the Q window starts mid-sequence.
        {"batch": 2, "sequence": 1000, "input_features": 320, "window": (64, 700)},
        # The last 128-row tile ends 64 rows past the padded storage.
        {"batch": 1, "sequence": 8256, "rotary_dim": 64, "routing": "mean", "window": (0, 8256)},
        # The Q tail tile reaches 64 rows past the sequence, whose block lengths it must not read.
        {
            "batch": 2,
            "sequence": 1024,
            "rotary_dim": 128,
            "block_lengths": True,
            "bias": True,
            "affine": False,
            "window": (64, 960),
        },
        {
            "batch": 1,
            "sequence": 200,
            "rotary_dim": 32,
            "window": (0, 77),
        },
    ],
)
def test_gluon_projections_match_the_triton_launchers(operation, case):
    torch.manual_seed(1109)
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
    mean_routing = case.get("routing") == "mean"
    options = {
        "routing_mode": _MEAN_ROUTING if mean_routing else _MINMAX_ROUTING,
        "block_lengths": (
            torch.randint(1, 65, (sequence // 64,), device="cuda", dtype=torch.int32)
            if case.get("block_lengths")
            else None
        ),
        "bias": torch.randn(3 * 128, device="cuda").bfloat16() if case.get("bias") else None,
        "window": case["window"],
    }
    allocation = {
        "storage": padded_sequence_length(case["window"][1] if operation == "query" else sequence),
        "batch": batch,
        "mean_routing": mean_routing,
    }
    expected = _outputs(operation, "cuda", **allocation)
    actual = _outputs(operation, "cuda", **allocation)

    _launch(
        projection,
        operation,
        operands,
        expected,
        execution_plan=policy.select_execution_plan(
            _SM89_TARGET,
            operation=operation,
            input_features=272,
            head_dim=128,
            operands_aligned=True,
        ),
        **options,
    )
    _launch(gluon, operation, operands, actual, **options)

    _assert_matches(actual, expected, operation)


@pytest.mark.parametrize("target", [_SM89_TARGET, _SM120_TARGET], ids=str)
@pytest.mark.parametrize("operand_index", [0, 2], ids=["input", "weight"])
@pytest.mark.parametrize("offset", [0, 1, 8, 16])
def test_execution_plan_checks_both_operand_pointer_alignments(target, operand_index, offset):
    operands = list(_operands("cpu", batch=1, heads=1))
    original = operands[operand_index]
    storage = torch.empty(original.numel() + offset, dtype=torch.int8)
    operands[operand_index] = storage[offset:].view_as(original)
    plan = dispatch.default_execution_plan(
        operands[0],
        operands[2],
        target=target,
        operation="query",
        head_dim=128,
        rotary_dim=96,
    )
    gluon_kernel = "gluon_tma" if target.is_cuda_capability(12, 0) else "gluon_async_copy"
    assert plan.kernel == (gluon_kernel if offset % 16 == 0 else "triton")


@pytest.mark.parametrize("target", [_SM89_TARGET, _SM120_TARGET], ids=str)
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not _MMA_V2_AVAILABLE,
    reason="requires NVIDIA SM80 or newer",
)
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("operand_index", [0, 2], ids=["input", "weight"])
@pytest.mark.parametrize("offset", [1, 8])
def test_unaligned_projections_use_the_triton_fallback(
    monkeypatch, operation, operand_index, offset
):
    torch.manual_seed(1109)
    operands = list(_operands("cuda", batch=1, sequence=193, heads=1))
    original = operands[operand_index]
    storage = torch.empty(original.numel() + offset, device="cuda", dtype=torch.int8)
    operands[operand_index] = storage[offset:].view_as(original).copy_(original)
    assert operands[operand_index].is_contiguous()
    guard = Mock(side_effect=AssertionError("unaligned operands reached Gluon"))
    monkeypatch.setattr(gluon, f"project_{operation}", guard)
    expected = _outputs(operation, "cuda", storage=256, batch=1, heads=1)
    actual = _outputs(operation, "cuda", storage=256, batch=1, heads=1)
    plan = policy.select_execution_plan(
        _SM89_TARGET, operation=operation, input_features=256, head_dim=128, operands_aligned=False
    )
    _launch(
        projection,
        operation,
        operands,
        expected,
        execution_plan=plan,
        window=(0, 193),
    )
    _launch(dispatch, operation, operands, actual, window=(0, 193))
    guard.assert_not_called()
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not _MMA_V2_AVAILABLE,
    reason="requires NVIDIA SM80 or newer",
)
@pytest.mark.parametrize("routing_mode", [_MEAN_ROUTING, _MINMAX_ROUTING])
@pytest.mark.parametrize("rows", [31, 32, 33, 64, 77, 128])
@pytest.mark.parametrize("lengths", [(64, 64, 64), (64, 17, 51)])
def test_gluon_query_window_intersects_block_lengths(routing_mode, rows, lengths):
    torch.manual_seed(1109)
    operands = _operands("cuda", batch=1, sequence=192, heads=1)
    start, storage = 64, padded_sequence_length(rows)
    actual = _outputs("query", "cuda", storage=storage, batch=1, heads=1)
    expected = _outputs("query", "cuda", storage=storage, batch=1, heads=1)
    effective_lengths = list(lengths)
    if rows % 64:
        final_block = (start + rows) // 64
        effective_lengths[final_block] = min(effective_lengths[final_block], rows % 64)
    for out, window_rows, valid_prefixes in (
        (actual, rows, lengths),
        (expected, storage, effective_lengths),
    ):
        _launch(
            dispatch,
            "query",
            operands,
            out,
            routing_mode=routing_mode,
            block_lengths=torch.tensor(valid_prefixes, device="cuda", dtype=torch.int32),
            window=(start, window_rows),
        )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0][:, :, rows:]) == 0
    assert torch.count_nonzero(actual[1][:, :, (rows + 31) // 32 :]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("input_features", [64, 128, 192, 256, 320, 4096, 5376])
@pytest.mark.parametrize("mean_routing", [False, True])
def test_sm120_tma_projections_match_async_copy(operation, input_features, mean_routing):
    torch.manual_seed(1901 + input_features)
    batch, sequence, heads = 2, 193, 3
    operands = _operands(
        "cuda", batch=batch, sequence=sequence, heads=heads, input_features=input_features
    )
    if mean_routing:
        operands = (*operands[:4], None, *operands[5:])
    lengths = torch.tensor([55, 64, 31, 1], device="cuda", dtype=torch.int32)
    bias = torch.randn(heads * 128, device="cuda").bfloat16() if mean_routing else None
    window = (64, 129)
    storage = padded_sequence_length(window[1] if operation == "query" else sequence)
    plan = dispatch.default_execution_plan(
        operands[0], operands[2], operation=operation, head_dim=128, rotary_dim=96
    )
    assert plan.kernel == "gluon_tma"
    outputs = []
    for execution_plan in (plan, replace(plan, kernel="gluon_async_copy")):
        out = _outputs(
            operation, "cuda", storage=storage, batch=batch, heads=heads, mean_routing=mean_routing
        )
        _launch(
            dispatch,
            operation,
            operands,
            out,
            block_lengths=lengths,
            bias=bias,
            window=window,
            routing_mode=_MEAN_ROUTING if mean_routing else _MINMAX_ROUTING,
            execution_plan=execution_plan,
        )
        outputs.append(out)
    for actual, expected in zip(*outputs, strict=True):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("operation", ["query", "key", "value"])
def test_sm120_tma_graph_capture_and_replay(operation):
    torch.manual_seed(2026)
    batch, sequence, heads = 2, 193, 3
    operands = _operands("cuda", batch=batch, sequence=sequence, heads=heads, input_features=256)
    storage = padded_sequence_length(sequence)
    plan = dispatch.default_execution_plan(
        operands[0], operands[2], operation=operation, head_dim=128, rotary_dim=96
    )
    assert plan.kernel == "gluon_tma"
    actual = _outputs(operation, "cuda", storage=storage, batch=batch, heads=heads)
    expected = _outputs(operation, "cuda", storage=storage, batch=batch, heads=heads)
    _launch(dispatch, operation, operands, actual, window=(0, sequence), execution_plan=plan)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(dispatch, operation, operands, actual, window=(0, sequence), execution_plan=plan)
    operands[0].random_(-127, 128)
    for tensor in actual:
        tensor.zero_()
    graph.replay()
    _launch(dispatch, operation, operands, expected, window=(0, sequence), execution_plan=plan)
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
