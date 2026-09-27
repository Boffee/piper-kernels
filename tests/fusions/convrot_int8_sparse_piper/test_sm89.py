"""SM89 projects D128 heads with Gluon kernels and other shapes with the Triton launchers."""

from contextlib import nullcontext
from unittest.mock import MagicMock, Mock

import pytest
import torch
import triton
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import _compile, _kernels
from piper_kernels.fusions.convrot_int8_sparse_piper import triton as projection
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import padded_sequence_length
from piper_kernels.fusions.convrot_int8_sparse_piper._nvidia import gluon_async_copy, sm89

_SM89 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)


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
    emit_block_mean=True,
):
    """Allocate sentinel-filled outputs, so unwritten elements compare equal."""

    def filled(shape, dtype=torch.float32):
        return torch.full(shape, 99 if dtype is torch.int8 else -7.0, device=device, dtype=dtype)

    if operation == "query":
        return (
            filled((batch, heads, storage, head_dim), torch.int8),
            filled((batch, heads, storage // 32)),
            filled((batch, heads, storage // 64, head_dim)),
        )
    if operation == "key":
        return (
            filled((batch, heads, storage, head_dim), torch.int8),
            filled((batch, heads, storage // 64)),
            filled((batch, heads, storage // 64, head_dim)),
            filled((batch, heads, 0 if mean_routing else storage // 64, head_dim)),
        )
    value_mean = filled((batch, heads, head_dim))
    return (
        filled((batch, heads, head_dim, storage), torch.int8),
        filled((batch, heads, storage // 64, 1)),
        value_mean,
        filled((batch, heads, storage // 64, head_dim)) if emit_block_mean else value_mean,
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
    emit_block_mean=True,
    **options,
):
    input_qdata, input_scale, weight_qdata, weight_scale, *_ = operands
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
        input_mean = input_qdata.new_zeros(
            (input_qdata.shape[0], input_qdata.shape[2]), dtype=torch.float32
        )
        module.project_value(
            input_qdata,
            input_scale,
            input_mean,
            weight_qdata,
            weight_scale,
            block_lengths,
            emit_block_mean=emit_block_mean,
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
def test_sm89_uses_gluon_only_for_supported_projections(
    monkeypatch, operation, head_dim, input_features, rotary_dim, supported
):
    gluon_launch, triton_launch = Mock(), Mock()
    monkeypatch.setattr(gluon_async_copy, f"project_{operation}", gluon_launch)
    monkeypatch.setattr(projection, f"project_{operation}", triton_launch)
    with FakeTensorMode():
        operands = _operands(
            "cuda:1", input_features=input_features, head_dim=head_dim, rotary_dim=rotary_dim
        )
        out = _outputs(operation, "cuda:1", storage=256, head_dim=head_dim)
        _launch(sm89, operation, operands, out)
    # V has no RoPE, so only its head and input widths matter.
    uses_gluon = supported or (
        operation == "value" and head_dim == 128 and input_features % 64 == 0
    )
    launched, idle = (gluon_launch, triton_launch) if uses_gluon else (triton_launch, gluon_launch)
    launched.assert_called_once()
    idle.assert_not_called()
    assert launched.call_args.kwargs["out"] is out
    if not uses_gluon:
        assert launched.call_args.kwargs["config"] is sm89._CONFIG


def test_compiler_cache_keys_include_the_sm89_projection_kernels():
    assert sm89.__file__ in _compile._source_files()
    assert gluon_async_copy.__file__ in _compile._source_files()


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("with_block_lengths", [False, True])
def test_sm89_gluon_launches_full_tiles_then_one_masked_tail(
    monkeypatch, operation, with_block_lengths
):
    function = getattr(gluon_async_copy, f"_{operation}_kernel")
    kernel = MagicMock()
    monkeypatch.setattr(gluon_async_copy, function.__name__, kernel)
    mean_kernel = MagicMock()
    monkeypatch.setattr(_kernels, "_project_prepared_input_mean_kernel", mean_kernel)
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(gluon_async_copy, "device_context", guard)
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
        _launch(
            gluon_async_copy, operation, operands, out, block_lengths=block_lengths, window=window
        )

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
    if operation == "value":
        mean_kernel.__getitem__.assert_called_once_with((3, 2))
        assert mean_kernel.__getitem__.return_value.call_args.kwargs["block_n"] == 128
    else:
        mean_kernel.__getitem__.assert_not_called()


_POINTER_TYPES = {
    "input_ptr": "*i8",
    "weight_ptr": "*i8",
    "query_ptr": "*i8",
    "key_ptr": "*i8",
    "value_ptr": "*i8",
    "block_lengths_ptr": "*i32",
    "norm_weight_ptr": "*bf16",
    "bias_ptr": "*bf16",
}


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(
    "variant", ["full_tiles", "masked_tail", "mean_routing", "block_lengths", "bias_without_norm"]
)
def test_sm89_gluon_kernels_compile_for_two_programs_per_sm(operation, variant):
    function = getattr(gluon_async_copy, f"_{operation}_kernel")
    constants = {
        "input_features": 256,
        "heads": 3,
        "rotary_dim": 96,
        "norm_epsilon": 1e-5,
        "softmax_scale": 128**-0.5,
        "mean_pool_summary": variant == "mean_routing",
        "mask_block_lengths": variant == "block_lengths",
        "mask_rows": variant != "full_tiles",
        "emit_block_mean": True,
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
        elif name.endswith("_ptr"):
            signature[name] = _POINTER_TYPES.get(name, "*fp32")
            # Launches specialize PyTorch's 16-byte-aligned allocations; cp.async relies on it.
            attributes[(index,)] = [["tt.divisibility", 16]]
        else:
            signature[name] = "i32"
    compiled = triton.compile(
        GluonASTSource(function, signature, constexprs=constants, attrs=attributes),
        target=GPUTarget("cuda", 89, 32),
        options={"num_warps": 4},
    )
    assert compiled.asm["cubin"]
    # Two programs share each SM's 99 KiB of shared memory.
    assert compiled.metadata.shared <= 99 * 1024 // 2
    assert "arith.truncf" not in compiled.asm["ttgir"]


def _assert_matches(actual, expected):
    codes = (actual[0].to(torch.int16) - expected[0].to(torch.int16)).abs()
    assert int(codes.max()) <= 1
    assert float((codes > 0).float().mean()) < 1e-4
    for left, right in zip(actual[1:], expected[1:], strict=True):
        # Q summaries add a block's maximum and minimum, so compare against the block scale.
        scale = float(right[torch.isfinite(right)].abs().max()) if right.numel() else 1.0
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6 * scale, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.skipif(not _SM89, reason="requires an SM89 GPU")
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
            "emit_block_mean": False,
            "window": (0, 77),
        },
    ],
)
def test_sm89_gluon_projections_match_the_triton_launchers(operation, case):
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
        "emit_block_mean": case.get("emit_block_mean", True),
    }
    allocation = {
        "storage": padded_sequence_length(case["window"][1] if operation == "query" else sequence),
        "batch": batch,
        "mean_routing": mean_routing,
        "emit_block_mean": options["emit_block_mean"],
    }
    expected = _outputs(operation, "cuda", **allocation)
    actual = _outputs(operation, "cuda", **allocation)

    _launch(projection, operation, operands, expected, config=sm89._CONFIG, **options)
    _launch(gluon_async_copy, operation, operands, actual, **options)

    _assert_matches(actual, expected)
