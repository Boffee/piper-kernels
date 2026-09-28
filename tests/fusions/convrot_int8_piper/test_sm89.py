"""SM89 projects dense D128 heads with Gluon kernels and other shapes with the Triton launchers."""

from contextlib import nullcontext
from unittest.mock import MagicMock, Mock

import pytest
import torch
import triton
from torch._subclasses.fake_tensor import FakeTensorMode
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import _compile, key, query, value
from piper_kernels.fusions.convrot_int8_piper import triton as projection
from piper_kernels.fusions.convrot_int8_piper._nvidia import dispatch, gluon_async_copy, policy
from piper_kernels.fusions.convrot_int8_projection._nvidia import _plan as nvidia_plan
from piper_kernels.fusions.convrot_int8_projection._nvidia import fragments

_SM89_TARGET = AcceleratorTarget("cuda", "sm89")
_NVIDIA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (8, 0)
)


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
    module, operation, operands, out, *, bias=None, window=(64, 129), causal=False, **options
):
    input_qdata, input_scale, weight_qdata, weight_scale, *_ = operands
    if module in (dispatch, gluon_async_copy) and "execution_plan" not in options:
        metadata = {
            "operation": operation,
            "head_dim": out[0].shape[2 if operation == "value" else 3],
            "rotary_dim": 0 if operation == "value" else operands[5].shape[1],
        }
        options["execution_plan"] = (
            dispatch.default_execution_plan(
                input_qdata, weight_qdata, target=_SM89_TARGET, **metadata
            )
            if module is dispatch
            else policy.select_execution_plan(
                _SM89_TARGET, input_features=input_qdata.shape[2], operands_aligned=True, **metadata
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
def test_sm89_uses_gluon_only_for_supported_projections(
    monkeypatch, operation, head_dim, input_features, rotary_dim, supported
):
    monkeypatch.setattr(dispatch.linear_gluon, "operands_aligned", lambda *operands: True)
    gluon_launch, triton_launch = Mock(), Mock()
    monkeypatch.setattr(gluon_async_copy, f"project_{operation}", gluon_launch)
    monkeypatch.setattr(projection, f"project_{operation}", triton_launch)
    with FakeTensorMode():
        operands = _operands(
            "cuda:1", input_features=input_features, head_dim=head_dim, rotary_dim=rotary_dim
        )
        out = _outputs(operation, "cuda:1", rows=193, head_dim=head_dim)
        _launch(dispatch, operation, operands, out)
    # V has no RoPE, so only its head and input widths matter.
    uses_gluon = supported or (
        operation == "value" and head_dim == 128 and input_features % 64 == 0
    )
    launched, idle = (gluon_launch, triton_launch) if uses_gluon else (triton_launch, gluon_launch)
    launched.assert_called_once()
    idle.assert_not_called()
    assert launched.call_args.kwargs["out"] is out
    plan = launched.call_args.kwargs["execution_plan"]
    assert plan.kernel == ("gluon_async_copy" if uses_gluon else "triton")
    if not uses_gluon:
        assert plan is policy.select_execution_plan(
            _SM89_TARGET,
            operation=operation,
            input_features=272,
            head_dim=128,
            operands_aligned=True,
        )
        assert plan.heads_per_program == 1


@pytest.mark.parametrize("operation", ["query", "key", "value"])
def test_sm89_unaligned_operands_use_the_triton_launchers(operation):
    plan = policy.select_execution_plan(
        _SM89_TARGET, operation=operation, input_features=256, head_dim=128, operands_aligned=False
    )
    assert plan.kernel == "triton"


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("aligned", [False, True])
def test_sm120_keeps_its_two_head_triton_plan(operation, aligned):
    plan = policy.select_execution_plan(
        AcceleratorTarget("cuda", "sm120"),
        operation=operation,
        input_features=5376,
        head_dim=128,
        rotary_dim=96,
        operands_aligned=aligned,
    )
    assert plan.kernel == "triton"
    assert (plan.heads_per_program, plan.num_warps, plan.block_k) == (2, 8, 128)


@pytest.mark.parametrize("architecture", ["sm80", "sm86", "sm90", "sm121"])
def test_other_nvidia_targets_have_no_dense_projection_policy(architecture):
    target = AcceleratorTarget("cuda", architecture)
    assert not policy.supports_target(target)
    with pytest.raises(ValueError, match="no NVIDIA policy"):
        policy.select_execution_plan(
            target, operation="value", input_features=256, head_dim=128, operands_aligned=True
        )


def test_compiler_cache_keys_include_the_sm89_projection_kernels():
    for module in (dispatch, policy, gluon_async_copy, fragments, nvidia_plan):
        assert module.__file__ in _compile._source_files()
    assert dispatch.linear_gluon.__file__ in _compile._source_files()


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("granularity", ["per_warp", "per_thread"])
@pytest.mark.parametrize("causal", [False, True])
def test_sm89_gluon_launches_full_tiles_then_one_masked_tail(  # noqa: PLR0915
    monkeypatch, operation, granularity, causal
):
    function = getattr(gluon_async_copy, f"_{operation}_kernel")
    kernel = MagicMock()
    monkeypatch.setattr(gluon_async_copy, function.__name__, kernel)
    mean_kernel = MagicMock()
    monkeypatch.setattr(gluon_async_copy, "project_prepared_input_mean_kernel", mean_kernel)
    monkeypatch.setattr(gluon_async_copy._ops, "dequantized_input_mean", Mock())
    guard = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(gluon_async_copy, "device_context", guard)
    finalize_mean, encode_key = Mock(), Mock()
    monkeypatch.setattr(gluon_async_copy.centered_projection, "finalize_mean", finalize_mean)
    monkeypatch.setattr(gluon_async_copy.qk_quantization, "prepare_key", encode_key)
    sequence, window = 448, (128, 300)
    rows = window[1] if operation == "query" else sequence
    storage = (rows + 63) // 64 * 64
    with FakeTensorMode():
        operands = _operands("cuda:1", sequence=sequence)
        out = _outputs(operation, "cuda:1", rows=rows, granularity=granularity)
        _launch(gluon_async_copy, operation, operands, out, window=window, causal=causal)

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
            assert arguments["is_causal"] is causal
    if operation == "value" and not causal:
        mean_kernel.__getitem__.assert_called_once_with((3 * 128 // 32, 2))
        assert mean_kernel.__getitem__.return_value.call_args.kwargs["block_n"] == 32
    else:
        mean_kernel.__getitem__.assert_not_called()
    if operation == "key":
        stored, partials = calls[0].args[8:10]
        assert stored.dtype is torch.bfloat16
        assert stored.shape == out[0].shape
        assert partials.shape == (2, 3, storage // 64, 128)
        finalize_mean.assert_called_once()
        assert finalize_mean.call_args.args == (partials, sequence)
        encode_key.assert_called_once()
        assert encode_key.call_args.kwargs["grouped"] is (granularity == "per_warp")
        assert encode_key.call_args.kwargs["storage_key_length"] == storage
        assert encode_key.call_args.kwargs["out"] == out
    else:
        finalize_mean.assert_not_called()
        encode_key.assert_not_called()


_POINTER_TYPES = {
    "input_ptr": "*i8",
    "weight_ptr": "*i8",
    "query_ptr": "*i8",
    "value_ptr": "*i8",
    "stored_ptr": "*bf16",
    "norm_weight_ptr": "*bf16",
    "bias_ptr": "*bf16",
}


@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize(
    "variant", ["full_tiles", "masked_tail", "per_warp", "causal", "bias_without_norm"]
)
def test_sm89_gluon_kernels_compile_for_two_programs_per_sm(operation, variant):
    function = getattr(gluon_async_copy, f"_{operation}_kernel")
    constants = {
        "input_features": 256,
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
    truncations = [line for line in compiled.asm["ttgir"].splitlines() if "arith.truncf" in line]
    # K rounds its transformed rows to BF16 for the centered encoder; V rounds log scales to FP16.
    assert bool(truncations) is (operation != "query")
    if operation == "key":
        assert all("f32" in line and "bf16" in line for line in truncations)


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


@pytest.mark.gpu
@pytest.mark.skipif(not _NVIDIA, reason="requires an NVIDIA GPU with cp.async and mma_v2")
@pytest.mark.parametrize("operation", ["query", "key", "value"])
@pytest.mark.parametrize("granularity", ["per_warp", "per_thread"])
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
def test_sm89_gluon_projections_match_the_triton_launchers(operation, granularity, case):
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
    triton_options = {"packed_amd": False, "mean_block_n": 32} if operation == "value" else {}

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
    _launch(gluon_async_copy, operation, operands, actual, **options)

    _assert_matches(operation, actual, expected)
