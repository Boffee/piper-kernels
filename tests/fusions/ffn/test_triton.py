"""Parametrized GPU tests for direct FFN custom ops and specialized Triton preparation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from _accelerators import exact_gfx1201_available, exact_sm120_available, sm8x_available
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.fusions.ffn import _convrot_int8 as int8_ffn
from piper_kernels.fusions.ffn import _nvfp4 as nvfp4_ffn
from piper_kernels.linear.convrot.int8 import _backend as int8_backend
from piper_kernels.linear.convrot.int8 import reference as int8_reference
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_backend
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend
from piper_kernels.weights.nvfp4 import triton as nvfp4_primitives

from .._convrot_int8 import Linear as Int8Linear
from ._helpers import (
    Activation,
    Int8Operands,
    assert_residual_add_matches_gated_updates,
    make_int8_operands,
    make_nvfp4_operands,
    materialized_int8,
    materialized_nvfp4,
    relative_l2,
)

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
_SM120 = pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
_FUSED_SWIGLU_GPU = pytest.mark.skipif(
    not (exact_sm120_available() or sm8x_available() or exact_gfx1201_available()),
    reason="requires NVIDIA SM8x/SM120 or ROCm gfx1201",
)


def _materialized_gated_updates(
    ffn: torch.Tensor,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
) -> torch.Tensor:
    hidden = base.float() + update_gate[gate_indices].float() * reusable_update.float()
    return (hidden + ffn_gate[gate_indices].float() * ffn.float()).to(base.dtype)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("rows", [127, 385], ids=["short", "ragged-multi-chunk"])
@pytest.mark.parametrize("chunk_rows", [64, 128])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_convrot_int8_chunked_ffn_matches_materialized(
    activation: Activation,
    rows: int,
    chunk_rows: int,
    bias_dtype: torch.dtype | None,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(501 + rows)
    operands = make_int8_operands(activation, rows=rows, bias_dtype=bias_dtype, dtype=dtype)
    expected = materialized_int8(operands)
    actual = int8_ffn.convrot_int8_ffn_op(**operands.arguments(chunk_rows))

    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < 0.01


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize(
    ("rows", "width", "intermediate", "columns", "group", "down_group", "chunk_rows"),
    [
        (1, 48, 80, 31, 16, 16, 4),
        (129, 768, 1024, 335, 64, 256, 64),
        (257, 5376, 9216, 513, 256, 64, 128),
        (65, 256, 32768, 257, 256, 256, 64),
    ],
)
def test_convrot_int8_shape_boundaries_match_portable_math(
    rows: int,
    width: int,
    intermediate: int,
    columns: int,
    group: int,
    down_group: int,
    chunk_rows: int,
) -> None:
    torch.manual_seed(213)
    operands = make_int8_operands(
        "swiglu",
        rows=rows,
        input_features=width,
        intermediate_features=intermediate,
        output_features=columns,
        group_size=group,
        down_group_size=down_group,
    )
    assert operands.gate is not None
    gate_result = int8_reference.linear(
        operands.input,
        operands.gate.qdata,
        operands.gate.scale,
        operands.gate.group_size,
        operands.gate.bias,
    )
    up_result = int8_reference.linear(
        operands.input,
        operands.up.qdata,
        operands.up.scale,
        operands.up.group_size,
        operands.up.bias,
    )
    expected = int8_reference.linear(
        torch.cat((up_result, gate_result), dim=-1),
        operands.down.qdata,
        operands.down.scale,
        operands.down.group_size,
        operands.down.bias,
        activation_fn="swiglu",
    )
    actual = int8_ffn.convrot_int8_ffn_op(**operands.arguments(chunk_rows))
    assert actual.shape == (rows, columns)
    assert actual.isfinite().all()
    assert relative_l2(actual, expected) < 0.01
    assert relative_l2(actual, materialized_int8(operands)) < 0.01


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("device_index", [0, 1], ids=["current-gpu", "noncurrent-gpu"])
def test_convrot_int8_preserves_operand_stream_and_caller_device(device_index: int) -> None:
    if torch.cuda.device_count() <= device_index:
        pytest.skip("requires a second GPU")
    with torch.cuda.device(0):
        device = torch.device("cuda", device_index)
        original_stream = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            operands = make_int8_operands("swiglu", rows=129)
            expected = int8_ffn.convrot_int8_ffn_op(**operands.arguments(64))
            with torch.cuda.device(0):
                caller_stream = torch.cuda.current_stream()
                actual = int8_ffn.convrot_int8_ffn_op(**operands.arguments(64))
                assert torch.cuda.current_device() == 0
                assert torch.cuda.current_stream() == caller_stream
                assert torch.cuda.current_stream(device) == stream
            stream.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.cuda.current_stream(device) == original_stream
        assert torch.cuda.current_device() == 0


@pytest.mark.gpu
@_GPU
def test_convrot_int8_accepts_adjacent_checkpoint_views() -> None:
    torch.manual_seed(209)
    operands = make_int8_operands("swiglu", rows=385, bias_dtype=torch.float32)
    assert operands.gate is not None
    assert operands.up.bias is not None
    assert operands.gate.bias is not None
    width = operands.up.qdata.shape[0]
    qdata = torch.cat((operands.up.qdata, operands.gate.qdata))
    scale = torch.cat((operands.up.scale, operands.gate.scale))
    bias = torch.cat((operands.up.bias, operands.gate.bias))
    up = Int8Linear(qdata[:width], scale[:width], bias[:width])
    gate = Int8Linear(qdata[width:], scale[width:], bias[width:])
    views = Int8Operands(operands.input, "swiglu", up, operands.down, gate)

    assert up.qdata.untyped_storage().data_ptr() == gate.qdata.untyped_storage().data_ptr()
    assert gate.qdata.storage_offset() > 0
    actual = int8_ffn.convrot_int8_ffn_op(**views.arguments(128))
    assert relative_l2(actual, materialized_int8(views)) < 0.01


@pytest.mark.gpu
@_FUSED_SWIGLU_GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [None, torch.float32])
def test_convrot_int8_fused_swiglu_ffn_matches_materialized(
    dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
) -> None:
    torch.manual_seed(509)
    # Chunk capacity triggers fused TMA (SM120), async-copy (SM8x), or paired RDNA4 (gfx1201).
    rows = 1_041
    input_features = 272 if (exact_sm120_available() or sm8x_available()) else 256
    intermediate_features, output_features = 2_560, 256
    operands = make_int8_operands(
        "swiglu",
        rows=rows,
        input_features=input_features,
        intermediate_features=intermediate_features,
        output_features=output_features,
        bias_dtype=bias_dtype,
        dtype=dtype,
        group_size=16,
        down_group_size=256,
    )
    expected = materialized_int8(operands)
    actual = int8_ffn.convrot_int8_ffn_op(**operands.arguments(1_024))
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < 0.01

    base = torch.randn(rows, output_features, dtype=dtype, device="cuda")
    reusable = torch.randn_like(base)
    update_gate = torch.randn(7, output_features, dtype=dtype, device="cuda")
    ffn_gate = torch.randn(5, output_features, dtype=dtype, device="cuda")
    gate_indices = torch.randint(-5, 5, (rows,), device="cuda")
    expected_update = _materialized_gated_updates(
        expected, base, reusable, update_gate, ffn_gate, gate_indices
    )
    actual_update = reusable.clone()
    result = int8_ffn.convrot_int8_ffn_update_op(
        **operands.arguments(1_024),
        output=actual_update,
        base=base,
        update_gate=update_gate,
        ffn_gate=ffn_gate,
        gate_indices=gate_indices,
        python_indexing=True,
    )
    assert result is None
    assert relative_l2(actual_update, expected_update) < 0.01

    if dtype is torch.bfloat16 and bias_dtype is None:
        up_scale = torch.tensor(0.03, device="cuda")
        gate_scale = torch.tensor(0.05, device="cuda")
        distinct_expected = materialized_int8(
            operands, up_input_scale=up_scale, gate_input_scale=gate_scale
        )
        distinct_actual = int8_ffn.convrot_int8_ffn_op(
            **operands.arguments(1_024, up_input_scale=up_scale, gate_input_scale=gate_scale)
        )
        assert relative_l2(distinct_actual, distinct_expected) < 0.01

        wide_out_features = intermediate_features + 256
        wide_operands = make_int8_operands(
            "swiglu",
            rows=rows,
            input_features=input_features,
            intermediate_features=intermediate_features,
            output_features=wide_out_features,
            bias_dtype=None,
            dtype=dtype,
            group_size=16,
            down_group_size=256,
        )
        wide_base = torch.randn(rows, wide_out_features, dtype=dtype, device="cuda")
        wide_reusable = torch.randn_like(wide_base)
        wide_update_gate = torch.randn(7, wide_out_features, dtype=dtype, device="cuda")
        wide_ffn_gate = torch.randn(5, wide_out_features, dtype=dtype, device="cuda")
        wide_expected = _materialized_gated_updates(
            materialized_int8(wide_operands),
            wide_base,
            wide_reusable,
            wide_update_gate,
            wide_ffn_gate,
            gate_indices,
        )
        wide_actual = wide_reusable.clone()
        wide_result = int8_ffn.convrot_int8_ffn_update_op(
            **wide_operands.arguments(1_024),
            output=wide_actual,
            base=wide_base,
            update_gate=wide_update_gate,
            ffn_gate=wide_ffn_gate,
            gate_indices=gate_indices,
            python_indexing=True,
        )
        assert wide_result is None
        assert relative_l2(wide_actual, wide_expected) < 0.01


@pytest.mark.gpu
@_FUSED_SWIGLU_GPU
def test_convrot_int8_fused_swiglu_ffn_halves_projection_scratch() -> None:
    chunk_rows = 2_048
    rows = 2 * chunk_rows
    width = 512
    intermediate = 8_192
    operands = make_int8_operands(
        "swiglu",
        rows=rows,
        input_features=width,
        intermediate_features=intermediate,
        output_features=width,
        bias_dtype=None,
        dtype=torch.bfloat16,
    )
    kwargs = operands.arguments(chunk_rows)
    with torch.inference_mode():
        warm = int8_ffn.convrot_int8_ffn_op(**kwargs)
        torch.cuda.synchronize()
        del warm
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = int8_ffn.convrot_int8_ffn_op(**kwargs)
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - baseline

    output_bytes = actual.numel() * actual.element_size()
    # Fused SwiGLU allocates 1 * intermediate BF16 projections (32 MiB) instead of 2x (64 MiB).
    projection_bytes = chunk_rows * intermediate * actual.element_size()
    prepared_bytes = chunk_rows * intermediate
    scale_bytes = chunk_rows * torch.float32.itemsize
    assert peak_extra <= output_bytes + projection_bytes + prepared_bytes + scale_bytes + 1_048_576


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize(
    "mode", ["shared", "alias", "distinct", "mixed", "static-down", "dynamic-down"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_convrot_int8_static_scales_share_only_compatible_inputs(
    monkeypatch: pytest.MonkeyPatch, mode: str, dtype: torch.dtype
) -> None:
    torch.manual_seed(1071)
    operands = make_int8_operands("swiglu", rows=129, dtype=dtype)
    scale = torch.tensor(0.02, device="cuda")
    down = torch.tensor(0.25, device="cuda")
    scale_map: dict[str, tuple[torch.Tensor | None, ...]] = {
        "shared": (scale, scale, down),
        "alias": (scale, scale.view(()), down),
        "distinct": (scale, torch.tensor(0.04, device="cuda"), down),
        "mixed": (scale, None, down),
        "static-down": (None, None, down),
        "dynamic-down": (scale, scale, None),
    }
    up_s, gate_s, down_s = scale_map[mode]
    expected = materialized_int8(
        operands, up_input_scale=up_s, gate_input_scale=gate_s, down_input_scale=down_s
    )
    backend = int8_backend.require_linear_backend(operands.input)
    prepare = Mock(wraps=backend.prepare_input)
    project = Mock(wraps=backend.linear_prepared)
    monkeypatch.setattr(backend, "prepare_input", prepare)
    monkeypatch.setattr(backend, "linear_prepared", project)

    actual = int8_ffn.convrot_int8_ffn_op(
        **operands.arguments(
            64, up_input_scale=up_s, gate_input_scale=gate_s, down_input_scale=down_s
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    shared = mode not in ("distinct", "mixed")
    assert prepare.call_count == (6 if shared else 9)
    assert project.call_count == prepare.call_count
    assert sum(
        call.kwargs.get("second_projection") is not None for call in project.call_args_list
    ) == (3 if shared else 0)


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize(
    ("up_group_size", "down_group_size"),
    [(None, None), (16, 64), (16, None), (None, 64)],
    ids=["standard", "convrot", "convrot-standard", "standard-convrot"],
)
@pytest.mark.parametrize("rows", [127, 385], ids=["short", "ragged-multi-chunk"])
@pytest.mark.parametrize("up_dynamic", [False, True], ids=["up-static", "up-dynamic"])
@pytest.mark.parametrize("down_dynamic", [False, True], ids=["down-static", "down-dynamic"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_chunked_ffn_matches_materialized(
    activation: Activation,
    up_group_size: int | None,
    down_group_size: int | None,
    rows: int,
    up_dynamic: bool,
    down_dynamic: bool,
    dtype: torch.dtype,
) -> None:
    operands = make_nvfp4_operands(
        activation,
        rows=rows,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        up_group_size=up_group_size,
        down_group_size=down_group_size,
        dtype=dtype,
        bias_dtype=torch.float32 if up_dynamic else torch.bfloat16,
        seed=963 + rows + up_dynamic + 10 * down_dynamic,
    )
    expected = materialized_nvfp4(operands)
    actual = nvfp4_ffn.nvfp4_ffn_op(**operands.arguments(128))
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < (0.1 if down_dynamic else 0.07)


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
@pytest.mark.parametrize("up_high_first", [False, True])
@pytest.mark.parametrize("down_high_first", [False, True])
def test_nvfp4_nibble_order_distinct_scales_and_unit_scale(
    activation: Activation,
    group_size: int | None,
    up_high_first: bool,
    down_high_first: bool,
) -> None:
    operands = make_nvfp4_operands(
        activation,
        rows=129,
        up_dynamic=False,
        down_dynamic=False,
        up_group_size=group_size,
        down_group_size=group_size,
        up_high_first=up_high_first,
        down_high_first=down_high_first,
        distinct_input_scales=activation == "swiglu",
        seed=967,
    )
    actual = nvfp4_ffn.nvfp4_ffn_op(**operands.arguments(128))
    assert relative_l2(actual, materialized_nvfp4(operands)) < 0.06

    # Implicit unit scale (None) matches explicit scalar ones bitwise.
    if not up_high_first and not down_high_first:

        def _with_scale(scale: torch.Tensor | None):
            return replace(
                operands,
                up=replace(operands.up, activation_scale=scale),
                down=replace(operands.down, activation_scale=scale),
                gate=None
                if operands.gate is None
                else replace(operands.gate, activation_scale=scale),
            )

        explicit = nvfp4_ffn.nvfp4_ffn_op(
            **_with_scale(torch.ones((), device="cuda")).arguments(128)
        )
        implicit = nvfp4_ffn.nvfp4_ffn_op(**_with_scale(None).arguments(128))
        torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
def test_nvfp4_all_zero_guard(group_size: int | None) -> None:
    operands = make_nvfp4_operands(
        "gelu_tanh",
        rows=129,
        up_dynamic=True,
        down_dynamic=True,
        up_group_size=group_size,
        down_group_size=group_size,
        bias_dtype=None,
        seed=969,
    )
    operands.input.zero_()
    actual = nvfp4_ffn.nvfp4_ffn_op(**operands.arguments(128))
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("with_weight_global_scale", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_source_affine_precision_in_strided_workspace(
    bias_dtype: torch.dtype | None,
    with_weight_global_scale: bool,
    dtype: torch.dtype,
) -> None:
    operands = make_nvfp4_operands(
        "swiglu",
        rows=127,
        up_dynamic=False,
        down_dynamic=False,
        dtype=dtype,
        bias_dtype=bias_dtype,
        seed=987,
    )
    linear = operands.up
    if not with_weight_global_scale:
        linear.weight.per_tensor_scale = None
    qdata, scale, global_scale = nvfp4_ops._prepare_compiled(
        operands.input, linear.activation_scale, linear.dynamic
    )
    width = linear.weight.shape[0]
    workspace = torch.full((127, 2 * width), float("nan"), device="cuda", dtype=dtype)
    actual = workspace[:, width:]
    nvfp4_ffn._project_affine_chunk(
        qdata,
        scale,
        global_scale,
        nvfp4_ffn.LinearOperands(
            linear.weight.qdata,
            linear.weight.scale,
            linear.weight.per_tensor_scale,
            linear.activation_scale,
            linear.bias,
            linear.dynamic,
            None,
            linear.weight.high_first,
        ),
        127,
        actual,
    )
    reference = nvfp4_reference.linear_prepared(
        qdata,
        scale,
        global_scale,
        linear.weight.qdata,
        linear.weight.scale,
        linear.weight.per_tensor_scale,
        linear.bias,
        dtype,
    )
    assert relative_l2(actual, reference) < 0.0001
    assert workspace[:, :width].isnan().all()


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
def test_nvfp4_dynamic_source_preparation_uses_one_global_scale_and_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch, group_size: int | None
) -> None:
    operands = make_nvfp4_operands(
        "swiglu",
        rows=385,
        up_dynamic=True,
        down_dynamic=True,
        up_group_size=group_size,
        down_group_size=group_size,
        seed=906,
    )
    scale_rows: list[int] = []
    prepared_rows: list[int] = []

    if group_size is None:
        orig_scale = nvfp4_primitives.dynamic_scale
        orig_prepare = nvfp4_backend.prepare_static_out

        def track_scale(x: torch.Tensor) -> torch.Tensor:
            scale_rows.append(x.shape[0])
            return orig_scale(x)

        def track_prepare(x: torch.Tensor, s: torch.Tensor, out, high_first: bool = False):
            prepared_rows.append(x.shape[0])
            return orig_prepare(x, s, out, high_first=high_first)

        monkeypatch.setattr(nvfp4_primitives, "dynamic_scale", track_scale)
        monkeypatch.setattr(nvfp4_backend, "prepare_static_out", track_prepare)
    else:
        orig_scale = convrot_nvfp4_backend.dynamic_scale
        orig_prepare = convrot_nvfp4_backend.prepare_static_out

        def track_scale(x: torch.Tensor, gs: int) -> torch.Tensor:
            scale_rows.append(x.shape[0])
            return orig_scale(x, gs)

        def track_prepare(x: torch.Tensor, s: torch.Tensor, gs: int, out, high_first: bool = False):
            prepared_rows.append(x.shape[0])
            return orig_prepare(x, s, gs, out, high_first=high_first)

        monkeypatch.setattr(convrot_nvfp4_backend, "dynamic_scale", track_scale)
        monkeypatch.setattr(convrot_nvfp4_backend, "prepare_static_out", track_prepare)

    nvfp4_ffn.nvfp4_ffn_op(**operands.arguments(128))
    assert scale_rows == [385]
    assert prepared_rows == [128, 128, 128, 1]


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("fmt", ["convrot_int8", "nvfp4", "convrot_nvfp4"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("python_indexing", [False, True])
@pytest.mark.parametrize("output_features", [384, 640], ids=["reused-workspace", "separate"])
def test_ffn_update_gated_updates_and_residual_add(
    fmt: str,
    activation: Activation,
    python_indexing: bool,
    output_features: int,
) -> None:
    if fmt != "convrot_int8" and not exact_sm120_available():
        pytest.skip("requires exact NVIDIA SM120")
    torch.manual_seed(505)
    if fmt == "convrot_int8":
        operands = make_int8_operands(
            activation, rows=129, output_features=output_features, bias_dtype=torch.float32
        )
        ffn_kwargs = operands.arguments(64)
        ref_ffn = materialized_int8(operands)
        update_op = int8_ffn.convrot_int8_ffn_update_op
        tol = 0.01
    else:
        group = 16 if fmt == "convrot_nvfp4" else None
        operands = make_nvfp4_operands(
            activation,
            rows=129,
            output_features=output_features,
            up_dynamic=True,
            down_dynamic=False,
            up_group_size=group,
            down_group_size=group,
            bias_dtype=torch.float32,
            seed=505,
        )
        ffn_kwargs = operands.arguments(128)
        ref_ffn = materialized_nvfp4(operands)
        update_op = nvfp4_ffn.nvfp4_ffn_update_op
        tol = 0.07

    rows = operands.input.shape[0]
    base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    reusable = torch.randn_like(base)
    update_gate = torch.randn(7, output_features, dtype=torch.bfloat16, device="cuda")
    ffn_gate = torch.randn(5, output_features, dtype=torch.bfloat16, device="cuda")
    gate_indices = torch.randint(-5 if python_indexing else 0, 5, (rows,), device="cuda")
    expected = _materialized_gated_updates(
        ref_ffn, base, reusable, update_gate, ffn_gate, gate_indices
    )
    actual = reusable.clone()
    result = update_op(
        **ffn_kwargs,
        output=actual,
        base=base,
        update_gate=update_gate,
        ffn_gate=ffn_gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    assert result is None
    assert relative_l2(actual, expected) < tol

    assert_residual_add_matches_gated_updates(
        update_op, ffn_kwargs, output_features, python_indexing=python_indexing
    )


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("fmt", ["convrot_int8", "nvfp4", "convrot_nvfp4"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
def test_all_four_custom_ops_pass_opcheck(fmt: str, activation: Activation) -> None:
    if fmt != "convrot_int8" and not exact_sm120_available():
        pytest.skip("requires exact NVIDIA SM120")
    torch.manual_seed(601)
    if fmt == "convrot_int8":
        operands = make_int8_operands(activation, rows=65, output_features=128)
        kwargs = operands.arguments(64)
        plain_op = int8_ffn.convrot_int8_ffn_op
        update_op = int8_ffn.convrot_int8_ffn_update_op
    else:
        group = 16 if fmt == "convrot_nvfp4" else None
        operands = make_nvfp4_operands(
            activation,
            rows=65,
            output_features=128,
            up_dynamic=False,
            down_dynamic=False,
            up_group_size=group,
            down_group_size=group,
            seed=601,
        )
        kwargs = operands.arguments(128)
        plain_op = nvfp4_ffn.nvfp4_ffn_op
        update_op = nvfp4_ffn.nvfp4_ffn_update_op

    base = torch.randn(65, 128, dtype=torch.bfloat16, device="cuda")
    ffn_gate = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    update_gate = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    gate_indices = torch.randint(0, 4, (65,), device="cuda")

    opcheck_utils: tuple[str, ...] = (
        (
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        )
        if fmt == "convrot_int8"
        # SchemaCheckMode calls allclose on float8_e4m3fn scales, and aot_dispatch_dynamic
        # assigns independent SymInts to input_features vs packed weight_qdata columns.
        else ("test_autograd_registration", "test_faketensor", "test_aot_dispatch_static")
    )

    with torch.inference_mode():
        assert set(
            torch.library.opcheck(plain_op, (), kwargs, test_utils=opcheck_utils).values()
        ) == {"SUCCESS"}
        assert set(
            torch.library.opcheck(
                update_op,
                (),
                {
                    **kwargs,
                    "output": base.clone(),
                    "ffn_gate": ffn_gate,
                    "gate_indices": gate_indices,
                    "python_indexing": False,
                },
                test_utils=opcheck_utils,
            ).values()
        ) == {"SUCCESS"}
        assert set(
            torch.library.opcheck(
                update_op,
                (),
                {
                    **kwargs,
                    "output": torch.randn_like(base),
                    "base": base,
                    "update_gate": update_gate,
                    "ffn_gate": ffn_gate,
                    "gate_indices": gate_indices,
                    "python_indexing": True,
                },
                test_utils=opcheck_utils,
            ).values()
        ) == {"SUCCESS"}


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("fmt", ["convrot_int8", "nvfp4", "convrot_nvfp4"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
def test_direct_op_runs_under_dynamic_fullgraph_compile(
    fmt: str,
    activation: Activation,
) -> None:
    if fmt != "convrot_int8" and not exact_sm120_available():
        pytest.skip("requires exact NVIDIA SM120")
    torch.manual_seed(507)
    if fmt == "convrot_int8":
        operands = make_int8_operands(activation, rows=257, bias_dtype=None)
        static_kwargs = {k: v for k, v in operands.arguments(128).items() if k != "input"}

        @torch.compile(fullgraph=True, dynamic=True)
        def run(source: torch.Tensor) -> torch.Tensor:
            return int8_ffn.convrot_int8_ffn_op(input=source, **static_kwargs)

        for rows in (257, 385):
            source = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
            current = replace(operands, input=source)
            assert relative_l2(run(source), materialized_int8(current)) < 0.01
        return

    group = 16 if fmt == "convrot_nvfp4" else None
    nv_operands = make_nvfp4_operands(
        activation,
        rows=257,
        up_dynamic=True,
        down_dynamic=True,
        up_group_size=group,
        down_group_size=group,
        bias_dtype=None,
        seed=507,
    )
    nv_static_kwargs = {k: v for k, v in nv_operands.arguments(128).items() if k != "input"}

    @torch.compile(fullgraph=True)
    def run_nv(source: torch.Tensor) -> torch.Tensor:
        return nvfp4_ffn.nvfp4_ffn_op(input=source, **nv_static_kwargs)

    for rows in (257, 385):
        source = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
        torch._dynamo.mark_dynamic(source, 0)
        current_nv = replace(nv_operands, input=source)
        assert relative_l2(run_nv(source), materialized_nvfp4(current_nv)) < 0.12


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("input_features", "intermediate_features", "output_features"),
    [(160, 224, 192), (224, 160, 192)],
    ids=["wider-intermediate", "wider-input"],
)
def test_nvfp4_multi_chunk_padded_scale_reuse_matches_materialized(
    group_size: int | None,
    activation: Activation,
    dynamic: bool,
    input_features: int,
    intermediate_features: int,
    output_features: int,
) -> None:
    operands = make_nvfp4_operands(
        activation,
        rows=2 * 128 + 17,
        input_features=input_features,
        intermediate_features=intermediate_features,
        output_features=output_features,
        up_dynamic=dynamic,
        down_dynamic=dynamic,
        up_group_size=group_size,
        down_group_size=group_size,
        bias_dtype=torch.bfloat16,
        seed=977 + input_features + output_features,
    )
    expected = materialized_nvfp4(operands)
    actual = nvfp4_ffn.nvfp4_ffn_op(**operands.arguments(128))
    assert actual.dtype is torch.bfloat16
    assert relative_l2(actual, expected) < (0.12 if dynamic else 0.08)


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_nvfp4_multi_chunk_ffn_reuses_activation_storage(
    group_size: int | None,
    activation: Activation,
    dynamic: bool,
) -> None:
    chunk_rows = 4_096
    rows = 2 * chunk_rows
    width = 512
    intermediate = 8_192
    operands = make_nvfp4_operands(
        activation,
        rows=rows,
        input_features=width,
        intermediate_features=intermediate,
        output_features=width,
        up_dynamic=dynamic,
        down_dynamic=dynamic,
        up_group_size=group_size,
        down_group_size=group_size,
        bias_dtype=None,
        seed=983,
    )
    kwargs = operands.arguments(chunk_rows)
    with torch.inference_mode():
        warm = nvfp4_ffn.nvfp4_ffn_op(**kwargs)
        torch.cuda.synchronize()
        del warm
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = nvfp4_ffn.nvfp4_ffn_op(**kwargs)
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - baseline

    output_bytes = actual.numel() * actual.element_size()
    projection_bytes = (
        chunk_rows * (2 if activation == "swiglu" else 1) * intermediate * actual.element_size()
    )
    shared_qdata_bytes = chunk_rows * (intermediate // 2)
    shared_scale_bytes = ((chunk_rows + 127) // 128 * 32) * ((intermediate + 63) // 64 * 16)
    overhead_margin = 512 * 1024 + torch.backends.cuda.cublaslt_workspace_size()
    assert (
        peak_extra
        <= output_bytes
        + projection_bytes
        + shared_qdata_bytes
        + shared_scale_bytes
        + overhead_margin
    )


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("rows", [1, 127, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_activation_dynamic_scale_matches_fp32_reference(
    activation: Activation, rows: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(961)
    width = 512 if activation == "gelu_tanh" else 1024
    projections = torch.randn(rows, width, device="cuda", dtype=dtype)
    if activation == "gelu_tanh":
        expected = per_tensor_amax_to_scale(
            F.gelu(projections.float(), approximate="tanh").abs().amax()
        )
    else:
        value, gate = projections.chunk(2, dim=-1)
        expected = per_tensor_amax_to_scale((value.float() * F.silu(gate.float())).abs().amax())
    torch.testing.assert_close(
        nvfp4_backend.dynamic_scale(projections, activation_fn=activation),
        expected,
        rtol=1e-6,
        atol=0,
    )
    zero_scale = nvfp4_backend.dynamic_scale(
        torch.zeros_like(projections),
        activation_fn=activation,
    )
    assert torch.equal(zero_scale, torch.zeros((), device="cuda", dtype=torch.float32))
    if dtype is torch.float16 and activation == "swiglu" and rows == 127:
        overflow_safe = torch.full((rows, width), 512.0, device="cuda", dtype=dtype)
        expected_fp32 = per_tensor_amax_to_scale(
            (512.0 * F.silu(torch.tensor(512.0, device="cuda"))).abs()
        )
        torch.testing.assert_close(
            nvfp4_backend.dynamic_scale(overflow_safe, activation_fn="swiglu"),
            expected_fp32,
            rtol=1e-6,
            atol=0,
        )
