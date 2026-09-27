"""Implementation selection preserves ConvRot INT8 dispatch and fallback contracts."""

import sys
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import _backend, dispatch
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._generic import mean as generic_mean
from piper_kernels.linear.convrot.int8._nvidia import dispatch as nvidia
from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia_kernels
from piper_kernels.weights.convrot.int8 import _backend as int8_updates
from piper_kernels.weights.convrot.int8 import _gguf as int8_gguf
from piper_kernels.weights.convrot.int8 import _update
from piper_kernels.weights.convrot.int8 import triton as int8_weight_triton


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm70"), False),
        (AcceleratorTarget("cuda", "sm75"), True),
        (AcceleratorTarget("cuda", "sm80"), True),
        (AcceleratorTarget("cuda", "sm89"), True),
        (AcceleratorTarget("cuda", "sm90"), True),
        (AcceleratorTarget("cuda", "sm100"), True),
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cuda", "sm121"), True),
        (AcceleratorTarget("cuda"), False),
        (AcceleratorTarget("hip", "gfx1201"), False),
        (AcceleratorTarget("hip", "gfx942"), False),
        (AcceleratorTarget("cpu"), False),
        (AcceleratorTarget("meta"), False),
    ],
)
def test_select_backend_preserves_supported_targets(monkeypatch, target, supported):
    implementation = ModuleType("test_convrot_implementation")
    monkeypatch.setattr(_backend, "_nvidia_backend", implementation)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)

    selected = _backend.select_linear_backend(torch.empty(1))

    assert selected is (implementation if supported else None)


def test_missing_triton_uses_reference_without_querying_hardware(monkeypatch):
    resolve_target = Mock(side_effect=AssertionError("unexpected hardware query"))
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", resolve_target)

    assert _backend.select_linear_backend(torch.empty(1)) is None
    resolve_target.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "target"),
    [
        (nvidia, AcceleratorTarget("cuda", "sm120")),
        (nvidia, AcceleratorTarget("cuda", "sm89")),
        pytest.param(
            amd,
            AcceleratorTarget("hip", "gfx1201"),
            marks=pytest.mark.skipif(
                sys.platform not in ("linux", "win32"), reason="requires Linux or Windows ROCm"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (0, 1024, 1024),
        (1024, 0, 1024),
        (1025, 1024, 1024),
        (100000, 1024, 1024),
        (1024, 1024, 1024),
        (512, 257, 256),
        (513, 257, 256),
        (513, 256, 272),
        (513, 257, 64),
    ],
)
def test_matmul_uses_one_launch_and_only_metadata(monkeypatch, backend, target, m, n, k):
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    kernel_backend = nvidia_kernels if backend is nvidia else backend
    monkeypatch.setattr(kernel_backend, "device_context", lambda device: nullcontext())
    kernel = MagicMock()
    gluon_kernel = MagicMock()
    monkeypatch.setattr(kernel_backend, "int8_matmul_kernel", kernel)
    if backend is nvidia:
        monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", kernel)
        monkeypatch.setattr(nvidia.gluon_async_copy, "_int8_matmul_kernel", gluon_kernel)
    value = torch.empty(m, k, device="meta", dtype=torch.int8)
    weight = torch.empty(n, k, device="meta", dtype=torch.int8)
    row_scale = torch.empty(m, device="meta")
    scale = torch.empty(n, 1, device="meta")
    plan = backend.default_execution_plan(weight, target=target)
    result = backend.execute_prepared_linear(
        value, row_scale, weight, scale, None, torch.bfloat16, plan
    )
    assert result.shape == (m, n)
    uses_gluon = getattr(plan, "matmul_kernel", "triton") == "gluon_async_copy"
    launched, unused = (gluon_kernel, kernel) if uses_gluon else (kernel, gluon_kernel)
    unused.__getitem__.assert_not_called()
    if not m or not n:
        launched.__getitem__.assert_not_called()
    else:
        row_tiles = (m + plan.matmul_block_m - 1) // plan.matmul_block_m
        column_tiles = (n + plan.matmul_block_n - 1) // plan.matmul_block_n
        assert launched.__getitem__.call_args_list == [call((row_tiles * column_tiles,))]
        launched.__getitem__.return_value.assert_called_once()
        if not uses_gluon:
            flags = kernel.__getitem__.return_value.call_args.kwargs
            # SM8x branches per tile instead of specializing on aligned M.
            sm8x = target.is_cuda_capability(8)
            assert flags["aligned_m"] == (not sm8x and m % plan.matmul_block_m == 0)
            assert flags["aligned_nk"] == (n % plan.matmul_block_n == k % plan.matmul_block_k == 0)


@pytest.mark.parametrize(
    ("architecture", "rows", "block_m"),
    [("sm120", None, 128), ("sm120", 1280, 64), ("sm89", None, 256), ("sm86", 256, 64)],
)
def test_nvidia_planner_needs_no_device_properties_with_explicit_target(
    monkeypatch, architecture, rows, block_m
):
    weight = SimpleNamespace(shape=(1024, 1024), device=torch.device("cuda"))
    properties = Mock(side_effect=AssertionError("planner queried device properties"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    plan = nvidia.default_execution_plan(
        weight, target=AcceleratorTarget("cuda", architecture), rows=rows
    )
    assert plan.matmul_block_m == block_m
    properties.assert_not_called()


@pytest.mark.parametrize("specialize_m", [False, True])
@pytest.mark.parametrize("explicit_bias_fma", [False, True])
def test_triton_tail_scheduling_is_independent_of_bias_rounding(
    monkeypatch, specialize_m, explicit_bias_fma
):
    monkeypatch.setattr(nvidia_kernels, "device_context", lambda device: nullcontext())
    kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "int8_matmul_kernel", kernel)
    monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", kernel)
    value = torch.empty(33, 256, device="meta", dtype=torch.int8)
    weight = torch.empty(64, 256, device="meta", dtype=torch.int8)
    plan = replace(
        nvidia.default_execution_plan(weight, target=AcceleratorTarget("cuda", "sm120"), rows=33),
        triton_specialize_m=specialize_m,
        triton_explicit_bias_fma=explicit_bias_fma,
    )
    nvidia.execute_prepared_linear(
        value,
        torch.empty(33, device="meta"),
        weight,
        torch.empty(64, 1, device="meta"),
        None,
        torch.bfloat16,
        plan,
    )
    flags = kernel.__getitem__.return_value.call_args.kwargs
    assert flags["explicit_bias_fma"] is explicit_bias_fma
    assert flags["per_tile_tail"] is not specialize_m


@pytest.mark.parametrize(
    ("architecture", "rows", "n", "tile", "group_m"),
    [
        # Production and explicit schedules both carry grouping in their plan.
        ("sm120", 4096, 1000, None, 16),
        ("sm120", 4096, 1000, (128, 128, 128, 4, 2), 0),
        ("sm120", 64, 1000, None, 0),
        ("sm89", 8192, 96, None, 16),
        ("sm89", 256, 1000, None, 0),
        ("sm89", 4, 1000, None, 0),
        ("sm86", 8192, 96, (128, 256, 128, 8, 3), 16),
    ],
)
@pytest.mark.parametrize("paired", [False, True])
def test_nvidia_launch_grouping_follows_the_plan(
    monkeypatch, architecture, rows, n, tile, group_m, paired
):
    target = AcceleratorTarget("cuda", architecture)
    monkeypatch.setattr(nvidia_kernels, "device_context", lambda device: nullcontext())
    kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "int8_matmul_kernel", kernel)
    monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", kernel)
    k = 1024
    value = torch.empty(rows, k, device="meta", dtype=torch.int8)
    weight = torch.empty(n, k, device="meta", dtype=torch.int8)
    plan = nvidia.default_execution_plan(weight, target=target, rows=rows)
    if tile is not None:
        block_m, block_n, block_k, num_warps, num_stages = tile
        plan = replace(
            plan,
            matmul_block_m=block_m,
            matmul_block_n=block_n,
            matmul_block_k=block_k,
            matmul_num_warps=num_warps,
            matmul_num_stages=num_stages,
            matmul_group_m=group_m,
            matmul_kernel="triton",
        )
    second = (weight, torch.empty(n, 1, device="meta"), None) if paired else None
    nvidia.execute_prepared_linear(
        value,
        torch.empty(rows, device="meta"),
        weight,
        torch.empty(n, 1, device="meta"),
        None,
        torch.bfloat16,
        plan,
        second_projection=second,
    )
    row_tiles = (rows + plan.matmul_block_m - 1) // plan.matmul_block_m
    column_tiles = (n + plan.matmul_block_n - 1) // plan.matmul_block_n * (2 if paired else 1)
    grid = (row_tiles * column_tiles,) if group_m else (row_tiles, column_tiles)
    assert kernel.__getitem__.call_args_list == [call(grid)]
    assert kernel.__getitem__.return_value.call_args.kwargs["group_m"] == group_m


@pytest.mark.parametrize(
    ("rows", "n", "block_m", "num_warps"), [(4096, 1000, 128, 4), (4096, 2000, 256, 8)]
)
@pytest.mark.parametrize("paired", [False, True])
def test_sm8x_wide_projections_launch_the_gluon_kernel(
    monkeypatch, rows, n, block_m, num_warps, paired
):
    target = AcceleratorTarget("cuda", "sm89")
    monkeypatch.setattr(nvidia_kernels, "device_context", lambda device: nullcontext())
    kernel = MagicMock()
    gluon_kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "int8_matmul_kernel", kernel)
    monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", kernel)
    monkeypatch.setattr(nvidia.gluon_async_copy, "_int8_matmul_kernel", gluon_kernel)
    k = 1024
    value = torch.empty(rows, k, device="meta", dtype=torch.int8)
    weight = torch.empty(n, k, device="meta", dtype=torch.int8)
    plan = nvidia.default_execution_plan(weight, target=target, rows=rows)
    second = (weight, torch.empty(n, 1, device="meta"), None) if paired else None
    nvidia.execute_prepared_linear(
        value,
        torch.empty(rows, device="meta"),
        weight,
        torch.empty(n, 1, device="meta"),
        None,
        torch.bfloat16,
        plan,
        second_projection=second,
    )
    tiles = (rows + block_m - 1) // block_m * ((n + 127) // 128) * (2 if paired else 1)
    assert plan.matmul_kernel == "gluon_async_copy"
    kernel.__getitem__.assert_not_called()
    assert gluon_kernel.__getitem__.call_args_list == [call((tiles,))]
    assert gluon_kernel.__getitem__.return_value.call_args.kwargs["num_warps"] == num_warps


@pytest.mark.parametrize("architecture", ["sm70", "sm75", "sm120"])
def test_auxiliary_operations_keep_their_own_support_rules(monkeypatch, architecture):
    target = AcceleratorTarget("cuda", architecture)
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    monkeypatch.setattr(runtime, "supports_device", lambda device: True)
    value = SimpleNamespace(device=torch.device("cuda"))

    assert int8_gguf.select_gguf_converter(value) is int8_weight_triton.convert_gguf_out
    assert _backend.select_dequantized_mean(value) is generic_mean.dequantized_input_mean
    assert int8_updates.select_add(value) is not None
    assert int8_updates.select_addmm(value) is not None


@pytest.mark.parametrize(
    ("backend", "target"),
    [
        (nvidia, AcceleratorTarget("cuda", "sm120")),
        pytest.param(
            amd,
            AcceleratorTarget("hip", "gfx1201"),
            marks=pytest.mark.skipif(
                sys.platform not in ("linux", "win32"), reason="requires Linux or Windows ROCm"
            ),
        ),
    ],
)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize(("rows", "columns"), [(2, 7), (64, 25600)])
def test_backend_owns_plans_and_forwards_preparation_and_projection_buffers(
    monkeypatch, backend, target, static, rows, columns
):
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    value = torch.empty(rows, 64)
    prepared = (torch.empty(rows, 32, dtype=torch.int8), torch.empty(rows))
    weight = torch.empty(columns, 32, dtype=torch.int8)
    scale = torch.empty(columns, 1)
    second = (torch.empty_like(weight), torch.empty_like(scale), torch.empty(columns))
    output = torch.empty(rows, 2 * columns + 4)[:, 2:-2]
    prepare = Mock(return_value=prepared)
    project = Mock(return_value=output)
    monkeypatch.setattr(backend, "prepare_input_with_plan", prepare)
    monkeypatch.setattr(backend, "execute_prepared_linear", project)
    implementation = _backend.require_linear_backend(value)
    assert implementation is backend

    input_scale = torch.tensor(0.02) if static else None
    assert implementation.prepare_input(value, 16, "swiglu", input_scale, out=prepared) is prepared
    assert prepare.call_args.kwargs["input_scale"] is input_scale
    assert prepare.call_args.args == (value, 32, 16)
    assert prepare.call_args.kwargs["out"] is prepared
    assert prepare.call_args.kwargs["activation_fn"] == "swiglu"
    assert prepare.call_args.kwargs["target"] == target
    expected_plan = backend.default_execution_plan(weight, target=target)
    assert prepare.call_args.kwargs["execution_plan"] == expected_plan

    result = implementation.linear_prepared(
        *prepared, weight, scale, None, torch.float32, out=output, second_projection=second
    )
    assert result is output
    if backend is nvidia:
        expected_plan = backend.default_execution_plan(weight, target=target, rows=rows)
    assert project.call_args.args[-1] == expected_plan
    assert project.call_args.kwargs == {"out": output, "second_projection": second}


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm70"), True),
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cpu"), False),
        (AcceleratorTarget("meta"), False),
        (AcceleratorTarget("hip", "gfx1201"), False),
    ],
)
def test_nvidia_fused_launcher_validates_target_before_launch(monkeypatch, target, supported):
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "rotate_quantize_rows_kernel", kernel)
    value = torch.empty(2, 512)
    qdata, scale = torch.full((2, 512), 99, dtype=torch.int8), torch.full((2,), -99.0)
    if supported:
        # SM70 preparation still works without INT8 matrix instructions.
        nvidia_kernels.fused_rotate_quantize_input(value, qdata, scale, 16, num_warps=4)
        kernel.__getitem__.assert_called_once_with((2,))
        launch = kernel.__getitem__.return_value
        launch.assert_called_once()
        assert launch.call_args.kwargs["chunk_count"] == 1
        assert launch.call_args.kwargs["chunk_size"] == 512
        assert launch.call_args.kwargs["accelerator_backend"] == "cuda"
    else:
        with pytest.raises(ValueError, match="preparation has no optimized policy"):
            nvidia_kernels.fused_rotate_quantize_input(value, qdata, scale, 16, num_warps=4)
        kernel.__getitem__.assert_not_called()
        assert (qdata == 99).all()
        assert (scale == -99.0).all()


@pytest.mark.parametrize("operation", ["linear", "add_", "addmm_"])
def test_validated_operations_call_the_selected_implementation(monkeypatch, operation):
    qdata = torch.zeros(7, 32, dtype=torch.int8)
    scale = torch.ones(7, 1)
    activation = torch.zeros(3, 64)
    bias = torch.ones(7)
    update = torch.zeros(7, 32)
    mat1, mat2 = torch.zeros(7, 3), torch.zeros(3, 32)
    expected_output = torch.ones(3, 7)
    execute = Mock(return_value=expected_output)
    implementation = ModuleType("test_convrot_implementation")
    monkeypatch.setattr(implementation, operation, execute, raising=False)
    monkeypatch.setattr(_backend, "_nvidia_backend", implementation)
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("cuda", "sm120")
    )

    if operation == "linear":
        actual = dispatch.linear(
            activation, qdata, scale, torch.float32, 16, bias, activation_fn="swiglu"
        )
        assert actual is expected_output
        execute.assert_called_once_with(activation, qdata, scale, bias, 16, "swiglu", None)
    elif operation == "add_":
        monkeypatch.setattr(int8_updates, "select_add", lambda value: execute)
        _update.add_(qdata, scale, torch.float32, 16, update, alpha=2, rounding_seed=2**64 - 1)
        execute.assert_called_once_with(qdata, scale, update, 16, 2.0, -1)
    else:
        monkeypatch.setattr(int8_updates, "select_addmm", lambda value: execute)
        _update.addmm_(
            qdata, scale, torch.float32, 16, mat1, mat2, beta=3, alpha=2, rounding_seed=2**64 - 1
        )
        execute.assert_called_once_with(qdata, scale, mat1, mat2, 16, 3.0, 2.0, -1)
