"""Implementation selection preserves ConvRot INT8 dispatch and fallback contracts."""

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch
from _accelerators import forbid_device_probes, mock_cuda_target, mock_triton_driver
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels._residual import ResidualAdd, ReusableUpdate
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import _backend, _functional
from piper_kernels.linear.convrot.int8._generic import mean as generic_mean
from piper_kernels.weights.convrot.int8 import _gguf as int8_gguf
from piper_kernels.weights.convrot.int8 import _update
from piper_kernels.weights.convrot.int8 import triton as int8_weight_triton

nvidia = _backend._nvidia_backend
amd = _backend._amd_backend
nvidia_kernels = nvidia.kernels


@pytest.mark.parametrize(
    ("target", "platform", "expected"),
    [
        (AcceleratorTarget("cuda", "sm70"), "linux", None),
        (AcceleratorTarget("cuda", "sm75"), "linux", nvidia),
        (AcceleratorTarget("cuda", "sm80"), "linux", nvidia),
        (AcceleratorTarget("cuda", "sm89"), "linux", nvidia),
        (AcceleratorTarget("cuda", "sm90"), "linux", nvidia),
        (AcceleratorTarget("cuda", "sm100"), "linux", nvidia),
        (AcceleratorTarget("cuda", "sm120"), "linux", nvidia),
        (AcceleratorTarget("cuda", "sm121"), "linux", nvidia),
        (AcceleratorTarget("cuda"), "linux", None),
        (AcceleratorTarget("hip", "gfx1201"), "linux", amd),
        (AcceleratorTarget("hip", "gfx942"), "win32", amd),
        (AcceleratorTarget("hip", "gfx1201"), "darwin", None),
        (AcceleratorTarget("hip", "gfx90a"), "linux", None),
        (AcceleratorTarget("cpu"), "linux", None),
        (AcceleratorTarget("meta"), "linux", None),
    ],
)
def test_select_backend_preserves_supported_targets(monkeypatch, target, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    assert _backend.select_linear_backend_for_target(target) is expected
    assert _backend.select_linear_backend(torch.empty(1)) is None


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
    kernel_backend = backend.kernels
    kernel = MagicMock()
    gluon_kernel = MagicMock()
    tma_kernel = MagicMock()
    monkeypatch.setattr(kernel_backend, "int8_matmul_kernel", kernel)
    if backend is nvidia:
        monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", kernel)
        monkeypatch.setattr(nvidia_kernels.gluon_async_copy, "_int8_matmul_kernel", gluon_kernel)
        monkeypatch.setattr(nvidia_kernels.gluon_tma, "_int8_tma_matmul_kernel", tma_kernel)
    value = torch.empty(m, k, device="meta", dtype=torch.int8)
    weight = torch.empty(n, k, device="meta", dtype=torch.int8)
    row_scale = torch.empty(m, device="meta")
    scale = torch.empty(n, 1, device="meta")
    plan = backend.default_execution_plan(weight, target=target)
    result = backend.execute_prepared_linear(
        value, row_scale, weight, scale, None, torch.bfloat16, plan, target=target
    )
    assert result.shape == (m, n)
    plan_kernel = getattr(plan, "matmul_kernel", "triton")
    if plan_kernel == "gluon_tma":
        launched, unused = tma_kernel, (kernel, gluon_kernel)
    elif plan_kernel == "gluon_async_copy":
        launched, unused = gluon_kernel, (kernel, tma_kernel)
    else:
        launched, unused = kernel, (gluon_kernel, tma_kernel)
    for other in unused:
        other.__getitem__.assert_not_called()
    if not m or not n:
        launched.__getitem__.assert_not_called()
    else:
        row_tiles = (m + plan.matmul_block_m - 1) // plan.matmul_block_m
        column_tiles = (n + plan.matmul_block_n - 1) // plan.matmul_block_n
        assert launched.__getitem__.call_args_list == [call((row_tiles * column_tiles,))]
        launched.__getitem__.return_value.assert_called_once()
        if plan_kernel == "triton":
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
        ("sm90", 4096, 1000, None, 16),
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
    kernel = MagicMock()
    gluon_kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "int8_matmul_kernel", kernel)
    monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", kernel)
    monkeypatch.setattr(nvidia_kernels.gluon_async_copy, "_int8_matmul_kernel", gluon_kernel)
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


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("k", [1024, 1032])
def test_sm120_large_projections_launch_tma_or_fall_back_to_grouped_triton(monkeypatch, paired, k):
    target = AcceleratorTarget("cuda", "sm120")
    triton_kernel = MagicMock()
    tma_kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "int8_matmul_kernel", triton_kernel)
    monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_matmul_kernel", triton_kernel)
    monkeypatch.setattr(nvidia_kernels.gluon_tma, "_int8_tma_matmul_kernel", tma_kernel)
    rows, n = 4096, 1000
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
    assert plan.matmul_kernel == "gluon_tma"
    if k % 16 == 0:
        tiles = (rows + 127) // 128 * ((n + 255) // 256) * (2 if paired else 1)
        triton_kernel.__getitem__.assert_not_called()
        assert tma_kernel.__getitem__.call_args_list == [call((tiles,))]
        assert tma_kernel.__getitem__.return_value.call_args.kwargs["num_warps"] == 8
    else:
        tma_kernel.__getitem__.assert_not_called()
        tiles = (rows + 127) // 128 * ((n + 63) // 64) * (2 if paired else 1)
        assert triton_kernel.__getitem__.call_args_list == [call((tiles,))]
        assert triton_kernel.__getitem__.return_value.call_args.kwargs["group_m"] == 16


@pytest.mark.parametrize("k", [256, 264])
def test_sm120_large_residual_launches_tma_or_falls_back_when_unaligned(monkeypatch, k):
    target = AcceleratorTarget("cuda", "sm120")
    triton_residual = MagicMock()
    tma_residual = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "int8_residual_matmul_kernel", triton_residual)
    monkeypatch.setattr(nvidia_kernels, "dynamic_m_int8_residual_matmul_kernel", triton_residual)
    monkeypatch.setattr(nvidia_kernels.gluon_tma, "_int8_tma_residual_matmul_kernel", tma_residual)
    rows, n = 1024, 2560
    value = torch.empty(rows, k, device="meta", dtype=torch.int8)
    weight = torch.empty(n, k, device="meta", dtype=torch.int8)
    base = torch.empty(rows, n, device="meta", dtype=torch.bfloat16)
    gate = torch.empty(2, n, device="meta", dtype=torch.bfloat16)
    indices = torch.empty(rows, device="meta", dtype=torch.int64)
    residual = ResidualAdd(base, gate, indices, True, ReusableUpdate(torch.empty_like(base), gate))
    nvidia.linear_prepared_residual(
        value,
        torch.empty(rows, device="meta"),
        weight,
        torch.empty(n, 1, device="meta"),
        None,
        residual,
        target=target,
    )
    if k % 16 == 0:
        tiles = (rows + 127) // 128 * ((n + 255) // 256)
        triton_residual.__getitem__.assert_not_called()
        assert tma_residual.__getitem__.call_args_list == [call((tiles,))]
        assert tma_residual.__getitem__.return_value.call_args.kwargs["num_warps"] == 8
    else:
        tma_residual.__getitem__.assert_not_called()
        tiles = (rows + 127) // 128 * ((n + 63) // 64)
        assert triton_residual.__getitem__.call_args_list == [call((tiles,))]
        assert triton_residual.__getitem__.return_value.call_args.kwargs["group_m"] == 16


def test_auxiliary_operations_keep_their_own_support_rules(monkeypatch):
    forbid_device_probes(monkeypatch, "capability probe")
    mock_triton_driver(monkeypatch, backend="cuda", arch=120)
    value = SimpleNamespace(device=torch.device("cuda"), shape=(2, 256), numel=lambda: 512)

    assert int8_gguf.select_gguf_converter(value) is int8_weight_triton.convert_gguf_out
    assert _backend.select_dequantized_mean(value) is generic_mean.dequantized_input_mean
    assert _update._use_triton(value)


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
    backend, target, static, rows, columns
):
    value = torch.empty(rows, 64)
    prepared = (torch.empty(rows, 32, dtype=torch.int8), torch.empty(rows))
    weight = torch.empty(columns, 32, dtype=torch.int8)
    scale = torch.empty(columns, 1)
    second = (torch.empty_like(weight), torch.empty_like(scale), torch.empty(columns))
    output = torch.empty(rows, 2 * columns + 4)[:, 2:-2]
    fake_kernels = SimpleNamespace(
        fused_rotate_quantize_input=Mock(),
        launch_prepared_linear=Mock(),
    )
    implementation = _backend.LinearBackend(backend.policy, fake_kernels)  # type: ignore[arg-type]

    input_scale = torch.tensor(0.02) if static else None
    assert (
        implementation.prepare_input(value, 16, "swiglu", input_scale, out=prepared, target=target)
        is prepared
    )
    prep_call = fake_kernels.fused_rotate_quantize_input.call_args
    assert [t.data_ptr() for t in prep_call.args[:3]] == [
        value.data_ptr(),
        prepared[0].data_ptr(),
        prepared[1].data_ptr(),
    ]
    assert [t.shape for t in prep_call.args[:3]] == [
        value.shape,
        prepared[0].shape,
        prepared[1].shape,
    ]
    assert prep_call.args[3] == 16
    expected_prep_plan = backend.default_execution_plan(weight, target=target)
    assert prep_call.kwargs == {
        "activation_fn": "swiglu",
        "input_scale": input_scale,
        "num_warps": expected_prep_plan.fused_num_warps,
        "target": target,
    }

    result = implementation.linear_prepared(
        *prepared,
        weight,
        scale,
        None,
        torch.float32,
        out=output,
        second_projection=second,
        target=target,
    )
    assert result is output
    launch_call = fake_kernels.launch_prepared_linear.call_args
    assert [launch_call.args[i].data_ptr() for i in (0, 1, 5)] == [
        prepared[0].data_ptr(),
        prepared[1].data_ptr(),
        output.data_ptr(),
    ]
    assert all(
        a is b
        for a, b in zip(
            (launch_call.args[2], launch_call.args[3], launch_call.args[4], *launch_call.args[6:9]),
            (weight, scale, None, *second),
            strict=True,
        )
    )
    expected_gemm_plan = backend.default_execution_plan(weight, target=target, rows=rows)
    assert launch_call.kwargs == {
        "paired": True,
        "logical_dtype": torch.float32,
        "execution_plan": expected_gemm_plan,
        "target": target,
    }


def test_backend_prepared_swiglu_falls_back_when_kernels_decline() -> None:
    rows, features, width = 3, 16, 4
    qdata = torch.empty((rows, features), dtype=torch.int8)
    row_scale = torch.empty(rows)
    weight = torch.empty((width, features), dtype=torch.int8)
    second_weight = torch.empty_like(weight)
    scale = torch.empty(width, 1)
    paired_values = torch.tensor(
        [[1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, -0.5]] * rows,
        dtype=torch.bfloat16,
    )

    def fake_paired_launch(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        output,
        second_w,
        second_s,
        second_b,
        **kwargs,
    ) -> None:
        output.copy_(paired_values)

    fake_kernels = SimpleNamespace(
        supports_prepared_swiglu=Mock(return_value=False),
        try_launch_prepared_swiglu=Mock(return_value=False),
        launch_prepared_linear=Mock(side_effect=fake_paired_launch),
    )
    implementation = _backend.LinearBackend(nvidia.policy, fake_kernels)  # type: ignore[arg-type]
    target = AcceleratorTarget("cuda", "sm89")
    plan = implementation.default_execution_plan(weight, target=target, rows=rows)
    assert not implementation.supports_prepared_swiglu(
        weight, second_weight, torch.bfloat16, execution_plan=plan, target=target
    )
    up_half, gate_half = paired_values.chunk(2, dim=-1)
    expected = (up_half.float() * torch.nn.functional.silu(gate_half.float())).to(torch.bfloat16)

    actual = implementation.linear_prepared_swiglu(
        qdata,
        row_scale,
        weight,
        scale,
        None,
        (second_weight, scale, None),
        torch.bfloat16,
        execution_plan=plan,
        target=target,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out_buf = torch.empty((rows, width + 4), dtype=torch.bfloat16)[:, 2:-2]
    returned = implementation.linear_prepared_swiglu(
        qdata,
        row_scale,
        weight,
        scale,
        None,
        (second_weight, scale, None),
        torch.bfloat16,
        out=out_buf,
        execution_plan=plan,
        target=target,
    )
    assert returned is out_buf
    torch.testing.assert_close(out_buf, expected, rtol=0, atol=0)
    assert fake_kernels.try_launch_prepared_swiglu.call_count == 2
    assert fake_kernels.launch_prepared_linear.call_count == 2


def test_nvidia_supports_prepared_swiglu_covers_sm8x_and_sm120_alignment() -> None:
    sm89 = AcceleratorTarget("cuda", "sm89")
    sm120 = AcceleratorTarget("cuda", "sm120")
    aligned_k = 256
    medium_weight = torch.empty((1024, aligned_k), device="meta", dtype=torch.int8)
    large_weight = torch.empty((2048, aligned_k), device="meta", dtype=torch.int8)
    unaligned_weight = torch.empty((1024, 264), device="meta", dtype=torch.int8)

    assert nvidia.supports_prepared_swiglu(
        medium_weight, medium_weight, torch.bfloat16, rows=4096, target=sm89
    )
    assert nvidia.supports_prepared_swiglu(
        large_weight, large_weight, torch.float16, rows=4096, target=sm89
    )
    assert not nvidia.supports_prepared_swiglu(
        medium_weight, medium_weight, torch.bfloat16, rows=16, target=sm89
    )
    assert not nvidia.supports_prepared_swiglu(
        unaligned_weight, unaligned_weight, torch.bfloat16, rows=4096, target=sm89
    )
    assert not nvidia.supports_prepared_swiglu(
        large_weight, large_weight, torch.float32, rows=4096, target=sm89
    )
    strided_weight = torch.empty((1024, 512), device="meta", dtype=torch.int8)[:, :256]
    assert not nvidia.supports_prepared_swiglu(
        strided_weight, strided_weight, torch.bfloat16, rows=4096, target=sm89
    )
    async_plan = nvidia.default_execution_plan(medium_weight, target=sm89, rows=4096)
    assert nvidia.supports_prepared_swiglu(
        medium_weight,
        medium_weight,
        torch.bfloat16,
        execution_plan=async_plan,
        target=sm120,
    )


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
    kernel = MagicMock()
    monkeypatch.setattr(nvidia_kernels, "rotate_quantize_rows_kernel", kernel)
    value = torch.empty(2, 512)
    qdata, scale = torch.full((2, 512), 99, dtype=torch.int8), torch.full((2,), -99.0)
    if supported:
        # SM70 preparation still works without INT8 matrix instructions.
        nvidia_kernels.fused_rotate_quantize_input(
            value, qdata, scale, 16, num_warps=4, target=target
        )
        kernel.__getitem__.assert_called_once_with((2,))
        launch = kernel.__getitem__.return_value
        launch.assert_called_once()
        assert launch.call_args.kwargs["chunk_count"] == 1
        assert launch.call_args.kwargs["chunk_size"] == 512
        assert launch.call_args.kwargs["accelerator_backend"] == "cuda"
    else:
        with pytest.raises(ValueError, match="preparation has no optimized policy"):
            nvidia_kernels.fused_rotate_quantize_input(
                value, qdata, scale, 16, num_warps=4, target=target
            )
        kernel.__getitem__.assert_not_called()
        assert (qdata == 99).all()
        assert (scale == -99.0).all()


@pytest.mark.parametrize("operation", ["linear", "add_", "addmm_"])
def test_validated_operations_call_the_selected_implementation(monkeypatch, operation):
    execute = Mock()
    implementation = ModuleType("test_convrot_implementation")
    monkeypatch.setattr(implementation, operation, execute, raising=False)

    if operation == "linear":
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class RecordCustomOp(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                resolved_kwargs = dict(kwargs or {})
                if func is torch.ops.piper_kernels.convrot_int8_linear.default:
                    calls.append((args, resolved_kwargs))
                return func(*args, **resolved_kwargs)

        with FakeTensorMode():
            qdata = torch.zeros(7, 32, device="cuda:0", dtype=torch.int8)
            scale = torch.ones(7, 1, device="cuda:0")
            activation = torch.zeros(3, 64, device="cuda:0")
            bias = torch.ones(7, device="cuda:0")
            expected_output = torch.ones(3, 7, device="cuda:0")
            execute.return_value = expected_output
            monkeypatch.setattr(_backend, "_nvidia_backend", implementation)
            mock_cuda_target(monkeypatch, (12, 0))
            with RecordCustomOp():
                _functional.linear(
                    activation, qdata, scale, torch.float32, 16, bias, activation_fn="swiglu"
                )
        assert len(calls) == 1
        call_args, call_kwargs = calls[0]
        assert len(call_args) == 6
        assert all(
            actual is expected
            for actual, expected in zip(
                call_args[:4], (activation, qdata, scale, bias), strict=True
            )
        )
        assert call_args[4:] == (16, "swiglu")
        assert call_kwargs == {}
        actual = _functional._ops.linear._init_fn(*call_args, **call_kwargs)
        assert actual is expected_output
        execute.assert_called_once_with(activation, qdata, scale, bias, 16, "swiglu", None)
        return

    with FakeTensorMode():
        qdata = torch.zeros(7, 32, device="cuda:0", dtype=torch.int8)
        scale = torch.ones(7, 1, device="cuda:0")
        update = torch.zeros(7, 32, device="cuda:0")
        mat1, mat2 = torch.zeros(7, 3, device="cuda:0"), torch.zeros(3, 32, device="cuda:0")
        monkeypatch.setattr(_update, "_triton_backend", implementation)
        if operation == "add_":
            _update.add_(
                qdata, scale, torch.float32, 16, update, beta=0.25, alpha=2, rounding_seed=2**64 - 1
            )
            execute.assert_called_once_with(qdata, scale, update, 16, 2.0, -1, beta=0.25)
        else:
            _update.addmm_(
                qdata,
                scale,
                torch.float32,
                16,
                mat1,
                mat2,
                beta=3,
                alpha=2,
                rounding_seed=2**64 - 1,
            )
            execute.assert_called_once_with(qdata, scale, mat1, mat2, 16, 3.0, 2.0, -1)
