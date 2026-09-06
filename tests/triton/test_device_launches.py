"""Repository-wide launch ownership, with device-one execution simulated on CPU."""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import triton
from torch._subclasses.fake_tensor import FakeTensorMode

import piper_kernels
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp import _policy as sage_policy
from piper_kernels.attention.sage_attention_2pp import triton as sage
from piper_kernels.fusions.swiglu_ffn import triton as gated_updates
from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.convrot import triton as rotation
from piper_kernels.linear.convrot.int8 import _generic, _gguf
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._generic import triton as generic
from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4
from piper_kernels.linear.nvfp4 import triton as nvfp4


def test_every_native_launch_and_descriptor_has_an_explicit_device_context():
    root = Path(piper_kernels.__file__).parent
    unguarded = []
    native_calls = []

    class Audit(ast.NodeVisitor):
        depth = 0

        def visit_FunctionDef(self, node):
            if any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list):
                return
            self.generic_visit(node)

        def visit_With(self, node):
            guarded = any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func) in ("device_context", "torch.cuda.device")
                for item in node.items
            )
            self.depth += guarded
            self.generic_visit(node)
            self.depth -= guarded

        def visit_Call(self, node):
            if isinstance(node.func, ast.Subscript) or (
                isinstance(node.func, ast.Name)
                and node.func.id in ("TensorDescriptor", "install_uint8_int8_dot_hook")
            ):
                location = f"{path.relative_to(root)}:{node.lineno}"
                native_calls.append(location)
                if not self.depth:
                    unguarded.append(location)
            self.generic_visit(node)

    for path in root.rglob("*.py"):
        Audit().visit(ast.parse(path.read_text()))
    assert native_calls
    assert not unguarded, unguarded


@pytest.fixture
def launches(monkeypatch):
    state = SimpleNamespace(current=0, calls=[], fail=False, backend="cuda", streams={0: 17, 1: 23})

    @contextmanager
    def guard(device):
        previous = state.current
        state.current = previous if device.index is None else device.index
        try:
            yield
        finally:
            state.current = previous

    monkeypatch.setattr(torch.cuda, "device", guard)
    monkeypatch.setattr(
        AcceleratorTarget,
        "from_device",
        lambda device: AcceleratorTarget(
            state.backend, "gfx1201" if state.backend == "hip" else "sm120"
        ),
    )
    driver = SimpleNamespace(
        get_active_torch_device=lambda: torch.device("cuda", state.current),
        get_current_target=lambda: SimpleNamespace(backend=state.backend, arch=120),
    )
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=driver))

    def watch(module, name):
        def launch(*args, **kwargs):
            assert state.current == 1
            assert state.streams[state.current] == 23
            state.calls.append(name)
            if state.fail:
                raise RuntimeError("simulated kernel failure")

        kernel = MagicMock()
        kernel.__getitem__.return_value.side_effect = launch
        monkeypatch.setattr(module, name, kernel)

    state.watch = watch
    return state


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_gguf_probes_and_launches_on_tensor_device_then_restores_caller(launches, tiled, fail):
    launches.fail = fail
    launches.backend = "hip"
    for name in (
        "rotate_quantize_rows_kernel",
        "convert_gguf_tiles_kernel",
        "gguf_row_scales_kernel",
    ):
        launches.watch(generic, name)
    with FakeTensorMode():
        data = torch.empty(2, 16384 if tiled else 256, device="cuda:1")
        error_context = (
            pytest.raises(RuntimeError, match="simulated kernel failure") if fail else nullcontext()
        )
        with error_context:
            result = _gguf.convert(
                data,
                quant_type=GGUFQuantizationType.F32,
                group_size=256,
                logical_dtype=torch.bfloat16,
            )
            assert not fail
            assert result[0].device == data.device
    assert launches.calls
    assert launches.current == 0
    if fail:
        assert len(launches.calls) == 1  # No retry after a potentially partial mutation.
    elif tiled:
        assert launches.calls == [
            "convert_gguf_tiles_kernel",
            "gguf_row_scales_kernel",
            "convert_gguf_tiles_kernel",
        ]


@pytest.mark.parametrize("operation", ["prepare_input", "add_", "addmm_"])
def test_generic_operations_keep_triton_on_noncurrent_gpu(launches, operation):
    for module, name in (
        (rotation, "rotate_groups_kernel"),
        (generic, "quantize_rows_kernel"),
        (generic, "requantize_update_rows_kernel"),
    ):
        launches.watch(module, name)
    with FakeTensorMode():
        value = torch.empty(2, 256, device="cuda:1")
        qdata = torch.empty_like(value, dtype=torch.int8)
        scale = torch.empty(2, 1, device=value.device)
        if operation == "prepare_input":
            _generic.prepare_input(value, 256)
        elif operation == "add_":
            _generic.add_(qdata, scale, value, 256, 0.5)
        else:
            _generic.addmm_(
                qdata,
                scale,
                value[:, :16],
                torch.empty(16, 256, device=value.device),
                256,
                1.0,
                0.5,
            )
    assert len(launches.calls) == 2
    assert launches.current == 0


@pytest.mark.parametrize("backend", [nvidia, amd])
@pytest.mark.parametrize("fail", [False, True])
def test_prepared_paired_projection_owns_context_for_all_launches(launches, backend, fail):
    launches.backend = "hip" if backend is amd else "cuda"
    launches.fail = fail
    launches.watch(backend, "int8_matmul_kernel")
    with FakeTensorMode():
        value = torch.empty(129, 256, dtype=torch.int8, device="cuda:1")
        scale = torch.empty(129, device=value.device)
        weight = torch.empty(256, 256, dtype=torch.int8, device=value.device)
        weight_scale = torch.empty(256, 1, device=value.device)
        output = torch.empty(129, 514, device=value.device)[:, 1:-1]
        error_context = (
            pytest.raises(RuntimeError, match="simulated kernel failure") if fail else nullcontext()
        )
        with error_context:
            result = backend.linear_prepared(
                value,
                scale,
                weight,
                weight_scale,
                None,
                torch.float32,
                out=output,
                second_projection=(weight, weight_scale, None),
            )
            assert not fail
            assert result is output
    assert launches.calls
    assert launches.current == 0


def test_nvfp4_scale_and_gguf_launchers_own_context(launches):
    launches.watch(nvfp4, "_amax_partial_kernel")
    launches.watch(nvfp4, "_amax_scale_kernel")
    launches.watch(convrot_nvfp4, "_rotate_quantize_nvfp4_kernel")
    with FakeTensorMode():
        value = torch.empty(3, 256, device="cuda:1")
        per_tensor_scale = nvfp4.dynamic_scale(value)
        convrot_nvfp4._gguf_prepare_out(
            value,
            int(GGUFQuantizationType.F32),
            256,
            torch.bfloat16,
            per_tensor_scale,
            torch.empty(3, 128, dtype=torch.uint8, device=value.device),
            torch.empty(3, 16, dtype=torch.float8_e4m3fn, device=value.device),
            is_swizzled_scales=False,
            high_first=False,
        )
    assert launches.calls[-1] == "_rotate_quantize_nvfp4_kernel"
    assert launches.current == 0


def test_attention_and_fusion_epilogue_launchers_own_context(launches):
    launches.watch(sage, "_sage_attention_2pp_kernel")
    launches.watch(gated_updates, "_gated_updates_kernel")
    with FakeTensorMode():
        value = torch.empty(1, 1, 64, 64, device="cuda:1")
        prepared = SimpleNamespace(
            query=value,
            key=value,
            value=value,
            output=torch.empty_like(value),
            query_scale=value,
            key_scale=value,
            value_scale=value,
            key_length=64,
            is_causal=False,
            plan=sage_policy.SageAttention2ppExecutionPlan(64, False, False),
        )
        assert sage._launch_sage_attention_2pp(prepared) is prepared.output
        updates = SimpleNamespace(
            update_gate=value, ffn_gate=value, gate_indices=value, python_indexing=False
        )
        layout = gated_updates.IndexedGatedUpdateLayout(64, 64, 1, 1)
        gated_updates.apply_indexed_gated_updates(
            value, value, torch.empty_like(value), updates, layout, 0
        )
    assert launches.calls == ["_sage_attention_2pp_kernel", "_gated_updates_kernel"]
    assert launches.current == 0
