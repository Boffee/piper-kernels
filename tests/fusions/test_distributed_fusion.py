"""Tests for distributed TP+SP fusion passes and 2-GPU execution."""

from __future__ import annotations

import gc
import operator
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _accelerators import mock_triton_driver
from _compile_capture import FunctionalTargetCapturePass
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn import functional as F  # noqa: N812

from piper_kernels import (
    SparsePiperAttention,
    compile_options,
    piper_attention,
    sparse_piper_coarse_residual,
)
from piper_kernels._distributed import _comm as distributed_comm
from piper_kernels._distributed import _fx as distributed_fx
from piper_kernels._distributed import _peer_exchange as peer_exchange
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    count_key_scales,
)
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.piper_attention import _quantized_dispatch as dense_quantized_dispatch
from piper_kernels.attention.sparse_piper_attention import _quantized_dispatch
from piper_kernels.attention.sparse_piper_attention._contracts import _SparsePiperRatioProfile
from piper_kernels.fusions.convrot_int8_piper import _backend as dense_fusion_backend
from piper_kernels.fusions.convrot_int8_piper import distributed_output as dense_distributed_output
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    _backend as fusion_backend,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    distributed_output,
    kv_gather,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    key as key_ops,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    value as value_ops,
)
from piper_kernels.linear.convrot.int8 import _backend as convrot_linear_backend
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.sharding import shard_quantized_weight

_ops = torch.ops.piper_kernels
_c10d = torch.ops._c10d_functional
_aten = torch.ops.aten
_AUTO_FUNCTIONALIZED = torch.ops.higher_order.auto_functionalized
_DENSE_RS_RESIDUAL_ADD = (
    _ops.convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_.default
)

_DEVICE = torch.device("cuda", 0)


def _relay_available() -> bool:
    try:
        from piper_offload.communication import register_relay_backend  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = float(expected.float().norm().item())
    if denom == 0.0:
        return float(actual.float().norm().item())
    return float((actual.float() - expected.float()).norm().item()) / denom


def _add_sequence_all_gather(
    graph: torch.fx.Graph,
    x_local: torch.fx.Node,
    *,
    batch: int,
    local_seq: int | torch.SymInt,
    full_seq_arg: torch.fx.Node | int,
    hidden: int,
    group_size: int = 2,
    group_name: str = "tp_0",
    batch_split_arg: torch.fx.Node | int | None = None,
) -> tuple[torch.fx.Node, torch.fx.Node]:
    x_val = x_local.meta["val"]
    wait_val = x_val.new_empty((batch * group_size, local_seq, hidden))
    full_val = x_val.new_empty((batch, local_seq * group_size, hidden))

    ag = graph.call_function(
        _c10d.all_gather_into_tensor.default,
        args=(x_local, group_size, group_name),
    )
    ag.meta["val"] = wait_val
    wait = graph.call_function(_c10d.wait_tensor.default, args=(ag,))
    wait.meta["val"] = wait_val

    if batch == 1:
        out = graph.call_function(
            _aten.reshape.default,
            args=(wait, [1, full_seq_arg, hidden]),
        )
    else:
        split_arg = batch if batch_split_arg is None else batch_split_arg
        split = graph.call_function(_aten.split.Tensor, args=(wait, split_arg))
        split.meta["val"] = [wait_val.new_empty((batch, local_seq, hidden))] * group_size
        chunks = []
        for idx in range(group_size):
            item = graph.call_function(operator.getitem, args=(split, idx))
            item.meta["val"] = wait_val.new_empty((batch, local_seq, hidden))
            chunks.append(item)
        out = graph.call_function(_aten.cat.default, args=(chunks, 1))
    out.meta["val"] = full_val
    return out, wait


def _add_sequence_reduce_scatter(
    graph: torch.fx.Graph,
    full_out: torch.fx.Node,
    *,
    batch: int,
    local_seq: int | torch.SymInt,
    hidden: int,
    group_size: int = 2,
    group_name: str = "tp_0",
) -> torch.fx.Node:
    full_val = full_out.meta["val"]
    repack_val = full_val.new_empty((batch * group_size, local_seq, hidden))
    local_val = full_val.new_empty((batch, local_seq, hidden))
    if batch == 1:
        repack = graph.call_function(
            _aten.reshape.default,
            args=(full_out, [group_size, local_seq, hidden]),
        )
    else:
        split = graph.call_function(_aten.split.Tensor, args=(full_out, local_seq, 1))
        split.meta["val"] = [local_val] * group_size
        chunks = []
        for idx in range(group_size):
            item = graph.call_function(operator.getitem, args=(split, idx))
            item.meta["val"] = local_val
            chunks.append(item)
        repack = graph.call_function(_aten.cat.default, args=(chunks, 0))
    repack.meta["val"] = repack_val
    rs = graph.call_function(
        _c10d.reduce_scatter_tensor.default,
        args=(repack, "sum", group_size, group_name),
    )
    rs.meta["val"] = local_val
    wait = graph.call_function(_c10d.wait_tensor.default, args=(rs,))
    wait.meta["val"] = local_val
    return wait


def _residual_add_op() -> torch._ops.OpOverload:
    from piper_kernels.fusions.residual_updates._graph import RESIDUAL_ADD  # noqa: PLC0415

    return RESIDUAL_ADD


@pytest.fixture
def fake_triton_device(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_triton_driver(monkeypatch, backend="cuda", arch=120)


def _run_post_grad_pass(
    graph: torch.fx.Graph,
    *,
    linear: bool = False,
    attention: bool = False,
    attention_output: bool = False,
) -> None:
    passes = compile_options(
        linear=linear,
        ffn=False,
        attention=attention,
        attention_output=attention_output,
    )["post_grad_custom_pre_pass"]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)


@pytest.mark.parametrize(
    ("batch", "dynamic", "prepare_args", "prepare_kwargs"),
    [
        pytest.param(1, False, (64, None), {}, id="explicit-activation"),
        pytest.param(2, False, (64,), {}, id="omitted-defaults"),
        pytest.param(1, True, (), {"group_size": 64}, id="keyword-group-size"),
        pytest.param(2, True, (64, None, None), {}, id="all-positional"),
    ],
)
def test_fold_quantized_all_gather_fx_shapes_and_symint(
    fake_triton_device: None,
    batch: int,
    dynamic: bool,
    prepare_args: tuple[int | None, ...],
    prepare_kwargs: dict[str, int],
) -> None:
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        graph = torch.fx.Graph()
        hidden = 128
        local_seq_concrete = 64
        x_local = graph.placeholder("x_local")
        if dynamic:
            t_local = shape_env.create_unbacked_symint()
            x_val = torch.empty((batch, t_local, hidden), device=_DEVICE, dtype=torch.bfloat16)
            x_local.meta["val"] = x_val
            local_seq_expr: int | torch.SymInt = x_val.shape[1]
            t_node = graph.call_function(_aten.sym_size.int, args=(x_local, 1))
            t_node.meta["val"] = local_seq_expr
            full_seq_node = graph.call_function(operator.mul, args=(t_node, 2))
            full_seq_node.meta["val"] = local_seq_expr * 2
            batch_node = graph.call_function(_aten.sym_size.int, args=(x_local, 0))
            batch_node.meta["val"] = batch
            full_seq_arg: torch.fx.Node | int = full_seq_node
            batch_split_arg: torch.fx.Node | int | None = batch_node
        else:
            x_val = torch.empty(
                (batch, local_seq_concrete, hidden), device=_DEVICE, dtype=torch.bfloat16
            )
            x_local.meta["val"] = x_val
            local_seq_expr = local_seq_concrete
            full_seq_arg = local_seq_concrete * 2
            batch_split_arg = batch

        ag_out, _ = _add_sequence_all_gather(
            graph,
            x_local,
            batch=batch,
            local_seq=local_seq_expr,
            full_seq_arg=full_seq_arg,
            hidden=hidden,
            batch_split_arg=batch_split_arg,
        )
        sym_seq = graph.call_function(_aten.sym_size.int, args=(ag_out, 1))
        sym_seq.meta["val"] = local_seq_expr * 2

        prep = graph.call_function(
            _ops.convrot_int8_prepare_input.default,
            args=(ag_out, *prepare_args),
            kwargs=prepare_kwargs,
        )
        q_full_val = torch.empty(
            (batch, local_seq_expr * 2, hidden), device=_DEVICE, dtype=torch.int8
        )
        s_full_val = torch.empty((batch, local_seq_expr * 2), device=_DEVICE, dtype=torch.float32)
        prep.meta["val"] = (q_full_val, s_full_val)
        q_item = graph.call_function(operator.getitem, args=(prep, 0))
        q_item.meta["val"] = q_full_val
        s_item = graph.call_function(operator.getitem, args=(prep, 1))
        s_item.meta["val"] = s_full_val
        graph.output((q_item, s_item, sym_seq))

        _run_post_grad_pass(graph, linear=True)
        graph.lint()
        targets = [n.target for n in graph.nodes if n.op == "call_function"]
        assert _c10d.all_gather_into_tensor.default not in targets
        assert _c10d.all_gather_into_tensor_coalesced.default in targets
        assert _ops.convrot_int8_prepare_input.default in targets
        # Wait count matches coalesced Q+S outputs only (original AG wait was erased).
        assert targets.count(_c10d.wait_tensor.default) == 2


def test_fold_quantized_all_gather_fx_noop_and_shared(fake_triton_device: None) -> None:
    with FakeTensorMode():
        # 1-GPU graph with no collectives short-circuits cleanly.
        g_noop = torch.fx.Graph()
        x = g_noop.placeholder("x")
        x.meta["val"] = torch.empty((1, 64, 128), device=_DEVICE, dtype=torch.bfloat16)
        g_noop.output(x)
        _run_post_grad_pass(g_noop, linear=True)
        assert not distributed_fx.has_c10d_collectives(g_noop)
        assert [n.target for n in g_noop.nodes if n.op == "call_function"] == []

        # Shared gathered BF16 consumer prevents rewrite.
        g_shared = torch.fx.Graph()
        x_local = g_shared.placeholder("x_local")
        x_local.meta["val"] = torch.empty((1, 64, 128), device=_DEVICE, dtype=torch.bfloat16)
        ag_out, _ = _add_sequence_all_gather(
            g_shared, x_local, batch=1, local_seq=64, full_seq_arg=128, hidden=128
        )
        prep = g_shared.call_function(
            _ops.convrot_int8_prepare_input.default, args=(ag_out, 64, None)
        )
        prep.meta["val"] = (
            torch.empty((1, 128, 128), device=_DEVICE, dtype=torch.int8),
            torch.empty((1, 128), device=_DEVICE, dtype=torch.float32),
        )
        q_item = g_shared.call_function(operator.getitem, args=(prep, 0))
        extra = g_shared.call_function(_aten.neg.default, args=(ag_out,))
        g_shared.output((q_item, extra))
        _run_post_grad_pass(g_shared, linear=True)
        assert _c10d.all_gather_into_tensor.default in [
            n.target for n in g_shared.nodes if n.op == "call_function"
        ]


def test_rewrite_sym_size_users_negative_dims_and_static_int(fake_triton_device: None) -> None:
    with FakeTensorMode():
        graph = torch.fx.Graph()
        x_local = graph.placeholder("x_local")
        x_local.meta["val"] = torch.empty((2, 64, 128), device=_DEVICE, dtype=torch.bfloat16)
        ag_out, _ = _add_sequence_all_gather(
            graph, x_local, batch=2, local_seq=64, full_seq_arg=128, hidden=128
        )
        all_gather_match = distributed_fx.match_sequence_all_gather(ag_out)
        assert all_gather_match is not None

        neg_seq = graph.call_function(_aten.sym_size.int, args=(ag_out, -2))
        neg_seq.meta["val"] = 128
        neg_batch = graph.call_function(_aten.sym_size.int, args=(ag_out, -3))
        neg_batch.meta["val"] = 2
        neg_hidden = graph.call_function(_aten.sym_size.int, args=(ag_out, -1))
        neg_hidden.meta["val"] = 128
        prep = graph.call_function(_ops.convrot_int8_prepare_input.default, args=(ag_out, 64, None))
        graph.output((neg_seq, neg_batch, neg_hidden, prep))

        sym_users = distributed_fx.collect_sym_size_users(ag_out, prep)
        assert sym_users == [neg_seq, neg_batch, neg_hidden]
        distributed_fx.rewrite_sym_size_users(graph, sym_users, all_gather_match)
        graph.lint()

        assert neg_seq not in graph.nodes
        assert neg_batch.args == (x_local, -3)
        assert neg_hidden.args == (x_local, -1)
        out_seq = graph.output_node().args[0][0]
        assert isinstance(out_seq, torch.fx.Node)
        assert out_seq.target is operator.mul
        assert out_seq.meta["val"] == 128
        local_seq_node = out_seq.args[0]
        assert isinstance(local_seq_node, torch.fx.Node)
        assert local_seq_node.args == (x_local, 1)
        assert local_seq_node.meta["val"] == 64

        bad_dim = graph.call_function(_aten.sym_size.int, args=(ag_out, 3))
        assert bad_dim.meta == {}
        assert distributed_fx.collect_sym_size_users(ag_out, prep) is None


def test_peer_exchange_world_size_3_keeps_zero_buffer_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeGroup:
        def rank(self) -> int:
            return 1

        def size(self) -> int:
            return 3

    recorded_sources: list[list[torch.Tensor]] = []

    class _FakeWork:
        def __init__(self, zero_tensor: torch.Tensor) -> None:
            self.zero_tensor = zero_tensor
            self.waited = False

        def wait(self) -> bool:
            assert int(self.zero_tensor.abs().sum().item()) == 0
            self.waited = True
            return True

    def fake_reduce_scatter(
        _output: torch.Tensor,
        sources: list[torch.Tensor],
        **_kwargs: object,
    ) -> _FakeWork:
        recorded_sources.append(list(sources))
        return _FakeWork(sources[1])

    monkeypatch.setattr(dist, "reduce_scatter", fake_reduce_scatter)

    output = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    fake_pg = _FakeGroup()
    work, offset, rows, _recv, keepalive = peer_exchange.run_peer_attention_exchange(
        fake_pg,  # type: ignore[arg-type]
        output,
        heads=1,
        head_dim=128,
        comm_chunk_rows=64,
        run_attention=lambda *_args: None,
        peer_projector=lambda _a, buf, _s, _r: buf.zero_(),
        apply_received=lambda *_args: None,
        reuse_direct_receive=True,
    )
    gc.collect()
    assert offset == 0
    assert rows == 64
    assert len(recorded_sources) == 1
    zero_buf = recorded_sources[0][1]
    assert len(keepalive) == 3
    assert any(
        tensor.untyped_storage().data_ptr() == zero_buf.untyped_storage().data_ptr()
        for tensor in keepalive
    )
    assert work.wait()
    assert work.waited


def test_partial_kv_launch_validation() -> None:
    from piper_kernels.fusions.convrot_int8_projection._plan import (  # noqa: PLC0415
        ProjectionExecutionPlan,
    )

    plan = ProjectionExecutionPlan(
        block_k=64,
        heads_per_program=1,
        num_warps=4,
        num_stages=2,
        block_m=64,
    )
    launches = plan.iter_row_block_launches(
        1,
        256,
        128,
        2,
        start_row_block=1,
        num_row_blocks=None,
    )
    assert launches == (((3, 2, 1), 1, True, True),)

    with pytest.raises(ValueError, match="full block_m alignment"):
        plan.iter_row_block_launches(
            1,
            200,
            128,
            2,
            start_row_block=1,
            num_row_blocks=None,
        )
    for bad_start, bad_count, expected_match in [
        (-1, 1, "non-negative"),
        (0, -1, "full block_m blocks"),
        (2, 3, "full block_m blocks"),
    ]:
        with pytest.raises(ValueError, match=expected_match):
            plan.iter_row_block_launches(
                1,
                256,
                128,
                2,
                start_row_block=bad_start,
                num_row_blocks=bad_count,
            )


def test_kv_gather_fake_validation() -> None:
    with FakeTensorMode():
        x_local = torch.empty((1, 64, 128), device=_DEVICE, dtype=torch.bfloat16)
        kw = torch.empty((256, 128), device=_DEVICE, dtype=torch.int8)
        ks = torch.empty((256, 1), device=_DEVICE, dtype=torch.float32)
        kn = torch.empty((128,), device=_DEVICE, dtype=torch.bfloat16)
        cos = torch.empty((128, 64), device=_DEVICE, dtype=torch.float32)
        sin = torch.empty((128, 64), device=_DEVICE, dtype=torch.float32)
        vw = torch.empty((256, 128), device=_DEVICE, dtype=torch.int8)
        vs = torch.empty((256, 1), device=_DEVICE, dtype=torch.float32)

        with pytest.raises(ValueError, match="routing"):
            _ops.convrot_int8_sparse_piper_prepare_gather_project_kv.default(
                x_local,
                2,
                "tp_0",
                64,
                None,
                kw,
                ks,
                kn,
                cos,
                sin,
                1e-5,
                99,
                None,
                None,
                None,
                vw,
                vs,
                None,
                128,
                None,
                0,
            )
        with pytest.raises(ValueError, match="comm_group_size"):
            kv_gather._prepare_gather_project_kv_op_fake(
                x_local,
                True,
                "tp_0",
                64,
                None,
                kw,
                ks,
                kn,
                cos,
                sin,
                1e-5,
                0,
                None,
                None,
                None,
                vw,
                vs,
                None,
                128,
                None,
                0,
            )


@pytest.mark.parametrize("world_size", [2, 3])
@pytest.mark.parametrize("reuse_direct_receive", [True, False])
def test_peer_exchange_multi_chunk_ping_pong(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    reuse_direct_receive: bool,
) -> None:
    events: list[tuple[Any, ...]] = []
    recv_ptrs: list[int] = []
    fake_pg = SimpleNamespace(rank=lambda: 0, size=lambda: world_size)

    class _FakeWork:
        def __init__(self, idx: int) -> None:
            self.idx = idx
            self.waited = False

        def wait(self) -> bool:
            self.waited = True
            events.append(("wait", self.idx))
            return True

    issued: list[tuple[_FakeWork, torch.Tensor | None]] = []

    def fake_exchange(
        *args: Any,
        **_kwargs: Any,
    ) -> _FakeWork:
        idx = len(issued)
        work = _FakeWork(idx)
        recv_tensor = args[0][1] if world_size == 2 else args[0]
        assert isinstance(recv_tensor, torch.Tensor)
        recv_ptrs.append(recv_tensor.untyped_storage().data_ptr())
        events.append(("start", idx))
        zero_or_send = args[1] if world_size == 3 else None
        buf = zero_or_send[0] if isinstance(zero_or_send, list) else None
        issued.append((work, buf))
        return work

    monkeypatch.setattr(dist, "all_gather" if world_size == 2 else "reduce_scatter", fake_exchange)

    output = torch.empty((1, 192, 128), dtype=torch.bfloat16)

    def record_apply(_tensor: torch.Tensor, off: int, cnt: int) -> None:
        events.append(("apply", off, cnt))

    work, offset, rows, recv, keepalive = peer_exchange.run_peer_attention_exchange(
        fake_pg,  # type: ignore[arg-type]
        output,
        heads=1,
        head_dim=128,
        comm_chunk_rows=64,
        run_attention=lambda *_args: None,
        peer_projector=lambda _a, buf, _s, _r: buf.zero_(),
        apply_received=record_apply,
        reuse_direct_receive=reuse_direct_receive,
    )
    gc.collect()
    assert len(issued) == 3
    assert [w.waited for w, _ in issued[:2]] == [True, True]
    assert not issued[2][0].waited
    assert (offset, rows) == (128, 64)
    assert len(keepalive) == (1 if world_size == 2 else 3)
    work.wait()
    assert issued[2][0].waited
    record_apply(recv, offset, rows)

    assert len(set(recv_ptrs)) == 1
    assert recv.untyped_storage().data_ptr() == recv_ptrs[0]
    assert (recv_ptrs[0] == output.untyped_storage().data_ptr()) is reuse_direct_receive
    assert events == [
        ("start", 0),
        ("wait", 0),
        ("apply", 0, 64),
        ("start", 1),
        ("wait", 1),
        ("apply", 64, 64),
        ("start", 2),
        ("wait", 2),
        ("apply", 128, 64),
    ]

    # Exception on chunk 1 drains in-flight chunk 0 collective before unwinding.
    issued.clear()
    calls = 0

    def fail_on_second_chunk(_start: int, _rows: int, _attention_chunk: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls > world_size - 1:
            raise RuntimeError("injected chunk failure")

    with pytest.raises(RuntimeError, match="injected chunk failure"):
        peer_exchange.run_peer_attention_exchange(
            fake_pg,  # type: ignore[arg-type]
            output,
            heads=1,
            head_dim=128,
            comm_chunk_rows=64,
            run_attention=fail_on_second_chunk,
            peer_projector=lambda _a, buf, _s, _r: buf.zero_(),
            apply_received=lambda *_args: None,
            reuse_direct_receive=reuse_direct_receive,
        )
    assert len(issued) == 1
    assert issued[0][0].waited


@pytest.mark.parametrize(
    (
        "local_seq",
        "query_chunk_rows",
        "tile_rows",
        "batch",
        "output_features",
        "element_size",
        "world_size",
        "reuse_direct_receive",
    ),
    [
        (8192, 8192, 64, 1, 5376, 2, 2, False),
        (8448, 8192, 128, 1, 6144, 2, 2, False),
        (10240, 8192, 64, 1, 5376, 2, 2, False),
        (55296, 8192, 64, 1, 5376, 2, 2, False),
        (27648, 8192, 64, 1, 5376, 2, 4, False),
        (13824, 8192, 64, 1, 5376, 2, 8, False),
        (10240, 8192, 64, 4, 5376, 2, 2, False),
        (16384, 8192, 64, 1, 12288, 2, 2, False),
        (16384, 8192, 64, 1, 12288, 2, 2, True),
        (16384, 8192, 64, 1, 5376, 4, 2, False),
        (128, 64, 64, 1, 512, 2, 2, False),
    ],
)
def test_choose_distributed_chunk_rows_bounds_comm_workspace(
    local_seq: int,
    query_chunk_rows: int,
    tile_rows: int,
    batch: int,
    output_features: int,
    element_size: int,
    world_size: int,
    reuse_direct_receive: bool,
) -> None:
    rows = peer_exchange.choose_distributed_chunk_rows(
        local_seq,
        query_chunk_rows,
        tile_rows=tile_rows,
        batch=batch,
        output_features=output_features,
        element_size=element_size,
        world_size=world_size,
        reuse_direct_receive=reuse_direct_receive,
    )
    limit = 256 * 1024 * 1024
    peer_count = max(1, world_size - 1)
    fixed_buffers = (0 if reuse_direct_receive else 1) + (0 if world_size <= 2 else 1)
    row_bytes = batch * output_features * element_size

    assert tile_rows <= rows <= min(local_seq, query_chunk_rows)
    assert rows % tile_rows == 0
    buffers = (
        (peer_count + fixed_buffers) if rows == local_seq else (2 * peer_count + fixed_buffers)
    )
    assert rows * row_bytes * buffers <= limit

    budget_rows = limit // (row_bytes * (2 * peer_count + fixed_buffers))
    cap = max(tile_rows, (min(local_seq, query_chunk_rows, budget_rows) // tile_rows) * tile_rows)
    expected_chunks = (local_seq + cap - 1) // cap
    actual_chunks = (local_seq + rows - 1) // rows
    assert actual_chunks == expected_chunks
    balanced = (local_seq + expected_chunks - 1) // expected_chunks
    assert rows == local_seq or rows <= balanced + tile_rows - 1


def test_choose_distributed_chunk_rows_boundaries_and_scaling() -> None:
    assert (
        peer_exchange.choose_distributed_chunk_rows(
            32,
            64,
            tile_rows=64,
            batch=1,
            output_features=512,
            element_size=2,
            world_size=2,
        )
        == 32
    )
    unaligned = peer_exchange.choose_distributed_chunk_rows(
        150,
        256,
        tile_rows=64,
        batch=1,
        output_features=512,
        element_size=2,
        world_size=2,
    )
    assert unaligned % 64 == 0
    assert 64 <= unaligned <= 150
    assert (
        peer_exchange.choose_distributed_chunk_rows(
            1024,
            8192,
            tile_rows=64,
            batch=1,
            output_features=512,
            element_size=2,
            world_size=2,
            max_comm_bytes=1,
        )
        == 64
    )

    wide_with_recv = peer_exchange.choose_distributed_chunk_rows(
        16384,
        8192,
        tile_rows=64,
        batch=1,
        output_features=12288,
        element_size=2,
        world_size=2,
        reuse_direct_receive=False,
    )
    wide_direct_recv = peer_exchange.choose_distributed_chunk_rows(
        16384,
        8192,
        tile_rows=64,
        batch=1,
        output_features=12288,
        element_size=2,
        world_size=2,
        reuse_direct_receive=True,
    )
    assert wide_direct_recv > wide_with_recv

    world_2, world_4, world_8 = (
        peer_exchange.choose_distributed_chunk_rows(
            27648,
            8192,
            tile_rows=64,
            batch=1,
            output_features=5376,
            element_size=2,
            world_size=ws,
        )
        for ws in (2, 4, 8)
    )
    assert world_2 > world_4 > world_8

    batch_1, batch_4 = (
        peer_exchange.choose_distributed_chunk_rows(
            10240,
            8192,
            tile_rows=64,
            batch=bs,
            output_features=5376,
            element_size=2,
            world_size=2,
        )
        for bs in (1, 4)
    )
    assert batch_1 > batch_4

    bf16_rows, fp32_rows = (
        peer_exchange.choose_distributed_chunk_rows(
            16384,
            8192,
            tile_rows=64,
            batch=1,
            output_features=5376,
            element_size=elem,
            world_size=2,
        )
        for elem in (2, 4)
    )
    assert bf16_rows > fp32_rows


def _make_distributed_output_meta_args(
    *,
    key_device: str | torch.device = "meta",
    base_device: str | torch.device = "meta",
    local_seq: int = 64,
    cos_rows: int = 128,
    q_out_features: int = 128,
    softmax_scale: float = 0.125,
    coarse_gate: torch.Tensor | None = None,
    input_group_size: int | None = None,
    fill_tensors: bool = False,
) -> tuple[tuple[Any, ...], torch.Tensor]:
    make = torch.zeros if fill_tensors else torch.empty
    seq, heads, head_dim, hidden, batch, tiles = 128, 2, 64, 128, 1, 2
    base = make((batch, local_seq, hidden), device=base_device, dtype=torch.bfloat16)
    gate = make((4, hidden), device=base_device, dtype=torch.bfloat16)
    gate_indices = torch.zeros(batch * local_seq, device=base_device, dtype=torch.int64)
    query_input = make(
        (batch, seq, hidden),
        device=key_device,
        dtype=torch.bfloat16 if input_group_size is not None else torch.int8,
    )
    query_scale = (
        None
        if input_group_size is not None
        else torch.ones((batch, seq), device=key_device, dtype=torch.float32)
        if fill_tensors
        else torch.empty((batch, seq), device=key_device, dtype=torch.float32)
    )
    key = make((batch, heads, seq, head_dim), device=key_device, dtype=torch.int8)
    args = (
        base,
        gate,
        gate_indices,
        False,
        2,
        "tp_0",
        query_input,
        query_scale,
        make((q_out_features, hidden), device=key_device, dtype=torch.int8),
        make((q_out_features, 1), device=key_device, dtype=torch.float32),
        make((head_dim,), device=key_device, dtype=torch.bfloat16),
        make((cos_rows, 32), device=key_device, dtype=torch.float32),
        make((cos_rows, 32), device=key_device, dtype=torch.float32),
        1e-5,
        softmax_scale,
        key,
        make((batch, heads, tiles), device=key_device, dtype=torch.float32),
        make((batch, heads, tiles, 2, head_dim), device=key_device, dtype=torch.int8),
        make((batch, heads, tiles, 2), device=key_device, dtype=torch.float32),
        make((batch, heads, tiles, head_dim), device=key_device, dtype=torch.float32),
        make((batch, heads, 0, head_dim), device=key_device, dtype=torch.float32),
        make((batch, heads, head_dim, seq), device=key_device, dtype=torch.int8),
        make((batch, heads, tiles, 1), device=key_device, dtype=torch.float32),
        make((batch, heads, tiles, head_dim), device=key_device, dtype=torch.bfloat16),
        _SparsePiperRatioProfile((500_000, 1_000_000)),
        tiles,
        seq,
        0,
        make((hidden, heads * head_dim), device=key_device, dtype=torch.int8),
        make((hidden, 1), device=key_device, dtype=torch.float32),
        None,
        64,
        64,
        None,
        coarse_gate,
        0.125 if coarse_gate is not None else None,
        tiles if coarse_gate is not None else None,
    )
    return args, base


def test_distributed_output_validation_and_fake() -> None:
    assert peer_exchange.validate_distributed_sequence(128, 2, "tp_0") == 64
    with pytest.raises(ValueError, match="comm_group_size >= 2"):
        peer_exchange.validate_distributed_sequence(128, 1, "tp_0")
    with pytest.raises(ValueError, match="comm_group_size >= 2"):
        peer_exchange.validate_distributed_sequence(128, True, "tp_0")
    with pytest.raises(TypeError, match="string comm_group_name"):
        peer_exchange.validate_distributed_sequence(128, 2, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divide evenly"):
        peer_exchange.validate_distributed_sequence(131, 2, "tp_0")

    with FakeTensorMode():
        valid_args, valid_base = _make_distributed_output_meta_args()
        assert (
            distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *valid_args
            )
            is None
        )
        standalone_out = distributed_output._projected_query_attention_reduce_scatter_op_fake(
            *valid_args[4:]
        )
        assert standalone_out.shape == (1, 64, 128)
        assert standalone_out.dtype is torch.bfloat16

        wrong_rows_args, _ = _make_distributed_output_meta_args(local_seq=128)
        with pytest.raises(ValueError, match="local rank share"):
            distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *wrong_rows_args
            )

        wrong_dev_args, _ = _make_distributed_output_meta_args(key_device="meta", base_device="cpu")
        with pytest.raises(ValueError, match="storage device"):
            distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *wrong_dev_args
            )

        bad_rs_args = (1, "tp_0", *valid_args[6:])
        with pytest.raises(ValueError, match="comm_group_size >= 2"):
            distributed_output._projected_query_attention_reduce_scatter_op_fake(*bad_rs_args)
        assert valid_base.shape == (1, 64, 128)


def test_distributed_output_pre_mutation_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeGroup:
        def rank(self) -> int:
            return 0

        def size(self) -> int:
            return 2

    comm_calls = 0

    def fail_comm(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal comm_calls
        comm_calls += 1
        raise AssertionError("collective launched despite invalid metadata")

    monkeypatch.setattr(distributed_comm, "resolve_process_group", lambda _name: _FakeGroup())
    monkeypatch.setattr(dist, "all_gather", fail_comm)
    monkeypatch.setattr(dist, "reduce_scatter", fail_comm)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", fail_comm)
    monkeypatch.setattr(fusion_backend, "require_projection_backend", lambda *_a, **_kw: object())
    monkeypatch.setattr(fusion_backend, "require_output_backend", lambda *_a, **_kw: object())
    monkeypatch.setattr(
        _quantized_dispatch._backend,
        "require_attention_backend",
        lambda _k: SimpleNamespace(
            value_format=ValueFormat.TRANSPOSED,
            skip_dense_routing=True,
            bind_context=lambda ctx: SimpleNamespace(context=ctx),
        ),
    )

    # Every invalid call must fail before collectives run and leave residual base untouched.
    cases = [
        ("RoPE cos/sin", {"cos_rows": 64, "input_group_size": None}),
        ("RoPE cos/sin", {"cos_rows": 64, "input_group_size": 64}),
        ("Q heads", {"q_out_features": 64}),
        ("softmax scale", {"softmax_scale": 0.0}),
        (
            "coarse gate",
            {"coarse_gate": torch.randn((1, 64, 2, 64), dtype=torch.bfloat16)},
        ),
        ("storage device", {"key_device": "meta", "base_device": "cpu"}),
    ]
    for pattern, overrides in cases:
        make_kwargs: dict[str, Any] = {
            "key_device": "cpu",
            "base_device": "cpu",
            "fill_tensors": True,
            **overrides,
        }
        args, base = _make_distributed_output_meta_args(**make_kwargs)
        if base.device.type == "cpu":
            torch.manual_seed(11)
            base.copy_(torch.randn_like(base))
        before = base.clone()
        igs = overrides.get("input_group_size")
        with pytest.raises(ValueError, match=pattern):
            distributed_output._projected_query_attention_reduce_scatter_residual_add_op(
                *args,
                input_group_size=igs,  # type: ignore[arg-type]
            )
        assert comm_calls == 0
        assert torch.equal(base, before)


def _setup_kv_gather_test_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[tuple[object, ...]],
    dict[str, Any],
    torch.Tensor,
    kv_gather._KeyProjectionSpec,
    kv_gather._ValueProjectionSpec,
]:
    events: list[tuple[object, ...]] = []
    state: dict[str, Any] = {"block_size": 64, "raise_own_key": False}

    class _FakeGroup:
        def rank(self) -> int:
            return 1

        def size(self) -> int:
            return 2

    class _FakeWork:
        def wait(self) -> bool:
            events.append(("wait_work",))
            return True

    def fake_prepare_input(
        _x: torch.Tensor, _gs: int, input_scale: torch.Tensor | None = None, *, out: Any = None
    ) -> Any:
        del input_scale
        events.append(("prepare_input",))
        if out is not None:
            out[0].zero_()
            out[1].fill_(1.0)
        return out

    def fake_project_key(
        *_a: Any,
        out: Any,
        start_row_block: int = 0,
        num_row_blocks: int | None = None,
        **_kw: Any,
    ) -> None:
        del out
        events.append(("project_key", start_row_block, num_row_blocks))
        if state["raise_own_key"] and start_row_block != 0:
            raise RuntimeError("own key launch failure")

    def fake_project_value(
        *_a: Any,
        out: Any,
        start_row_block: int = 0,
        num_row_blocks: int | None = None,
        **_kw: Any,
    ) -> None:
        del out
        events.append(("project_value", start_row_block, num_row_blocks))

    linear_stub = SimpleNamespace(prepare_input=fake_prepare_input)
    kv_backend = SimpleNamespace(
        partial_row_block_size=lambda *_a, **_kw: state["block_size"],
        project_key=fake_project_key,
        project_value=fake_project_value,
    )
    monkeypatch.setattr(distributed_comm, "resolve_process_group", lambda _name: _FakeGroup())
    monkeypatch.setattr(
        distributed_comm,
        "start_coalesced_all_gather",
        lambda _pg, _out, _in: events.append(("start_coalesced_all_gather",)) or (_FakeWork(),),
    )

    def fake_wait_all(works: Any) -> None:
        events.append(("wait_all", len(works)))
        for work in works:
            work.wait()

    monkeypatch.setattr(distributed_comm, "wait_all", fake_wait_all)
    monkeypatch.setattr(convrot_linear_backend, "require_linear_backend", lambda _t: linear_stub)
    monkeypatch.setattr(fusion_backend, "require_projection_backend", lambda *_a, **_kw: kv_backend)

    x_local = torch.empty((1, 128, 128), dtype=torch.bfloat16)
    key_spec = kv_gather._KeyProjectionSpec(
        weight_qdata=torch.empty((256, 128), dtype=torch.int8),
        weight_scale=torch.empty((256, 1), dtype=torch.float32),
        norm_weight=torch.empty((128,), dtype=torch.bfloat16),
        cos=torch.empty((256, 64), dtype=torch.float32),
        sin=torch.empty((256, 64), dtype=torch.float32),
        norm_epsilon=1e-5,
        routing_mode=0,
        block_lengths=None,
        bias=None,
        head_dim=128,
    )
    value_spec = kv_gather._ValueProjectionSpec(
        weight_qdata=torch.empty((256, 128), dtype=torch.int8),
        weight_scale=torch.empty((256, 1), dtype=torch.float32),
        block_lengths=None,
        head_dim=128,
        bias=None,
        storage_format=ValueFormat.TRANSPOSED,
    )
    return events, state, x_local, key_spec, value_spec


def test_kv_gather_pre_collective_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    events, _state, x_local, key_spec, value_spec = _setup_kv_gather_test_stubs(monkeypatch)
    with pytest.raises(ValueError, match="contiguous"):
        kv_gather._run_prepare_gather_project_kv(
            x_local.transpose(1, 2), 2, "tp_0", 64, None, key_spec, value_spec
        )
    with pytest.raises(ValueError, match="expected 4"):
        kv_gather._run_prepare_gather_project_kv(x_local, 4, "tp_0", 64, None, key_spec, value_spec)
    bad_val_spec = kv_gather._ValueProjectionSpec(
        weight_qdata=torch.empty((256, 64), dtype=torch.int8),
        weight_scale=value_spec.weight_scale,
        block_lengths=None,
        head_dim=128,
        bias=None,
        storage_format=ValueFormat.TRANSPOSED,
    )
    with pytest.raises(ValueError, match="complete"):
        kv_gather._run_prepare_gather_project_kv(
            x_local, 2, "tp_0", 64, None, key_spec, bad_val_spec
        )
    assert events == []


def test_kv_gather_overlap_and_fallback_order(monkeypatch: pytest.MonkeyPatch) -> None:
    events, state, x_local, key_spec, value_spec = _setup_kv_gather_test_stubs(monkeypatch)

    # Overlapped order: prepare -> start_all_gather -> own K/V -> wait_all -> peer K/V.
    kv_gather._run_prepare_gather_project_kv(x_local, 2, "tp_0", 64, None, key_spec, value_spec)
    assert events == [
        ("prepare_input",),
        ("start_coalesced_all_gather",),
        ("project_key", 2, 2),
        ("project_value", 2, 2),
        ("wait_all", 1),
        ("wait_work",),
        ("project_key", 0, 2),
        ("project_value", 0, 2),
    ]

    # Non-overlapped Gluon order (`partial_row_block_size` -> None): wait_all before K/V.
    events.clear()
    state["block_size"] = None
    kv_gather._run_prepare_gather_project_kv(x_local, 2, "tp_0", 64, None, key_spec, value_spec)
    assert events == [
        ("prepare_input",),
        ("start_coalesced_all_gather",),
        ("wait_all", 1),
        ("wait_work",),
        ("project_key", 0, None),
        ("project_value", 0, None),
    ]

    # Exceptions during in-flight own projection still drain collectives before unwinding.
    events.clear()
    state["block_size"] = 64
    state["raise_own_key"] = True
    with pytest.raises(RuntimeError, match="own key launch failure"):
        kv_gather._run_prepare_gather_project_kv(x_local, 2, "tp_0", 64, None, key_spec, value_spec)
    assert ("wait_all", 1) in events
    assert ("wait_work",) in events


def test_nvidia_partial_kv_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    from piper_kernels.fusions.convrot_int8_sparse_piper._nvidia import (  # noqa: PLC0415
        dispatch as nvidia_dispatch,
    )

    qdata = torch.empty((1, 128, 128), dtype=torch.int8)
    scale = torch.empty((1, 128), dtype=torch.float32)
    weight = torch.empty((256, 128), dtype=torch.int8)
    weight_scale = torch.empty((256, 1), dtype=torch.float32)
    cos = torch.empty((128, 64), dtype=torch.float32)
    sin = torch.empty((128, 64), dtype=torch.float32)
    k_out = key_ops.allocate_key_output(qdata, 1, 2, 128, 128, 0)
    v_out = value_ops.allocate_value_output(qdata, 1, 2, 128, 128, ValueFormat.TRANSPOSED)

    sm120 = AcceleratorTarget("cuda", "sm120")
    sm89 = AcceleratorTarget("cuda", "sm89")
    orig_default_plan = nvidia_dispatch.default_execution_plan
    sm120_key_plan = orig_default_plan(
        qdata, weight, operation="key", head_dim=128, rotary_dim=64, target=sm120
    )
    sm120_value_plan = orig_default_plan(
        qdata, weight, operation="value", head_dim=128, target=sm120
    )

    monkeypatch.setattr(
        nvidia_dispatch, "default_execution_plan", partial(orig_default_plan, target=sm120)
    )
    assert (
        nvidia_dispatch.partial_row_block_size(
            qdata, weight, operation="key", head_dim=128, rotary_dim=64
        )
        is None
    )
    assert (
        nvidia_dispatch.partial_row_block_size(
            qdata,
            weight,
            operation="value",
            head_dim=128,
            value_format=ValueFormat.TRANSPOSED,
        )
        is None
    )
    # Non-transposed AMD_WMMA falls back to Triton so its partial row block is exposed.
    assert (
        nvidia_dispatch.partial_row_block_size(
            qdata,
            weight,
            operation="value",
            head_dim=128,
            value_format=ValueFormat.AMD_WMMA,
        )
        == 128
    )
    with pytest.raises(ValueError, match="Triton execution plan"):
        nvidia_dispatch.project_key(
            qdata,
            scale,
            weight,
            weight_scale,
            None,
            cos,
            sin,
            1e-5,
            0,
            None,
            out=k_out,
            execution_plan=sm120_key_plan,
            start_row_block=1,
            num_row_blocks=1,
        )
    with pytest.raises(ValueError, match="Triton execution plan"):
        nvidia_dispatch.project_value(
            qdata,
            scale,
            weight,
            weight_scale,
            None,
            value_format=ValueFormat.TRANSPOSED,
            out=v_out,
            execution_plan=sm120_value_plan,
            start_row_block=1,
            num_row_blocks=1,
        )

    monkeypatch.setattr(
        nvidia_dispatch, "default_execution_plan", partial(orig_default_plan, target=sm89)
    )
    triton_qdata = torch.empty((1, 128, 144), dtype=torch.int8)
    triton_weight = torch.empty((256, 144), dtype=torch.int8)
    assert (
        nvidia_dispatch.partial_row_block_size(
            triton_qdata, triton_weight, operation="key", head_dim=128, rotary_dim=64
        )
        == 64
    )
    assert (
        nvidia_dispatch.partial_row_block_size(
            triton_qdata,
            triton_weight,
            operation="value",
            head_dim=128,
            value_format=ValueFormat.TRANSPOSED,
        )
        == 64
    )


def _sentinel_clone(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    filled = tuple(torch.empty_like(t) for t in tensors)
    for tensor in filled:
        fill_val: int | float = (
            17 if tensor.dtype is torch.int8 else -99.0 if tensor.dtype is torch.float32 else -7.5
        )
        tensor.fill_(fill_val)
    return filled


def _assert_dim2_partial(
    part: torch.Tensor,
    ref: torch.Tensor,
    start: int,
    end: int,
    sentinel: int | float,
) -> None:
    assert torch.equal(part[:, :, start:end], ref[:, :, start:end])
    assert torch.all(part[:, :, :start] == sentinel)
    assert torch.all(part[:, :, end:] == sentinel)


@pytest.mark.gpu
def test_partial_kv_launch_preserves_guard_blocks() -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("requires at least 1 CUDA/ROCm GPU")
    device = _DEVICE
    batch, heads, head_dim, input_features, rotary_dim = 1, 2, 128, 128, 64
    probe_q = torch.empty((batch, 64, input_features), device=device, dtype=torch.int8)
    probe_w = torch.empty((heads * head_dim, input_features), device=device, dtype=torch.int8)
    key_backend = fusion_backend.require_projection_backend(probe_q, head_dim=head_dim)
    value_backend = fusion_backend.require_projection_backend(probe_q, head_dim=head_dim)
    key_block_m = key_backend.partial_row_block_size(
        probe_q, probe_w, operation="key", head_dim=head_dim, rotary_dim=rotary_dim
    )
    value_format = _quantized_dispatch._backend.require_attention_backend(
        torch.empty((batch, heads, 64, head_dim), device=device, dtype=torch.int8)
    ).value_format
    value_block_m = value_backend.partial_row_block_size(
        probe_q, probe_w, operation="value", head_dim=head_dim, value_format=value_format
    )
    if key_block_m is None or value_block_m is None:
        pytest.skip("backend uses whole-sequence Gluon projection instead of partial row blocks")

    seq = 4 * max(key_block_m, value_block_m)
    torch.manual_seed(321)
    x_qdata = torch.randint(
        -127, 128, (batch, seq, input_features), device=device, dtype=torch.int8
    )
    x_scale = torch.rand((batch, seq), device=device, dtype=torch.float32).add_(0.1)
    k_w = torch.randint(
        -127, 128, (heads * head_dim, input_features), device=device, dtype=torch.int8
    )
    v_w = torch.randint(
        -127, 128, (heads * head_dim, input_features), device=device, dtype=torch.int8
    )
    k_s = torch.rand((heads * head_dim, 1), device=device, dtype=torch.float32).add_(0.01)
    v_s = torch.rand((heads * head_dim, 1), device=device, dtype=torch.float32).add_(0.01)
    k_norm = torch.rand((head_dim,), device=device, dtype=torch.bfloat16).add_(0.5)
    angles = torch.rand((seq, rotary_dim), device=device, dtype=torch.float32) * (2 * torch.pi)
    cos, sin = angles.cos().contiguous(), angles.sin().contiguous()

    # Include minmax routing (`routing_mode=1`) so `key_aux` is non-empty and guard-checked.
    ref_k = key_ops.allocate_key_output(x_qdata, batch, heads, seq, head_dim, 1)
    ref_v = value_ops.allocate_value_output(x_qdata, batch, heads, seq, head_dim, value_format)
    key_backend.project_key(x_qdata, x_scale, k_w, k_s, k_norm, cos, sin, 1e-5, 1, None, out=ref_k)
    value_backend.project_value(
        x_qdata, x_scale, v_w, v_s, None, value_format=value_format, out=ref_v
    )

    part_k, part_v = _sentinel_clone(ref_k), _sentinel_clone(ref_v)
    k_total, v_total = seq // key_block_m, seq // value_block_m
    k_ptr, v_ptr = part_k[0].data_ptr(), part_v[0].data_ptr()
    key_backend.project_key(
        x_qdata,
        x_scale,
        k_w,
        k_s,
        k_norm,
        cos,
        sin,
        1e-5,
        1,
        None,
        out=part_k,
        start_row_block=1,
        num_row_blocks=k_total - 2,
    )
    value_backend.project_value(
        x_qdata,
        x_scale,
        v_w,
        v_s,
        None,
        value_format=value_format,
        out=part_v,
        start_row_block=1,
        num_row_blocks=v_total - 2,
    )
    assert part_k[0].data_ptr() == k_ptr
    assert part_v[0].data_ptr() == v_ptr

    k_row_start, k_row_end = key_block_m, (k_total - 1) * key_block_m
    _assert_dim2_partial(part_k[0], ref_k[0], k_row_start, k_row_end, 17)
    for idx, sentinel in [(1, -99.0), (2, 17), (3, -99.0), (4, -99.0), (5, -99.0)]:
        _assert_dim2_partial(part_k[idx], ref_k[idx], k_row_start // 64, k_row_end // 64, sentinel)

    v_row_start, v_row_end = value_block_m, (v_total - 1) * value_block_m
    if value_format is ValueFormat.AMD_WMMA:
        _assert_dim2_partial(part_v[0], ref_v[0], v_row_start // 64, v_row_end // 64, 17)
    else:
        assert torch.equal(
            part_v[0][:, :, :, v_row_start:v_row_end], ref_v[0][:, :, :, v_row_start:v_row_end]
        )
        assert torch.all(part_v[0][:, :, :, :v_row_start] == 17)
        assert torch.all(part_v[0][:, :, :, v_row_end:] == 17)
    for idx, sentinel in [(1, -99.0), (2, -7.5)]:
        _assert_dim2_partial(part_v[idx], ref_v[idx], v_row_start // 64, v_row_end // 64, sentinel)


def _make_kv_fx_graph(include_value: bool) -> torch.fx.Graph:
    shape_env = ShapeEnv()
    graph = torch.fx.Graph()
    t_local = shape_env.create_unbacked_symint()
    x_local = graph.placeholder("x_local")
    x_val = torch.empty((1, t_local, 128), device=_DEVICE, dtype=torch.bfloat16)
    x_local.meta["val"] = x_val
    t_node = graph.call_function(_aten.sym_size.int, args=(x_local, 1))
    t_node.meta["val"] = x_val.shape[1]
    full_t = graph.call_function(operator.mul, args=(t_node, 2))
    full_t.meta["val"] = x_val.shape[1] * 2

    ag_out, _ = _add_sequence_all_gather(
        graph, x_local, batch=1, local_seq=x_val.shape[1], full_seq_arg=full_t, hidden=128
    )
    sym_seq = graph.call_function(_aten.sym_size.int, args=(ag_out, 1))
    sym_seq.meta["val"] = x_val.shape[1] * 2
    specs = {
        "cos": torch.empty((1024, 96), device=_DEVICE, dtype=torch.float32),
        "sin": torch.empty((1024, 96), device=_DEVICE, dtype=torch.float32),
        "kw": torch.empty((256, 128), device=_DEVICE, dtype=torch.int8),
        "ks": torch.empty((256, 2), device=_DEVICE, dtype=torch.float32),
        "kn": torch.empty((128,), device=_DEVICE, dtype=torch.bfloat16),
        "vw": torch.empty((256, 128), device=_DEVICE, dtype=torch.int8),
        "vs": torch.empty((256, 2), device=_DEVICE, dtype=torch.float32),
    }
    p_nodes = {name: graph.placeholder(name) for name in specs}
    for name, val in specs.items():
        p_nodes[name].meta["val"] = val

    cos_s = graph.call_function(_aten.slice.Tensor, args=(p_nodes["cos"], 0, 0, sym_seq))
    cos_s.meta["val"] = torch.empty((x_val.shape[1] * 2, 96), device=_DEVICE)
    sin_s = graph.call_function(_aten.slice.Tensor, args=(p_nodes["sin"], 0, 0, sym_seq))
    sin_s.meta["val"] = torch.empty((x_val.shape[1] * 2, 96), device=_DEVICE)

    prep = graph.call_function(
        _ops.convrot_int8_prepare_input.default, kwargs={"input": ag_out, "group_size": 64}
    )
    q_val = torch.empty((1, x_val.shape[1] * 2, 128), device=_DEVICE, dtype=torch.int8)
    s_val = torch.empty((1, x_val.shape[1] * 2), device=_DEVICE, dtype=torch.float32)
    prep.meta["val"] = (q_val, s_val)
    q_item = graph.call_function(operator.getitem, args=(prep, 0))
    q_item.meta["val"] = q_val
    s_item = graph.call_function(operator.getitem, args=(prep, 1))
    s_item.meta["val"] = s_val

    k_node = graph.call_function(
        _ops.convrot_int8_sparse_piper_project_key.default,
        args=(q_item, s_item, p_nodes["kw"], p_nodes["ks"], p_nodes["kn"], cos_s, sin_s, 1e-5, 0),
    )
    k_val = (
        torch.empty((1, 2, 128, 128), device=_DEVICE, dtype=torch.int8),
        torch.empty((1, 2, 2), device=_DEVICE, dtype=torch.float32),
        torch.empty((1, 2, 2, 2, 128), device=_DEVICE, dtype=torch.int8),
        torch.empty((1, 2, 2, 2), device=_DEVICE, dtype=torch.float32),
        torch.empty((1, 2, 2, 128), device=_DEVICE, dtype=torch.float32),
        torch.empty((1, 2, 2, 128), device=_DEVICE, dtype=torch.float32),
    )
    k_node.meta["val"] = k_val
    k0 = graph.call_function(operator.getitem, args=(k_node, 0))
    k0.meta["val"] = k_val[0]

    outs: list[torch.fx.Node] = [q_item, s_item, k0]
    if include_value:
        v_node = graph.call_function(
            _ops.convrot_int8_sparse_piper_project_value.default,
            args=(q_item, s_item, p_nodes["vw"], p_nodes["vs"]),
        )
        v_val = (
            torch.empty((1, 2, 128, 128), device=_DEVICE, dtype=torch.int8),
            torch.empty((1, 2, 2, 1), device=_DEVICE, dtype=torch.float32),
            torch.empty((1, 2, 2, 128), device=_DEVICE, dtype=torch.bfloat16),
        )
        v_node.meta["val"] = v_val
        v0 = graph.call_function(operator.getitem, args=(v_node, 0))
        v0.meta["val"] = v_val[0]
        outs.append(v0)
    graph.output(tuple(outs))
    return graph


@pytest.mark.parametrize("include_value", [True, False])
def test_fold_prepare_gather_project_kv_fx(
    fake_triton_device: None,
    include_value: bool,
) -> None:
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        graph = _make_kv_fx_graph(include_value)
        _run_post_grad_pass(graph, attention=True)
        graph.lint()
        targets = [n.target for n in graph.nodes if n.op == "call_function"]
        if include_value:
            assert not distributed_fx.has_c10d_collectives(graph)
            assert _ops.convrot_int8_sparse_piper_prepare_gather_project_kv.default in targets
            assert _ops.convrot_int8_prepare_input.default not in targets
            assert _ops.convrot_int8_sparse_piper_project_key.default not in targets
            assert _ops.convrot_int8_sparse_piper_project_value.default not in targets
        else:
            assert distributed_fx.has_c10d_collectives(graph)
            assert _ops.convrot_int8_sparse_piper_prepare_gather_project_kv.default not in targets


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("residual", [True, False])
def test_fold_distributed_attention_output_fx(
    fake_triton_device: None,
    batch: int,
    residual: bool,
) -> None:
    with FakeTensorMode():
        graph = torch.fx.Graph()
        placeholders = {
            name: graph.placeholder(name)
            for name in (
                "qi",
                "qw",
                "qs",
                "cos",
                "sin",
                "k0",
                "k1",
                "k2",
                "k3",
                "k4",
                "k5",
                "v0",
                "v1",
                "v2",
                "ow",
                "os",
            )
        }
        full_val = torch.empty((batch, 128, 128), device=_DEVICE, dtype=torch.bfloat16)
        attn_out = graph.call_function(
            _ops.convrot_int8_sparse_piper_projected_query_attention_output.default,
            args=(
                placeholders["qi"],
                None,
                placeholders["qw"],
                placeholders["qs"],
                None,
                placeholders["cos"],
                placeholders["sin"],
                1e-5,
                0.088,
                placeholders["k0"],
                placeholders["k1"],
                placeholders["k2"],
                placeholders["k3"],
                placeholders["k4"],
                placeholders["k5"],
                placeholders["v0"],
                placeholders["v1"],
                placeholders["v2"],
                None,
                2,
                128,
                0,
                placeholders["ow"],
                placeholders["os"],
                None,
                64,
            ),
        )
        attn_out.meta["val"] = full_val
        wait_node = _add_sequence_reduce_scatter(
            graph, attn_out, batch=batch, local_seq=64, hidden=128
        )
        if residual:
            base = graph.placeholder("base")
            gate = graph.placeholder("gate")
            idx = graph.placeholder("idx")
            base_val = torch.empty((batch, 64, 128), device=_DEVICE, dtype=torch.bfloat16)
            base.meta["val"] = base_val
            call = graph.call_function(
                _AUTO_FUNCTIONALIZED,
                args=(_residual_add_op(),),
                kwargs={
                    "base": base,
                    "update": wait_node,
                    "gate": gate,
                    "gate_indices": idx,
                    "python_indexing": False,
                },
            )
            call.meta["val"] = (None, base_val)
            out_node = graph.call_function(operator.getitem, args=(call, 1))
            out_node.meta["val"] = base_val
        else:
            out_node = wait_node
        graph.output(out_node)

        _run_post_grad_pass(graph, attention=True)
        graph.lint()
        assert not distributed_fx.has_c10d_collectives(graph)
        targets = [n.target for n in graph.nodes if n.op == "call_function"]
        if residual:
            assert (
                _ops.convrot_int8_sparse_piper_projected_query_attention_output.default
                not in targets
            )
            op_prefix = "convrot_int8_sparse_piper_projected_query_attention"
            expected_res_op = getattr(_ops, f"{op_prefix}_reduce_scatter_residual_add_").default
            assert any(
                n.target is _AUTO_FUNCTIONALIZED and n.args and n.args[0] is expected_res_op
                for n in graph.nodes
            )
        else:
            assert (
                _ops.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter.default
                in targets
            )


def test_fold_distributed_attention_output_fx_shared_negative(fake_triton_device: None) -> None:
    with FakeTensorMode():
        graph = torch.fx.Graph()
        p = graph.placeholder("p")
        attn_out = graph.call_function(
            _ops.convrot_int8_sparse_piper_projected_query_attention_output.default,
            args=tuple([p] * 26),
        )
        attn_out.meta["val"] = torch.empty((1, 128, 128), device=_DEVICE, dtype=torch.bfloat16)
        wait_node = _add_sequence_reduce_scatter(graph, attn_out, batch=1, local_seq=64, hidden=128)
        extra = graph.call_function(_aten.neg.default, args=(attn_out,))
        graph.output((wait_node, extra))
        _run_post_grad_pass(graph, attention=True)
        assert distributed_fx.has_c10d_collectives(graph)
        targets = [n.target for n in graph.nodes if n.op == "call_function"]
        assert (
            _ops.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter.default
            not in targets
        )


def _make_dense_distributed_output_meta_args(
    *,
    key_device: str | torch.device = "meta",
    base_device: str | torch.device = "meta",
    batch: int = 1,
    seq: int = 256,
    local_seq: int = 128,
    softmax_scale: float = 0.125,
    query_chunk_rows: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
    fill_tensors: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any], torch.Tensor]:
    make = torch.zeros if fill_tensors else torch.empty
    heads, head_dim, hidden = 2, 64, 128
    storage = (seq + 63) // 64 * 64
    chunks = (seq + 1023) // 1024
    num_key_scales = count_key_scales(storage, "per_warp")
    base = make((batch, local_seq, hidden), device=base_device, dtype=output_dtype)
    gate = make((4, hidden), device=base_device, dtype=output_dtype)
    gate_indices = torch.zeros(batch * local_seq, device=base_device, dtype=torch.int64)
    input_qdata = make((batch, seq, hidden), device=key_device, dtype=torch.int8)
    input_scale = (
        torch.ones((batch, seq), device=key_device, dtype=torch.float32)
        if fill_tensors
        else torch.empty((batch, seq), device=key_device, dtype=torch.float32)
    )
    key = make((batch, heads, storage, head_dim), device=key_device, dtype=torch.int8)
    args = (
        base,
        gate,
        gate_indices,
        False,
        2,
        "tp_0",
        input_qdata,
        input_scale,
        make((hidden, hidden), device=key_device, dtype=torch.int8),
        make((hidden, 1), device=key_device, dtype=torch.float32),
        make((head_dim,), device=key_device, dtype=output_dtype),
        make((seq, 32), device=key_device, dtype=torch.float32),
        make((seq, 32), device=key_device, dtype=torch.float32),
        1e-5,
        softmax_scale,
        None,
        key,
        make((batch, heads, num_key_scales), device=key_device, dtype=torch.float32),
        make((batch, heads, chunks, head_dim), device=key_device, dtype=torch.float32),
        make((batch, heads, head_dim, storage), device=key_device, dtype=torch.int8),
        make((batch, heads, storage), device=key_device, dtype=torch.float32),
        make((batch, heads, storage), device=key_device, dtype=torch.float32),
        make((batch, heads, chunks, head_dim), device=key_device, dtype=torch.float32),
        seq,
        False,
        make((hidden, heads * head_dim), device=key_device, dtype=torch.int8),
        make((hidden, 1), device=key_device, dtype=torch.float32),
        None,
        64,
        None,
    )
    kwargs = {"head_dim": head_dim, "query_chunk_rows": query_chunk_rows}
    return args, kwargs, base


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("residual", [True, False])
def test_fold_dense_distributed_attention_output_fx(
    fake_triton_device: None,
    batch: int,
    residual: bool,
) -> None:
    with FakeTensorMode():
        graph = torch.fx.Graph()
        names = (
            "iq",
            "isc",
            "qw",
            "qs",
            "cos",
            "sin",
            "k0",
            "k1",
            "k2",
            "v0",
            "v1",
            "v2",
            "v3",
            "ow",
            "os",
        )
        p = {name: graph.placeholder(name) for name in names}
        full_val = torch.empty((batch, 256, 128), device=_DEVICE, dtype=torch.bfloat16)
        attn_out = graph.call_function(
            _ops.convrot_int8_piper_projected_query_attention_output.default,
            args=(
                p["iq"],
                p["isc"],
                p["qw"],
                p["qs"],
                None,
                p["cos"],
                p["sin"],
                1e-5,
                0.088,
                None,
                p["k0"],
                p["k1"],
                p["k2"],
                p["v0"],
                p["v1"],
                p["v2"],
                p["v3"],
                256,
                False,
                torch.bfloat16,
                p["ow"],
                p["os"],
                None,
                64,
                None,
            ),
            kwargs={"head_dim": 64},
        )
        attn_out.meta["val"] = full_val
        wait_node = _add_sequence_reduce_scatter(
            graph, attn_out, batch=batch, local_seq=128, hidden=128
        )
        if residual:
            base = graph.placeholder("base")
            gate = graph.placeholder("gate")
            idx = graph.placeholder("idx")
            base_val = torch.empty((batch, 128, 128), device=_DEVICE, dtype=torch.bfloat16)
            base.meta["val"] = base_val
            call = graph.call_function(
                _AUTO_FUNCTIONALIZED,
                args=(_residual_add_op(),),
                kwargs={
                    "base": base,
                    "update": wait_node,
                    "gate": gate,
                    "gate_indices": idx,
                    "python_indexing": False,
                },
            )
            call.meta["val"] = (None, base_val)
            out_node = graph.call_function(operator.getitem, args=(call, 1))
            out_node.meta["val"] = base_val
        else:
            out_node = wait_node
        graph.output(out_node)

        _run_post_grad_pass(graph, attention_output=True)
        graph.lint()
        assert not distributed_fx.has_c10d_collectives(graph)
        targets = [n.target for n in graph.nodes if n.op == "call_function"]
        assert _ops.convrot_int8_piper_projected_query_attention_output.default not in targets
        if residual:
            assert any(
                n.target is _AUTO_FUNCTIONALIZED and n.args and n.args[0] is _DENSE_RS_RESIDUAL_ADD
                for n in graph.nodes
            )
        else:
            assert (
                _ops.convrot_int8_piper_projected_query_attention_reduce_scatter.default in targets
            )


def test_fold_dense_distributed_attention_output_fx_negative(fake_triton_device: None) -> None:
    with FakeTensorMode():
        # Non-exclusive producer rejects folding.
        g_shared = torch.fx.Graph()
        p = g_shared.placeholder("p")
        attn_out = g_shared.call_function(
            _ops.convrot_int8_piper_projected_query_attention_output.default,
            args=tuple([p] * 25),
            kwargs={"head_dim": 64},
        )
        attn_out.meta["val"] = torch.empty((1, 256, 128), device=_DEVICE, dtype=torch.bfloat16)
        wait_node = _add_sequence_reduce_scatter(
            g_shared, attn_out, batch=1, local_seq=128, hidden=128
        )
        extra = g_shared.call_function(_aten.neg.default, args=(attn_out,))
        g_shared.output((wait_node, extra))
        _run_post_grad_pass(g_shared, attention_output=True)
        assert distributed_fx.has_c10d_collectives(g_shared)
        assert _ops.convrot_int8_piper_projected_query_attention_output.default in [
            n.target for n in g_shared.nodes if n.op == "call_function"
        ]

        # Residual dtype mismatch rejects folding into residual_add_.
        g_dtype = torch.fx.Graph()
        arg_nodes = [g_dtype.placeholder(f"p{idx}") for idx in range(25)]
        arg_nodes[19] = torch.bfloat16  # type: ignore[assignment]
        mismatch_out = g_dtype.call_function(
            _ops.convrot_int8_piper_projected_query_attention_output.default,
            args=tuple(arg_nodes),
            kwargs={"head_dim": 64},
        )
        mismatch_out.meta["val"] = torch.empty((1, 256, 128), device=_DEVICE, dtype=torch.bfloat16)
        wait_mismatch = _add_sequence_reduce_scatter(
            g_dtype, mismatch_out, batch=1, local_seq=128, hidden=128
        )
        base = g_dtype.placeholder("base")
        base.meta["val"] = torch.empty((1, 128, 128), device=_DEVICE, dtype=torch.float16)
        call = g_dtype.call_function(
            _AUTO_FUNCTIONALIZED,
            args=(_residual_add_op(),),
            kwargs={
                "base": base,
                "update": wait_mismatch,
                "gate": arg_nodes[0],
                "gate_indices": arg_nodes[1],
                "python_indexing": False,
            },
        )
        out = g_dtype.call_function(operator.getitem, args=(call, 1))
        g_dtype.output(out)
        # Residual fold skips on dtype mismatch, then standalone fold rewrites the reduce-scatter.
        _run_post_grad_pass(g_dtype, attention_output=True)
        assert not distributed_fx.has_c10d_collectives(g_dtype)
        targets = [n.target for n in g_dtype.nodes if n.op == "call_function"]
        assert _ops.convrot_int8_piper_projected_query_attention_reduce_scatter.default in targets
        assert not any(
            n.target is _AUTO_FUNCTIONALIZED and n.args and n.args[0] is _DENSE_RS_RESIDUAL_ADD
            for n in g_dtype.nodes
        )


def test_dense_distributed_output_validation_and_fake() -> None:
    with FakeTensorMode():
        args, kwargs, valid_base = _make_dense_distributed_output_meta_args()
        assert (
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *args, **kwargs
            )
            is None
        )
        rs_args = (*args[4:25], torch.bfloat16, *args[25:])
        standalone = dense_distributed_output._projected_query_attention_reduce_scatter_op_fake(
            *rs_args, **kwargs
        )
        assert standalone.shape == (1, 128, 128)
        assert standalone.dtype is torch.bfloat16

        # Batch == 0 validates metadata cleanly.
        empty_args, empty_kw, _ = _make_dense_distributed_output_meta_args(batch=0)
        assert (
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *empty_args, **empty_kw
            )
            is None
        )
        empty_rs = (*empty_args[4:25], torch.bfloat16, *empty_args[25:])
        assert dense_distributed_output._projected_query_attention_reduce_scatter_op_fake(
            *empty_rs, **empty_kw
        ).shape == (0, 128, 128)

        wrong_rows_args, wrong_kw, _ = _make_dense_distributed_output_meta_args(local_seq=256)
        with pytest.raises(ValueError, match="local rank share"):
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *wrong_rows_args, **wrong_kw
            )

        wrong_dev_args, dev_kw, _ = _make_dense_distributed_output_meta_args(
            key_device="meta", base_device="cpu"
        )
        with pytest.raises(ValueError, match="storage device"):
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *wrong_dev_args, **dev_kw
            )

        bad_chunk_args, bad_chunk_kw, _ = _make_dense_distributed_output_meta_args(
            query_chunk_rows=64
        )
        with pytest.raises(ValueError, match="multiples of 128"):
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *bad_chunk_args, **bad_chunk_kw
            )

        bad_rs_args = (1, "tp_0", *rs_args[2:])
        with pytest.raises(ValueError, match="comm_group_size >= 2"):
            dense_distributed_output._projected_query_attention_reduce_scatter_op_fake(
                *bad_rs_args, **kwargs
            )
        odd_args, odd_kw, _ = _make_dense_distributed_output_meta_args(seq=257, local_seq=128)
        with pytest.raises(ValueError, match="divide evenly"):
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op_fake(
                *odd_args, **odd_kw
            )
        assert valid_base.shape == (1, 128, 128)


def test_dense_distributed_output_pre_mutation_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeGroup:
        def rank(self) -> int:
            return 0

        def size(self) -> int:
            return 2

    comm_calls = 0

    def fail_comm(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal comm_calls
        comm_calls += 1
        raise AssertionError("collective launched despite invalid metadata or empty batch")

    monkeypatch.setattr(distributed_comm, "resolve_process_group", lambda _name: _FakeGroup())
    monkeypatch.setattr(dist, "all_gather", fail_comm)
    monkeypatch.setattr(dist, "reduce_scatter", fail_comm)
    monkeypatch.setattr(dist, "reduce_scatter_tensor", fail_comm)
    monkeypatch.setattr(
        convrot_linear_backend, "require_linear_backend", lambda *_a, **_kw: object()
    )
    monkeypatch.setattr(
        dense_fusion_backend, "require_projection_backend", lambda *_a, **_kw: object()
    )

    cases = [
        ("softmax scale", {"softmax_scale": 0.0}),
        ("multiples of 128", {"query_chunk_rows": 64}),
        ("storage device", {"key_device": "meta", "base_device": "cpu"}),
        ("local rank share", {"local_seq": 64}),
    ]
    for pattern, overrides in cases:
        make_kwargs: dict[str, Any] = {
            "key_device": "cpu",
            "base_device": "cpu",
            "fill_tensors": True,
            **overrides,
        }
        args, kwargs, base = _make_dense_distributed_output_meta_args(**make_kwargs)
        if base.device.type == "cpu":
            torch.manual_seed(17)
            base.copy_(torch.randn_like(base))
        before = base.clone()
        with pytest.raises(ValueError, match=pattern):
            dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op(
                *args, **kwargs
            )
        assert comm_calls == 0
        assert torch.equal(base, before)

    mismatch_args, mismatch_kw, mismatch_base = _make_dense_distributed_output_meta_args(
        key_device="cpu", base_device="cpu", seq=256, local_seq=64, fill_tensors=True
    )
    torch.manual_seed(19)
    mismatch_base.copy_(torch.randn_like(mismatch_base))
    before_mismatch = mismatch_base.clone()
    with pytest.raises(ValueError, match="must match resolved group"):
        dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op(
            *mismatch_args[:4], 4, *mismatch_args[5:], **mismatch_kw
        )
    assert comm_calls == 0
    assert torch.equal(mismatch_base, before_mismatch)

    # Batch == 0 returns cleanly without launching collectives.
    empty_args, empty_kw, empty_base = _make_dense_distributed_output_meta_args(
        key_device="cpu", base_device="cpu", batch=0, fill_tensors=True
    )
    assert (
        dense_distributed_output._projected_query_attention_reduce_scatter_residual_add_op(
            *empty_args, **empty_kw
        )
        is None
    )
    empty_rs_args = (*empty_args[4:25], torch.bfloat16, *empty_args[25:])
    empty_out = dense_distributed_output._projected_query_attention_reduce_scatter_op(
        *empty_rs_args, **empty_kw
    )
    assert empty_out.shape == (0, 128, 128)
    assert empty_base.shape == (0, 128, 128)
    assert comm_calls == 0


# --- Live 2-GPU end-to-end tests ---

_LIVE_HIDDEN = 512
_LIVE_HEADS = 4
_LIVE_HEAD_DIM = 128
_LIVE_ROTARY_DIM = 64
_LIVE_INTERMEDIATE = 1024
_LIVE_MAX_SEQ = 1024


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)


_ATTN_COLWISE = frozenset({"attn.to_q", "attn.to_k", "attn.to_v"})
_ATTN_ROWWISE = frozenset({"attn.to_out.0"})


class _RotaryAttn(nn.Module):
    def __init__(
        self,
        heads: int,
        *,
        use_projected_gate: bool = False,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = _LIVE_HEAD_DIM
        self.rotary_dim = _LIVE_ROTARY_DIM
        self.to_q = nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
        self.to_k = nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
        self.to_v = nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
        self.to_gate = (
            nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
            if use_projected_gate
            else None
        )
        self.norm_q = _RMSNorm(self.head_dim, dtype=dtype)
        self.norm_k = _RMSNorm(self.head_dim, dtype=dtype)
        self.to_out = nn.ModuleList(
            [nn.Linear(heads * self.head_dim, _LIVE_HIDDEN, bias=bias, dtype=dtype)]
        )
        ratios = tuple(0.5 if idx % 2 == 0 else 1.0 for idx in range(heads))
        self.sparse_attention = SparsePiperAttention(ratios, routing="mean")

    def _norm_rope(
        self, proj: torch.Tensor, norm: _RMSNorm, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = proj.shape
        x = norm(proj.view(batch, seq, self.heads, self.head_dim))
        rotary = x[..., : self.rotary_dim]
        half1, half2 = rotary.chunk(2, dim=-1)
        rot_cat = torch.cat((-half2, half1), dim=-1)
        c_slice = cos.to(proj.dtype)[None, :, None, :]
        s_slice = sin.to(proj.dtype)[None, :, None, :]
        rotary = rotary * c_slice + rot_cat * s_slice
        return torch.cat((rotary, x[..., self.rotary_dim :]), dim=-1).contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        cos_seq = cos[:seq]
        sin_seq = sin[:seq]
        q = self._norm_rope(self.to_q(hidden), self.norm_q, cos_seq, sin_seq)
        k = self._norm_rope(self.to_k(hidden), self.norm_k, cos_seq, sin_seq)
        v = self.to_v(hidden).view(batch, seq, self.heads, self.head_dim)
        attn = self.sparse_attention(
            q,
            k,
            v,
            sparse_key_blocks=2,
            block_lengths=block_lengths,
        )
        active_gate = (
            self.to_gate(hidden).view(batch, seq, self.heads, self.head_dim)
            if self.to_gate is not None
            else coarse_gate
        )
        if active_gate is not None:
            attn = attn + sparse_piper_coarse_residual(
                q,
                k,
                v,
                active_gate,
                routing="mean",
                coarse_key_blocks=2,
                coarse_scale=0.125,
                block_lengths=block_lengths,
            )
        return self.to_out[0](attn.reshape(batch, seq, self.heads * self.head_dim))


class _SwiGLUFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            _LIVE_HIDDEN, _LIVE_INTERMEDIATE, bias=False, dtype=torch.bfloat16
        )
        self.value_proj = nn.Linear(
            _LIVE_HIDDEN, _LIVE_INTERMEDIATE, bias=False, dtype=torch.bfloat16
        )
        self.down_proj = nn.Linear(
            _LIVE_INTERMEDIATE, _LIVE_HIDDEN, bias=False, dtype=torch.bfloat16
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.value_proj(hidden))


class _LiveH3Block(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.attn = _RotaryAttn(heads)
        self.ff = _SwiGLUFFN()
        self.attn_gates = nn.Parameter(
            torch.full((4, _LIVE_HIDDEN), 0.5, dtype=torch.bfloat16), requires_grad=False
        )
        self.ffn_gates = nn.Parameter(
            torch.full((4, _LIVE_HIDDEN), 0.5, dtype=torch.bfloat16), requires_grad=False
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        gate_indices: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.attn_gates[gate_indices] * self.attn(
            hidden, cos, sin, block_lengths, coarse_gate
        )
        hidden = hidden + self.ffn_gates[gate_indices] * self.ff(hidden)
        return hidden


def _quantize_module_linears(
    module: nn.Module,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    *,
    biases: dict[str, torch.Tensor] | None = None,
    mesh: DeviceMesh | None = None,
    rank: int = 0,
    world: int = 1,
    device: torch.device,
    colwise_names: AbstractSet[str] = frozenset(),
    rowwise_names: AbstractSet[str] = frozenset(),
) -> None:
    for name, raw in weights.items():
        quant_full = ConvRotInt8Tensor.from_hp(raw.to(device), group_size=64)
        sub = module.get_submodule(name)
        assert isinstance(sub, nn.Linear)
        if mesh is not None and name in colwise_names:
            length = raw.shape[0] // world
            sharded = shard_quantized_weight(quant_full, dim=0, start=rank * length, length=length)
            param: torch.Tensor = DTensor.from_local(sharded, mesh, [Shard(0)], run_check=False)
        elif mesh is not None and name in rowwise_names:
            length = raw.shape[1] // world
            sharded = shard_quantized_weight(quant_full, dim=1, start=rank * length, length=length)
            param = DTensor.from_local(sharded, mesh, [Shard(1)], run_check=False)
        else:
            param = quant_full
        sub.weight = nn.Parameter(param, requires_grad=False)

        if biases is not None and name in biases:
            raw_bias = biases[name].to(device)
            if mesh is not None and name in colwise_names:
                length = raw_bias.shape[0] // world
                local_bias = raw_bias.narrow(0, rank * length, length).contiguous()
                bias_param: torch.Tensor = DTensor.from_local(
                    local_bias, mesh, [Shard(0)], run_check=False
                )
            elif mesh is not None and name in rowwise_names:
                bias_param = DTensor.from_local(raw_bias, mesh, [Replicate()], run_check=False)
            else:
                bias_param = raw_bias
            sub.bias = nn.Parameter(bias_param, requires_grad=False)

    for name, norm_w in norms.items():
        sub = module.get_submodule(name)
        assert hasattr(sub, "weight")
        sub.weight = nn.Parameter(norm_w.to(device), requires_grad=False)


def _parallelize_attention_module(
    module: nn.Module,
    mesh: DeviceMesh,
    *,
    include_gate: bool = False,
) -> None:
    plan: dict[str, Any] = {
        "attn": PrepareModuleInput(
            input_layouts=(Shard(1), None, None, None, None),
            desired_input_layouts=(Replicate(), None, None, None, None),
            use_local_output=True,
        ),
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
    }
    if include_gate:
        plan["attn.to_gate"] = ColwiseParallel()
    parallelize_module(module, mesh, plan, src_data_rank=None)


@contextmanager
def _count_distributed_paths() -> Iterator[dict[str, int]]:
    counts = {"peer_exchange": 0, "fallback": 0}
    orig_peer = peer_exchange.run_peer_chunk_exchange
    orig_fallback = peer_exchange.run_distributed_fallback

    def wrapped_peer(*args: Any, **kwargs: Any) -> Any:
        counts["peer_exchange"] += 1
        return orig_peer(*args, **kwargs)

    def wrapped_fallback(*args: Any, **kwargs: Any) -> Any:
        counts["fallback"] += 1
        return orig_fallback(*args, **kwargs)

    peer_exchange.run_peer_chunk_exchange = wrapped_peer  # type: ignore[assignment]
    peer_exchange.run_distributed_fallback = wrapped_fallback  # type: ignore[assignment]
    try:
        yield counts
    finally:
        peer_exchange.run_peer_chunk_exchange = orig_peer  # type: ignore[assignment]
        peer_exchange.run_distributed_fallback = orig_fallback  # type: ignore[assignment]


_KV_GATHER_OP = "piper_kernels.convrot_int8_sparse_piper_prepare_gather_project_kv.default"
_ATTN_RS_OP = (
    "piper_kernels.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter.default"
)
_ATTN_RS_RESIDUAL_OP = (
    "piper_kernels."
    "convrot_int8_sparse_piper_projected_query_attention_reduce_scatter_residual_add_.default"
)


def _build_compile_options(capture: FunctionalTargetCapturePass) -> dict[str, Any]:
    opts = compile_options(ffn=True, attention=True, residual_updates=True)
    return capture.wrap_options(opts)


def _gather_seq(local: torch.Tensor, world: int, pg: dist.ProcessGroup) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local.contiguous(), group=pg)
    return torch.cat(parts, dim=1)


def _check_quantized_all_gather_sibling_linears(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    class _SiblingPair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_a = nn.Linear(_LIVE_HIDDEN, _LIVE_HIDDEN, bias=True, dtype=torch.bfloat16)
            self.proj_b = nn.Linear(_LIVE_HIDDEN, _LIVE_HIDDEN, bias=True, dtype=torch.bfloat16)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.proj_a(x), self.proj_b(x)

    class _WrapPair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pair = _SiblingPair()

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.pair(x)

    torch.manual_seed(77)
    pair_weights = {
        "pair.proj_a": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.bfloat16) * 0.02,
        "pair.proj_b": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.bfloat16) * 0.02,
    }
    pair_biases = {
        "pair.proj_a": torch.randn(_LIVE_HIDDEN, dtype=torch.bfloat16) * 0.05,
        "pair.proj_b": torch.randn(_LIVE_HIDDEN, dtype=torch.bfloat16) * 0.05,
    }

    dist_mod = _WrapPair().to(device).eval()
    _quantize_module_linears(
        dist_mod,
        pair_weights,
        {},
        biases=pair_biases,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=frozenset({"pair.proj_a", "pair.proj_b"}),
    )
    parallelize_module(
        dist_mod,
        mesh,
        {
            "pair": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_output=True,
            ),
            "pair.proj_a": ColwiseParallel(),
            "pair.proj_b": ColwiseParallel(),
        },
        src_data_rank=None,
    )
    capture = FunctionalTargetCapturePass()
    compiled_dist = torch.compile(dist_mod, options=capture.wrap_options(compile_options()))

    ref_fn = None
    if rank == 0:
        ref_mod = _WrapPair().to(device).eval()
        _quantize_module_linears(ref_mod, pair_weights, {}, biases=pair_biases, device=device)
        ref_fn = torch.compile(ref_mod, options=compile_options())

    seq = 256
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(88)
    full_x = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    out_a, out_b = compiled_dist(full_x[:, mine].contiguous().to(device))
    parts_a = [torch.empty_like(out_a) for _ in range(world)]
    parts_b = [torch.empty_like(out_b) for _ in range(world)]
    dist.all_gather(parts_a, out_a.contiguous(), group=pg)
    dist.all_gather(parts_b, out_b.contiguous(), group=pg)
    if rank == 0:
        assert ref_fn is not None
        ref_a, ref_b = ref_fn(full_x.to(device))
        assert _rel_l2(torch.cat(parts_a, dim=-1), ref_a) < 0.005
        assert _rel_l2(torch.cat(parts_b, dim=-1), ref_b) < 0.005

    assert capture.calls == 1
    assert "piper_kernels.convrot_int8_prepare_input.default" in capture.target_names
    assert "piper_kernels.convrot_int8_linear_prepared.default" in capture.target_names
    assert "_c10d_functional.all_gather_into_tensor_coalesced.default" in capture.target_names
    assert "_c10d_functional.all_gather_into_tensor.default" not in capture.target_names


def _check_dynamic_tpsp_blocks(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
    ref_fn: Any,
) -> None:
    block = _LiveH3Block(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        block,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(block, mesh)
    capture = FunctionalTargetCapturePass()
    opts = _build_compile_options(capture)

    def _dist_step(
        h: torch.Tensor,
        c: torch.Tensor,
        s: torch.Tensor,
        gi: torch.Tensor,
        bl: torch.Tensor | None,
        cg: torch.Tensor | None,
        _blk: nn.Module = block,
    ) -> torch.Tensor:
        return h.copy_(_blk(h, c, s, gi, bl, cg))

    compiled = torch.compile(_dist_step, dynamic=True, options=opts)

    with _count_distributed_paths() as path_counts:
        # Keep cos_all/sin_all shape fixed; only sequence dims are dynamic -> capture.calls == 1.
        for seq in (1024, 512):
            torch.manual_seed(100 + seq)
            full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
            full_gi = torch.arange(seq, dtype=torch.int64) % 4
            local_seq = seq // world
            mine = slice(rank * local_seq, (rank + 1) * local_seq)

            local_h = full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device)
            local_gi = full_gi[mine].contiguous().to(device)
            torch._dynamo.mark_dynamic(local_h, 1)
            torch._dynamo.mark_dynamic(local_gi, 0)
            compiled(local_h, cos_all, sin_all, local_gi, None, None)
            gathered = _gather_seq(local_h, world, pg)

            if rank == 0:
                assert ref_fn is not None
                ref_h = full_h.clone().to(device)
                ref_gi = full_gi.clone().to(device)
                torch._dynamo.mark_dynamic(ref_h, 1)
                torch._dynamo.mark_dynamic(ref_gi, 0)
                ref_fn(ref_h, cos_all, sin_all, ref_gi, None, None)
                assert _rel_l2(gathered, ref_h) < 0.005, (
                    f"hybrid seq={seq} rel_l2={_rel_l2(gathered, ref_h)}"
                )

    assert path_counts == {"peer_exchange": 2, "fallback": 0}
    assert capture.calls == 1, f"hybrid recompiled: calls={capture.calls}"
    assert not any(name.startswith("_c10d_functional.") for name in capture.target_names), (
        f"hybrid leftover collectives: {capture.target_names}"
    )
    assert _KV_GATHER_OP in capture.target_names, f"hybrid: {capture.target_names}"
    assert _ATTN_RS_RESIDUAL_OP in capture.target_names, f"hybrid: {capture.target_names}"
    assert "piper_kernels.convrot_int8_ffn_update_.default" in capture.target_names, (
        f"hybrid: {capture.target_names}"
    )


def _check_block_lengths(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
    ref_fn: Any,
) -> None:
    block = _LiveH3Block(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        block,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(block, mesh)
    capture = FunctionalTargetCapturePass()
    compiled_block = torch.compile(
        lambda h, c, s, gi, bl, cg: h.copy_(block(h, c, s, gi, bl, cg)),
        dynamic=True,
        options=_build_compile_options(capture),
    )

    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(202)
    full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    full_gi = torch.arange(seq, dtype=torch.int64) % 4
    bl_full = torch.tensor([64, 64, 64, 64, 64, 64, 64, 52], device=device, dtype=torch.int32)
    lh = full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device)
    lgi = full_gi[mine].contiguous().to(device)
    with _count_distributed_paths() as path_counts:
        compiled_block(lh, cos_all, sin_all, lgi, bl_full, None)
    gathered = _gather_seq(lh, world, pg)
    if rank == 0:
        assert ref_fn is not None
        rh = full_h.clone().to(device)
        rgi = full_gi.clone().to(device)
        ref_fn(rh, cos_all, sin_all, rgi, bl_full, None)
        assert _rel_l2(gathered, rh) < 0.005
    assert path_counts == {"peer_exchange": 1, "fallback": 0}
    assert capture.calls == 1
    assert not any(name.startswith("_c10d_functional.") for name in capture.target_names)
    assert _KV_GATHER_OP in capture.target_names
    assert _ATTN_RS_RESIDUAL_OP in capture.target_names


def _check_batched(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
    ref_fn: Any,
) -> None:
    block = _LiveH3Block(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        block,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(block, mesh)
    capture = FunctionalTargetCapturePass()
    compiled_block = torch.compile(
        lambda h, c, s, gi, bl, cg: h.copy_(block(h, c, s, gi, bl, cg)),
        dynamic=True,
        options=_build_compile_options(capture),
    )

    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(303)
    full_h_b2 = torch.randn(2, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    full_gi_b2 = torch.arange(seq, dtype=torch.int64) % 4
    lh_b2 = full_h_b2[:, mine].clone(memory_format=torch.contiguous_format).to(device)
    lgi_b2 = full_gi_b2[mine].contiguous().to(device)
    with _count_distributed_paths() as path_counts:
        compiled_block(lh_b2, cos_all, sin_all, lgi_b2, None, None)
    gathered_b2 = _gather_seq(lh_b2, world, pg)
    if rank == 0:
        assert ref_fn is not None
        rh_b2 = full_h_b2.clone().to(device)
        ref_fn(rh_b2, cos_all, sin_all, full_gi_b2.clone().to(device), None, None)
        assert _rel_l2(gathered_b2, rh_b2) < 0.005
    assert path_counts == {"peer_exchange": 1, "fallback": 0}
    assert capture.calls == 1
    assert not any(name.startswith("_c10d_functional.") for name in capture.target_names)
    assert _KV_GATHER_OP in capture.target_names
    # 1D sequence-length `gate_indices` broadcast over batch=2 intentionally lowers without
    # `residual_add_`, exercising batched standalone `projected_query_attention_reduce_scatter`.
    assert _ATTN_RS_OP in capture.target_names
    assert "piper_kernels.convrot_int8_ffn.default" in capture.target_names


def _check_unaligned_local_seq_fallback(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
    ref_fn: Any,
) -> None:
    # seq=192 -> local_seq=96 (% 64 != 0): exercises kv_gather and distributed_output fallbacks.
    block = _LiveH3Block(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        block,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(block, mesh)
    capture = FunctionalTargetCapturePass()
    compiled_block = torch.compile(
        lambda h, c, s, gi, bl, cg: h.copy_(block(h, c, s, gi, bl, cg)),
        options=_build_compile_options(capture),
    )
    seq = 192
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(606)
    full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    full_gi = torch.arange(seq, dtype=torch.int64) % 4
    lh = full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device)
    lgi = full_gi[mine].contiguous().to(device)
    with _count_distributed_paths() as path_counts:
        compiled_block(lh, cos_all, sin_all, lgi, None, None)
    gathered = _gather_seq(lh, world, pg)
    if rank == 0:
        assert ref_fn is not None
        rh = full_h.clone().to(device)
        ref_fn(rh, cos_all, sin_all, full_gi.clone().to(device), None, None)
        assert _rel_l2(gathered, rh) < 0.005
    assert path_counts == {"peer_exchange": 0, "fallback": 1}
    assert capture.calls == 1
    assert not any(name.startswith("_c10d_functional.") for name in capture.target_names)
    assert _KV_GATHER_OP in capture.target_names
    assert _ATTN_RS_RESIDUAL_OP in capture.target_names


def _check_coarse(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
    ref_fn: Any,
) -> None:
    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)

    coarse_dist = _LiveH3Block(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        coarse_dist,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(coarse_dist, mesh)
    cap_coarse = FunctionalTargetCapturePass()
    compiled_coarse = torch.compile(
        lambda h, c, s, gi, bl, cg: h.copy_(coarse_dist(h, c, s, gi, bl, cg)),
        options=_build_compile_options(cap_coarse),
    )
    torch.manual_seed(404)
    ch = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    cgi = torch.arange(seq, dtype=torch.int64) % 4
    coarse_g = torch.sigmoid(
        torch.randn(1, seq, _LIVE_HEADS, _LIVE_HEAD_DIM, dtype=torch.bfloat16)
    ).to(device)
    l_cg = coarse_g[
        :, :, rank * (_LIVE_HEADS // world) : (rank + 1) * (_LIVE_HEADS // world)
    ].contiguous()
    l_ch = ch[:, mine].clone(memory_format=torch.contiguous_format).to(device)
    l_cgi = cgi[mine].contiguous().to(device)
    with _count_distributed_paths() as coarse_counts:
        compiled_coarse(l_ch, cos_all, sin_all, l_cgi, None, l_cg)
    gathered_c = _gather_seq(l_ch, world, pg)
    if rank == 0:
        assert ref_fn is not None
        rc = ch.clone().to(device)
        ref_fn(rc, cos_all, sin_all, cgi.clone().to(device), None, coarse_g)
        assert _rel_l2(gathered_c, rc) < 0.005
    assert coarse_counts == {"peer_exchange": 1, "fallback": 0}
    assert cap_coarse.calls == 1
    assert not any(name.startswith("_c10d_functional.") for name in cap_coarse.target_names)
    assert _KV_GATHER_OP in cap_coarse.target_names
    assert _ATTN_RS_RESIDUAL_OP in cap_coarse.target_names


def _check_standalone(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)

    class _WrapStandalone(nn.Module):
        def __init__(self, heads: int) -> None:
            super().__init__()
            self.attn = _RotaryAttn(heads)
            self.ff = _SwiGLUFFN()

        def forward(self, h: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            a = self.attn(h, c, s, None, None)
            return a + self.ff(a)

    standalone = _WrapStandalone(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        standalone,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(standalone, mesh)
    cap_sa = FunctionalTargetCapturePass()
    compiled_sa = torch.compile(
        standalone,
        options=cap_sa.wrap_options(compile_options(ffn=True, attention=True)),
    )
    torch.manual_seed(505)
    sa_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    with _count_distributed_paths() as sa_counts:
        sa_out = compiled_sa(
            sa_h[:, mine].clone(memory_format=torch.contiguous_format).to(device),
            cos_all,
            sin_all,
        )
    gathered_sa = _gather_seq(sa_out, world, pg)
    if rank == 0:
        sa_ref = _WrapStandalone(_LIVE_HEADS).to(device).eval()
        _quantize_module_linears(sa_ref, weights, norms, device=device)
        sa_ref_opts = compile_options(ffn=True, attention=True)
        compiled_sa_ref = torch.compile(sa_ref, options=sa_ref_opts)
        ref_sa_out = compiled_sa_ref(sa_h.to(device), cos_all, sin_all)
        rel = _rel_l2(gathered_sa, ref_sa_out)
        assert rel < 0.02, f"standalone rel_l2={rel}"
    assert sa_counts == {"peer_exchange": 1, "fallback": 0}
    assert cap_sa.calls == 1
    assert not any(name.startswith("_c10d_functional.") for name in cap_sa.target_names)
    assert _KV_GATHER_OP in cap_sa.target_names
    assert _ATTN_RS_OP in cap_sa.target_names
    assert "piper_kernels.convrot_int8_ffn.default" in cap_sa.target_names


def _check_bias_fp16_and_projected_gate(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    class _WrapAttnBiasGate(nn.Module):
        def __init__(self, heads: int) -> None:
            super().__init__()
            self.attn = _RotaryAttn(
                heads,
                use_projected_gate=True,
                bias=True,
                dtype=torch.float16,
            )

        def forward(
            self,
            h: torch.Tensor,
            c: torch.Tensor,
            s: torch.Tensor,
            bl: torch.Tensor | None = None,
            cg: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.attn(h, c, s, bl, cg)

    torch.manual_seed(909)
    scale = 0.02
    fp16_weights = {
        "attn.to_q": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_k": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_v": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_gate": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_out.0": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
    }
    fp16_biases = {
        name: torch.randn(_LIVE_HIDDEN, dtype=torch.float16) * 0.05 for name in fp16_weights
    }
    fp16_norms = {
        "attn.norm_q": torch.rand(_LIVE_HEAD_DIM, dtype=torch.float16).add_(0.5),
        "attn.norm_k": torch.rand(_LIVE_HEAD_DIM, dtype=torch.float16).add_(0.5),
    }

    dist_mod = _WrapAttnBiasGate(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        dist_mod,
        fp16_weights,
        fp16_norms,
        biases=fp16_biases,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=frozenset({*_ATTN_COLWISE, "attn.to_gate"}),
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_attention_module(dist_mod, mesh, include_gate=True)
    capture = FunctionalTargetCapturePass()
    compiled_dist = torch.compile(
        dist_mod,
        options=capture.wrap_options(compile_options(attention=True)),
    )

    ref_fn = None
    if rank == 0:
        ref_mod = _WrapAttnBiasGate(_LIVE_HEADS).to(device).eval()
        _quantize_module_linears(
            ref_mod, fp16_weights, fp16_norms, biases=fp16_biases, device=device
        )
        ref_fn = torch.compile(
            ref_mod,
            options=compile_options(attention=True),
        )

    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(910)
    full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.float16)
    with _count_distributed_paths() as path_counts:
        dist_out = compiled_dist(full_h[:, mine].contiguous().to(device), cos_all, sin_all)
    gathered = _gather_seq(dist_out, world, pg)
    if rank == 0:
        assert ref_fn is not None
        ref_out = ref_fn(full_h.to(device), cos_all, sin_all)
        rel = _rel_l2(gathered, ref_out)
        assert rel < 0.02, f"fp16+bias+projected_gate rel_l2={rel}"
    assert path_counts == {"peer_exchange": 1, "fallback": 0}
    assert capture.calls == 1
    assert not any(name.startswith("_c10d_functional.") for name in capture.target_names)
    assert _KV_GATHER_OP in capture.target_names
    assert _ATTN_RS_OP in capture.target_names
    # Assert `to_gate` folded into the attention output op rather than staying separate.
    assert "piper_kernels.convrot_int8_linear.default" not in capture.target_names
    assert "piper_kernels.convrot_int8_linear_prepared.default" not in capture.target_names


def _prepare_sharded_attn_weights(
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
) -> tuple[
    tuple[ConvRotInt8Tensor, ConvRotInt8Tensor, ConvRotInt8Tensor, ConvRotInt8Tensor],
    tuple[ConvRotInt8Tensor, ConvRotInt8Tensor, ConvRotInt8Tensor, ConvRotInt8Tensor],
]:
    head_len = (_LIVE_HEADS // world) * _LIVE_HEAD_DIM
    full = (
        ConvRotInt8Tensor.from_hp(weights["attn.to_q"].to(device), group_size=64),
        ConvRotInt8Tensor.from_hp(weights["attn.to_k"].to(device), group_size=64),
        ConvRotInt8Tensor.from_hp(weights["attn.to_v"].to(device), group_size=64),
        ConvRotInt8Tensor.from_hp(weights["attn.to_out.0"].to(device), group_size=64),
    )
    local = (
        shard_quantized_weight(full[0], dim=0, start=rank * head_len, length=head_len),
        shard_quantized_weight(full[1], dim=0, start=rank * head_len, length=head_len),
        shard_quantized_weight(full[2], dim=0, start=rank * head_len, length=head_len),
        shard_quantized_weight(full[3], dim=1, start=rank * head_len, length=head_len),
    )
    return full, local


def _check_direct_multichunk_and_inplace_ownership(
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    batch, seq = 2, 256
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    (q_full, k_full, v_full, o_full), (q_local, k_local, v_local, o_local) = (
        _prepare_sharded_attn_weights(rank, world, device, weights)
    )
    q_norm, k_norm = norms["attn.norm_q"].to(device), norms["attn.norm_k"].to(device)

    torch.manual_seed(707)
    full_h = torch.randn(batch, seq, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    full_base = torch.randn(batch, seq, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    gate = torch.randn(4, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    full_gi_2d = (torch.arange(batch * seq, device=device, dtype=torch.int64) % 4).view(batch, seq)
    cos_seq, sin_seq = cos_all[:seq].contiguous(), sin_all[:seq].contiguous()

    qdata_2d, scale_1d = _ops.convrot_int8_prepare_input.default(
        full_h.view(batch * seq, _LIVE_HIDDEN), 64, None
    )
    full_qdata, full_scale = qdata_2d.view(batch, seq, _LIVE_HIDDEN), scale_1d.view(batch, seq)
    k_out = _ops.convrot_int8_sparse_piper_project_key.default(
        full_qdata, full_scale, k_local.qdata, k_local.scale, k_norm, cos_seq, sin_seq, 1e-5, 0
    )
    v_format = int(_quantized_dispatch._backend.require_attention_backend(k_out[0]).value_format)
    v_out = _ops.convrot_int8_sparse_piper_project_value.default(
        full_qdata, full_scale, v_local.qdata, v_local.scale, value_format=v_format
    )

    local_base = full_base[:, mine].clone(memory_format=torch.contiguous_format)
    local_gi = full_gi_2d[:, mine].reshape(-1).contiguous()
    base_ptr = local_base.data_ptr()
    local_profile = _SparsePiperRatioProfile((500_000, 1_000_000))
    common_args = (
        full_qdata,
        full_scale,
        q_local.qdata,
        q_local.scale,
        q_norm,
        cos_seq,
        sin_seq,
        1e-5,
        _LIVE_HEAD_DIM**-0.5,
        *k_out,
        *v_out,
        local_profile,
        2,
        seq,
        0,
        o_local.qdata,
        o_local.scale,
        None,
        64,
        64,
    )

    exchange_calls = 0
    orig_start = distributed_comm.start_peer_exchange

    def counted_exchange(*args: Any, **kwargs: Any) -> Any:
        nonlocal exchange_calls
        exchange_calls += 1
        return orig_start(*args, **kwargs)

    distributed_comm.start_peer_exchange = counted_exchange
    try:
        with _count_distributed_paths() as path_counts:
            _ops.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter_residual_add_.default(
                local_base, gate, local_gi, False, world, pg.group_name, *common_args
            )
    finally:
        distributed_comm.start_peer_exchange = orig_start

    assert local_base.data_ptr() == base_ptr
    assert path_counts == {"peer_exchange": 1, "fallback": 0}
    assert exchange_calls == 2
    gathered_base = _gather_seq(local_base, world, pg)

    if rank == 0:
        ref_k_out = _ops.convrot_int8_sparse_piper_project_key.default(
            full_qdata, full_scale, k_full.qdata, k_full.scale, k_norm, cos_seq, sin_seq, 1e-5, 0
        )
        ref_v_out = _ops.convrot_int8_sparse_piper_project_value.default(
            full_qdata, full_scale, v_full.qdata, v_full.scale, value_format=v_format
        )
        ref_base = full_base.clone()
        full_profile = _SparsePiperRatioProfile((500_000, 1_000_000, 500_000, 1_000_000))
        _ops.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default(
            ref_base,
            gate,
            full_gi_2d.reshape(-1).contiguous(),
            False,
            full_qdata,
            full_scale,
            q_full.qdata,
            q_full.scale,
            q_norm,
            cos_seq,
            sin_seq,
            1e-5,
            _LIVE_HEAD_DIM**-0.5,
            *ref_k_out,
            *ref_v_out,
            full_profile,
            2,
            seq,
            0,
            o_full.qdata,
            o_full.scale,
            None,
            64,
            64,
        )
        assert _rel_l2(gathered_base, ref_base) < 0.005

    before_fail = local_base.clone()
    bad_args = (*common_args[:5], cos_seq[:-1], sin_seq[:-1], *common_args[7:])
    with pytest.raises(ValueError, match="RoPE"):
        _ops.convrot_int8_sparse_piper_projected_query_attention_reduce_scatter_residual_add_.default(
            local_base, gate, local_gi, False, world, pg.group_name, *bad_args
        )
    assert local_base.data_ptr() == base_ptr
    assert torch.equal(local_base, before_fail)


_DENSE_KEY_OP = "piper_kernels.convrot_int8_piper_project_key.default"
_DENSE_VALUE_OP = "piper_kernels.convrot_int8_piper_project_value.default"
_DENSE_ATTN_RS_OP = (
    "piper_kernels.convrot_int8_piper_projected_query_attention_reduce_scatter.default"
)
_DENSE_ATTN_RS_RESIDUAL_OP = (
    "piper_kernels."
    "convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_.default"
)


class _DenseRotaryAttn(nn.Module):
    def __init__(
        self,
        heads: int,
        *,
        is_causal: bool = False,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = _LIVE_HEAD_DIM
        self.rotary_dim = _LIVE_ROTARY_DIM
        self.is_causal = is_causal
        self.to_q = nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
        self.to_k = nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
        self.to_v = nn.Linear(_LIVE_HIDDEN, heads * self.head_dim, bias=bias, dtype=dtype)
        self.norm_q = _RMSNorm(self.head_dim, dtype=dtype)
        self.norm_k = _RMSNorm(self.head_dim, dtype=dtype)
        self.to_out = nn.ModuleList(
            [nn.Linear(heads * self.head_dim, _LIVE_HIDDEN, bias=bias, dtype=dtype)]
        )

    def _norm_rope(
        self, proj: torch.Tensor, norm: _RMSNorm, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = proj.shape
        x = norm(proj.view(batch, seq, self.heads, self.head_dim))
        rotary = x[..., : self.rotary_dim]
        half1, half2 = rotary.chunk(2, dim=-1)
        rot_cat = torch.cat((-half2, half1), dim=-1)
        c_slice = cos.to(proj.dtype)[None, :, None, :]
        s_slice = sin.to(proj.dtype)[None, :, None, :]
        rotary = rotary * c_slice + rot_cat * s_slice
        return torch.cat((rotary, x[..., self.rotary_dim :]), dim=-1).contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        cos_seq = cos[:seq]
        sin_seq = sin[:seq]
        q = self._norm_rope(self.to_q(hidden), self.norm_q, cos_seq, sin_seq)
        k = self._norm_rope(self.to_k(hidden), self.norm_k, cos_seq, sin_seq)
        v = self.to_v(hidden).view(batch, seq, self.heads, self.head_dim)
        attn = piper_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.is_causal,
        ).transpose(1, 2)
        return self.to_out[0](attn.reshape(batch, seq, self.heads * self.head_dim))


class _LiveDenseBlock(nn.Module):
    def __init__(self, heads: int, *, is_causal: bool = False) -> None:
        super().__init__()
        self.attn = _DenseRotaryAttn(heads, is_causal=is_causal)
        self.ff = _SwiGLUFFN()
        self.attn_gates = nn.Parameter(
            torch.full((4, _LIVE_HIDDEN), 0.5, dtype=torch.bfloat16), requires_grad=False
        )
        self.ffn_gates = nn.Parameter(
            torch.full((4, _LIVE_HIDDEN), 0.5, dtype=torch.bfloat16), requires_grad=False
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attn_gates[gate_indices] * self.attn(hidden, cos, sin)
        hidden = hidden + self.ffn_gates[gate_indices] * self.ff(hidden)
        return hidden


def _parallelize_dense_attention_module(module: nn.Module, mesh: DeviceMesh) -> None:
    plan: dict[str, Any] = {
        "attn": PrepareModuleInput(
            input_layouts=(Shard(1), None, None),
            desired_input_layouts=(Replicate(), None, None),
            use_local_output=True,
        ),
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
    }
    parallelize_module(module, mesh, plan, src_data_rank=None)


def _build_dense_compile_options(capture: FunctionalTargetCapturePass | None) -> dict[str, Any]:
    opts = compile_options(ffn=True, attention_output=True, residual_updates=True)
    return opts if capture is None else capture.wrap_options(opts)


def _compile_dense_block(
    mesh: DeviceMesh,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    *,
    is_causal: bool = False,
    dynamic: bool = False,
) -> tuple[Any, FunctionalTargetCapturePass]:
    block = _LiveDenseBlock(_LIVE_HEADS // world, is_causal=is_causal).to(device).eval()
    _quantize_module_linears(
        block,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_dense_attention_module(block, mesh)
    capture = FunctionalTargetCapturePass()
    compiled = torch.compile(
        lambda h, c, s, gi: h.copy_(block(h, c, s, gi)),
        dynamic=dynamic,
        options=_build_dense_compile_options(capture),
    )
    return compiled, capture


def _check_dense_dynamic_tpsp_blocks(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    compiled_dist, capture = _compile_dense_block(
        mesh, rank, world, device, weights, norms, is_causal=False, dynamic=True
    )
    ref_fn = None
    if rank == 0:
        ref_block = _LiveDenseBlock(_LIVE_HEADS, is_causal=False).to(device).eval()
        _quantize_module_linears(ref_block, weights, norms, device=device)
        ref_fn = torch.compile(
            lambda h, c, s, gi: h.copy_(ref_block(h, c, s, gi)),
            dynamic=True,
            options=_build_dense_compile_options(None),
        )

    for seq in (256, 512):
        local_seq = seq // world
        mine = slice(rank * local_seq, (rank + 1) * local_seq)
        torch.manual_seed(1100 + seq)
        full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
        full_gi = torch.arange(seq, dtype=torch.int64) % 4
        local_h = full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device)
        local_gi = full_gi[mine].contiguous().to(device)
        with _count_distributed_paths() as path_counts:
            compiled_dist(local_h, cos_all, sin_all, local_gi)
        gathered = _gather_seq(local_h, world, pg)
        if rank == 0:
            assert ref_fn is not None
            ref_h = full_h.clone().to(device)
            ref_fn(ref_h, cos_all, sin_all, full_gi.clone().to(device))
            assert _rel_l2(gathered, ref_h) < 0.005
        assert path_counts == {"peer_exchange": 1, "fallback": 0}

    assert capture.calls == 1
    assert "_c10d_functional.all_gather_into_tensor_coalesced.default" in capture.target_names
    assert "_c10d_functional.all_gather_into_tensor.default" not in capture.target_names
    assert "_c10d_functional.reduce_scatter_tensor.default" not in capture.target_names
    assert _DENSE_KEY_OP in capture.target_names
    assert _DENSE_VALUE_OP in capture.target_names
    assert _DENSE_ATTN_RS_RESIDUAL_OP in capture.target_names
    assert "piper_kernels.convrot_int8_ffn_update_.default" in capture.target_names


def _check_dense_causal_and_unaligned(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    compiled_causal, cap_causal = _compile_dense_block(
        mesh, rank, world, device, weights, norms, is_causal=True
    )
    compiled_unaligned, cap_unaligned = _compile_dense_block(
        mesh, rank, world, device, weights, norms, is_causal=False
    )
    ref_causal = None
    ref_noncausal = None
    if rank == 0:
        rc_block = _LiveDenseBlock(_LIVE_HEADS, is_causal=True).to(device).eval()
        _quantize_module_linears(rc_block, weights, norms, device=device)
        ref_causal = torch.compile(
            lambda h, c, s, gi: h.copy_(rc_block(h, c, s, gi)),
            options=_build_dense_compile_options(None),
        )
        rn_block = _LiveDenseBlock(_LIVE_HEADS, is_causal=False).to(device).eval()
        _quantize_module_linears(rn_block, weights, norms, device=device)
        ref_noncausal = torch.compile(
            lambda h, c, s, gi: h.copy_(rn_block(h, c, s, gi)),
            options=_build_dense_compile_options(None),
        )

    for seq, expected_paths, is_causal, fn, cap, ref_fn in (
        (512, {"peer_exchange": 1, "fallback": 0}, True, compiled_causal, cap_causal, ref_causal),
        (
            192,
            {"peer_exchange": 0, "fallback": 1},
            False,
            compiled_unaligned,
            cap_unaligned,
            ref_noncausal,
        ),
    ):
        local_seq = seq // world
        mine = slice(rank * local_seq, (rank + 1) * local_seq)
        torch.manual_seed(1200 + seq + int(is_causal))
        full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
        full_gi = torch.arange(seq, dtype=torch.int64) % 4
        local_h = full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device)
        local_gi = full_gi[mine].contiguous().to(device)
        with _count_distributed_paths() as path_counts:
            fn(local_h, cos_all, sin_all, local_gi)
        gathered = _gather_seq(local_h, world, pg)
        if rank == 0:
            assert ref_fn is not None
            ref_h = full_h.clone().to(device)
            ref_fn(ref_h, cos_all, sin_all, full_gi.clone().to(device))
            assert _rel_l2(gathered, ref_h) < 0.005
        assert path_counts == expected_paths
        assert cap.calls == 1
        assert _DENSE_ATTN_RS_RESIDUAL_OP in cap.target_names
        assert "_c10d_functional.reduce_scatter_tensor.default" not in cap.target_names


def _check_dense_batched_and_standalone(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    compiled_b2, cap_b2 = _compile_dense_block(mesh, rank, world, device, weights, norms)
    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(1303)
    full_h_b2 = torch.randn(2, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    full_gi_b2 = torch.arange(seq, dtype=torch.int64) % 4
    lh_b2 = full_h_b2[:, mine].clone(memory_format=torch.contiguous_format).to(device)
    lgi_b2 = full_gi_b2[mine].contiguous().to(device)
    with _count_distributed_paths() as b2_counts:
        compiled_b2(lh_b2, cos_all, sin_all, lgi_b2)
    gathered_b2 = _gather_seq(lh_b2, world, pg)
    if rank == 0:
        ref_b2 = _LiveDenseBlock(_LIVE_HEADS).to(device).eval()
        _quantize_module_linears(ref_b2, weights, norms, device=device)
        ref_b2_fn = torch.compile(
            lambda h, c, s, gi: h.copy_(ref_b2(h, c, s, gi)),
            options=_build_dense_compile_options(None),
        )
        rh_b2 = full_h_b2.clone().to(device)
        ref_b2_fn(rh_b2, cos_all, sin_all, full_gi_b2.clone().to(device))
        assert _rel_l2(gathered_b2, rh_b2) < 0.005
    assert b2_counts == {"peer_exchange": 1, "fallback": 0}
    assert cap_b2.calls == 1
    assert _DENSE_ATTN_RS_OP in cap_b2.target_names

    class _WrapDenseStandalone(nn.Module):
        def __init__(self, heads: int) -> None:
            super().__init__()
            self.attn = _DenseRotaryAttn(heads)
            self.ff = _SwiGLUFFN()

        def forward(self, h: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            a = self.attn(h, c, s)
            return a + self.ff(a)

    sa_mod = _WrapDenseStandalone(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        sa_mod,
        weights,
        norms,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_dense_attention_module(sa_mod, mesh)
    cap_sa = FunctionalTargetCapturePass()
    sa_opts = compile_options(ffn=True, attention_output=True)
    compiled_sa = torch.compile(sa_mod, options=cap_sa.wrap_options(sa_opts))
    torch.manual_seed(1404)
    sa_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.bfloat16)
    with _count_distributed_paths() as sa_counts:
        sa_out = compiled_sa(
            sa_h[:, mine].clone(memory_format=torch.contiguous_format).to(device), cos_all, sin_all
        )
    gathered_sa = _gather_seq(sa_out, world, pg)
    if rank == 0:
        sa_ref = _WrapDenseStandalone(_LIVE_HEADS).to(device).eval()
        _quantize_module_linears(sa_ref, weights, norms, device=device)
        ref_sa_out = torch.compile(sa_ref, options=sa_opts)(sa_h.to(device), cos_all, sin_all)
        assert _rel_l2(gathered_sa, ref_sa_out) < 0.02
    assert sa_counts == {"peer_exchange": 1, "fallback": 0}
    assert cap_sa.calls == 1
    assert _DENSE_ATTN_RS_OP in cap_sa.target_names
    assert "_c10d_functional.reduce_scatter_tensor.default" not in cap_sa.target_names


def _check_dense_direct_multichunk_and_ownership(
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    batch, seq = 2, 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    (q_full, k_full, v_full, o_full), (q_local, k_local, v_local, o_local) = (
        _prepare_sharded_attn_weights(rank, world, device, weights)
    )
    q_norm, k_norm = norms["attn.norm_q"].to(device), norms["attn.norm_k"].to(device)
    cos_seq, sin_seq = cos_all[:seq].contiguous(), sin_all[:seq].contiguous()
    scale = _LIVE_HEAD_DIM**-0.5

    torch.manual_seed(1505)
    full_h = torch.randn(batch, seq, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    full_base = torch.randn(batch, seq, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    gate = torch.randn(4, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    full_gi_2d = (torch.arange(batch * seq, device=device, dtype=torch.int64) % 4).view(batch, seq)

    qdata_2d, scale_1d = _ops.convrot_int8_prepare_input.default(
        full_h.view(batch * seq, _LIVE_HIDDEN), 64, None
    )
    full_qdata, full_scale = qdata_2d.view(batch, seq, _LIVE_HIDDEN), scale_1d.view(batch, seq)
    qk_granularity = dense_quantized_dispatch.select_qk_quantization(
        AcceleratorTarget.from_device(device)
    )
    k_out = _ops.convrot_int8_piper_project_key.default(
        full_qdata,
        full_scale,
        k_local.qdata,
        k_local.scale,
        k_norm,
        cos_seq,
        sin_seq,
        1e-5,
        qk_quantization=qk_granularity,
    )
    v_out = _ops.convrot_int8_piper_project_value.default(
        full_qdata,
        full_scale,
        v_local.qdata,
        v_local.scale,
        head_dim=_LIVE_HEAD_DIM,
        is_causal=False,
    )

    local_base = full_base[:, mine].clone(memory_format=torch.contiguous_format)
    local_gi = full_gi_2d[:, mine].reshape(-1).contiguous()
    base_ptr = local_base.data_ptr()
    common_args = (
        full_qdata,
        full_scale,
        q_local.qdata,
        q_local.scale,
        q_norm,
        cos_seq,
        sin_seq,
        1e-5,
        scale,
        None,
        *k_out,
        *v_out,
        seq,
        False,
        o_local.qdata,
        o_local.scale,
        None,
        64,
        None,
    )

    exchange_calls = 0
    orig_start = distributed_comm.start_peer_exchange

    def counted_exchange(*args: Any, **kwargs: Any) -> Any:
        nonlocal exchange_calls
        exchange_calls += 1
        return orig_start(*args, **kwargs)

    distributed_comm.start_peer_exchange = counted_exchange
    try:
        with _count_distributed_paths() as path_counts:
            _ops.convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_.default(
                local_base,
                gate,
                local_gi,
                False,
                world,
                pg.group_name,
                *common_args,
                head_dim=_LIVE_HEAD_DIM,
                query_chunk_rows=128,
            )
    finally:
        distributed_comm.start_peer_exchange = orig_start

    assert local_base.data_ptr() == base_ptr
    assert path_counts == {"peer_exchange": 1, "fallback": 0}
    assert exchange_calls == 2
    gathered_base = _gather_seq(local_base, world, pg)
    if rank == 0:
        ref_k = _ops.convrot_int8_piper_project_key.default(
            full_qdata,
            full_scale,
            k_full.qdata,
            k_full.scale,
            k_norm,
            cos_seq,
            sin_seq,
            1e-5,
            qk_quantization=qk_granularity,
        )
        ref_v = _ops.convrot_int8_piper_project_value.default(
            full_qdata,
            full_scale,
            v_full.qdata,
            v_full.scale,
            head_dim=_LIVE_HEAD_DIM,
            is_causal=False,
        )
        ref_base = full_base.clone()
        _ops.convrot_int8_piper_projected_query_attention_residual_add_.default(
            ref_base,
            gate,
            full_gi_2d.reshape(-1).contiguous(),
            False,
            full_qdata,
            full_scale,
            q_full.qdata,
            q_full.scale,
            q_norm,
            cos_seq,
            sin_seq,
            1e-5,
            scale,
            None,
            *ref_k,
            *ref_v,
            seq,
            False,
            o_full.qdata,
            o_full.scale,
            None,
            64,
            None,
            head_dim=_LIVE_HEAD_DIM,
            query_chunk_rows=128,
        )
        assert _rel_l2(gathered_base, ref_base) < 0.005

    before_fail = local_base.clone()
    with pytest.raises(ValueError, match="softmax scale"):
        _ops.convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_.default(
            local_base,
            gate,
            local_gi,
            False,
            world,
            pg.group_name,
            *common_args[:8],
            0.0,
            *common_args[9:],
            head_dim=_LIVE_HEAD_DIM,
            query_chunk_rows=128,
        )
    assert local_base.data_ptr() == base_ptr
    assert torch.equal(local_base, before_fail)


def _check_dense_fp16_bias(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    class _WrapDenseAttnBias(nn.Module):
        def __init__(self, heads: int) -> None:
            super().__init__()
            self.attn = _DenseRotaryAttn(heads, bias=True, dtype=torch.float16)

        def forward(
            self,
            h: torch.Tensor,
            c: torch.Tensor,
            s: torch.Tensor,
        ) -> torch.Tensor:
            return self.attn(h, c, s)

    torch.manual_seed(1601)
    scale = 0.02
    fp16_weights = {
        "attn.to_q": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_k": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_v": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
        "attn.to_out.0": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.float16) * scale,
    }
    fp16_biases = {
        name: torch.randn(_LIVE_HIDDEN, dtype=torch.float16) * 0.05 for name in fp16_weights
    }
    fp16_norms = {
        "attn.norm_q": torch.rand(_LIVE_HEAD_DIM, dtype=torch.float16).add_(0.5),
        "attn.norm_k": torch.rand(_LIVE_HEAD_DIM, dtype=torch.float16).add_(0.5),
    }

    dist_mod = _WrapDenseAttnBias(_LIVE_HEADS // world).to(device).eval()
    _quantize_module_linears(
        dist_mod,
        fp16_weights,
        fp16_norms,
        biases=fp16_biases,
        mesh=mesh,
        rank=rank,
        world=world,
        device=device,
        colwise_names=_ATTN_COLWISE,
        rowwise_names=_ATTN_ROWWISE,
    )
    _parallelize_dense_attention_module(dist_mod, mesh)
    capture = FunctionalTargetCapturePass()
    fp16_opts = compile_options(attention_output=True)
    compiled_dist = torch.compile(dist_mod, options=capture.wrap_options(fp16_opts))

    ref_fn = None
    if rank == 0:
        ref_mod = _WrapDenseAttnBias(_LIVE_HEADS).to(device).eval()
        _quantize_module_linears(
            ref_mod, fp16_weights, fp16_norms, biases=fp16_biases, device=device
        )
        ref_fn = torch.compile(ref_mod, options=fp16_opts)

    seq = 512
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(1602)
    full_h = torch.randn(1, seq, _LIVE_HIDDEN, dtype=torch.float16)
    with _count_distributed_paths() as path_counts:
        dist_out = compiled_dist(
            full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device),
            cos_all,
            sin_all,
        )
    gathered = _gather_seq(dist_out, world, pg)
    if rank == 0:
        assert ref_fn is not None
        ref_out = ref_fn(full_h.to(device), cos_all, sin_all)
        rel = _rel_l2(gathered, ref_out)
        assert rel < 0.02, f"dense fp16+bias rel_l2={rel}"
    assert path_counts == {"peer_exchange": 1, "fallback": 0}
    assert capture.calls == 1
    assert "_c10d_functional.all_gather_into_tensor_coalesced.default" in capture.target_names
    assert "_c10d_functional.all_gather_into_tensor.default" not in capture.target_names
    assert "_c10d_functional.reduce_scatter_tensor.default" not in capture.target_names
    assert _DENSE_KEY_OP in capture.target_names
    assert _DENSE_VALUE_OP in capture.target_names
    assert _DENSE_ATTN_RS_OP in capture.target_names


def _check_dense_narrow_output(
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    batch, seq = 2, 512
    local_seq = seq // world
    narrow_features = 128
    torch.manual_seed(1603)
    o_narrow_raw = torch.randn(narrow_features, _LIVE_HIDDEN, dtype=torch.bfloat16) * 0.02
    o_narrow_full = ConvRotInt8Tensor.from_hp(o_narrow_raw.to(device), group_size=64)
    head_len = (_LIVE_HEADS // world) * _LIVE_HEAD_DIM
    o_narrow_local = shard_quantized_weight(
        o_narrow_full, dim=1, start=rank * head_len, length=head_len
    )

    (q_full, k_full, v_full, _o_full), (q_local, k_local, v_local, _o_local) = (
        _prepare_sharded_attn_weights(rank, world, device, weights)
    )
    q_norm, k_norm = norms["attn.norm_q"].to(device), norms["attn.norm_k"].to(device)
    cos_seq, sin_seq = cos_all[:seq].contiguous(), sin_all[:seq].contiguous()
    attn_scale = _LIVE_HEAD_DIM**-0.5

    torch.manual_seed(1604)
    full_h_narrow = torch.randn(batch, seq, _LIVE_HIDDEN, dtype=torch.bfloat16, device=device)
    qdata_2d, scale_1d = _ops.convrot_int8_prepare_input.default(
        full_h_narrow.view(batch * seq, _LIVE_HIDDEN), 64, None
    )
    full_qdata, full_scale = qdata_2d.view(batch, seq, _LIVE_HIDDEN), scale_1d.view(batch, seq)
    qk_granularity = dense_quantized_dispatch.select_qk_quantization(
        AcceleratorTarget.from_device(device)
    )
    k_out = _ops.convrot_int8_piper_project_key.default(
        full_qdata,
        full_scale,
        k_local.qdata,
        k_local.scale,
        k_norm,
        cos_seq,
        sin_seq,
        1e-5,
        qk_quantization=qk_granularity,
    )
    v_out = _ops.convrot_int8_piper_project_value.default(
        full_qdata,
        full_scale,
        v_local.qdata,
        v_local.scale,
        head_dim=_LIVE_HEAD_DIM,
        is_causal=False,
    )

    exchange_calls = 0
    orig_start = distributed_comm.start_peer_exchange

    def counted_exchange(*args: Any, **kwargs: Any) -> Any:
        nonlocal exchange_calls
        exchange_calls += 1
        return orig_start(*args, **kwargs)

    distributed_comm.start_peer_exchange = counted_exchange
    try:
        with _count_distributed_paths() as narrow_counts:
            narrow_out = _ops.convrot_int8_piper_projected_query_attention_reduce_scatter.default(
                world,
                pg.group_name,
                full_qdata,
                full_scale,
                q_local.qdata,
                q_local.scale,
                q_norm,
                cos_seq,
                sin_seq,
                1e-5,
                attn_scale,
                None,
                *k_out,
                *v_out,
                seq,
                False,
                torch.bfloat16,
                o_narrow_local.qdata,
                o_narrow_local.scale,
                None,
                64,
                None,
                head_dim=_LIVE_HEAD_DIM,
                query_chunk_rows=128,
            )
    finally:
        distributed_comm.start_peer_exchange = orig_start

    assert narrow_counts == {"peer_exchange": 1, "fallback": 0}
    assert exchange_calls == 2
    assert narrow_out.shape == (batch, local_seq, narrow_features)
    gathered_narrow = _gather_seq(narrow_out, world, pg)
    if rank == 0:
        ref_k = _ops.convrot_int8_piper_project_key.default(
            full_qdata,
            full_scale,
            k_full.qdata,
            k_full.scale,
            k_norm,
            cos_seq,
            sin_seq,
            1e-5,
            qk_quantization=qk_granularity,
        )
        ref_v = _ops.convrot_int8_piper_project_value.default(
            full_qdata,
            full_scale,
            v_full.qdata,
            v_full.scale,
            head_dim=_LIVE_HEAD_DIM,
            is_causal=False,
        )
        ref_narrow = _ops.convrot_int8_piper_projected_query_attention_output.default(
            full_qdata,
            full_scale,
            q_full.qdata,
            q_full.scale,
            q_norm,
            cos_seq,
            sin_seq,
            1e-5,
            attn_scale,
            None,
            *ref_k,
            *ref_v,
            seq,
            False,
            torch.bfloat16,
            o_narrow_full.qdata,
            o_narrow_full.scale,
            None,
            64,
            None,
            head_dim=_LIVE_HEAD_DIM,
            query_chunk_rows=128,
        )
        assert _rel_l2(gathered_narrow, ref_narrow) < 0.02


def _check_dense_fp16_bias_and_narrow_output(
    mesh: DeviceMesh,
    pg: dist.ProcessGroup,
    rank: int,
    world: int,
    device: torch.device,
    weights: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    cos_all: torch.Tensor,
    sin_all: torch.Tensor,
) -> None:
    _check_dense_fp16_bias(mesh, pg, rank, world, device, cos_all, sin_all)
    _check_dense_narrow_output(pg, rank, world, device, weights, norms, cos_all, sin_all)


def _2gpu_block_worker(rank: int, world: int, store_path: str, backend: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    init_kwargs: dict[str, Any] = {
        "backend": backend,
        "rank": rank,
        "world_size": world,
        "store": dist.FileStore(store_path, world),
        "timeout": timedelta(minutes=5),
    }
    if backend == "piper_relay":
        from piper_offload.communication import (  # noqa: PLC0415
            RelayOptions,
            register_relay_backend,
        )

        register_relay_backend()
        init_kwargs["pg_options"] = RelayOptions(staging_bytes=16 << 20, transport="shared")

    dist.init_process_group(**init_kwargs)
    try:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("tp",))
        pg = mesh.get_group("tp")
        torch.manual_seed(42)
        scale = 0.02
        weights = {
            "attn.to_q": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.bfloat16) * scale,
            "attn.to_k": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.bfloat16) * scale,
            "attn.to_v": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.bfloat16) * scale,
            "attn.to_out.0": torch.randn(_LIVE_HIDDEN, _LIVE_HIDDEN, dtype=torch.bfloat16) * scale,
            "ff.gate_proj": torch.randn(_LIVE_INTERMEDIATE, _LIVE_HIDDEN, dtype=torch.bfloat16)
            * scale,
            "ff.value_proj": torch.randn(_LIVE_INTERMEDIATE, _LIVE_HIDDEN, dtype=torch.bfloat16)
            * scale,
            "ff.down_proj": torch.randn(_LIVE_HIDDEN, _LIVE_INTERMEDIATE, dtype=torch.bfloat16)
            * scale,
        }
        norms = {
            "attn.norm_q": torch.rand(_LIVE_HEAD_DIM, dtype=torch.bfloat16).add_(0.5),
            "attn.norm_k": torch.rand(_LIVE_HEAD_DIM, dtype=torch.bfloat16).add_(0.5),
        }
        angles = torch.rand(_LIVE_MAX_SEQ, _LIVE_ROTARY_DIM, dtype=torch.float32) * (2 * torch.pi)
        cos_all = angles.cos().to(device)
        sin_all = angles.sin().to(device)

        with torch.inference_mode():
            ref_fn = None
            if rank == 0:
                ref_block = _LiveH3Block(_LIVE_HEADS).to(device).eval()
                _quantize_module_linears(ref_block, weights, norms, device=device)
                ref_opts = compile_options(ffn=True, attention=True, residual_updates=True)
                ref_fn = torch.compile(
                    lambda h, c, s, gi, bl, cg: h.copy_(ref_block(h, c, s, gi, bl, cg)),
                    dynamic=True,
                    options=ref_opts,
                )

            _check_quantized_all_gather_sibling_linears(mesh, pg, rank, world, device)
            _check_dynamic_tpsp_blocks(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all, ref_fn
            )
            _check_block_lengths(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all, ref_fn
            )
            _check_batched(mesh, pg, rank, world, device, weights, norms, cos_all, sin_all, ref_fn)
            _check_unaligned_local_seq_fallback(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all, ref_fn
            )
            _check_coarse(mesh, pg, rank, world, device, weights, norms, cos_all, sin_all, ref_fn)
            _check_standalone(mesh, pg, rank, world, device, weights, norms, cos_all, sin_all)
            _check_bias_fp16_and_projected_gate(mesh, pg, rank, world, device, cos_all, sin_all)
            _check_direct_multichunk_and_inplace_ownership(
                pg, rank, world, device, weights, norms, cos_all, sin_all
            )
            _check_dense_dynamic_tpsp_blocks(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all
            )
            _check_dense_causal_and_unaligned(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all
            )
            _check_dense_batched_and_standalone(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all
            )
            _check_dense_direct_multichunk_and_ownership(
                pg, rank, world, device, weights, norms, cos_all, sin_all
            )
            _check_dense_fp16_bias_and_narrow_output(
                mesh, pg, rank, world, device, weights, norms, cos_all, sin_all
            )

            if backend == "piper_relay":
                from piper_offload import host_pin_manager  # noqa: PLC0415

                host_pin_manager.clear()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
def test_distributed_tp_sp_end_to_end(tmp_path: Path) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA/ROCm GPUs")
    backend = "piper_relay" if _relay_available() else "nccl" if dist.is_nccl_available() else None
    if backend is None:
        pytest.skip("requires piper_offload relay or NCCL distributed backend")
    mp.spawn(_2gpu_block_worker, args=(2, str(tmp_path / "store"), backend), nprocs=2, join=True)
