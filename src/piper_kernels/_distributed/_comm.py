"""Runtime collective helpers shared by distributed fusion custom operators.

Resolves functional c10d process groups by name and issues asynchronous coalesced
all-gathers and bounded chunk exchanges without changing process-group state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.distributed as dist


def resolve_process_group(group_name: str) -> dist.ProcessGroup:
    """Resolve one functional collective ``group_name`` to its ``ProcessGroup``."""
    # PyTorch's _C._distributed_c10d registry is intentionally untyped in stubs.
    c10d_cpp = cast(Any, torch._C)._distributed_c10d
    return cast(dist.ProcessGroup, c10d_cpp._resolve_process_group(group_name))


def start_coalesced_all_gather(
    process_group: dist.ProcessGroup,
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
) -> tuple[dist.Work, ...]:
    """Start async equal-size all-gathers for multiple tensors, coalescing when supported.

    Callers may alias ``inputs[i]`` with its exact own-rank slice of ``outputs[i]``.
    """
    coalesced = getattr(process_group, "all_gather_single_coalesced", None)
    if callable(coalesced):
        return (cast(dist.Work, coalesced(list(outputs), list(inputs))),)
    works: list[dist.Work] = []
    for output, source in zip(outputs, inputs, strict=True):
        work = dist.all_gather_into_tensor(output, source, group=process_group, async_op=True)
        assert work is not None
        works.append(work)
    return tuple(works)


def wait_all(works: Sequence[dist.Work]) -> None:
    """Block until every asynchronous collective work handle completes."""
    for work in works:
        work.wait()


def start_peer_exchange(
    process_group: dist.ProcessGroup,
    peer_send_buffers: Sequence[torch.Tensor],
    receive_buffer: torch.Tensor,
    zero_local_buffer: torch.Tensor | None = None,
) -> dist.Work:
    """Exchange one bounded peer-only chunk across ranks.

    ``peer_send_buffers`` holds one contiguous tensor per peer rank (all ranks except
    ``process_group.rank()``, in ascending rank order). Each rank receives into
    ``receive_buffer`` the sum of all peers' chunks destined for this rank.

    When ``world_size == 2``, uses ``dist.all_gather`` with own-slice aliasing so
    backends that skip self-copies avoid separate reduction buffers and kernels.
    For ``world_size >= 3``, uses ``dist.reduce_scatter`` with ``zero_local_buffer``
    as this rank's local slice; callers must keep ``zero_local_buffer`` alive until
    the returned work completes.
    """
    rank = process_group.rank()
    world_size = process_group.size()
    if world_size == 2:
        send = peer_send_buffers[0]
        output_list = [send, receive_buffer] if rank == 0 else [receive_buffer, send]
        work = dist.all_gather(output_list, send, group=process_group, async_op=True)
        assert work is not None
        return work

    assert zero_local_buffer is not None
    sources: list[torch.Tensor] = []
    peer_iter = iter(peer_send_buffers)
    for target_rank in range(world_size):
        sources.append(zero_local_buffer if target_rank == rank else next(peer_iter))
    work = dist.reduce_scatter(
        receive_buffer, sources, op=dist.ReduceOp.SUM, group=process_group, async_op=True
    )
    assert work is not None
    return work


__all__ = [
    "resolve_process_group",
    "start_coalesced_all_gather",
    "start_peer_exchange",
    "wait_all",
]
