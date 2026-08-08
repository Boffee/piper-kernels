"""Host-side attention scheduling policies."""

import torch
import triton


def select_query_block(
    query: torch.Tensor,
    batch: int,
    heads: int,
    query_length: int,
) -> int:
    """Choose the largest tile that launches at least one CTA per SM."""
    num_sms = torch.cuda.get_device_properties(query.device).multi_processor_count
    parallelism = batch * heads
    for block_m in (128, 64):
        if triton.cdiv(query_length, block_m) * parallelism >= num_sms:
            return block_m
    return 32
