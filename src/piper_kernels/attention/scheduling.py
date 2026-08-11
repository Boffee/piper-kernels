"""Host-side attention scheduling policies."""

import torch
import triton

BLOCK_M_VALUES = (32, 64, 128)
NUM_WARPS_VALUES = (2, 4, 8)
NUM_STAGES_VALUES = (1, 2, 3, 4)
LOOP_NUM_STAGES_VALUES = (None, 1, 2, 3, 4)


def select_query_block(
    query: torch.Tensor,
    batch: int,
    heads: int,
    query_length: int,
) -> int:
    """Choose the largest tile that launches at least one CTA per SM."""
    num_sms = torch.cuda.get_device_properties(query.device).multi_processor_count
    parallelism = batch * heads
    for block_m in reversed(BLOCK_M_VALUES[1:]):
        if triton.cdiv(query_length, block_m) * parallelism >= num_sms:
            return block_m
    return BLOCK_M_VALUES[0]
