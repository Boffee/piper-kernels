"""Public API and validation tests for SageAttention2++."""

import pytest
import torch

from piper_kernels import sage_attention_2pp


def test_rejects_grouped_query_attention() -> None:
    query = torch.randn(1, 4, 16, 64, dtype=torch.float16)
    key = torch.randn(1, 2, 16, 64, dtype=torch.float16)

    with pytest.raises(ValueError, match="equal batch and head dimensions"):
        sage_attention_2pp(query, key, key)
