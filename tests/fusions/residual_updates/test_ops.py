"""The generic residual add updates its base row by row."""

import pytest
import torch

from piper_kernels.fusions.residual_updates import _ops


def _operands(device, *, rows=37, features=96, dtype=torch.bfloat16):
    torch.manual_seed(1301)
    base = torch.randn(1, rows, features, device=device, dtype=dtype)
    update = torch.randn_like(base)
    gate_storage = torch.randn(5, 3 * features, device=device, dtype=dtype)
    gate = gate_storage[:, features : 2 * features]
    gate_indices = torch.randint(-5, 5, (rows,), device=device)
    return base, update, gate, gate_indices


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_residual_add_rounds_each_row_once(dtype):
    base, update, gate, gate_indices = _operands("cuda", dtype=dtype)
    expected = (base.float() + gate[gate_indices].float() * update.float()).to(base.dtype)

    result = _ops._residual_add_op(base, update, gate, gate_indices, True)

    assert result is None
    # The FP32 multiply-add may contract differently from the eager reference.
    torch.testing.assert_close(base, expected, atol=1e-6, rtol=2**-7)


@pytest.mark.parametrize(
    "update",
    [
        lambda base: base[:, :-1],
        lambda base: base.float(),
        lambda base: base.transpose(1, 2).contiguous().transpose(1, 2),
    ],
    ids=["shape", "dtype", "layout"],
)
def test_residual_add_rejects_mismatched_update_from_metadata(update):
    base, _, gate, gate_indices = _operands("meta")
    with pytest.raises(ValueError, match="update must be contiguous and match base"):
        _ops._residual_add_op_fake(base, update(base), gate, gate_indices, True)
