"""Residual operand validation needs only metadata."""

from dataclasses import replace

import pytest
import torch

from piper_kernels._residual import ResidualAdd, ReusableUpdate, validate_residual_add


def _residual():
    base = torch.empty(2, 7, 16, device="meta", dtype=torch.bfloat16)
    gate = torch.empty(3, 48, device="meta", dtype=base.dtype)[:, 16:32]
    indices = torch.empty(14, device="meta", dtype=torch.int64)
    return ResidualAdd(base, gate, indices, python_indexing=False)


@pytest.mark.parametrize("reusable", [False, True])
def test_residual_validates_without_tensor_contents(reusable):
    residual = _residual()
    if reusable:
        update = ReusableUpdate(torch.empty_like(residual.base), residual.gate)
        residual = replace(residual, reusable=update)
    validate_residual_add(residual.base, residual, 16)
    assert residual.output is (residual.reusable.update if reusable else residual.base)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: replace(r, gate_indices=r.gate_indices[:7]), "indices must be"),
        (lambda r: replace(r, gate_indices=r.gate_indices.float()), "indices must be"),
        (lambda r: replace(r, gate=r.gate[:, ::2]), "gates must be"),
        (lambda r: replace(r, gate=r.gate.float()), "gates must be"),
        (lambda r: replace(r, base=r.base[:, :-1]), "base must be"),
        (
            lambda r: replace(r, reusable=ReusableUpdate(r.base.float(), r.gate)),
            "reusable update must be",
        ),
        (
            lambda r: replace(r, reusable=ReusableUpdate(torch.empty_like(r.base), r.gate.float())),
            "gates must be",
        ),
    ],
)
def test_residual_rejects_invalid_operand_metadata(change, message):
    residual = _residual()
    with pytest.raises(ValueError, match=message):
        validate_residual_add(residual.base, change(residual), 16)
