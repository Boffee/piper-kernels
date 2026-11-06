"""Each attention's head slicing names arguments its operators take."""

import pytest
import torch

from piper_kernels.fusions.convrot_int8_piper import _compile as dense_compile
from piper_kernels.fusions.convrot_int8_sparse_piper import _compile as sparse_compile

_kernels = torch.ops.piper_kernels
_SLICINGS = {
    "dense": (
        dense_compile._HEAD_SLICING,
        _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default,
    ),
    "sparse": (
        sparse_compile._HEAD_SLICING,
        _kernels.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default,
    ),
}


def _names(operator: torch._ops.OpOverload) -> set[str]:
    return {argument.name for argument in operator._schema.arguments}


@pytest.mark.parametrize("name", _SLICINGS)
def test_head_slicing_names_arguments_its_operators_take(name):
    slicing, residual_add = _SLICINGS[name]
    add = _names(residual_add)

    assert {
        "base",
        *slicing.rows,
        slicing.columns,
        *slicing.once,
        slicing.alignment,
        *slicing.unsliced,
        *([] if slicing.window_rows is None else [slicing.window_rows]),
        *(slicing.position or ()),
    } <= add
    producers = [item for producer in slicing.producers for item in producer.items()]
    # HeadSlicing reads head_dim from the add or a producer.
    assert "head_dim" in add or all("head_dim" in _names(target) for target, _ in producers)
    for target, outputs in producers:
        assert set(slicing.producer_rows) <= _names(target)
        assert set(outputs) <= add
        assert len(outputs) == len(target._schema.returns)
