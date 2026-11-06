"""Attention residual-add operators take the add's arguments, then their output's."""

import pytest

from piper_kernels.fusions.convrot_int8_piper import output as dense
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse

_PAIRS = {
    "dense": (
        dense._projected_query_attention_output_op,
        dense._projected_query_attention_residual_add_op,
    ),
    "sparse": (
        sparse._projected_query_attention_output_op,
        sparse._projected_query_attention_residual_add_op,
    ),
}


@pytest.mark.parametrize("name", _PAIRS)
def test_residual_add_takes_the_add_then_the_output_arguments(name):
    output, residual_add = (operation._opoverload._schema for operation in _PAIRS[name])
    output_names = [
        argument.name for argument in output.arguments if argument.name != "output_dtype"
    ]

    assert [argument.name for argument in residual_add.arguments] == [
        "base",
        "gate",
        "gate_indices",
        "python_indexing",
        *output_names,
    ]
