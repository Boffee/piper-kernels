"""A fused dense attention computed inside its residual add runs in two head groups."""

import re

import pytest
import torch
from _compile_capture import FunctionalTargetCapturePass
from torch._inductor.utils import run_and_get_code

from piper_kernels.fusions.convrot_int8_piper import _compile as dense_compile

from ..residual_updates.test_compile import (
    _assert_updates_input_in_place,
    _compile_copy_back,
    _inputs,
    _options,
    _ResidualBlock,
)
from .test_compile import _available

_kernels = torch.ops.piper_kernels
_RESIDUAL_ADD = _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default
_KEY = _kernels.convrot_int8_piper_project_key.default
_VALUE = _kernels.convrot_int8_piper_project_value.default


class _MultiQueryResidualBlock(_ResidualBlock):
    kv_heads = 1


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_each_head_group_projects_its_context_before_its_add():
    model = _ResidualBlock()
    capture = FunctionalTargetCapturePass()
    compiled = _compile_copy_back(model, _options(capture))
    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_RESIDUAL_ADD) == 2
    assert capture.targets.count(_KEY) == 2
    assert capture.targets.count(_VALUE) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_each_head_group_projects_its_context_after_the_previous_group_adds():
    model = _ResidualBlock()
    compiled = _compile_copy_back(model, _options(FunctionalTargetCapturePass()))
    with torch.no_grad():
        _, (code, *_) = run_and_get_code(compiled, *_inputs(model, 300))

    calls = re.findall(
        r"torch\.ops\.piper_kernels\.convrot_int8_piper_"
        r"(project_key|project_value|projected_query_attention_residual_add_)\.default\(",
        code,
    )
    # Inductor frees the first group's K and V before the second group allocates its own.
    first_add = calls.index("projected_query_attention_residual_add_")
    assert sorted(calls[:first_add]) == ["project_key", "project_value"]
    assert sorted(calls[first_add + 1 :]) == [
        "project_key",
        "project_value",
        "projected_query_attention_residual_add_",
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_attention_whose_heads_do_not_split_evenly_stays_one_group():
    model = _MultiQueryResidualBlock()
    capture = FunctionalTargetCapturePass()
    compiled = _compile_copy_back(model, _options(capture))
    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_RESIDUAL_ADD) == 1
    assert capture.targets.count(_KEY) == 1


def test_head_split_names_arguments_its_operators_take():
    split = dense_compile._HEAD_SPLIT

    def names(operator):
        return {argument.name for argument in operator._schema.arguments}

    assert {*split.rows, split.columns, *split.once, split.alignment} <= names(_RESIDUAL_ADD)
    for producer, outputs in split.producers:
        assert set(split.producer_rows) <= names(producer)
        assert set(outputs) <= names(_RESIDUAL_ADD)
        assert len(outputs) == len(producer._schema.returns)
