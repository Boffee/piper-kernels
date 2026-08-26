"""Public sparse Piper Attention contract tests."""

import pytest
import torch

from piper_kernels import (
    prepare_sparse_piper_attention_plan,
    sparse_piper_attention,
)


def _inputs(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(52)
    shape = (1, 3 * 64, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    return query, key, value


def test_every_query_uses_sparse_prefix_plus_dense_suffix_on_cpu() -> None:
    query, key, value = _inputs()
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], dtype=torch.int32),
        sparse_key_blocks=2,
    )

    with torch.no_grad():
        output = sparse_piper_attention(
            query,
            key,
            value,
            plan,
            suffix_start=2 * 64,
        )

    assert output.shape == query.shape
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output).all()


def test_plan_accepts_arbitrary_per_head_budgets() -> None:
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([3, 1, 4, 2], dtype=torch.int64),
        sparse_key_blocks=5,
        query_chunk_blocks=17,
    )

    assert plan.keep_blocks.tolist() == [3, 1, 4, 2]
    assert plan.head_offsets.tolist() == [0, 3, 4, 8, 10]
    assert plan.routes_per_query == 10
    assert plan.query_chunk_blocks == 17


def test_dense_suffix_is_included_for_prefix_and_suffix_queries() -> None:
    shape = (1, 3 * 64, 2, 128)
    query = torch.zeros(shape, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.empty_like(query)
    value[:, :64] = 1
    value[:, 64:128] = 2
    value[:, 128:] = 10
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], dtype=torch.int32),
        sparse_key_blocks=2,
    )

    with torch.no_grad():
        output = sparse_piper_attention(query, key, value, plan, suffix_start=128)

    expected_by_head = torch.tensor([5.5, 13 / 3], dtype=torch.bfloat16)
    torch.testing.assert_close(output[0, 0, :, 0], expected_by_head)
    torch.testing.assert_close(output[0, -1, :, 0], expected_by_head)


def test_contract_rejects_misaligned_sparse_prefix() -> None:
    query, key, value = _inputs()
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 1], dtype=torch.int32),
        sparse_key_blocks=2,
    )

    with pytest.raises(ValueError, match="suffix_start"):
        sparse_piper_attention(query, key, value, plan, suffix_start=129)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_path_runs_and_writes_engine_layout() -> None:
    query, key, value = _inputs("cuda")
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        sparse_key_blocks=2,
    )

    with torch.no_grad():
        output = sparse_piper_attention(query, key, value, plan, suffix_start=2 * 64)

    assert output.shape == query.shape
    assert output.is_contiguous()
    assert torch.isfinite(output).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_int8_error_stays_within_the_selected_quality_gate() -> None:
    query, key, value = _inputs()
    keep = torch.tensor([1, 2], dtype=torch.int32)
    cpu_plan = prepare_sparse_piper_attention_plan(keep, sparse_key_blocks=2)
    cuda_plan = prepare_sparse_piper_attention_plan(keep.cuda(), sparse_key_blocks=2)

    with torch.no_grad():
        reference = sparse_piper_attention(
            query,
            key,
            value,
            cpu_plan,
            suffix_start=2 * 64,
        )
        actual = sparse_piper_attention(
            query.cuda(),
            key.cuda(),
            value.cuda(),
            cuda_plan,
            suffix_start=2 * 64,
        ).cpu()

    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.025


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_padded_tail_cannot_change_valid_outputs() -> None:
    query, key, value = _inputs("cuda")
    valid_length = 2 * 64 + 17
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        sparse_key_blocks=2,
    )
    changed_query = query.clone()
    changed_key = key.clone()
    changed_value = value.clone()
    changed_query[:, valid_length:] = 40
    changed_key[:, valid_length:] = -50
    changed_value[:, valid_length:] = 60

    with torch.no_grad():
        expected = sparse_piper_attention(
            query,
            key,
            value,
            plan,
            suffix_start=2 * 64,
            valid_sequence_length=valid_length,
        )
        actual = sparse_piper_attention(
            changed_query,
            changed_key,
            changed_value,
            plan,
            suffix_start=2 * 64,
            valid_sequence_length=valid_length,
        )

    torch.testing.assert_close(actual[:, :valid_length], expected[:, :valid_length], atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_operator_is_opaque_to_a_full_compile_graph() -> None:
    query, key, value = _inputs("cuda")
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        sparse_key_blocks=2,
    )

    def run(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return sparse_piper_attention(query, key, value, plan, suffix_start=2 * 64)

    compiled = torch.compile(run, fullgraph=True)
    with torch.no_grad():
        expected = run(query, key, value)
        actual = compiled(query, key, value)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
