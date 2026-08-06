import pytest
import torch
from benchmark_mixed_int8_dot import _reference_output


def test_reference_output_validates_every_tile() -> None:
    probability = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ],
        dtype=torch.int8,
    )
    value = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 2], [3, 4]],
            [[-1, 1], [2, -2]],
        ],
        dtype=torch.int8,
    )

    actual = _reference_output(probability, value, tile_batch=2)

    expected = torch.stack(
        [probability[index].to(torch.int32) @ value[index].to(torch.int32) for index in range(3)]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tile_batch", [0, -1])
def test_reference_output_requires_positive_tile_batch(tile_batch: int) -> None:
    values = torch.ones((1, 2, 2), dtype=torch.int8)

    with pytest.raises(ValueError, match="tile batch must be positive"):
        _reference_output(values, values, tile_batch=tile_batch)
