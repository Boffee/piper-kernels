"""CLI controls shared by the attention execution-plan tuners."""

from __future__ import annotations

import argparse

from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)

from .attention import ATTENTION_DTYPE_NAMES
from .tuning import add_tuning_arguments, parse_optional_integer, validate_tuning_arguments


def add_attention_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add workload and shared launch-plan axes for an attention tuner."""
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--kv-sequence", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=ATTENTION_DTYPE_NAMES, default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--block-m", type=int, choices=BLOCK_M_VALUES, nargs="+")
    parser.add_argument("--num-warps", type=int, choices=NUM_WARPS_VALUES, nargs="+")
    parser.add_argument("--num-stages", type=int, choices=NUM_STAGES_VALUES, nargs="+")
    parser.add_argument(
        "--use-tensor-descriptors",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--reverse-causal-blocks",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--loop-num-stages",
        type=parse_optional_integer,
        choices=LOOP_NUM_STAGES_VALUES,
        metavar="{0,1,2,3,4}",
        nargs="+",
        help="loop pipeline stages; 0 uses compiler default; omitted retains production",
    )
    parser.add_argument(
        "--loop-licm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-packed-probability-conversion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    add_tuning_arguments(parser)


def validate_attention_tuning_arguments(arguments: argparse.Namespace) -> None:
    """Validate workload and launch controls shared by attention tuners."""
    lengths = (arguments.sequence, arguments.kv_sequence or arguments.sequence)
    if any(length <= 0 for length in lengths):
        raise SystemExit("attention sequence lengths must be positive")
    if arguments.batch_size <= 0 or arguments.heads <= 0:
        raise SystemExit("batch size and heads must be positive")
    if arguments.causal and lengths[0] != lengths[1]:
        raise SystemExit("causal attention requires equal query and key/value lengths")
    if not arguments.causal and arguments.reverse_causal_blocks is True:
        raise SystemExit("reverse causal-block order requires causal attention")
    validate_tuning_arguments(arguments)
