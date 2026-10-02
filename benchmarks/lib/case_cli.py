"""Named workload selection for tuners and stage diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .cases import AttentionCase, LinearCase, named_case


def apply_case(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    *,
    attention: bool = False,
    preparation: bool = False,
) -> argparse.Namespace:
    """Use a catalog case unchanged; custom shape flags describe separate experiments."""
    if args.case is None:
        return args
    try:
        case = named_case(args.case)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if attention:
        if not isinstance(case, AttentionCase) or case.keep_ratio is not None:
            raise SystemExit("--case requires a dense attention case")
        values: dict[str, object] = {
            "sequence": case.sequence,
            "kv_sequence": case.sequence,
            "batch_size": case.batch,
            "heads": case.heads,
            "kv_heads": case.kv_heads,
            "head_dim": case.head_dim,
            "causal": case.causal,
        }
    else:
        if not isinstance(case, LinearCase):
            raise SystemExit("--case requires a linear case")
        values = {
            "rows": case.rows,
            "in_features": [case.in_features] if preparation else case.in_features,
            "input_activation": None,
        }
        if not preparation:
            values.update(out_features=case.out_features, bias=case.bias)
    values.update(dtype=case.dtype, seed=case.seed)
    flags = {"--" + key.replace("_", "-") for key in values}
    flags.update(
        "--no-" + key.replace("_", "-") for key, value in values.items() if isinstance(value, bool)
    )
    supplied = {part.split("=", 1)[0] for part in (sys.argv[1:] if argv is None else argv)}
    conflicts = flags & supplied
    if conflicts:
        raise SystemExit(
            f"--case cannot be combined with workload overrides: {', '.join(sorted(conflicts))}"
        )
    for key, value in values.items():
        setattr(args, key, value)
    return args
