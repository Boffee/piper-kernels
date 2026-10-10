"""Named workload selection for tuners and stage diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence

from .cases import AttentionCase, Case, LinearCase, named_case


def require_case[CaseT: Case](identity: str, kind: type[CaseT] | tuple[type[CaseT], ...]) -> CaseT:
    """Resolve a CLI identity and reject cases outside the tool's operation family."""
    try:
        case = named_case(identity)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not isinstance(case, kind):
        raise SystemExit(f"case {identity!r} does not support this diagnostic or tuner")
    return case


def set_case_arguments(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    values: Mapping[str, object],
) -> None:
    """Apply catalog values only when no workload override was supplied.

    Callers disable argparse abbreviation so explicit flags have unambiguous names.
    Both positive and negative boolean forms, and --name=value, count as overrides.
    """
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
    vars(args).update(values)


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
    if attention:
        case = require_case(args.case, AttentionCase)
        if case.keep_ratio is not None:
            raise SystemExit("--case requires a dense attention case")
        values: dict[str, object] = {
            "sequence": case.sequence,
            "kv_sequence": case.sequence,
            "batch_size": case.batch,
            "heads": case.heads,
            "kv_heads": case.kv_heads,
            "head_dim": case.head_dim,
            "causal": case.causal,
            "scale": None,
        }
    else:
        case = require_case(args.case, LinearCase)
        values = {
            "rows": case.rows,
            "in_features": [case.in_features] if preparation else case.in_features,
            "input_activation": None,
        }
        if not preparation:
            values.update(out_features=case.out_features, bias=case.bias)
    values.update(dtype=case.dtype, seed=case.seed)
    set_case_arguments(args, argv, values)
    return args
