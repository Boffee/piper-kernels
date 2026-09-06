"""Device-independent sparse-Piper routing names and static operator modes."""

_MINMAX_ROUTING = 0
_MEAN_ROUTING = 1
_ROUTING_MODE_BY_NAME = {
    "minmax": _MINMAX_ROUTING,
    "mean": _MEAN_ROUTING,
}
_ROUTING_NAME_BY_MODE = {mode: name for name, mode in _ROUTING_MODE_BY_NAME.items()}


def validate_routing_mode(routing_mode: int) -> None:
    """Reject routing modes outside the internal static operator contract."""
    if not is_valid_routing_mode(routing_mode):
        raise ValueError("sparse Piper routing mode must be minmax or mean")


def routing_mode_from_name(routing: str) -> int:
    """Resolve a public routing name to its static operator mode."""
    try:
        return _ROUTING_MODE_BY_NAME[routing]
    except (KeyError, TypeError) as exc:
        raise ValueError("sparse Piper routing must be 'mean' or 'minmax'") from exc


def is_valid_routing_mode(routing_mode: int) -> bool:
    """Return whether a value names a supported static routing policy."""
    return routing_mode in _ROUTING_NAME_BY_MODE
