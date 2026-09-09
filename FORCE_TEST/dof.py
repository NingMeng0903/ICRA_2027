"""Session-level seven/eight degree-of-freedom selection for FORCE_TEST.

The experiment chooses the mechanism once at session start.  Individual
SERVO_TWIST calls inherit that choice; they do not carry a secondary policy.
This module is deliberately small so the public API remains owned by
``peirastic.api.PeirasticArm`` while the test scripts share one handshake and
one provenance record.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_DOF = 8
_LAST: dict[str, Any] = {
    "dof_requested": DEFAULT_DOF,
    "dof_active": float("nan"),
    "dof_config_source": "FORCE_TEST --dof (default=8)",
    "controller_api_version": "unknown",
}


def add_session_args(parser) -> None:
    parser.add_argument(
        "--dof",
        type=int,
        choices=(7, 8),
        default=DEFAULT_DOF,
        help="controller mechanism for this session (default: 8; 7 locks rail)",
    )
    from paths import add_run_arg

    add_run_arg(parser)


def _version(arm: Any) -> str:
    for obj in (arm, getattr(arm, "client", None), getattr(arm, "controller", None)):
        if obj is None:
            continue
        for name in (
            "controller_api_version",
            "api_version",
            "controller_version",
            "version",
            "__version__",
        ):
            value = getattr(obj, name, None)
            if value not in (None, ""):
                return str(value)
    try:
        import peirastic

        value = getattr(peirastic, "__version__", None)
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    return "set_dof/get_dof"


def _active_value(value: Any, *, requested: int | None = None) -> int:
    """Decode ``get_dof`` and require a successful, finite status/value pair.

    The facade returns ``(status, active_dof)``.  Treating only the second
    element as authoritative can turn ``(ERR_TIMEOUT, nan)`` into a
    successful session record, so a structural transition is never accepted
    without both an OK status and the requested active value.
    """
    status: Any
    active: Any
    if isinstance(value, dict):
        status = value.get("status", value.get("code", value.get("ret")))
        active = next((value[key] for key in ("dof", "active_dof", "value") if key in value), None)
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        # PeirasticArm.get_dof() returns (status, active_dof).
        status, active = value[0], value[1]
    else:
        raise RuntimeError("get_dof() must return (status, active_dof)")
    try:
        from peirastic.api.codes import CODE_NAMES, OK
    except ImportError:
        OK = 0
        CODE_NAMES = {}
    try:
        status_i = int(status)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(f"get_dof() returned invalid status {status!r}") from None
    if status_i != int(OK):
        raise RuntimeError(
            f"get_dof() failed: {status_i} ({CODE_NAMES.get(status_i, status_i)})"
        )
    try:
        active_f = float(active)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError(f"get_dof() returned invalid active DOF {active!r}") from None
    if not math.isfinite(active_f):
        raise RuntimeError(f"get_dof() returned non-finite active DOF {active!r}")
    if active_f != math.trunc(active_f):
        raise RuntimeError(f"get_dof() returned non-integral active DOF {active!r}")
    active_i = int(active_f)
    if active_i not in (7, 8):
        raise RuntimeError(f"get_dof() returned unsupported active DOF {active_i}")
    if requested is not None and active_i != int(requested):
        raise RuntimeError(
            f"controller reports DOF={active_i}, requested {int(requested)}"
        )
    return active_i


def set_session_dof(arm: Any, dof: int, *, source: str = "FORCE_TEST --dof") -> dict[str, Any]:
    global _LAST
    requested = int(dof)
    if requested not in (7, 8):
        raise ValueError(f"dof must be 7 or 8, got {requested}")
    setter = getattr(arm, "set_dof", None)
    getter = getattr(arm, "get_dof", None)
    if not callable(setter) or not callable(getter):
        raise RuntimeError("controller API must provide arm.set_dof() and arm.get_dof()")
    # A script may call go_mid again to return from a press.  Once the
    # requested structure is confirmed, inherit it instead of issuing a
    # second structural transition in the same session.
    previous = _LAST.get("dof_active")
    if _LAST.get("dof_requested") == requested and isinstance(previous, int):
        active = _active_value(getter(), requested=requested)
        if active == requested:
            _LAST["dof_config_source"] = str(source)
            _LAST["controller_api_version"] = _version(arm)
            print(
                f"[DOF] inherited={active} source={source} "
                f"api={_LAST['controller_api_version']}",
                flush=True,
            )
            return dict(_LAST)
    # The new API queues the structural switch at a safe task boundary.  The
    # block argument is intentional: MOVEJ/SERVO must inherit a settled mode.
    # New facades expose the explicit after_current switch.  The currently
    # deployed facade implied it and only accepted block; retain a narrow
    # signature fallback while the rolling API migration is in progress.
    try:
        ret = setter(requested, after_current=True, block=1)
    except TypeError as exc:
        if "after_current" not in str(exc):
            raise
        ret = setter(requested, block=1)
    try:
        from peirastic.api.codes import CODE_NAMES, OK

        if ret not in (None, OK):
            raise RuntimeError(f"set_dof({requested}) -> {ret} ({CODE_NAMES.get(ret, ret)})")
    except ImportError:
        if ret not in (None, 0):
            raise RuntimeError(f"set_dof({requested}) -> {ret}")
    active = _active_value(getter(), requested=requested)
    _LAST = {
        "dof_requested": requested,
        "dof_active": active,
        "dof_config_source": str(source),
        "controller_api_version": _version(arm),
    }
    print(
        f"[DOF] requested={requested} active={active} source={source} "
        f"api={_LAST['controller_api_version']}",
        flush=True,
    )
    return dict(_LAST)


def set_requested_dof(dof: int, *, source: str = "FORCE_TEST --dof") -> dict[str, Any]:
    """Record an offline request before an analyze-only command."""
    requested = int(dof)
    if requested not in (7, 8):
        raise ValueError(f"dof must be 7 or 8, got {requested}")
    global _LAST
    _LAST = {
        "dof_requested": requested,
        "dof_active": float("nan"),
        "dof_config_source": str(source),
        "controller_api_version": "offline",
    }
    return dict(_LAST)


def current_metadata() -> dict[str, Any]:
    out = dict(_LAST)
    for key in ("dof_requested", "dof_active"):
        value = out.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
    return out
