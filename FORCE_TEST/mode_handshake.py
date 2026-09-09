"""Shared mode-install handshake for the hardware-facing FORCE_TEST loggers.

``ack_seq`` only says that Window A consumed a mailbox request.  A logger may
send its first twist only after ``install_seq`` has reached that request and
the requested mode is the live mode.  Keeping the polling loop here prevents
each experiment logger from accidentally implementing a weaker handshake.

The module has no controller imports at import time so its protocol can be
tested offline.  The concrete PEIRASTIC ``Mode`` and ``ModeRequest`` classes
are imported only when a logger requests SERVO_TWIST installation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Mapping
from typing import Any, Callable


DEFAULT_INSTALL_TIMEOUT_S = 2.0
DEFAULT_INSTALL_POLL_S = 0.01


class ModeInstallError(RuntimeError):
    """The requested mode was not proven live by the controller."""


class ModeInstallTimeout(ModeInstallError, TimeoutError):
    """The controller never published an install ACK before the deadline."""


@dataclass(frozen=True)
class ModeInstallAck:
    """Provenance for one successfully installed mode request."""

    mode: int
    mode_name: str
    request_seq: int
    install_seq: int
    status: Any
    ack_seq: int | None


def _positive_seq(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, bytes, bytearray)):
        raise ModeInstallError(f"{name} must be a positive integer, got {value!r}")
    try:
        number = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModeInstallError(f"{name} must be a positive integer, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric != float(number) or number <= 0:
        raise ModeInstallError(f"{name} must be a positive integer, got {value!r}")
    return number


def _optional_seq(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, bytes, bytearray)):
        return None
    try:
        number = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric != float(number) or number < 0:
        return None
    return number


def _mode_number(request: Any) -> int:
    try:
        return int(request.mode)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModeInstallError(f"mode request has no valid mode: {request!r}") from exc


def _mode_name(request: Any, number: int) -> str:
    mode = getattr(request, "mode", number)
    name = getattr(mode, "name", None)
    if name:
        return str(name)
    try:
        from peirastic.core.modes import MODE_LABEL

        return str(MODE_LABEL.get(number, number))
    except Exception:
        return str(number)


def _terminal_status(status: Any) -> str | None:
    """Return a terminal status label without requiring controller imports."""

    if isinstance(status, str):
        label = status.strip().upper()
        if label in {"ERROR", "STOPPED", "ESTOP"}:
            return label
        return None
    try:
        value = int(status)
    except (TypeError, ValueError, OverflowError):
        return None
    # Current PEIRASTIC Status: ERROR=3, STOPPED=4, ESTOP=5.  Importing the
    # enum when available avoids coupling the offline helper to those values.
    try:
        from peirastic.core.ipc import Status

        labels = {
            int(Status.ERROR): "ERROR",
            int(Status.STOPPED): "STOPPED",
            int(Status.ESTOP): "ESTOP",
        }
    except Exception:
        labels = {3: "ERROR", 4: "STOPPED", 5: "ESTOP"}
    return labels.get(value)


def _status_name(status: Any) -> str | None:
    """Map the live status to the protocol label when it is available."""

    if isinstance(status, str):
        label = status.strip().upper()
        return label or None
    try:
        value = int(status)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        from peirastic.core.ipc import Status

        labels = {int(item): item.name for item in Status}
    except Exception:
        labels = {
            0: "IDLE",
            1: "RUNNING",
            2: "DONE",
            3: "ERROR",
            4: "STOPPED",
            5: "ESTOP",
        }
    return labels.get(value)


def _validate_snapshot_freshness(
    snapshot: Mapping[str, Any],
    *,
    now: float,
    mode_name: str,
    request_seq: int,
) -> None:
    """Reject an explicitly stale controller snapshot.

    Older offline fakes do not publish ``t_mono`` and remain supported.  The
    versioned PEIRASTIC snapshot does publish the controller monotonic clock;
    when present, it must be finite and within one second of this process.
    """

    if "t_mono" not in snapshot or snapshot.get("t_mono") is None:
        return
    try:
        stamp = float(snapshot["t_mono"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModeInstallError(
            f"{mode_name} mode request_seq={request_seq} has invalid snapshot t_mono="
            f"{snapshot.get('t_mono')!r}"
        ) from exc
    if not math.isfinite(stamp):
        raise ModeInstallError(
            f"{mode_name} mode request_seq={request_seq} has invalid snapshot t_mono={stamp!r}"
        )
    age = abs(float(now) - stamp)
    if age > 1.0:
        raise ModeInstallError(
            f"{mode_name} mode request_seq={request_seq} snapshot is stale: "
            f"t_mono_age_s={age:.3f}"
        )


def wait_for_mode_install(
    client: Any,
    request: Any,
    *,
    timeout_s: float = DEFAULT_INSTALL_TIMEOUT_S,
    poll_s: float = DEFAULT_INSTALL_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> ModeInstallAck:
    """Send ``request`` and wait for its *install* ACK.

    The mailbox ACK is deliberately not treated as an installation ACK.  A
    snapshot without ``status``/``install_seq`` is a protocol failure, and a
    sequence that is negative, invalid, or still below the request cannot
    authorize a logger tick.  Once the sequence reaches the request, the
    snapshot must also report the exact live mode and a RUNNING/DONE status.
    ``clock`` and ``sleeper`` are injectable for offline tests.
    """

    try:
        timeout = float(timeout_s)
        poll = float(poll_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mode install timeout and poll must be finite numbers") from exc
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError(f"mode install timeout must be non-negative, got {timeout_s!r}")
    if not math.isfinite(poll) or poll <= 0.0:
        raise ValueError(f"mode install poll must be positive, got {poll_s!r}")

    mode_number = _mode_number(request)
    mode_name = _mode_name(request, mode_number)
    try:
        raw_request_seq = client.set_mode(request)
    except Exception as exc:
        raise ModeInstallError(f"{mode_name} mode request failed before install ACK: {exc}") from exc
    request_seq = _positive_seq(raw_request_seq, name="mode request_seq")

    deadline = float(clock()) + timeout
    while True:
        try:
            snapshot = client.snapshot()
        except Exception as exc:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} snapshot failed while waiting for install ACK: {exc}"
            ) from exc
        if not isinstance(snapshot, Mapping):
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} snapshot is not a mapping: {snapshot!r}"
            )
        _validate_snapshot_freshness(
            snapshot,
            now=float(clock()),
            mode_name=mode_name,
            request_seq=request_seq,
        )
        if "status" not in snapshot:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} snapshot is missing status"
            )
        if "install_seq" not in snapshot:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} snapshot is missing install_seq; "
                "mailbox ack_seq is not sufficient"
            )
        install_raw = snapshot.get("install_seq")
        if isinstance(install_raw, (bool, bytes, bytearray)):
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} has invalid install_seq={install_raw!r}"
            )
        try:
            install_seq = int(install_raw)
            install_numeric = float(install_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} has invalid install_seq={install_raw!r}"
            ) from exc
        if not math.isfinite(install_numeric) or install_numeric != float(install_seq) or install_seq < 0:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} has invalid install_seq={install_raw!r}"
            )

        status = snapshot.get("status")
        status_name = _status_name(status)
        if status_name is None:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} has invalid status={status!r}"
            )
        terminal = _terminal_status(status)
        ack_seq = _optional_seq(snapshot.get("ack_seq"), name="ack_seq")
        if terminal in {"STOPPED", "ESTOP"}:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} install refused: "
                f"status={terminal} install_seq={install_seq} ack_seq={ack_seq}"
            )
        if terminal == "ERROR":
            detail = str(snapshot.get("msg") or "controller error")
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} install refused: "
                f"status=ERROR install_seq={install_seq} ack_seq={ack_seq} {detail}"
            )

        if install_seq > request_seq:
            raise ModeInstallError(
                f"{mode_name} mode request_seq={request_seq} was superseded by install_seq={install_seq}"
            )
        if install_seq == request_seq:
            if "mode" not in snapshot:
                raise ModeInstallError(
                    f"{mode_name} mode request_seq={request_seq} install_seq={install_seq} "
                    "snapshot is missing live mode"
                )
            try:
                live_mode = int(snapshot["mode"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ModeInstallError(
                    f"{mode_name} mode request_seq={request_seq} has invalid live mode "
                    f"{snapshot['mode']!r} at install_seq={install_seq}"
                ) from exc
            if live_mode != mode_number:
                raise ModeInstallError(
                    f"{mode_name} mode request_seq={request_seq} install_seq={install_seq} "
                    f"but controller reports live mode={live_mode}"
                )
            if status_name not in {"RUNNING", "DONE"}:
                raise ModeInstallError(
                    f"{mode_name} mode request_seq={request_seq} install_seq={install_seq} "
                    f"but controller status={status_name}; expected RUNNING or DONE"
                )
            return ModeInstallAck(
                mode=mode_number,
                mode_name=mode_name,
                request_seq=request_seq,
                install_seq=install_seq,
                status=status,
                ack_seq=ack_seq,
            )

        now = float(clock())
        if now >= deadline:
            ack_text = "missing" if ack_seq is None else str(ack_seq)
            raise ModeInstallTimeout(
                f"{mode_name} mode install ACK timeout: request_seq={request_seq} "
                f"ack_seq={ack_text} install_seq={install_seq} status={status!r}"
            )
        sleeper(min(poll, max(deadline - now, 0.0)))


def install_servo_twist(
    client: Any,
    *,
    filter_enabled: bool = False,
    timeout_s: float = DEFAULT_INSTALL_TIMEOUT_S,
    poll_s: float = DEFAULT_INSTALL_POLL_S,
) -> ModeInstallAck:
    """Install SERVO_TWIST and return the common provenance record."""

    from peirastic.core.modes import Mode, ModeRequest

    return wait_for_mode_install(
        client,
        ModeRequest(Mode.SERVO_TWIST, {"filter": bool(filter_enabled)}),
        timeout_s=timeout_s,
        poll_s=poll_s,
    )


def mode_provenance(ack: ModeInstallAck | None) -> dict[str, str]:
    """CSV-safe request/install provenance shared by both logger classes."""

    if ack is None:
        return {
            "mode_name": "",
            "mode_request_seq": "",
            "mode_install_seq": "",
            "mode_install_status": "not_installed",
        }
    return {
        "mode_name": str(ack.mode_name),
        "mode_request_seq": str(int(ack.request_seq)),
        "mode_install_seq": str(int(ack.install_seq)),
        "mode_install_status": "installed",
    }


__all__ = [
    "DEFAULT_INSTALL_POLL_S",
    "DEFAULT_INSTALL_TIMEOUT_S",
    "ModeInstallAck",
    "ModeInstallError",
    "ModeInstallTimeout",
    "install_servo_twist",
    "mode_provenance",
    "wait_for_mode_install",
]
