"""MOVEJ to the taught mid-stroke, then wait for nullspace to finish.

Same target as ``peirastic.DEMO.movej``: rail 400 mm + taught arm angles.
After arrival, hold 5 s so the null-space posture can settle before any
SERVO_TWIST ID.  Force loop stays off.  Window A must already be running.
"""

from __future__ import annotations

import time

from paths import active_run_id, add_playground, set_active_run_id, stamp
from dof import add_session_args, set_session_dof

SETTLE_S = 5.0


def add_movej_args(parser) -> None:
    add_session_args(parser)
    parser.add_argument(
        "--skip-movej",
        action="store_true",
        help="already at mid-stroke; do not MOVEJ (still waits settle)",
    )
    parser.add_argument("--movej-v", type=float, default=0.4, help="MOVEJ speed fraction (0,1]")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=SETTLE_S,
        help="hold after arrival so nullspace can finish (default 5)",
    )
    parser.add_argument("--skip-settle", action="store_true")


def go_mid_from_args(args) -> int:
    set_active_run_id(getattr(args, "run_id", "") or active_run_id(stamp()))
    settle = 0.0 if bool(getattr(args, "skip_settle", False)) else float(getattr(args, "settle_s", SETTLE_S))
    return go_mid(
        prefix=str(getattr(args, "shm_prefix", "")),
        skip=bool(getattr(args, "skip_movej", False)),
        v=float(getattr(args, "movej_v", 0.4)),
        settle_s=settle,
        dof=int(getattr(args, "dof", 8)),
    )


def stop_for_dof_boundary(arm) -> int:
    """Terminate a continuous task before requesting a structural switch."""
    add_playground()
    try:
        from peirastic.api.codes import CODE_NAMES, OK
    except ImportError:
        # Hardware runs import the facade's codes.  Keep the offline FORCE_TEST
        # boundary test usable when optional native dependencies are absent.
        CODE_NAMES, OK = {}, 0

    stopper = getattr(arm, "set_arm_stop", None)
    if not callable(stopper):
        raise RuntimeError("controller API must provide set_arm_stop() before set_dof()")
    ret = stopper()
    if ret not in (None, OK):
        raise RuntimeError(
            f"stop previous task -> {ret} ({CODE_NAMES.get(ret, ret)})"
        )
    return OK if ret is None else ret


def go_mid(
    *,
    prefix: str = "",
    skip: bool = False,
    v: float = 0.4,
    settle_s: float = SETTLE_S,
    dof: int = 8,
) -> int:
    """Block until MOVEJ arrives, then wait ``settle_s``.  Return 0 on success."""

    add_playground()
    from peirastic.api import PeirasticArm
    from peirastic.api.codes import CODE_NAMES, OK
    from peirastic.DEMO.movej import _fmt_q, q_target_rad

    # Structural DOF is selected once before MOVEJ (or before a skipped-MOVEJ
    # session) and inherited by every subsequent mode request.
    arm = PeirasticArm(prefix=str(prefix))
    try:
        # SERVO_TWIST is continuous even after its twist bus is zeroed.  End
        # that task explicitly before a 7/8-DOF boundary request; SET_DOF's
        # dedicated ACK then proves the stationary transition completed.
        try:
            ret_stop = stop_for_dof_boundary(arm)
        except RuntimeError as exc:
            print(
                f"[ERR] {exc}",
                flush=True,
            )
            return 1
        set_session_dof(arm, int(dof))
        if skip:
            print("[MOVEJ] skipped", flush=True)
        else:
            q = q_target_rad()
            print(f"[MOVEJ] mid-stroke  {_fmt_q(q)}  v={v:.2f}", flush=True)
            ret = arm.movej(q, v=float(v), r=0, connect=0, block=1)
            if ret != OK:
                print(f"[ERR] MOVEJ -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
                return 1
            print("[OK] MOVEJ mid-stroke", flush=True)
    except KeyboardInterrupt:
        arm.set_arm_stop()
        print("[MOVEJ] interrupted", flush=True)
        return 130
    finally:
        arm.close()
    if settle_s > 0.0:
        print(f"[SETTLE] {settle_s:.1f}s for nullspace", flush=True)
        time.sleep(float(settle_s))
    return 0
