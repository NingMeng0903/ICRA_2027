"""MOVEJ to the taught mid-stroke, then wait for nullspace to finish.

Same target as ``peirastic.DEMO.movej``: rail 400 mm + taught arm angles.
After arrival, hold 5 s so the null-space posture can settle before any
SERVO_TWIST ID.  Force loop stays off.  Window A must already be running.
"""

from __future__ import annotations

import time

from paths import add_playground

SETTLE_S = 5.0


def add_movej_args(parser) -> None:
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
    settle = 0.0 if bool(getattr(args, "skip_settle", False)) else float(getattr(args, "settle_s", SETTLE_S))
    return go_mid(
        prefix=str(getattr(args, "shm_prefix", "")),
        skip=bool(getattr(args, "skip_movej", False)),
        v=float(getattr(args, "movej_v", 0.4)),
        settle_s=settle,
    )


def go_mid(
    *,
    prefix: str = "",
    skip: bool = False,
    v: float = 0.4,
    settle_s: float = SETTLE_S,
) -> int:
    """Block until MOVEJ arrives, then wait ``settle_s``.  Return 0 on success."""

    if skip:
        print("[MOVEJ] skipped", flush=True)
    else:
        add_playground()
        from peirastic.api import PeirasticArm
        from peirastic.api.codes import CODE_NAMES, OK
        from peirastic.DEMO.movej import _fmt_q, q_target_rad

        q = q_target_rad()
        print(f"[MOVEJ] mid-stroke  {_fmt_q(q)}  v={v:.2f}", flush=True)
        arm = PeirasticArm(prefix=str(prefix))
        try:
            ret = arm.movej(q, v=float(v), r=0, connect=0, block=1)
        except KeyboardInterrupt:
            arm.set_arm_stop()
            print("[MOVEJ] interrupted", flush=True)
            return 130
        finally:
            arm.close()
        if ret != OK:
            print(f"[ERR] MOVEJ -> {ret} ({CODE_NAMES.get(ret, ret)})", flush=True)
            return 1
        print("[OK] MOVEJ mid-stroke", flush=True)
    if settle_s > 0.0:
        print(f"[SETTLE] {settle_s:.1f}s for nullspace", flush=True)
        time.sleep(float(settle_s))
    return 0
