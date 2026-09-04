#!/usr/bin/env python3
"""Paper §II.A excitation: random-phase multisine on the velocity servo.

Samuel et al. put a multisine on the inner-loop command and record the
achieved motion, then average the FRF.  Their inner loop is position;
ours is velocity, so the command is vel_ff (Vr), the measurement is
v_ach (Vm).  Force loop stays off.

This is the Q-band record CDYOB needs before anyone talks about
active_model_validated.  A log chirp (03) is a substitute, not the paper
method.  One pose per run — change the pose and run again to average T(s).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goto_mid import add_movej_args, go_mid_from_args
from paths import add_playground, dry_exit, kind_dirs, stamp


def make_multisine(
    *,
    dt: float,
    amp_m_s: float,
    f0: float,
    f1: float,
    period_s: float,
    n_periods: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(round(float(period_s) / dt))
    t = np.arange(n) * dt
    freqs = np.arange(f0, f1 + 1e-12, 1.0 / max(period_s, 1e-6))
    if freqs.size == 0:
        freqs = np.array([f0])
    sig = np.zeros(n)
    for f in freqs:
        sig += np.sin(2.0 * math.pi * float(f) * t + float(rng.uniform(0.0, 2.0 * math.pi)))
    peak = float(np.max(np.abs(sig))) or 1.0
    one = (amp_m_s / peak) * sig
    return np.tile(one, int(max(n_periods, 1)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--amp-mm-s", type=float, default=15.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=10.0)
    p.add_argument("--period-s", type=float, default=20.0)
    p.add_argument("--periods", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    if args.csv:
        if dry_exit(args):
            return 0
        from importlib import import_module

        import_module("03_chirp_gv").analyze(
            Path(args.csv),
            when=when,
            f1=args.f1,
            amp_mm_s=args.amp_mm_s,
            prefix="09_multisine",
        )
        return 0
    dt = 1.0 / max(float(args.hz), 1.0)
    u = make_multisine(
        dt=dt,
        amp_m_s=args.amp_mm_s / 1000.0,
        f0=args.f0,
        f1=args.f1,
        period_s=args.period_s,
        n_periods=args.periods,
        seed=args.seed,
    )
    print(
        f"[PLAN] MOVEJ mid-stroke, then SERVO_TWIST multisine {args.amp_mm_s:.1f} mm/s  "
        f"{args.f0:.2f}–{args.f1:.1f} Hz  {args.periods}×{args.period_s:.0f}s  "
        f"lines≈{int((args.f1 - args.f0) * args.period_s)}  force loop OFF",
        flush=True,
    )
    print(
        "[PLAN] paper §II.A.  Repeat at other poses and compare T0/Tp.  "
        "Do not average in this script.",
        flush=True,
    )
    if dry_exit(args):
        return 0
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("09_multisine")
    log = data / "multisine.csv"
    srv = ServoLogger(prefix=args.shm_prefix, hz=args.hz, log_csv=log)
    try:
        srv.start_twist()
        if not srv.hold(0.0, 0.4, "rest"):
            return 130
        if not srv.play(u, "multisine"):
            return 130
        srv.tick(0.0, "done")
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
    finally:
        srv.close()
    if log.is_file() and srv.n_rows > 64:
        from importlib import import_module

        import_module("03_chirp_gv").analyze(
            log,
            when=when,
            f1=args.f1,
            amp_mm_s=args.amp_mm_s,
            prefix="09_multisine",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
