#!/usr/bin/env python3
"""Same inner Gv, but already in light contact.  Force loop still OFF.

What: seek until F>contact_n, then one small velocity chirp.  Abort if F high.
Why: check that air T0/Tp still hold at ~1 N.  This is still vel_ff → v_ach,
     not a force controller.  Hard first-touch (4–10 N) is out of scope.

If the chirp unloads the pad, the file is still Gv of the servo, just with
a changing load.  Compare T0 to 03_chirp_gv.py, not to a force Bode.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goto_mid import add_movej_args, go_mid_from_args
from paths import add_playground, dry_exit, kind_dirs, stamp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="", help="analyze with 03_chirp_gv.py --csv")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--amp-mm-s", type=float, default=5.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=10.0)
    p.add_argument("--chirp-s", type=float, default=45.0)
    p.add_argument("--seek-mm-s", type=float, default=8.0)
    p.add_argument("--contact-n", type=float, default=0.40)
    p.add_argument("--abort-n", type=float, default=3.50)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    if not args.csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, then seek {args.seek_mm_s:.1f} mm/s until "
            f"F>{args.contact_n:.2f} N, then {args.amp_mm_s:.1f} mm/s chirp  "
            f"force loop OFF  abort F>{args.abort_n:.1f}",
            flush=True,
        )
    if dry_exit(args):
        return 0
    if args.csv:
        from importlib import import_module

        import_module("03_chirp_gv").analyze(
            Path(args.csv),
            when=stamp(),
            f1=args.f1,
            amp_mm_s=args.amp_mm_s,
            prefix="07_contact_gv",
        )
        return 0
    rc_m = go_mid_from_args(args)
    if rc_m:
        return rc_m
    add_playground()
    from servo_log import ServoLogger

    when = stamp()
    data, _visu = kind_dirs("07_contact_gv")
    log = data / "contact_gv.csv"
    srv = ServoLogger(
        prefix=args.shm_prefix, hz=args.hz, log_csv=log, abort_n=args.abort_n
    )
    rc = 0
    try:
        srv.start_twist()
        t_seek = time.monotonic()
        latched = False
        while time.monotonic() - t_seek < 12.0:
            if not srv.tick(args.seek_mm_s / 1000.0, "seek"):
                rc = 130
                break
            if math.isfinite(srv.last_fz) and srv.last_fz > args.contact_n:
                latched = True
                print(f"[CONTACT] Fz={srv.last_fz:.2f} N", flush=True)
                break
            time.sleep(srv.dt)
        if rc != 0:
            return rc
        if not latched:
            print("[ERR] no contact", flush=True)
            return 2
        if not srv.chirp(args.amp_mm_s / 1000.0, args.f0, args.f1, args.chirp_s, "chirp"):
            rc = 0 if srv.aborted else 130
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
            prefix="07_contact_gv",
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
