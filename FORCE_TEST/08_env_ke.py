#!/usr/bin/env python3
"""Open-loop environment stiffness Ke.  Force loop OFF.  Not the velocity servo.

What: seek, then a constant +press velocity.  Ke ≈ ΔF / Δx.
Why: later force work needs a contact stiffness number.  This is the
     pad/tissue, not an inner-loop parameter.  Do not start hybrid.

After the press: retract tool −Z until F drops, then MOVEJ mid-stroke.
DATA/08_ke/ keeps only the latest take.  Each Ke and --site is appended
to DATA/ke_sites.csv, which wipe does not touch.

This is the only script in this folder that is not a velocity-servo ID.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import load_pair, mpl, save_fig
from io_csv import col, write_json
from goto_mid import add_movej_args, go_mid, go_mid_from_args
from paths import DATA, add_playground, dry_exit, kind_dirs, stamp, write_readme

SITES_CSV = DATA / "ke_sites.csv"
SITE_FIELDS = (
    "collected_at",
    "site",
    "ke_n_m",
    "fz0_n",
    "fz1_n",
    "dF_n",
    "dx_mm",
    "press_mm_s",
    "press_s",
    "aborted",
)


def append_site_row(row: dict) -> Path:
    """Append one Ke/site row.  Lives outside DATA/08_ke/ so wipe keeps it."""
    DATA.mkdir(parents=True, exist_ok=True)
    fresh = not SITES_CSV.is_file()
    with SITES_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SITE_FIELDS))
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SITE_FIELDS})
    print(f"[SITES] {row.get('site') or '(no --site)'}  Ke={row.get('ke_n_m')}  {SITES_CSV}", flush=True)
    return SITES_CSV


def analyze(
    csv_path: Path,
    *,
    when: str,
    site: str = "",
    press_mm_s: float | None = None,
    press_s: float | None = None,
    aborted: bool = False,
    record_site: bool = True,
) -> dict:
    t, u, y, _fb, dt_col, rows = load_pair(csv_path)
    fz = col(rows, "fz")
    dt = np.where(np.isfinite(dt_col) & (dt_col > 1e-6), dt_col, 0.005)
    press = np.array([str(r.get("phase") or "").startswith("press") for r in rows])
    if not np.any(press):
        press = np.isfinite(fz) & (fz > 0.4) & (u > 0.001)
    dx = np.cumsum(np.nan_to_num(u, nan=0.0) * dt)
    if int(np.count_nonzero(press)) >= 4:
        fz0 = float(fz[press][0])
        fz1 = float(fz[press][-1])
        dF = float(fz1 - fz0)
        dxp = float(dx[press][-1] - dx[press][0])
        ke = float(dF / dxp) if abs(dxp) > 1e-6 else float("nan")
    else:
        fz0 = fz1 = dF = dxp = float("nan")
        ke = float("nan")
    payload = {
        "csv": str(csv_path),
        "site": site,
        "ke_n_m": ke,
        "fz0_n": fz0,
        "fz1_n": fz1,
        "dF_n": dF,
        "dx_mm": 1e3 * dxp if math.isfinite(dxp) else float("nan"),
        "press_mm_s": press_mm_s,
        "press_s": press_s,
        "aborted": bool(aborted),
        "what": "environment stiffness, not inner-loop Gv",
        "collected_at": when,
        "sites_csv": str(SITES_CSV),
    }
    data, visu = kind_dirs("08_ke", preserve=csv_path)
    write_json(data / "ke.json", payload)
    plt = mpl()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    t0 = float(t[0]) if t.size else 0.0
    ax.plot(t - t0, fz)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Fz (N)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, visu, "press.png")
    plt.close(fig)
    site_txt = site if site else "（未标部位，用 --site）"
    write_readme(
        visu,
        f"""# 08_ke — 开环环境刚度

采集：`{when}` · filter OFF · 力环关 · 开环下压 · 部位 **{site_txt}**

## 结论

- **Ke ≈ {ke:.1f} N/m**（ΔF/Δx，press 段）。这是垫/组织，不是内环 Gv。
- 接触后 Fz {fz0:.2f} → {fz1:.2f} N。最新一拍在 `DATA/08_ke/`（会被下次覆盖）。
- Ke 与部位追加在 `{SITES_CSV.name}`，wipe 不删。

不要开混合。不要写进 yaml，除非明确要写。
""",
    )
    if record_site:
        append_site_row(
            {
                "collected_at": when,
                "site": site,
                "ke_n_m": f"{ke:.3f}" if math.isfinite(ke) else "",
                "fz0_n": f"{fz0:.3f}" if math.isfinite(fz0) else "",
                "fz1_n": f"{fz1:.3f}" if math.isfinite(fz1) else "",
                "dF_n": f"{dF:.3f}" if math.isfinite(dF) else "",
                "dx_mm": f"{1e3 * dxp:.3f}" if math.isfinite(dxp) else "",
                "press_mm_s": "" if press_mm_s is None else f"{press_mm_s:.3f}",
                "press_s": "" if press_s is None else f"{press_s:.3f}",
                "aborted": "1" if aborted else "0",
            }
        )
    print(f"[KE] {ke:.1f} N/m  site={site_txt}  data={data}", flush=True)
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="", help="analyze an existing press CSV")
    p.add_argument("--site", default="", help="body site or pad name; appended to DATA/ke_sites.csv")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--seek-mm-s", type=float, default=10.0)
    p.add_argument("--press-mm-s", type=float, default=3.0)
    p.add_argument("--press-s", type=float, default=2.0)
    p.add_argument("--retract-mm-s", type=float, default=8.0)
    p.add_argument("--retract-s", type=float, default=2.5)
    p.add_argument("--contact-n", type=float, default=0.40)
    p.add_argument("--abort-n", type=float, default=4.50)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument(
        "--skip-return-movej",
        action="store_true",
        help="do not MOVEJ mid-stroke after the press",
    )
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    site = str(args.site).strip()
    if not args.csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, seek + press {args.press_mm_s:.1f} mm/s "
            f"{args.press_s:.1f}s, retract −Z, MOVEJ mid  "
            f"site={site or '(tell me after)'}  abort F≥{args.abort_n:.1f} N  "
            f"force loop OFF  log→{SITES_CSV.name}",
            flush=True,
        )
    if dry_exit(args):
        return 0
    if args.csv:
        analyze(
            Path(args.csv),
            when=when,
            site=site,
            press_mm_s=args.press_mm_s,
            press_s=args.press_s,
        )
        return 0
    rc_m = go_mid_from_args(args)
    if rc_m:
        return rc_m
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("08_ke")
    log = data / "press.csv"
    srv = ServoLogger(
        prefix=args.shm_prefix, hz=args.hz, log_csv=log, abort_n=args.abort_n
    )
    rc = 0
    latched = False
    estop = False
    try:
        srv.start_twist()
        t_seek = time.monotonic()
        while time.monotonic() - t_seek < 12.0:
            if not srv.tick(args.seek_mm_s / 1000.0, "seek"):
                estop = not srv.aborted
                rc = 130
                break
            if math.isfinite(srv.last_fz) and srv.last_fz > args.contact_n:
                latched = True
                print(f"[CONTACT] Fz={srv.last_fz:.2f} N — press", flush=True)
                break
            time.sleep(srv.dt)
        if rc == 0 and not latched:
            print("[ERR] no contact", flush=True)
            rc = 2
        if latched and rc == 0:
            if not srv.hold(args.press_mm_s / 1000.0, args.press_s, "press"):
                estop = not srv.aborted
                rc = 0 if srv.aborted else 130
        if latched and not estop:
            srv.retract_z(
                args.retract_mm_s / 1000.0,
                args.retract_s,
                args.contact_n,
            )
        srv.tick(0.0, "done", check_abort=False)
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
        rc = 130
    finally:
        srv.close()
    if not args.skip_return_movej and not estop:
        print("[MOVEJ] return mid-stroke after press", flush=True)
        rc_back = go_mid(
            prefix=str(args.shm_prefix),
            skip=False,
            v=float(args.movej_v),
            settle_s=0.0,
        )
        if rc_back and rc == 0:
            rc = rc_back
    elif estop:
        print("[MOVEJ] skipped (ESTOP)", flush=True)
    if log.is_file() and srv.n_rows > 16:
        analyze(
            log,
            when=when,
            site=site,
            press_mm_s=args.press_mm_s,
            press_s=args.press_s,
            aborted=bool(srv.aborted),
        )
        if not site:
            print("[SITES] no --site; tell me the site and I will fill that row", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
