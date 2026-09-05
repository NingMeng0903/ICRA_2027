#!/usr/bin/env python3
"""Open-loop environment stiffness Ke.  Force loop OFF.

What: seek, then a constant +press velocity.  Ke ≈ ΔF / Δx_toolz.
Why: later force work needs a contact stiffness number.  This is the
     pad/tissue, not an inner-loop parameter.  Do not start hybrid.

Δx is the Window A TCP pose projected on the first-press tool-Z axis.
Command-velocity integration is not Ke — it folds the 28–40 ms delay
into the stiffness.

After the press: retract tool −Z until F drops, then MOVEJ mid-stroke.
DATA/08_ke/ keeps only the latest take.  Each Ke and --site is appended
to DATA/ke_sites.csv, which wipe does not touch.
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

from goto_mid import add_movej_args, go_mid, go_mid_from_args
from id_common import load_aligned, stash_window_a
from io_csv import col, write_json
from paper_fig import ACH, MINUS, mpl, panel_tag, save
from paths import DATA, add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import AlignmentError, add_window_a_arg, fmt_finite, phase_mask, tool_z_displacement

SITES_CSV = DATA / "ke_sites.csv"
SITE_FIELDS = (
    "collected_at",
    "site",
    "ke_n_m",
    "fz0_n",
    "fz1_n",
    "dF_n",
    "dx_mm",
    "dx_source",
    "press_mm_s",
    "press_s",
    "aborted",
    "align_gap_ms",
)
TAU_L_S = 0.040
V_E_EXAMPLE_M_S = 0.010


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


def _press_slice(rows, fz: np.ndarray) -> np.ndarray:
    press = phase_mask(rows, "press")
    if int(np.count_nonzero(press)) >= 4:
        return press
    return np.isfinite(fz) & (fz > 0.4)


def analyze(
    csv_path: Path,
    *,
    when: str,
    site: str = "",
    press_mm_s: float | None = None,
    press_s: float | None = None,
    aborted: bool = False,
    record_site: bool = True,
    window_a_csv: str = "",
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    fz = col(rows, "fz", "fz_raw_comp")
    pose_dx = tool_z_displacement(
        np.column_stack(
            [
                col(rows, "pose_x", "pose_meas_x"),
                col(rows, "pose_y", "pose_meas_y"),
                col(rows, "pose_z", "pose_meas_z"),
                col(rows, "pose_rx", "pose_meas_rx"),
                col(rows, "pose_ry", "pose_meas_ry"),
                col(rows, "pose_rz", "pose_meas_rz"),
            ]
        )
    )
    press = _press_slice(rows, fz)
    if int(np.count_nonzero(press)) >= 4:
        fz0 = float(fz[press][0])
        fz1 = float(fz[press][-1])
        dF = float(fz1 - fz0)
        x_p = pose_dx[press]
        if np.isfinite(x_p).sum() >= 2:
            dxp = float(x_p[np.isfinite(x_p)][-1] - x_p[np.isfinite(x_p)][0])
        else:
            dxp = float("nan")
        ke = float(dF / dxp) if math.isfinite(dxp) and abs(dxp) > 1e-6 else float("nan")
    else:
        fz0 = fz1 = dF = dxp = ke = float("nan")
    bound_n = (
        2.0 * float(ke) * V_E_EXAMPLE_M_S * TAU_L_S
        if math.isfinite(ke)
        else float("nan")
    )
    payload = {
        "csv": str(csv_path),
        "window_a_csv": align.get("window_a_csv"),
        "site": site,
        "ke_n_m": ke,
        "fz0_n": fz0,
        "fz1_n": fz1,
        "dF_n": dF,
        "dx_mm": 1e3 * dxp if math.isfinite(dxp) else float("nan"),
        "dx_source": "pose_tool_z",
        "press_mm_s": press_mm_s,
        "press_s": press_s,
        "aborted": bool(aborted),
        "align": align,
        "example_delay_bound_n": bound_n,
        "example_bound_note": "2 Ke * 10 mm/s * 40 ms; not a measured ve",
        "what": "environment stiffness from actual pose, not inner-loop Gv",
        "collected_at": when,
        "sites_csv": str(SITES_CSV),
    }
    data, visu = kind_dirs("08_ke", preserve=csv_path)
    write_json(data / "ke.json", payload)
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 3.80), sharex=True, constrained_layout=True)
    t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
    tt = t - t0
    axes[0].plot(tt, fz, color=ACH, lw=1.15)
    axes[1].plot(tt, 1e3 * pose_dx, color=MINUS, lw=1.15)
    if np.any(press):
        axes[0].axvspan(float(tt[press][0]), float(tt[press][-1]), color=ACH, alpha=0.08)
        axes[1].axvspan(float(tt[press][0]), float(tt[press][-1]), color=MINUS, alpha=0.08)
    axes[0].set_ylabel("Fz (N)")
    axes[1].set_ylabel("tool-Z travel (mm)")
    axes[1].set_xlabel("time (s)")
    for letter, ax in zip("ab", axes):
        ax.grid(True, alpha=0.28)
        panel_tag(ax, letter)
    save(fig, visu, "press.png")
    plt.close(fig)
    site_txt = site if site else "（未标部位，用 --site）"
    hard = math.isfinite(ke) and ke >= 500.0
    write_readme(
        visu,
        f"""# 08_ke — 开环环境刚度

采集：`{when}` · filter OFF · 力环关 · 开环下压 · 部位 **{site_txt}**

## 结论

- **Ke ≈ {fmt_finite(ke, '.1f')} N/m**（ΔF / 实际 tool-Z 位移，press 段）。这是垫/组织，不是内环 Gv。
- 位移来源是 Window A 位姿，**不是**命令积分。对齐中位间隙 {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms。
- 接触后 Fz {fmt_finite(fz0, '.2f')} → {fmt_finite(fz1, '.2f')} N，Δx = {fmt_finite(1e3 * dxp if math.isfinite(dxp) else float('nan'), '.2f')} mm。
- 量纲示例 2 Ke ve τℓ（ve=10 mm/s，τℓ=40 ms）≈ {fmt_finite(bound_n, '.2f')} N。{'硬垫，C1 下界有机会紧。' if hard else '若这是软垫，主实验必须另采硬垫才谈不可实现。'}
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
                "dx_source": "pose_tool_z",
                "press_mm_s": "" if press_mm_s is None else f"{press_mm_s:.3f}",
                "press_s": "" if press_s is None else f"{press_s:.3f}",
                "aborted": "1" if aborted else "0",
                "align_gap_ms": f"{float(align.get('gap_median_ms', float('nan'))):.2f}",
            }
        )
    print(
        f"[KE] {ke:.1f} N/m  dx={1e3 * dxp:.2f} mm pose  site={site_txt}  data={data}",
        flush=True,
    )
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
    add_window_a_arg(p)
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    site = str(args.site).strip()
    if not args.csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, seek + press {args.press_mm_s:.1f} mm/s "
            f"{args.press_s:.1f}s, retract −Z, MOVEJ mid  "
            f"site={site or '(tell me after)'}  abort F≥{args.abort_n:.1f} N  "
            f"force loop OFF  Δx=Window A pose  log→{SITES_CSV.name}",
            flush=True,
        )
        print("[PLAN] Window A must be started with --log-csv; pass --window-a-csv", flush=True)
    if dry_exit(args):
        return 0
    if args.csv:
        try:
            analyze(
                Path(args.csv),
                when=when,
                site=site,
                press_mm_s=args.press_mm_s,
                press_s=args.press_s,
                window_a_csv=args.window_a_csv,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        return 0
    if not args.window_a_csv:
        print("[ERR] --window-a-csv is required so Ke uses actual pose", flush=True)
        return 2
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
    try:
        stash_window_a(args.window_a_csv, data)
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    if log.is_file() and srv.n_rows > 16:
        try:
            analyze(
                log,
                when=when,
                site=site,
                press_mm_s=args.press_mm_s,
                press_s=args.press_s,
                aborted=bool(srv.aborted),
                window_a_csv=args.window_a_csv,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        if not site:
            print("[SITES] no --site; tell me the site and I will fill that row", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
