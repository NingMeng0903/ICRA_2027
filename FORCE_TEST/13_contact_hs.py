#!/usr/bin/env python3
"""Force–moment–scan coupling Hs.  Force loop OFF.

What: quasi-static z/tilt, hold (relaxation), flat vs known-slope scans
      at more than one ρ, both directions, then a small tilt-sign probe.
Why: C2 needs a physical Hs ρ term.  If ρ and uz stay collinear the
      regression is rank-deficient.  90° is not a preset law.

Torque columns need the playground Window A patch (tx, ty, tz).
Without them the moment half degrades and the JSON says so.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_log import ContactLogger, axis_twist
from goto_mid import go_mid_from_args
from id_common import add_contact_args, load_aligned, stash_window_a
from io_csv import col, write_json
from paper_fig import ACH, MINUS, mpl, panel_tag, save
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import (
    AlignmentError,
    fmt_finite,
    has_torque,
    integrate_along_axis,
    phase_mask,
    pose6,
    tool_z_displacement,
    twist_ach6,
    wrench6,
)

KIND = "13_contact_hs"


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _slope(x: np.ndarray, y: np.ndarray) -> dict:
    xx, yy = _finite_xy(np.asarray(x, float), np.asarray(y, float))
    if xx.size < 8 or float(np.std(xx)) < 1e-6:
        return {"n": int(xx.size), "slope": float("nan"), "r2": float("nan")}
    a = np.vstack([xx - xx[0], np.ones(xx.size)]).T
    coef, *_ = np.linalg.lstsq(a, yy, rcond=None)
    pred = a @ coef
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"n": int(xx.size), "slope": float(coef[0]), "r2": r2}


def _mean_rate(t: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    tt = t[mask]
    yy = y[mask]
    m = np.isfinite(tt) & np.isfinite(yy)
    if int(np.count_nonzero(m)) < 6:
        return float("nan")
    dt = float(tt[m][-1] - tt[m][0])
    if abs(dt) < 1e-3:
        return float("nan")
    return float((yy[m][-1] - yy[m][0]) / dt)


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    slope_deg: float | None = None,
    scan_axis: int = 0,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    pose = pose6(rows)
    wrench = wrench6(rows)
    ach = twist_ach6(rows)
    dx_z = tool_z_displacement(pose)
    ds = integrate_along_axis(pose, scan_axis)
    fz = wrench[:, 2]
    tau = wrench[:, 4]
    torque_ok = has_torque(rows)
    quasi_z = _slope(dx_z[phase_mask(rows, "quasi_z")], fz[phase_mask(rows, "quasi_z")])
    quasi_th = _slope(
        col(rows, "pose_ry", "pose_meas_ry")[phase_mask(rows, "quasi_th")],
        tau[phase_mask(rows, "quasi_th")],
    )
    hold_press = _mean_rate(t, fz, phase_mask(rows, "hold_press"))
    hold_unload = _mean_rate(t, fz, phase_mask(rows, "hold_unload"))
    scans = {}
    for name in ("scan_flat_slow", "scan_flat_fast", "scan_slope_slow", "scan_slope_fast", "scan_rev"):
        mask = phase_mask(rows, name)
        scans[name] = {
            "df_ds": _slope(ds[mask], fz[mask]),
            "rho_proxy_m_s": float(np.nanmedian(np.abs(ach[mask, scan_axis]))) if np.any(mask) else float("nan"),
            "n": int(np.count_nonzero(mask)),
        }
    hs_flat = float(scans["scan_flat_fast"]["df_ds"].get("slope") or float("nan"))
    hs_slope = float(scans["scan_slope_fast"]["df_ds"].get("slope") or float("nan"))
    hs_identifiable = (
        math.isfinite(hs_flat)
        and math.isfinite(hs_slope)
        and abs(hs_slope - hs_flat) > 20.0
    )
    tilt_up = _mean_rate(t, fz, phase_mask(rows, "tilt_up"))
    tilt_dn = _mean_rate(t, fz, phase_mask(rows, "tilt_dn"))
    sign_ok = math.isfinite(tilt_up) and math.isfinite(tilt_dn) and (tilt_up * tilt_dn) < 0.0
    # Rank of [vz, ωθ, ρ] vs df on slope+flat scans.
    mask_reg = phase_mask(rows, "scan_flat_fast", "scan_slope_fast", "scan_rev")
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt) & (dt > 1e-4), dt, 0.005)
    df = np.diff(fz, prepend=fz[0]) / dt
    X = np.column_stack([ach[:, 2], ach[:, 4], ach[:, scan_axis]])[mask_reg]
    y = df[mask_reg]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    rank = int(np.linalg.matrix_rank(X[ok], tol=1e-6)) if int(np.count_nonzero(ok)) > 12 else 0
    cond = float(np.linalg.cond(X[ok])) if rank >= 2 else float("nan")
    payload = {
        "csv": str(csv_path),
        "align": align,
        "torque_present": bool(torque_ok),
        "ke_quasi_n_m": quasi_z.get("slope"),
        "ke_quasi_r2": quasi_z.get("r2"),
        "k_theta_nm_per_rad": quasi_th.get("slope"),
        "hold_press_n_s": hold_press,
        "hold_unload_n_s": hold_unload,
        "scans": scans,
        "Hs_flat_n_m": hs_flat,
        "Hs_slope_n_m": hs_slope,
        "Hs_identifiable": bool(hs_identifiable),
        "tilt_up_dfdt": tilt_up,
        "tilt_dn_dfdt": tilt_dn,
        "tilt_sign_opposite": bool(sign_ok),
        "reg_rank": rank,
        "reg_cond": cond,
        "slope_deg": slope_deg,
        "scan_axis": scan_axis,
        "collected_at": when,
        "what": "contact Hs / tilt sign, not a 90-deg bounce law",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "hs.json", payload)
    _plot_13(ds, fz, phase_mask(rows, "scan_flat_fast"), phase_mask(rows, "scan_slope_fast"), visu)
    hs_note = (
        "坠角与平面的 df/ds 能分开，Hsρ 可以进调速。"
        if hs_identifiable
        else "Hs 几乎看不出来：ρ 先当保守扰动包络，不能当理论核心。"
    )
    write_readme(
        visu,
        f"""# 13_contact_hs — 力–力矩–扫描耦合

采集：`{when}` · filter OFF · 力环关 · 力矩列 {'在' if torque_ok else '**缺失**（旧 Window A 或未打补丁）'}

## 结论

- 准静态 Ke ≈ {fmt_finite(quasi_z.get('slope'), '.1f')} N/m（R² {fmt_finite(quasi_z.get('r2'), '.2f')}），位移是实际 pose。
- 平面 df/ds ≈ {fmt_finite(hs_flat, '.1f')} N/m，坠角 ≈ {fmt_finite(hs_slope, '.1f')} N/m。{hs_note}
- 回归矩阵秩 {rank}，条件数 {fmt_finite(cond, '.1f')}。秩 < 2 时 uz 与 ρ 共线，模型不可辨。
- 倾角增大/减小的 df/dt 符号{('相反，可做定性调节。' if sign_ok else '没有可靠反号，不要写硬攻角约束。')}
- 不要把 90° 写成物理定律。阈值从这些符号和已知楔角出。

## 图

`hs_scan.png`：平面 vs 坠角扫描的 Fz 对路径弧长。
""",
    )
    print(
        f"[13] Ke={quasi_z.get('slope')}  Hs_id={hs_identifiable}  rank={rank}  data={data}",
        flush=True,
    )
    return payload


def _plot_13(ds, fz, flat, slope, visu: Path) -> None:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    if np.any(flat):
        ax.plot(1e3 * ds[flat], fz[flat], color=ACH, lw=1.15)
    if np.any(slope):
        ax.plot(1e3 * ds[slope], fz[slope], color=MINUS, lw=1.15)
    ax.set_xlabel("path travel (mm)")
    ax.set_ylabel("Fz (N)")
    ax.grid(True, alpha=0.28)
    save(fig, visu, "hs_scan.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.40)
    p.add_argument("--slope-deg", type=float, default=float("nan"), help="known wedge angle, for the log only")
    p.add_argument("--scan-mm-s", type=float, default=8.0)
    p.add_argument("--scan-fast-mm-s", type=float, default=16.0)
    p.add_argument("--scan-s", type=float, default=1.2)
    p.add_argument("--quasi-mm-s", type=float, default=2.0)
    p.add_argument("--tilt-deg-s", type=float, default=6.0)
    args = p.parse_args()
    when = stamp()
    print(
        f"[PLAN] MOVEJ mid, seek, quasi z/tilt, hold, flat+slope scans both "
        f"speeds and reverse, tilt sign  force loop OFF  "
        f"slope_deg={args.slope_deg}",
        flush=True,
    )
    if dry_exit(args):
        return 0
    if args.csv:
        try:
            analyze(
                Path(args.csv),
                when=when,
                window_a_csv=args.window_a_csv,
                slope_deg=args.slope_deg,
                scan_axis=args.scan_axis,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        return 0
    if not args.window_a_csv:
        print("[ERR] --window-a-csv is required", flush=True)
        return 2
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    data, _visu = kind_dirs(KIND)
    log = data / "hs.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
    )
    slow = args.scan_mm_s / 1000.0
    fast = args.scan_fast_mm_s / 1000.0
    qz = args.quasi_mm_s / 1000.0
    wth = math.radians(args.tilt_deg_s)
    try:
        srv.start_twist()
        if not srv.seek_contact(0.008, contact_n=args.contact_n):
            return 2
        if not srv.hold(axis_twist(2, qz), 1.0, "quasi_z"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.theta_axis, 0.5 * wth), 0.8, "quasi_th"):
            return 0 if srv.aborted else 130
        if not srv.hold(0.0, 1.5, "hold_press", check_abort=False):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, -qz), 0.5, "hold_unload", check_abort=False):
            return 0 if srv.aborted else 130
        if not srv.hold(0.0, 1.0, "hold_unload", check_abort=False):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, slow), args.scan_s, "scan_flat_slow"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, fast), args.scan_s, "scan_flat_fast"):
            return 0 if srv.aborted else 130
        print("[SCAN] put the probe on the known slope, then the next two moves run", flush=True)
        if not srv.hold(0.0, 1.0, "pause_slope", check_abort=False):
            return 130
        if not srv.hold(axis_twist(args.scan_axis, slow), args.scan_s, "scan_slope_slow"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, fast), args.scan_s, "scan_slope_fast"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, -slow), args.scan_s, "scan_rev"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.theta_axis, wth), 0.5, "tilt_up"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.theta_axis, -wth), 0.5, "tilt_dn"):
            return 0 if srv.aborted else 130
        srv.retract_z(0.008, 2.5, args.contact_n)
        srv.tick(0.0, "done", check_abort=False)
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
        return 130
    finally:
        srv.close()
    try:
        stash_window_a(args.window_a_csv, data)
        if log.is_file() and srv.n_rows > 64:
            analyze(
                log,
                when=when,
                window_a_csv=args.window_a_csv,
                slope_deg=args.slope_deg,
                scan_axis=args.scan_axis,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
