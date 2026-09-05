#!/usr/bin/env python3
"""Force–moment–scan coupling H.  Force loop OFF.

What: one known attack angle per take.  Seek, settle, then scans with
      independent δvz, δωθ on two ρ and reverse, plus ±tilt for sign.
Why: H = H(α, F, surface).  Flat and a wedge in the same take must not
     share one regression.  Run α = −10/−5/0/+5/+10 separately.

Moment is τ_C = τ_TCP − r_TC × F.  Tilt angle is
θ = Log(R_ref^T R)^∨ · e_θ, not global Euler ry.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_log import ContactLogger, axis_twist
from goto_mid import go_mid_from_args
from id_common import add_contact_args, load_aligned, stash_window_a
from id_math import dump_jsonable, scan_perturb_twists, wrench_at_offset
from io_csv import col, write_json
from paper_fig import ACH, MINUS, mpl, save
from paths import DATA, add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import (
    AlignmentError,
    fmt_finite,
    has_torque,
    integrate_along_axis,
    phase_mask,
    pose6,
    relative_axis_angle,
    tool_z_displacement,
    twist_ach6,
    twist_cmd6,
    wrench6,
)

KIND = "13_contact_hs"
ALPHA_CSV = DATA / "hs_alphas.csv"
ALPHA_FIELDS = (
    "collected_at",
    "alpha_deg",
    "site",
    "F_mean_n",
    "tau_mean_nm",
    "df_ds_n_m",
    "Hz",
    "Hth",
    "Hs",
    "reg_rank",
    "tilt_sign_opposite",
)


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


def _fit_H(vz, wth, rho, df) -> dict:
    X = np.column_stack([vz, wth, rho])
    y = np.asarray(df, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = int(np.count_nonzero(ok))
    if n < 24:
        return {"n": n, "rank": 0, "cond": float("nan"), "Hz": float("nan"), "Hth": float("nan"), "Hs": float("nan")}
    Xo, yo = X[ok], y[ok]
    rank = int(np.linalg.matrix_rank(Xo, tol=1e-6))
    cond = float(np.linalg.cond(Xo)) if rank >= 1 else float("nan")
    coef, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    return {
        "n": n,
        "rank": rank,
        "cond": cond,
        "Hz": float(coef[0]),
        "Hth": float(coef[1]),
        "Hs": float(coef[2]),
    }


def append_alpha_row(row: dict) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    fresh = not ALPHA_CSV.is_file()
    with ALPHA_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALPHA_FIELDS))
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in ALPHA_FIELDS})
    print(f"[ALPHA] α={row.get('alpha_deg')}  {ALPHA_CSV}", flush=True)
    return ALPHA_CSV


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    alpha_deg: float | None = None,
    slope_deg: float | None = None,
    scan_axis: int = 0,
    theta_axis: int = 4,
    site: str = "",
    r_tcp_m: tuple[float, float, float] = (0.0, 0.0, 0.03),
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    pose = pose6(rows)
    wrench = wrench6(rows)
    ach = twist_ach6(rows)
    cmd = twist_cmd6(rows)
    dx_z = tool_z_displacement(pose)
    ds = integrate_along_axis(pose, scan_axis)
    fz = wrench[:, 2]
    r = np.asarray(r_tcp_m, dtype=float).reshape(3)
    wrench_c = wrench_at_offset(wrench, r)
    e_th = np.zeros(3, dtype=float)
    e_th[int(theta_axis) % 3] = 1.0
    tau_th = wrench_c[:, 3:6] @ e_th
    theta = relative_axis_angle(pose, int(theta_axis) % 3)
    torque_ok = has_torque(rows)
    if alpha_deg is None or not math.isfinite(float(alpha_deg)):
        alpha_deg = slope_deg
    quasi_z = _slope(dx_z[phase_mask(rows, "quasi_z")], fz[phase_mask(rows, "quasi_z")])
    quasi_th = _slope(theta[phase_mask(rows, "quasi_th")], tau_th[phase_mask(rows, "quasi_th")])
    hold_press = _mean_rate(t, fz, phase_mask(rows, "hold_press"))
    hold_unload = _mean_rate(t, fz, phase_mask(rows, "hold_unload"))
    scans = {}
    for name in ("scan_slow", "scan_fast", "scan_rev"):
        mask = phase_mask(rows, name)
        scans[name] = {
            "df_ds": _slope(ds[mask], fz[mask]),
            "rho_proxy_m_s": float(np.nanmedian(np.abs(ach[mask, scan_axis]))) if np.any(mask) else float("nan"),
            "n": int(np.count_nonzero(mask)),
        }
    hs_this = float(scans["scan_fast"]["df_ds"].get("slope") or float("nan"))
    tilt_up = _mean_rate(t, fz, phase_mask(rows, "tilt_up"))
    tilt_dn = _mean_rate(t, fz, phase_mask(rows, "tilt_dn"))
    sign_ok = math.isfinite(tilt_up) and math.isfinite(tilt_dn) and (tilt_up * tilt_dn) < 0.0
    mask_reg = phase_mask(rows, "scan_slow", "scan_fast", "scan_rev")
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt) & (dt > 1e-4), dt, 0.005)
    df = np.diff(fz, prepend=fz[0]) / dt
    # Prefer achieved motion; fall back to command if Window A ωθ is empty.
    vz = ach[:, 2]
    wth = ach[:, int(theta_axis)]
    if not np.isfinite(wth[mask_reg]).any():
        wth = cmd[:, int(theta_axis)]
    rho = ach[:, int(scan_axis)]
    H = _fit_H(vz[mask_reg], wth[mask_reg], rho[mask_reg], df[mask_reg])
    rank_ok = int(H.get("rank") or 0) >= 3
    F_mean = float(np.nanmean(fz[mask_reg])) if np.any(mask_reg) else float("nan")
    tau_mean = float(np.nanmean(tau_th[mask_reg])) if np.any(mask_reg) else float("nan")
    hs_identifiable = bool(rank_ok and math.isfinite(float(H.get("Hs") or float("nan"))))
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
        "Hs_n_m": hs_this,
        "Hs_identifiable": hs_identifiable,
        "H": H,
        "Hz": H.get("Hz"),
        "Hth": H.get("Hth"),
        "Hs": H.get("Hs"),
        "reg_rank": H.get("rank"),
        "reg_cond": H.get("cond"),
        "rank3": bool(rank_ok),
        "tilt_up_dfdt": tilt_up,
        "tilt_dn_dfdt": tilt_dn,
        "tilt_sign_opposite": bool(sign_ok),
        "alpha_deg": alpha_deg,
        "r_C_tool_m": [float(r[0]), float(r[1]), float(r[2])],
        "theta_source": "so3_log_relative",
        "tau_source": "tau_C = tau_TCP - r × F",
        "features": {"F_mean_n": F_mean, "tau_C_mean_nm": tau_mean, "df_ds_n_m": hs_this},
        "scan_axis": scan_axis,
        "theta_axis": theta_axis,
        "site": site,
        "collected_at": when,
        "what": "H(α) from one surface per take; not a mixed flat+wedge plant",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "hs.json", dump_jsonable(payload))
    _plot_13(ds, fz, phase_mask(rows, "scan_slow"), phase_mask(rows, "scan_fast"), visu)
    if rank_ok:
        h_note = (
            f"这一拍 α = {fmt_finite(float(alpha_deg) if alpha_deg is not None else float('nan'), '.1f')}°，"
            f"秩 3，H = [{fmt_finite(H.get('Hz'), '.1f')}, "
            f"{fmt_finite(H.get('Hth'), '.1f')}, {fmt_finite(H.get('Hs'), '.1f')}]。"
        )
    else:
        h_note = (
            f"这一拍 α = {fmt_finite(float(alpha_deg) if alpha_deg is not None else float('nan'), '.1f')}°，"
            f"回归秩 {H.get('rank')}（条件数 {fmt_finite(H.get('cond'), '.1f')}）。还不能把完整 H 写进 QP。"
        )
    write_readme(
        visu,
        f"""# 13_contact_hs — 力–力矩–扫描耦合

采集：`{when}` · filter OFF · 力环关 · **一拍一个 α** = {fmt_finite(float(alpha_deg) if alpha_deg is not None else float('nan'), '.1f')}° · 力矩列 {'在' if torque_ok else '**缺失**'}

## 结论

- 准静态 Ke ≈ {fmt_finite(quasi_z.get('slope'), '.1f')} N/m。k_θ 用 τ_C 对相对 Log 角，不是 global ry。
- {h_note}
- 倾角增大/减小的 df/dt 符号{('相反，可做定性调节。' if sign_ok else '没有可靠反号，不要写硬攻角约束。')}
- 特征 (F̄, τ_C̄, df/ds) = ({fmt_finite(F_mean, '.2f')}, {fmt_finite(tau_mean, '.3f')}, {fmt_finite(hs_this, '.1f')})。
  粗 α 映射要 −10/−5/0/+5/+10° **各一拍**，见 `{ALPHA_CSV.name}`。不要在同一 take 里换表面。
- r_C = {r.tolist()} m（tool）。不要把 90° 写成物理定律。

## 图

`hs_scan.png`：这一拍两个 ρ 的 Fz 对路径弧长。
""",
    )
    append_alpha_row(
        {
            "collected_at": when,
            "alpha_deg": "" if alpha_deg is None or not math.isfinite(float(alpha_deg)) else f"{float(alpha_deg):.2f}",
            "site": site,
            "F_mean_n": f"{F_mean:.3f}" if math.isfinite(F_mean) else "",
            "tau_mean_nm": f"{tau_mean:.4f}" if math.isfinite(tau_mean) else "",
            "df_ds_n_m": f"{hs_this:.3f}" if math.isfinite(hs_this) else "",
            "Hz": f"{float(H.get('Hz') or float('nan')):.4f}" if math.isfinite(float(H.get("Hz") or float("nan"))) else "",
            "Hth": f"{float(H.get('Hth') or float('nan')):.4f}" if math.isfinite(float(H.get("Hth") or float("nan"))) else "",
            "Hs": f"{float(H.get('Hs') or float('nan')):.4f}" if math.isfinite(float(H.get("Hs") or float("nan"))) else "",
            "reg_rank": str(H.get("rank") or 0),
            "tilt_sign_opposite": "1" if sign_ok else "0",
        }
    )
    print(
        f"[13] rank={H.get('rank')}  α={alpha_deg}  Hs={H.get('Hs')}  data={data}",
        flush=True,
    )
    return payload


def _plot_13(ds, fz, slow, fast, visu: Path) -> None:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    if np.any(slow):
        ax.plot(1e3 * ds[slow], fz[slow], color=ACH, lw=1.15)
    if np.any(fast):
        ax.plot(1e3 * ds[fast], fz[fast], color=MINUS, lw=1.15)
    ax.set_xlabel("path travel (mm)")
    ax.set_ylabel("Fz (N)")
    ax.grid(True, alpha=0.28)
    save(fig, visu, "hs_scan.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.40)
    p.add_argument("--alpha-deg", type=float, default=float("nan"), help="known attack angle of THIS take; required to collect")
    p.add_argument("--slope-deg", type=float, default=float("nan"), help="alias of --alpha-deg")
    p.add_argument("--rx-mm", type=float, default=0.0)
    p.add_argument("--ry-mm", type=float, default=0.0)
    p.add_argument("--rz-mm", type=float, default=30.0, help="tool-frame contact offset r_C")
    p.add_argument("--site", default="")
    p.add_argument("--scan-mm-s", type=float, default=8.0)
    p.add_argument("--scan-fast-mm-s", type=float, default=16.0)
    p.add_argument("--scan-s", type=float, default=2.0)
    p.add_argument("--dvz-mm-s", type=float, default=1.5)
    p.add_argument("--dth-deg-s", type=float, default=3.0)
    p.add_argument("--quasi-mm-s", type=float, default=2.0)
    p.add_argument("--tilt-deg-s", type=float, default=6.0)
    args = p.parse_args()
    when = stamp()
    alpha = args.alpha_deg if math.isfinite(args.alpha_deg) else args.slope_deg
    print(
        f"[PLAN] MOVEJ mid, one-α take α={alpha}°, seek, quasi, hold, "
        f"scan ρ1/ρ2/reverse + independent δvz/δωθ, ±tilt  "
        f"r_C=[{args.rx_mm:.1f},{args.ry_mm:.1f},{args.rz_mm:.1f}] mm  force loop OFF  "
        "do not change the wedge mid-take",
        flush=True,
    )
    if dry_exit(args):
        return 0
    r_tcp = (args.rx_mm / 1000.0, args.ry_mm / 1000.0, args.rz_mm / 1000.0)
    if args.csv:
        try:
            analyze(
                Path(args.csv),
                when=when,
                window_a_csv=args.window_a_csv,
                alpha_deg=alpha,
                slope_deg=args.slope_deg,
                scan_axis=args.scan_axis,
                theta_axis=args.theta_axis,
                site=args.site,
                r_tcp_m=r_tcp,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        return 0
    if not math.isfinite(float(alpha)):
        print("[ERR] --alpha-deg is required for a collect (0 = flat). One angle per take.", flush=True)
        return 2
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
        secondary=args.secondary,
    )
    slow = args.scan_mm_s / 1000.0
    fast = args.scan_fast_mm_s / 1000.0
    qz = args.quasi_mm_s / 1000.0
    wth = math.radians(args.tilt_deg_s)
    dvz = args.dvz_mm_s / 1000.0
    dth = math.radians(args.dth_deg_s)

    def _pert(rho, seconds, seed, phase):
        seq = scan_perturb_twists(
            srv.dt,
            seconds,
            rho,
            dvz,
            dth,
            scan_axis=args.scan_axis,
            theta_axis=args.theta_axis,
            seed=seed,
        )
        return srv.play(seq, phase)

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
        if not _pert(slow, args.scan_s, 3, "scan_slow"):
            return 0 if srv.aborted else 130
        if not _pert(fast, args.scan_s, 4, "scan_fast"):
            return 0 if srv.aborted else 130
        if not _pert(-slow, args.scan_s, 5, "scan_rev"):
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
                alpha_deg=alpha,
                slope_deg=args.slope_deg,
                scan_axis=args.scan_axis,
                theta_axis=args.theta_axis,
                site=args.site,
                r_tcp_m=r_tcp,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
