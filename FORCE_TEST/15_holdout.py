#!/usr/bin/env python3
"""Independent hold-out of the 08/11–14 envelopes.  Force loop OFF.

What: a mixed take that is *not* a copy of any ID script.  Analysis
      loads G, W, Ke_bar, T_tail, H, ē_P from DATA/ and only asks
      whether the new measurements sit inside those sets.
Why: coverage on the identification take is not a certificate.
     15 never writes a new G, Ke, H, or w̄.

Missing ID JSON → that check is skipped and reported, not fitted.
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
from id_common import add_contact_args, load_aligned, load_id_bundle, stash_window_a
from id_math import (
    dump_jsonable,
    port_power,
    predict_2x2,
    prefix_covers,
    prefix_debt,
    prefix_work,
)
from io_csv import col, write_json
from paper_fig import ACH, MINUS, mpl, save
import paths
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import (
    AlignmentError,
    fmt_finite,
    has_torque,
    pose6,
    pose_body_twist,
    twist_ach6,
    twist_cmd6,
    wrench6,
)

KIND = "15_holdout"


def _dt(t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0] + 0.005)
    if t.size > 2:
        dt[0] = float(np.nanmedian(np.diff(t[np.isfinite(t)])))
    return np.where(np.isfinite(dt) & (dt > 1e-4), dt, 0.005)


def _phase_edge(rows, name: str) -> int | None:
    idx = [i for i, r in enumerate(rows) if str(r.get("phase") or "").startswith(name)]
    return idx[0] if idx else None


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    id_root: str = "",
    theta_axis: int = 4,
    scan_axis: int = 0,
    horizon_s: float = 0.25,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    bundle = load_id_bundle(root=Path(id_root) if id_root else paths.DATA)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt = _dt(t)
    cmd = twist_cmd6(rows)
    ach = twist_ach6(rows)
    wrench = wrench6(rows)
    pose_tw = pose_body_twist(pose6(rows), dt)
    th = int(theta_axis)
    checks: dict[str, dict] = {}

    g = (bundle.get("11_exec_2dof") or {}).get("G")
    tube = (bundle.get("11_exec_2dof") or {}).get("tube") or {}
    if g:
        vz_hat, th_hat = predict_2x2(cmd[:, 2], cmd[:, th], float(np.median(dt)), g)
        ev = ach[:, 2] - vz_hat
        ew = ach[:, th] - th_hat
        bar_v = float(tube.get("e_vz_bar") or (tube.get("e_vz") or {}).get("bar") or float("nan"))
        bar_w = float(tube.get("e_wth_bar") or (tube.get("e_wth") or {}).get("bar") or float("nan"))
        cov_v = float(np.mean(np.abs(ev[np.isfinite(ev)]) <= bar_v + 1e-12)) if math.isfinite(bar_v) else float("nan")
        cov_w = float(np.mean(np.abs(ew[np.isfinite(ew)]) <= bar_w + 1e-12)) if math.isfinite(bar_w) else float("nan")
        checks["exec"] = {
            "cover_vz": cov_v,
            "cover_wth": cov_w,
            "e_vz_p95": float(np.nanpercentile(np.abs(ev), 95)),
            "e_wth_p95": float(np.nanpercentile(np.abs(ew), 95)),
            "bar_vz": bar_v,
            "bar_wth": bar_w,
            "ok": bool(math.isfinite(cov_v) and cov_v >= 0.95 and math.isfinite(cov_w) and cov_w >= 0.90),
        }
    else:
        checks["exec"] = {"skipped": True, "reason": "11_exec_2dof/exec.json missing"}

    ke08 = bundle.get("08_ke") or {}
    tail12 = bundle.get("12_stop_tail") or {}
    w_bar = float(tail12.get("w_bar_n") or float("nan"))
    ke_bar = float(ke08.get("ke_bar_n_m") or ke08.get("ke_n_m") or float("nan"))
    i0 = _phase_edge(rows, "stop_hold")
    if i0 is None:
        i0 = _phase_edge(rows, "stop_")
    if i0 is not None and math.isfinite(w_bar):
        n = max(int(round(horizon_s / float(np.median(dt)))), 8)
        fz = wrench[:, 2]
        f0 = float(fz[i0])
        peak = float(np.nanmax(fz[i0 : i0 + n]))
        df = peak - f0
        checks["force_tail"] = {
            "df_peak_n": df,
            "w_bar_n": w_bar,
            "inside": bool(math.isfinite(df) and abs(df) <= w_bar + 1e-9),
            "ke_bar_n_m": ke_bar,
        }
    else:
        checks["force_tail"] = {"skipped": True, "reason": "no stop phase or no 12 w_bar"}

    H = (bundle.get("13_contact_hs") or {}).get("H")
    if H and int(H.get("rank") or 0) >= 3:
        dti = dt
        df = np.diff(wrench[:, 2], prepend=wrench[0, 2]) / dti
        pred = (
            float(H["Hz"]) * ach[:, 2]
            + float(H["Hth"]) * ach[:, th]
            + float(H["Hs"]) * ach[:, int(scan_axis)]
        )
        resid = df - pred
        checks["H"] = {
            "resid_p95": float(np.nanpercentile(np.abs(resid), 95)),
            "ok": bool(np.nanpercentile(np.abs(resid), 95) < 25.0),
        }
    else:
        checks["H"] = {"skipped": True, "reason": "13 H missing or rank<3"}

    e14 = bundle.get("14_port_energy") or {}
    e_v = float(e14.get("e_v_bar") or float("nan"))
    e_w = float(e14.get("e_w_bar") or 0.0)
    if math.isfinite(e_v) and e14.get("bound_source") == "11_exec_2dof":
        trans_only = not has_torque(rows)
        p_true = port_power(wrench, pose_tw, trans_only=trans_only)
        f_n = np.linalg.norm(wrench[:, :3], axis=1)
        tau_n = np.linalg.norm(wrench[:, 3:6], axis=1) if not trans_only else np.zeros(len(rows))
        p_hat = port_power(wrench, ach, trans_only=trans_only)
        p_bound = p_hat - f_n * e_v - tau_n * (e_w if math.isfinite(e_w) else 0.0)
        cover = prefix_covers(prefix_debt(prefix_work(p_true, dt)), prefix_debt(prefix_work(p_bound, dt)))
        checks["energy"] = {**cover, "ok": bool(cover["all_ok"])}
    else:
        checks["energy"] = {"skipped": True, "reason": "14 bound not from 11 contract"}

    scored = [c for c in checks.values() if "ok" in c]
    payload = {
        "csv": str(csv_path),
        "align": align,
        "id_used": {k: None if v is None else v.get("_path") for k, v in bundle.items()},
        "checks": checks,
        "n_scored": len(scored),
        "n_ok": sum(1 for c in scored if c.get("ok")),
        "holdout_ok": bool(scored) and all(c.get("ok") for c in scored),
        "fitted_anything": False,
        "collected_at": when,
        "what": "hold-out coverage only; no new G/Ke/H/w_bar",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "holdout.json", dump_jsonable(payload))
    _plot_15(checks, visu)
    lines = []
    for name, chk in checks.items():
        if chk.get("skipped"):
            lines.append(f"- {name}：跳过（{chk.get('reason')}）。")
        elif name == "exec":
            lines.append(
                f"- exec：vz 覆盖 {fmt_finite(chk.get('cover_vz'), '.2f')}，"
                f"ωθ {fmt_finite(chk.get('cover_wth'), '.2f')}。"
            )
        elif name == "force_tail":
            lines.append(
                f"- force_tail：ΔF = {fmt_finite(chk.get('df_peak_n'), '.2f')} N，"
                f"w̄ = {fmt_finite(chk.get('w_bar_n'), '.2f')} N，"
                f"{'在管内' if chk.get('inside') else '出管'}。"
            )
        elif name == "energy":
            lines.append(
                f"- energy：prefix 覆盖 {fmt_finite(chk.get('frac'), '.2f')}，"
                f"{'∀k 通过' if chk.get('ok') else '有违约'}。"
            )
        else:
            lines.append(f"- {name}：{'覆盖' if chk.get('ok') else '出管'}。")
    write_readme(
        visu,
        f"""# 15_holdout — 独立覆盖

采集：`{when}` · filter OFF · 力环关 · **没有拟合任何新参数**

## 结论

- 评分项 {payload['n_scored']}，通过 {payload['n_ok']}。总体 hold-out {'通过' if payload['holdout_ok'] else '未通过 / 不完整'}。
{chr(10).join(lines)}

独立数据上的覆盖不好，就不要把 11–14 的管子写成 certified envelope。

## 图

`holdout_cover.png`：各项覆盖指示。
""",
    )
    print(
        f"[15] scored={payload['n_scored']} ok={payload['n_ok']}  "
        f"holdout_ok={payload['holdout_ok']}  data={data}",
        flush=True,
    )
    return payload


def _plot_15(checks: dict, visu: Path) -> None:
    plt = mpl()
    names = list(checks)
    vals = []
    for name in names:
        chk = checks[name]
        if chk.get("skipped"):
            vals.append(float("nan"))
        elif name == "exec":
            vals.append(float(chk.get("cover_vz") or float("nan")))
        elif name == "force_tail":
            vals.append(1.0 if chk.get("inside") else 0.0)
        elif name == "energy":
            vals.append(float(chk.get("frac") or float("nan")))
        else:
            vals.append(1.0 if chk.get("ok") else 0.0)
    fig, ax = plt.subplots(figsize=(3.50, 2.20), constrained_layout=True)
    x = np.arange(len(names))
    colors = [ACH if math.isfinite(v) and v >= 0.95 else MINUS for v in vals]
    ax.bar(x, [0.0 if not math.isfinite(v) else v for v in vals], color=colors, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("coverage")
    ax.grid(True, axis="y", alpha=0.28)
    save(fig, visu, "holdout_cover.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.80)
    p.add_argument("--id-root", default="", help="DATA root with 08/11–14 JSON")
    p.add_argument("--f0-n", type=float, default=1.8)
    p.add_argument("--press-mm-s", type=float, default=10.0)
    p.add_argument("--scan-mm-s", type=float, default=12.0)
    p.add_argument("--tilt-deg-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=15)
    args = p.parse_args()
    when = stamp()
    print(
        "[PLAN] MOVEJ mid, mixed hold-out (air pulse, preload, uz, tilt, "
        f"scan both ways, press-stop, output/recover)  F0≈{args.f0_n:.1f} N  "
        "force loop OFF  analyze never fits",
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
                id_root=args.id_root,
                theta_axis=args.theta_axis,
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
    log = data / "holdout.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
    )
    rng = np.random.default_rng(int(args.seed))
    vz = abs(args.press_mm_s) / 1000.0 * float(rng.choice([0.8, 1.0, 1.2]))
    vs = abs(args.scan_mm_s) / 1000.0 * float(rng.choice([-1.0, 1.0]))
    wth = math.radians(args.tilt_deg_s) * float(rng.choice([-1.0, 1.0]))
    try:
        srv.start_twist()
        if not srv.chirp_axis(2, 0.008, 0.4, 2.5, 4.0, "air_z"):
            return 0 if srv.aborted else 130
        if not srv.seek_contact(0.007, contact_n=args.contact_n):
            return 2
        if not srv.seek_force(args.f0_n, vel_m_s=0.004, band_n=0.20, phase="preload"):
            return 2
        if not srv.hold(0.0, 0.4, "preload"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, 0.6 * vz), 0.45, "run_z"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.theta_axis, wth), 0.5, "run_th"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, vs), 0.9, "scan_fwd"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, -vs), 0.9, "scan_rev"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, vz), 0.55, "run_press"):
            return 0 if srv.aborted else 130
        if not srv.hold(0.0, 0.35, "stop_hold", check_abort=False):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, 0.005), 0.45, "output"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, -0.005), 0.45, "recover", check_abort=False):
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
                id_root=args.id_root,
                theta_axis=args.theta_axis,
                scan_axis=args.scan_axis,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
