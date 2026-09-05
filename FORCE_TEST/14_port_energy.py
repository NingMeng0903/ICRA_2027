#!/usr/bin/env python3
"""Physical port power and prefix-wise energy debt.  Force loop OFF.

What: from the 11 *contact* contract,
          V̂ = Ĝ_contact u,   P̂ = Wᵀ V̂,
          P_lower = P̂ − ‖F‖ ē_v − ‖τ‖ ē_ω.
      Check P_pose ≥ P_lower via D_pose(k) ≤ D_lower(k) for every prefix.
Why: 11's residual is V_ach − Ĝ u, not V_true − V_ach.  Subtracting ē_G
     from P_ach is the wrong bound.  Route A closes 11 → 14 for the QP.

TCP↔contact invariance is an adjoint self-check, not a lever calibration.
Missing 11 G_contact, missing torque, or any prefix leak → no passivity claim.
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
from id_common import add_contact_args, load_aligned, load_kind_json, stash_window_a
from id_math import (
    dump_jsonable,
    port_power,
    power_invariance_rel_err,
    predicted_twist,
    prefix_covers,
    prefix_debt,
    prefix_work,
    twist_at_offset,
    wrench_at_offset,
)
from io_csv import col, write_json
from paper_fig import ACH, CMD, MINUS, mpl, save
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

KIND = "14_port_energy"
T0_S = 0.028


def _dt(t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0] + 0.005)
    if t.size > 2:
        dt[0] = float(np.nanmedian(np.diff(t[np.isfinite(t)])))
    return np.where(np.isfinite(dt) & (dt > 1e-4), dt, 0.005)


def _shift(arr: np.ndarray, ticks: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    k = max(int(ticks), 0)
    if k == 0:
        return arr.copy()
    if k < arr.size:
        out[k:] = arr[:-k]
    return out


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    t0_s: float = T0_S,
    r_tcp_m: tuple[float, float, float] = (0.0, 0.0, 0.03),
    id_root: str = "",
    theta_axis: int = 4,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt = _dt(t)
    pose = pose6(rows)
    wrench = wrench6(rows)
    cmd = twist_cmd6(rows)
    ach = twist_ach6(rows)
    pose_tw = pose_body_twist(pose, dt)
    trans_only = not has_torque(rows)
    r = np.asarray(r_tcp_m, dtype=float).reshape(3)
    inv_rel = power_invariance_rel_err(wrench, ach, r, trans_only=trans_only)
    inv_ok = math.isfinite(inv_rel) and inv_rel < 1e-9

    p_cmd = port_power(wrench, cmd, trans_only=trans_only)
    delay_ticks = int(round(float(t0_s) / float(np.median(dt))))
    p_cmd_d = port_power(wrench, _shift(cmd, delay_ticks), trans_only=trans_only)
    p_ach = port_power(wrench, ach, trans_only=trans_only)
    p_pose = port_power(wrench, pose_tw, trans_only=trans_only)
    p_contact = port_power(
        wrench_at_offset(wrench, r), twist_at_offset(ach, r), trans_only=trans_only
    )

    exec11 = load_kind_json("11_exec_2dof", root=Path(id_root) if id_root else None)
    G = None
    bound_source = "missing_11_contract"
    if exec11:
        G = exec11.get("G_contact") or None
        if G is not None:
            bound_source = "11_G_contact"
        elif exec11.get("G"):
            G = exec11.get("G")
            bound_source = "11_G_unspecified"
        th = int(exec11.get("theta_axis") or theta_axis)
    else:
        th = int(theta_axis)
    tube = (exec11 or {}).get("tube_contact") or (exec11 or {}).get("tube") or {}
    e_v = float((tube.get("e_vz") or {}).get("bar") or tube.get("e_vz_bar") or float("nan"))
    e_w = float((tube.get("e_wth") or {}).get("bar") or tube.get("e_wth_bar") or float("nan"))
    if bound_source.startswith("11_") and not math.isfinite(e_v):
        bound_source = "missing_11_tube"
        G = None
    e_v_same = float(np.nanpercentile(np.linalg.norm(ach[:, :3] - cmd[:, :3], axis=1), 95))
    reasons: list[str] = []
    if trans_only:
        reasons.append("torque columns missing")
    if not inv_ok:
        reasons.append(f"TCP↔contact adjoint self-check failed ({inv_rel})")
    if G is None or not bound_source.startswith("11_G"):
        reasons.append("need 11 G_contact and W_contact; do not subtract ē from P_ach")
        e_v_use = float("nan")
        e_w_use = float("nan")
        p_hat = np.full_like(p_ach, np.nan)
        p_bound = np.full_like(p_ach, np.nan)
    else:
        e_v_use = e_v
        e_w_use = e_w if math.isfinite(e_w) else 0.0
        v_hat = predicted_twist(cmd, float(np.median(dt)), G, theta_axis=th)
        p_hat = port_power(wrench, v_hat, trans_only=trans_only)
        f_n = np.linalg.norm(wrench[:, :3], axis=1)
        tau_n = np.linalg.norm(wrench[:, 3:6], axis=1) if not trans_only else np.zeros(wrench.shape[0])
        p_bound = p_hat - f_n * e_v_use - tau_n * e_w_use

    work = {
        "cmd": float(np.nansum(p_cmd * dt)),
        "cmd_delayed": float(np.nansum(p_cmd_d * dt)),
        "achieved": float(np.nansum(p_ach * dt)),
        "pose": float(np.nansum(p_pose * dt)),
        "contact": float(np.nansum(p_contact * dt)),
        "G_u": float(np.nansum(p_hat * dt)) if np.isfinite(p_hat).any() else float("nan"),
        "bound": float(np.nansum(p_bound * dt)) if np.isfinite(p_bound).any() else float("nan"),
    }
    d_pose = prefix_debt(prefix_work(p_pose, dt))
    d_ach = prefix_debt(prefix_work(p_ach, dt))
    d_bound = prefix_debt(prefix_work(p_bound, dt)) if np.isfinite(p_bound).any() else np.full_like(d_ach, np.nan)
    d_true = d_pose if np.isfinite(p_pose).any() else d_ach
    cover = prefix_covers(d_true, d_bound)
    if not cover["all_ok"]:
        reasons.append(
            f"prefix coverage failed (frac={cover['frac']}, max_viol={cover['max_violation']})"
        )
    pose_ge_bound = float("nan")
    if np.isfinite(p_bound).any() and np.isfinite(p_pose).any():
        okp = np.isfinite(p_pose) & np.isfinite(p_bound)
        if np.any(okp):
            pose_ge_bound = float(np.mean(p_pose[okp] + 1e-12 >= p_bound[okp]))
    claim = bool(inv_ok and not trans_only and bound_source == "11_G_contact" and cover["all_ok"])
    payload = {
        "csv": str(csv_path),
        "align": align,
        "torque_present": not trans_only,
        "trans_only": bool(trans_only),
        "t0_s": t0_s,
        "delay_ticks": delay_ticks,
        "r_C_tool_m": [float(r[0]), float(r[1]), float(r[2])],
        "frames": {
            "wrench": "tool/TCP Window A fx,fy,fz,tx,ty,tz",
            "twist_achieved": "tool Window A twist_achieved_*",
            "pose_twist": "tool body: v=R^T pdot, ω=Log(R_k^T R_{k+1})^∨/Δt",
            "invariance": "algebraic adjoint self-check; does not calibrate r_C or sensor frame",
        },
        "invariance_rel_err": inv_rel,
        "invariance_ok": bool(inv_ok),
        "bound_source": bound_source,
        "bound_formula": "P_lower = W^T Ĝ_contact u − ||F|| ē_v − ||τ|| ē_ω",
        "e_v_bar": e_v_use if G is not None else float("nan"),
        "e_w_bar": e_w_use if G is not None else float("nan"),
        "e_v_same_take_p95": e_v_same,
        "pose_ge_bound_frac": pose_ge_bound,
        "work_j": work,
        "prefix_cover": cover,
        "prefix_debt_max_j": {
            "achieved": float(np.nanmax(d_ach)) if d_ach.size else float("nan"),
            "pose": float(np.nanmax(d_pose)) if d_pose.size else float("nan"),
            "bound": float(np.nanmax(d_bound)) if np.isfinite(d_bound).any() else float("nan"),
        },
        "passivity_claim_allowed": bool(claim),
        "claim_blockers": reasons,
        "exec_json": None if exec11 is None else exec11.get("_path"),
        "collected_at": when,
        "what": "P_lower = W^T G_contact u − ||F||ē; never P_ach − ||F||ē_G",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "energy.json", dump_jsonable(payload))
    _plot_14(t, p_hat, p_pose, p_bound, d_true, d_bound, visu)
    if claim:
        claim_txt = "P_lower = Wᵀ Ĝ_contact u − ‖F‖ē_v − ‖τ‖ē_ω 在每一个 prefix 上包住位姿债务。仍只是辨识证书，不是闭环无源。"
    else:
        claim_txt = "不能声称 passivity：" + ("；".join(reasons) if reasons else "条件不足") + "。"
    write_readme(
        visu,
        f"""# 14_port_energy — 从命令预测端口功率

采集：`{when}` · filter OFF · 力环关 · {'六维 wrench' if not trans_only else '**无力矩列**'} · 边界来源 **{bound_source}**

## 结论

- r_C = {r.tolist()} m（tool）。TCP↔接触点不变性相对误差 {fmt_finite(inv_rel, '.2e')} 是伴随自检，**不是**杠杆标定。
- 角速度来自 \(\\mathrm{{Log}}(R_k^\\top R_{{k+1}})^\\vee/\\Delta t\)。wrench / twist_ach / pose twist 都按 tool。
- 净功 (J)：\(\\hat P=W^\\top\\hat G u\) {fmt_finite(work['G_u'], '.4f')}，位姿 {fmt_finite(work['pose'], '.4f')}，下界 {fmt_finite(work['bound'], '.4f')}。**不是** \(P_{{\\rm ach}}-\\|F\\|\\bar e_G\)。
- \(P_{{\\rm pose}}\\ge P_{{\\rm lower}}\) 的样本比例 {fmt_finite(pose_ge_bound, '.3f')}。前缀覆盖：全部通过 = {cover['all_ok']}，比例 {fmt_finite(cover['frac'], '.3f')}。
- {claim_txt}

## 图

`port_power.png`：\(\\hat P\)、位姿功率、下界。`prefix_debt.png`：D_pose(k) 对 D_lower(k)。
""",
    )
    print(
        f"[14] claim={claim}  src={bound_source}  cover={cover['frac']}  "
        f"inv={inv_rel:.2e}  data={data}",
        flush=True,
    )
    return payload


def _plot_14(t, p_hat, p_pose, p_bound, d_true, d_bound, visu: Path) -> None:
    t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
    tt = t - t0
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    if np.isfinite(p_hat).any():
        ax.plot(tt, p_hat, color=CMD, lw=0.9)
    ax.plot(tt, p_pose, color=ACH, lw=1.15)
    if np.isfinite(p_bound).any():
        ax.plot(tt, p_bound, color=MINUS, lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (W)")
    ax.grid(True, alpha=0.28)
    save(fig, visu, "port_power.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    ax.plot(tt[: d_true.size], d_true, color=ACH, lw=1.15)
    if np.isfinite(d_bound).any():
        ax.plot(tt[: d_bound.size], d_bound, color=MINUS, lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("prefix debt (J)")
    ax.grid(True, alpha=0.28)
    save(fig, visu, "prefix_debt.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.40)
    p.add_argument("--t0-ms", type=float, default=28.0)
    p.add_argument("--amp-mm-s", type=float, default=8.0)
    p.add_argument("--scan-mm-s", type=float, default=10.0)
    p.add_argument("--tilt-deg-s", type=float, default=6.0)
    p.add_argument("--rx-mm", type=float, default=0.0)
    p.add_argument("--ry-mm", type=float, default=0.0)
    p.add_argument("--rz-mm", type=float, default=30.0, help="tool-frame r_C; not a calibrated contact point")
    p.add_argument("--lever-mm", type=float, default=float("nan"), help="alias for --rz-mm")
    p.add_argument("--id-root", default="", help="DATA root that holds 11_exec_2dof/exec.json")
    args = p.parse_args()
    when = stamp()
    print(
        "[PLAN] MOVEJ mid, air z, contact z, tilt, scan, press-then-retract  "
        f"force loop OFF  T0={args.t0_ms:.0f} ms  "
        f"P_lower = W^T G_contact u − ||F||ē  "
        "passivity_claim_allowed stays false without prefix-∀k coverage",
        flush=True,
    )
    if dry_exit(args):
        return 0
    rz = args.lever_mm if math.isfinite(args.lever_mm) else args.rz_mm
    r_tcp = (args.rx_mm / 1000.0, args.ry_mm / 1000.0, rz / 1000.0)
    if args.csv:
        try:
            analyze(
                Path(args.csv),
                when=when,
                window_a_csv=args.window_a_csv,
                t0_s=args.t0_ms / 1000.0,
                r_tcp_m=r_tcp,
                id_root=args.id_root,
                theta_axis=args.theta_axis,
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
    log = data / "energy.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
        secondary=args.secondary,
    )
    vz = args.amp_mm_s / 1000.0
    vs = args.scan_mm_s / 1000.0
    wth = math.radians(args.tilt_deg_s)
    try:
        srv.start_twist()
        if not srv.chirp_axis(2, vz, 0.3, 3.0, 8.0, "air_z"):
            return 0 if srv.aborted else 130
        if not srv.seek_contact(0.008, contact_n=args.contact_n):
            return 2
        if not srv.chirp_disp_axis(2, 0.0004, 0.3, 3.0, 6.0, "contact_z", min_n=0.25):
            if not (srv.aborted or srv.unloaded):
                return 130
        if not srv.hold(axis_twist(args.theta_axis, wth), 0.8, "contact_tilt"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.scan_axis, vs), 1.2, "contact_scan"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, 0.006), 0.6, "output"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, -0.006), 0.6, "recover", check_abort=False):
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
                t0_s=args.t0_ms / 1000.0,
                r_tcp_m=r_tcp,
                id_root=args.id_root,
                theta_axis=args.theta_axis,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
