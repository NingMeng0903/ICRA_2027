#!/usr/bin/env python3
"""Physical port power and prefix-wise energy debt.  Force loop OFF.

What: compare TCP power, contact-point power (same wrench/twist, offset
      adjoint), achieved-twist power, and SO(3)-log pose twist power.
      The conservative bound is
          P ≥ P̂ − ‖F‖ ē_v − ‖τ‖ ē_ω
      with (ē_v, ē_ω) from the 11 execution contract, not ‖v_ach−v_cmd‖.
Why: tank-in-QP is not a contribution.  passivity_claim_allowed is True
     only if invariance holds, the bound comes from 11, and
     D_true(k) ≤ D_bound(k) for every prefix k.

Missing 11 JSON, missing torque, or any prefix leak → no passivity claim.
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
    lever_m: float = 0.03,
    id_root: str = "",
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
    r = np.array([0.0, 0.0, float(lever_m)], dtype=float)
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
    tube = (exec11 or {}).get("tube") or {}
    e_v = float((tube.get("e_vz") or {}).get("bar") or tube.get("e_vz_bar") or float("nan"))
    e_w = float((tube.get("e_wth") or {}).get("bar") or tube.get("e_wth_bar") or float("nan"))
    bound_source = "11_exec_2dof" if exec11 and math.isfinite(e_v) else "missing_11_contract"
    # Same-take ‖v_ach−v_cmd‖ is recorded but is *not* a physical bound.
    e_v_same = float(np.nanpercentile(np.linalg.norm(ach[:, :3] - cmd[:, :3], axis=1), 95))
    reasons: list[str] = []
    if trans_only:
        reasons.append("torque columns missing")
    if not inv_ok:
        reasons.append(f"TCP↔contact power invariance failed ({inv_rel})")
    if bound_source != "11_exec_2dof":
        reasons.append("no 11 execution-contract ē_v, ē_ω")
        e_v_use = float("nan")
        e_w_use = float("nan")
        p_bound = np.full_like(p_ach, np.nan)
    else:
        e_v_use = e_v
        e_w_use = e_w if math.isfinite(e_w) else 0.0
        f_n = np.linalg.norm(wrench[:, :3], axis=1)
        tau_n = np.linalg.norm(wrench[:, 3:6], axis=1) if not trans_only else np.zeros(wrench.shape[0])
        p_bound = p_ach - f_n * e_v_use - tau_n * e_w_use

    work = {
        "cmd": float(np.nansum(p_cmd * dt)),
        "cmd_delayed": float(np.nansum(p_cmd_d * dt)),
        "achieved": float(np.nansum(p_ach * dt)),
        "pose": float(np.nansum(p_pose * dt)),
        "contact": float(np.nansum(p_contact * dt)),
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
    claim = bool(inv_ok and not trans_only and bound_source == "11_exec_2dof" and cover["all_ok"])
    payload = {
        "csv": str(csv_path),
        "align": align,
        "torque_present": not trans_only,
        "trans_only": bool(trans_only),
        "t0_s": t0_s,
        "delay_ticks": delay_ticks,
        "lever_m": lever_m,
        "invariance_rel_err": inv_rel,
        "invariance_ok": bool(inv_ok),
        "bound_source": bound_source,
        "e_v_bar": e_v_use if bound_source == "11_exec_2dof" else float("nan"),
        "e_w_bar": e_w_use if bound_source == "11_exec_2dof" else float("nan"),
        "e_v_same_take_p95": e_v_same,
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
        "what": "port calibration; passivity only if 11-bound covers every prefix",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "energy.json", dump_jsonable(payload))
    _plot_14(t, p_cmd, p_ach, p_pose, p_bound, d_true, d_bound, visu)
    if claim:
        claim_txt = "11 的 ē 给出的下界在每一个 prefix 上包住位姿债务。仍只是辨识证书，不是闭环无源。"
    else:
        claim_txt = "不能声称 passivity：" + ("；".join(reasons) if reasons else "条件不足") + "。"
    write_readme(
        visu,
        f"""# 14_port_energy — 真实端口与逐段前缀债务

采集：`{when}` · filter OFF · 力环关 · {'六维 wrench' if not trans_only else '**无力矩列**'} · 边界来源 **{bound_source}**

## 结论

- TCP↔接触点功率不变性相对误差 {fmt_finite(inv_rel, '.2e')}（应接近机器精度）。
- 角速度来自 \({{(\\mathrm{{Log}}\\,R_k^\\top R_{{k+1}})^\\vee / \\Delta t}}\)，不是 Euler 差分。
- 净功 (J)：实际 twist {fmt_finite(work['achieved'], '.4f')}，位姿 {fmt_finite(work['pose'], '.4f')}，接触点 {fmt_finite(work['contact'], '.4f')}，下界 {fmt_finite(work['bound'], '.4f')}。
- 前缀覆盖：{cover['n']} 点，全部通过 = {cover['all_ok']}，比例 {fmt_finite(cover['frac'], '.3f')}，最大违约 {fmt_finite(cover['max_violation'], '.4f')} J。
- {claim_txt}
- 同文件 ‖v_ach−v_cmd‖ p95 = {fmt_finite(e_v_same, '.4f')} m/s，**不是** conservative bound。

## 图

`port_power.png`：功率。`prefix_debt.png`：D_true(k) 对 D_bound(k)。
""",
    )
    print(
        f"[14] claim={claim}  inv={inv_rel:.2e}  cover={cover['frac']}  "
        f"src={bound_source}  data={data}",
        flush=True,
    )
    return payload


def _plot_14(t, p_cmd, p_ach, p_pose, p_bound, d_true, d_bound, visu: Path) -> None:
    t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
    tt = t - t0
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    ax.plot(tt, p_cmd, color=CMD, lw=0.9)
    ax.plot(tt, p_ach, color=ACH, lw=1.15)
    ax.plot(tt, p_pose, color=MINUS, lw=0.9)
    if np.isfinite(p_bound).any():
        ax.plot(tt, p_bound, color="#009E73", lw=0.8)
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
    p.add_argument("--lever-mm", type=float, default=30.0)
    p.add_argument("--id-root", default="", help="DATA root that holds 11_exec_2dof/exec.json")
    args = p.parse_args()
    when = stamp()
    print(
        "[PLAN] MOVEJ mid, air z, contact z, tilt, scan, press-then-retract  "
        f"force loop OFF  T0={args.t0_ms:.0f} ms  bound from 11 if present  "
        "passivity_claim_allowed stays false without prefix-∀k coverage",
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
                t0_s=args.t0_ms / 1000.0,
                lever_m=args.lever_mm / 1000.0,
                id_root=args.id_root,
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
                lever_m=args.lever_mm / 1000.0,
                id_root=args.id_root,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
