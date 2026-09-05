#!/usr/bin/env python3
"""Physical port power and prefix energy debt.  Force loop OFF.

What: air z, contact z, tilt, tangential scan, then press-then-retract
      (output then recover).  Compare same-tick command power, delayed
      command power, achieved-twist power, and pose-increment power.
Why: a tank-in-QP is not a contribution.  This file only asks whether a
      conservative ζ ≤ ζ_ref exists.  If it fails, the paper drops
      passivity and keeps the name “energy-aware scheduler”.

Six-D wrench needs tx, ty, tz on Window A.  Missing torque degrades
the certificate to translation-only.
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
from paper_fig import ACH, CMD, MINUS, mpl, panel_tag, save
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import (
    AlignmentError,
    fmt_finite,
    has_torque,
    pose6,
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


def _power(wrench: np.ndarray, twist: np.ndarray, *, trans_only: bool) -> np.ndarray:
    p = np.sum(wrench[:, :3] * twist[:, :3], axis=1)
    if not trans_only:
        p = p + np.sum(wrench[:, 3:6] * twist[:, 3:6], axis=1)
    return p


def _pose_twist(pose: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros((pose.shape[0], 6), dtype=float)
    dp = np.diff(pose, axis=0)
    out[1:, :3] = dp[:, :3] / dt[1:, None]
    out[1:, 3:6] = dp[:, 3:6] / dt[1:, None]
    out[0] = out[1] if out.shape[0] > 1 else 0.0
    return out


def _shift(arr: np.ndarray, ticks: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    k = max(int(ticks), 0)
    if k == 0:
        return arr.copy()
    if k < arr.size:
        out[k:] = arr[:-k]
    return out


def _prefix_debt(p: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Running max of −cumsum(p dt).  First-out-then-recover shows up here."""

    work = np.cumsum(np.where(np.isfinite(p), p, 0.0) * dt)
    debt = np.maximum.accumulate(np.maximum(-work, 0.0))
    return debt


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    t0_s: float = T0_S,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt = _dt(t)
    pose = pose6(rows)
    wrench = wrench6(rows)
    cmd = twist_cmd6(rows)
    ach = twist_ach6(rows)
    trans_only = not has_torque(rows)
    p_cmd = _power(wrench, cmd, trans_only=trans_only)
    delay_ticks = int(round(float(t0_s) / float(np.median(dt))))
    p_cmd_d = _power(wrench, _shift(cmd, delay_ticks), trans_only=trans_only)
    p_ach = _power(wrench, ach, trans_only=trans_only)
    p_pose = _power(wrench, _pose_twist(pose, dt), trans_only=trans_only)
    # Conservative measured power: achieved twist minus a sync envelope.
    e_v = np.linalg.norm(ach[:, :3] - cmd[:, :3], axis=1)
    f_n = np.linalg.norm(wrench[:, :3], axis=1)
    p_cons = p_ach - f_n * e_v
    work = {
        "cmd": float(np.nansum(p_cmd * dt)),
        "cmd_delayed": float(np.nansum(p_cmd_d * dt)),
        "achieved": float(np.nansum(p_ach * dt)),
        "pose": float(np.nansum(p_pose * dt)),
        "conservative": float(np.nansum(p_cons * dt)),
    }
    debt = {
        "cmd": float(np.nanmax(_prefix_debt(p_cmd, dt))),
        "cmd_delayed": float(np.nanmax(_prefix_debt(p_cmd_d, dt))),
        "achieved": float(np.nanmax(_prefix_debt(p_ach, dt))),
        "pose": float(np.nanmax(_prefix_debt(p_pose, dt))),
        "conservative": float(np.nanmax(_prefix_debt(p_cons, dt))),
    }
    # Coverage: conservative integral never exceeds achieved by a large positive leak
    # (we want ζ_cons ≤ ζ_ref + margin).  Here "ref" is pose increment.
    leak = work["conservative"] - work["pose"]
    covers = math.isfinite(leak) and leak <= 0.05
    prefix_ok = debt["conservative"] + 1e-9 >= debt["pose"]
    tcp_vs_contact = abs(work["achieved"] - work["pose"])
    payload = {
        "csv": str(csv_path),
        "align": align,
        "torque_present": not trans_only,
        "trans_only": bool(trans_only),
        "t0_s": t0_s,
        "delay_ticks": delay_ticks,
        "work_j": work,
        "prefix_debt_j": debt,
        "conservative_covers_pose_work": bool(covers),
        "prefix_debt_covers_pose": bool(prefix_ok),
        "tcp_vs_pose_work_abs_j": tcp_vs_contact,
        "passivity_claim_allowed": bool(covers and prefix_ok and not trans_only),
        "collected_at": when,
        "what": "port calibration, not a tank theorem",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "energy.json", payload)
    _plot_14(t, p_cmd, p_ach, p_pose, p_cons, visu)
    claim = (
        "保守功包住位姿参考，且前缀债务不漏中途透支。仍只是辨识证书，不是闭环无源。"
        if payload["passivity_claim_allowed"]
        else "覆盖失败或只有平移功率：主文取消无源性声明，只留 energy-aware scheduler。"
    )
    write_readme(
        visu,
        f"""# 14_port_energy — 真实端口与前缀能量

采集：`{when}` · filter OFF · 力环关 · {'六维 wrench' if not trans_only else '**无力矩列，只结算 F·v**'}

## 结论

- 净功 (J)：命令 {fmt_finite(work['cmd'], '.4f')}，延迟命令 {fmt_finite(work['cmd_delayed'], '.4f')}，实际 twist {fmt_finite(work['achieved'], '.4f')}，位姿增量 {fmt_finite(work['pose'], '.4f')}，保守 {fmt_finite(work['conservative'], '.4f')}。
- 最大前缀债务 (J)：实际 {fmt_finite(debt['achieved'], '.4f')}，位姿 {fmt_finite(debt['pose'], '.4f')}，保守 {fmt_finite(debt['conservative'], '.4f')}。
- {claim}
- 有限罐会被摩擦路径耗尽。不要把 tank 和“永远保持接触”写成同一个无限时域定理。

## 图

`port_power.png`：四种功率。先输出后回收会在积分里先负后正。
""",
    )
    print(
        f"[14] W_ach={work['achieved']:.4f} J  cover={payload['passivity_claim_allowed']}  data={data}",
        flush=True,
    )
    return payload


def _plot_14(t, p_cmd, p_ach, p_pose, p_cons, visu: Path) -> None:
    t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
    tt = t - t0
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    ax.plot(tt, p_cmd, color=CMD, lw=0.9)
    ax.plot(tt, p_ach, color=ACH, lw=1.15)
    ax.plot(tt, p_pose, color=MINUS, lw=0.9)
    ax.plot(tt, p_cons, color="#009E73", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (W)")
    ax.grid(True, alpha=0.28)
    save(fig, visu, "port_power.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.40)
    p.add_argument("--t0-ms", type=float, default=28.0)
    p.add_argument("--amp-mm-s", type=float, default=8.0)
    p.add_argument("--scan-mm-s", type=float, default=10.0)
    p.add_argument("--tilt-deg-s", type=float, default=6.0)
    args = p.parse_args()
    when = stamp()
    print(
        "[PLAN] MOVEJ mid, air z, contact z, tilt, scan, press-then-retract  "
        f"force loop OFF  T0={args.t0_ms:.0f} ms for delayed-command power",
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
        if not srv.chirp_axis(2, 0.5 * vz, 0.3, 2.0, 6.0, "contact_z"):
            return 0 if srv.aborted else 130
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
            analyze(log, when=when, window_a_csv=args.window_a_csv, t0_s=args.t0_ms / 1000.0)
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
