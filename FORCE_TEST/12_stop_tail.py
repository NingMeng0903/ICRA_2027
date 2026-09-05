#!/usr/bin/env python3
"""Committed-queue / stop-tail envelopes.  Force loop OFF.

What: press-stop, retract-stop, reverse, and small tilt-stop on the pad.
      Same current (F, v) is approached with two different command histories.
Why: C1's sufficient contract is old queue ⊕ candidate ⊕ backup tail.
     A zero command is not a contact hold.  Measure remaining press after
     the stop edge, and the late force peak.

This is still open-loop identification, not the filter.
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
    phase_mask,
    pose6,
    tool_z_displacement,
    twist_ach6,
    twist_cmd6,
    wrench6,
)

KIND = "12_stop_tail"


def _edge_index(mask: np.ndarray) -> int | None:
    if not np.any(mask):
        return None
    return int(np.flatnonzero(mask)[0])


def _prefix_after(values: np.ndarray, i0: int, n: int) -> np.ndarray:
    sl = values[i0 : min(values.size, i0 + n)]
    return sl


def _trial_metrics(rows, *, stop_phase: str, horizon_s: float = 0.25) -> dict:
    t = col(rows, "t_wall_s", "t_mono_s")
    dt = float(np.nanmedian(np.diff(t[np.isfinite(t)]))) if t.size > 2 else 0.005
    if not math.isfinite(dt) or dt <= 1e-4:
        dt = 0.005
    n = max(int(round(horizon_s / dt)), 8)
    fz = wrench6(rows)[:, 2]
    tau = wrench6(rows)[:, 4]
    vz = twist_ach6(rows)[:, 2]
    uz = twist_cmd6(rows)[:, 2]
    dx = tool_z_displacement(pose6(rows))
    i0 = _edge_index(phase_mask(rows, stop_phase))
    if i0 is None:
        return {"phase": stop_phase, "n": 0}
    f0 = float(fz[i0]) if i0 < fz.size else float("nan")
    v0 = float(vz[i0]) if i0 < vz.size else float("nan")
    u0 = float(uz[i0]) if i0 < uz.size else float("nan")
    f_pref = _prefix_after(fz, i0, n)
    x_pref = _prefix_after(dx, i0, n)
    tau_pref = _prefix_after(tau, i0, n)
    peak_f = float(np.nanmax(f_pref)) if f_pref.size else float("nan")
    min_f = float(np.nanmin(f_pref)) if f_pref.size else float("nan")
    dx_tail = (
        float(x_pref[np.isfinite(x_pref)][-1] - x_pref[np.isfinite(x_pref)][0])
        if np.isfinite(x_pref).sum() >= 2
        else float("nan")
    )
    late_peak = math.isfinite(peak_f) and math.isfinite(f0) and peak_f > f0 + 0.15
    return {
        "phase": stop_phase,
        "i0": i0,
        "t0_s": float(t[i0]) if i0 < t.size else float("nan"),
        "f0_n": f0,
        "v0_m_s": v0,
        "u_at_edge_m_s": u0,
        "f_peak_n": peak_f,
        "f_min_n": min_f,
        "df_peak_n": peak_f - f0 if math.isfinite(peak_f) and math.isfinite(f0) else float("nan"),
        "dx_tail_mm": 1e3 * dx_tail if math.isfinite(dx_tail) else float("nan"),
        "dtau_nm": (
            float(np.nanmax(np.abs(tau_pref - tau_pref[0])))
            if tau_pref.size and np.isfinite(tau_pref).any()
            else float("nan")
        ),
        "late_peak": bool(late_peak),
        "horizon_s": horizon_s,
        "n": int(f_pref.size),
    }


def analyze(csv_path: Path, *, when: str, window_a_csv: str = "") -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    names = [
        "stop_press_slow",
        "stop_press_fast",
        "stop_retract",
        "stop_reverse",
        "stop_tilt",
        "hold_high",
        "hold_low",
    ]
    trials = [_trial_metrics(rows, stop_phase=name) for name in names]
    present = [tr for tr in trials if tr.get("n", 0) >= 4]
    late = [tr for tr in present if tr.get("late_peak")]
    # Same (F,v) split: slow vs fast press-stop.
    slow = next((tr for tr in trials if tr["phase"] == "stop_press_slow"), {})
    fast = next((tr for tr in trials if tr["phase"] == "stop_press_fast"), {})
    same_state = (
        math.isfinite(float(slow.get("f0_n") or float("nan")))
        and math.isfinite(float(fast.get("f0_n") or float("nan")))
        and abs(float(slow["f0_n"]) - float(fast["f0_n"])) < 0.6
        and abs(float(slow.get("v0_m_s") or 0.0) - float(fast.get("v0_m_s") or 0.0)) < 0.008
    )
    queue_splits = False
    if same_state and math.isfinite(float(slow.get("dx_tail_mm") or float("nan"))) and math.isfinite(
        float(fast.get("dx_tail_mm") or float("nan"))
    ):
        queue_splits = abs(float(slow["dx_tail_mm"]) - float(fast["dx_tail_mm"])) > 0.15
    w_bar = float("nan")
    dfs = [abs(float(tr["df_peak_n"])) for tr in present if math.isfinite(float(tr.get("df_peak_n") or float("nan")))]
    if dfs:
        w_bar = float(np.percentile(dfs, 90))
    payload = {
        "csv": str(csv_path),
        "align": align,
        "trials": trials,
        "n_trials": len(present),
        "n_late_peak": len(late),
        "same_fv_two_queues": bool(same_state),
        "queue_splits_tail": bool(queue_splits),
        "w_bar_proxy_n": w_bar,
        "collected_at": when,
        "what": "stop-tail envelope, not a controller",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "tail.json", payload)
    _plot_12(rows, present, visu)
    verdict = (
        "同 (F,v) 不同队列把尾巴切开了，延迟增广必要。"
        if queue_splits
        else (
            "还没有做成同 (F,v) 对照，或尾巴差太小；C1 的队列项尚未被这拍数据撑住。"
            if not same_state
            else "同 (F,v) 下尾巴几乎一样：延迟增广可能不是这套垫/速度上的主因。"
        )
    )
    write_readme(
        visu,
        f"""# 12_stop_tail — 旧队列与停止尾巴

采集：`{when}` · filter OFF · 力环关 · 对齐中位 {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms

## 结论

- 有效停止边 **{len(present)}** 条；其中晚到力峰 **{len(late)}** 条。
- 停止后 250 ms 力增量的 p90 ≈ {fmt_finite(w_bar, '.2f')} N（这是 \(\\bar w\) 的数据代理，不是定理里的盒扰动）。
- {verdict}
- 零命令尾巴若仍压入，backup 不能写成 u=0。

## 图

`stop_tail.png`：**a** 力，**b** tool-Z 位移。竖线是各 stop 边。
""",
    )
    print(
        f"[12] trials={len(present)} late={len(late)} queue_split={queue_splits}  data={data}",
        flush=True,
    )
    return payload


def _plot_12(rows, trials: list[dict], visu: Path) -> None:
    t = col(rows, "t_wall_s", "t_mono_s")
    t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
    tt = t - t0
    fz = wrench6(rows)[:, 2]
    dx = 1e3 * tool_z_displacement(pose6(rows))
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 3.80), sharex=True, constrained_layout=True)
    axes[0].plot(tt, fz, color=ACH, lw=1.15)
    axes[1].plot(tt, dx, color=MINUS, lw=1.15)
    for tr in trials:
        t_edge = float(tr.get("t0_s") or float("nan")) - t0
        if math.isfinite(t_edge):
            axes[0].axvline(t_edge, color=ACH, lw=0.6, alpha=0.45)
            axes[1].axvline(t_edge, color=MINUS, lw=0.6, alpha=0.45)
    axes[0].set_ylabel("Fz (N)")
    axes[1].set_ylabel("tool-Z travel (mm)")
    axes[1].set_xlabel("time (s)")
    for letter, ax in zip("ab", axes):
        ax.grid(True, alpha=0.28)
        panel_tag(ax, letter)
    save(fig, visu, "stop_tail.png")
    plt.close(fig)


def _press_then(srv: ContactLogger, vel: float, seconds: float, phase_run: str, phase_stop: str, stop_s: float) -> bool:
    if not srv.hold(axis_twist(2, vel), seconds, phase_run):
        return False
    return srv.hold(axis_twist(2, 0.0), stop_s, phase_stop, check_abort=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.40)
    p.add_argument("--slow-mm-s", type=float, default=6.0)
    p.add_argument("--fast-mm-s", type=float, default=15.0)
    p.add_argument("--press-s", type=float, default=0.8)
    p.add_argument("--stop-s", type=float, default=0.35)
    p.add_argument("--tilt-deg-s", type=float, default=8.0)
    args = p.parse_args()
    when = stamp()
    print(
        f"[PLAN] MOVEJ mid, seek, then press-stop slow/fast, retract-stop, "
        f"reverse, tilt-stop, hold-high/low  force loop OFF  "
        f"abort F≥{args.abort_n:.1f} N",
        flush=True,
    )
    if dry_exit(args):
        return 0
    if args.csv:
        try:
            analyze(Path(args.csv), when=when, window_a_csv=args.window_a_csv)
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
    log = data / "tail.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
    )
    slow = args.slow_mm_s / 1000.0
    fast = args.fast_mm_s / 1000.0
    tilt = math.radians(args.tilt_deg_s)
    try:
        srv.start_twist()
        if not srv.seek_contact(0.008, contact_n=args.contact_n):
            return 2
        if not _press_then(srv, slow, args.press_s, "run_press_slow", "stop_press_slow", args.stop_s):
            return 0 if srv.aborted else 130
        if not _press_then(srv, fast, args.press_s, "run_press_fast", "stop_press_fast", args.stop_s):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, -slow), args.press_s, "run_retract"):
            return 0 if srv.aborted else 130
        if not srv.hold(0.0, args.stop_s, "stop_retract", check_abort=False):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, slow), 0.45 * args.press_s, "run_reverse"):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(2, -slow), args.stop_s, "stop_reverse", check_abort=False):
            return 0 if srv.aborted else 130
        if not srv.hold(axis_twist(args.theta_axis, tilt), 0.4, "run_tilt"):
            return 0 if srv.aborted else 130
        if not srv.hold(0.0, args.stop_s, "stop_tilt", check_abort=False):
            return 0 if srv.aborted else 130
        if math.isfinite(srv.last_fz) and srv.last_fz > 2.5:
            if not srv.hold(axis_twist(2, -0.004), args.stop_s, "hold_high", check_abort=False):
                return 0 if srv.aborted else 130
        else:
            srv.hold(0.0, 0.15, "hold_high", check_abort=False)
        if math.isfinite(srv.last_fz) and srv.last_fz < 1.2:
            if not srv.hold(axis_twist(2, 0.003), args.stop_s, "hold_low"):
                return 0 if srv.aborted else 130
        else:
            srv.hold(0.0, 0.15, "hold_low", check_abort=False)
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
            analyze(log, when=when, window_a_csv=args.window_a_csv)
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
