#!/usr/bin/env python3
"""Matched-state committed-queue / stop-tail.  Force loop OFF.

What: construct pairs that share (F, x, v, u) at t0 but not the previous
      50–100 ms of command history, then apply the *same* backup.
Why: (F, v) is not a sufficient state.  A slow-then-fast sequential
      press that happens to land near the same F is not this experiment.

Live matching uses MotionBus v_tcp_z, integrated x_proxy, and Fz.
Analysis re-checks the match on Window A pose.  w̄ is max|ΔF|+δ on
matched train pairs, not a p90 of leftover stop edges.

Force feedback is used only to reach the preload.  Every hist/stop
phase is open-loop commanded twist.  Soft-pad settle also waits for
|Ḟ| below a threshold so viscoelastic relaxation is not the hidden state.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_log import ContactLogger, axis_twist
from goto_mid import go_mid_from_args
from id_common import add_contact_args, load_aligned, stash_window_a
from id_math import dump_jsonable
from io_csv import col, write_json
from paper_fig import ACH, CMD, MINUS, mpl, panel_tag, save
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
PAIR_RE = re.compile(r"^(hist|stop|miss)_([AB])(\d+)$")


def _edge_index(mask: np.ndarray) -> int | None:
    if not np.any(mask):
        return None
    return int(np.flatnonzero(mask)[0])


def _prefix_after(values: np.ndarray, i0: int, n: int) -> np.ndarray:
    return values[i0 : min(values.size, i0 + n)]


def _trial_metrics(rows, *, stop_phase: str, hist_phase: str, horizon_s: float, hist_s: float) -> dict:
    t = col(rows, "t_wall_s", "t_mono_s")
    dt = float(np.nanmedian(np.diff(t[np.isfinite(t)]))) if t.size > 2 else 0.005
    if not math.isfinite(dt) or dt <= 1e-4:
        dt = 0.005
    n = max(int(round(horizon_s / dt)), 8)
    n_hist = max(int(round(hist_s / dt)), 4)
    fz = wrench6(rows)[:, 2]
    vz = twist_ach6(rows)[:, 2]
    uz = twist_cmd6(rows)[:, 2]
    dx = tool_z_displacement(pose6(rows))
    i0 = _edge_index(phase_mask(rows, stop_phase))
    if i0 is None:
        return {"phase": stop_phase, "n": 0}
    i_hist0 = max(i0 - n_hist, 0)
    f_pref = _prefix_after(fz, i0, n)
    x_pref = _prefix_after(dx, i0, n)
    peak_f = float(np.nanmax(f_pref)) if f_pref.size else float("nan")
    f0 = float(fz[i0]) if i0 < fz.size else float("nan")
    dx_tail = (
        float(x_pref[np.isfinite(x_pref)][-1] - x_pref[np.isfinite(x_pref)][0])
        if np.isfinite(x_pref).sum() >= 2
        else float("nan")
    )
    return {
        "phase": stop_phase,
        "hist_phase": hist_phase,
        "i0": i0,
        "t0_s": float(t[i0]) if i0 < t.size else float("nan"),
        "f0_n": f0,
        "x0_m": float(dx[i0]) if i0 < dx.size else float("nan"),
        "v0_m_s": float(vz[i0]) if i0 < vz.size else float("nan"),
        "u0_m_s": float(uz[i0]) if i0 < uz.size else float("nan"),
        "u_hist": uz[i_hist0:i0].tolist() if i0 > i_hist0 else [],
        "f_peak_n": peak_f,
        "df_peak_n": peak_f - f0 if math.isfinite(peak_f) and math.isfinite(f0) else float("nan"),
        "dx_tail_mm": 1e3 * dx_tail if math.isfinite(dx_tail) else float("nan"),
        "late_peak": bool(math.isfinite(peak_f) and math.isfinite(f0) and peak_f > f0 + 0.15),
        "horizon_s": horizon_s,
        "n": int(f_pref.size),
    }


def _match_ok(a: dict, b: dict, *, eps_f: float, eps_x: float, eps_v: float, eps_u: float) -> dict:
    def _d(key):
        va = float(a.get(key) or float("nan"))
        vb = float(b.get(key) or float("nan"))
        if not (math.isfinite(va) and math.isfinite(vb)):
            return float("nan")
        return abs(va - vb)

    d = {
        "dF_n": _d("f0_n"),
        "dx_m": _d("x0_m"),
        "dv_m_s": _d("v0_m_s"),
        "du_m_s": _d("u0_m_s"),
    }
    ok = (
        math.isfinite(d["dF_n"])
        and d["dF_n"] < eps_f
        and math.isfinite(d["dx_m"])
        and d["dx_m"] < eps_x
        and math.isfinite(d["dv_m_s"])
        and d["dv_m_s"] < eps_v
        and math.isfinite(d["du_m_s"])
        and d["du_m_s"] < eps_u
    )
    d["ok"] = bool(ok)
    return d


def _hist_rms(a: dict, b: dict) -> float:
    ua = np.asarray(a.get("u_hist") or [], dtype=float)
    ub = np.asarray(b.get("u_hist") or [], dtype=float)
    n = min(ua.size, ub.size)
    if n < 4:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(ua[-n:] - ub[-n:]))))


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    hist_ms: float = 80.0,
    eps_f: float = 0.25,
    eps_x_mm: float = 0.35,
    eps_v_mm_s: float = 4.0,
    eps_u_mm_s: float = 1.5,
    w_slack: float = 0.05,
    hist_min_mm_s: float = 2.0,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    phases = sorted({str(r.get("phase") or "") for r in rows})
    ids = sorted({int(m.group(3)) for p in phases if (m := PAIR_RE.match(p))})
    pairs = []
    for i in ids:
        tag = f"{i:02d}"
        a = _trial_metrics(
            rows, stop_phase=f"stop_A{tag}", hist_phase=f"hist_A{tag}", horizon_s=0.25, hist_s=hist_ms / 1000.0
        )
        b = _trial_metrics(
            rows, stop_phase=f"stop_B{tag}", hist_phase=f"hist_B{tag}", horizon_s=0.25, hist_s=hist_ms / 1000.0
        )
        if a.get("n", 0) < 4 or b.get("n", 0) < 4:
            pairs.append({"i": i, "matched": False, "reason": "missing stop_B or too short", "A": a, "B": b})
            continue
        match = _match_ok(
            a, b, eps_f=eps_f, eps_x=eps_x_mm / 1000.0, eps_v=eps_v_mm_s / 1000.0, eps_u=eps_u_mm_s / 1000.0
        )
        hrms = _hist_rms(a, b)
        hist_diff = math.isfinite(hrms) and hrms > hist_min_mm_s / 1000.0
        split = False
        if match["ok"] and hist_diff:
            da = float(a.get("dx_tail_mm") or float("nan"))
            db = float(b.get("dx_tail_mm") or float("nan"))
            split = math.isfinite(da) and math.isfinite(db) and abs(da - db) > 0.15
        pairs.append(
            {
                "i": i,
                "matched": bool(match["ok"]),
                "hist_diff": bool(hist_diff),
                "hist_rms_mm_s": 1e3 * hrms if math.isfinite(hrms) else float("nan"),
                "queue_splits_tail": bool(split),
                "match": match,
                "A": {k: v for k, v in a.items() if k != "u_hist"},
                "B": {k: v for k, v in b.items() if k != "u_hist"},
            }
        )
    matched = [p for p in pairs if p["matched"] and p["hist_diff"]]
    splits = [p for p in matched if p["queue_splits_tail"]]
    dfs = []
    for p in matched:
        for side in ("A", "B"):
            df = float(p[side].get("df_peak_n") or float("nan"))
            if math.isfinite(df):
                dfs.append(abs(df))
    w_bar = float(max(dfs) + w_slack) if dfs else float("nan")
    extras = []
    for name in ("env_retract", "env_reverse", "env_tilt"):
        extras.append(_trial_metrics(rows, stop_phase=name, hist_phase="", horizon_s=0.25, hist_s=hist_ms / 1000.0))
    payload = {
        "csv": str(csv_path),
        "align": align,
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_matched_different_queue": len(matched),
        "n_queue_splits_tail": len(splits),
        "queue_splits_tail": bool(splits),
        "w_bar_n": w_bar,
        "w_bar_note": "max_|dF| + slack on Window-A-matched train pairs; not p90",
        "w_slack_n": w_slack,
        "eps": {"F_n": eps_f, "x_mm": eps_x_mm, "v_mm_s": eps_v_mm_s, "u_mm_s": eps_u_mm_s, "hist_ms": hist_ms},
        "envelope_extras": extras,
        "collected_at": when,
        "what": "matched-state stop tail; sequential slow/fast is not this claim",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "tail.json", dump_jsonable(payload))
    _plot_12(rows, pairs, visu)
    if matched and splits:
        verdict = f"{len(splits)}/{len(matched)} 对同状态不同队列把尾巴切开了，延迟增广必要。"
    elif matched:
        verdict = (
            f"{len(matched)} 对通过了 Window A 的 (F,x,v,u) 匹配且历史不同，"
            "但尾巴差 < 0.15 mm：这套垫/速度上队列可能不是主因。"
        )
    elif pairs:
        verdict = "没有一对在 Window A 上同时满足同状态且历史不同。不能声称 same-state different queue。"
    else:
        verdict = "日志里没有 hist_A/stop_B 对。旧的 slow/fast 顺序不能拿来当这个实验。"
    write_readme(
        visu,
        f"""# 12_stop_tail — 同状态不同队列

采集：`{when}` · filter OFF · 力环关 · 对齐中位 {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms

## 结论

- 设计对数 {len(pairs)}，Window A 匹配且历史不同 **{len(matched)}**，尾巴切开 **{len(splits)}**。
- {verdict}
- 训练包络 \(\\bar w\) = {fmt_finite(w_bar, '.2f')} N（max|ΔF|+{w_slack:.2f}，不是 p90）。覆盖率留给 15。
- 匹配门：|ΔF|<{eps_f:.2f} N，|Δx|<{eps_x_mm:.2f} mm，|Δv|<{eps_v_mm_s:.1f} mm/s，|Δu|<{eps_u_mm_s:.1f} mm/s。
- 力反馈只用于 preload 初始化；hist / stop 全程开环 twist。软垫 settle 还要求 |Ḟ| 持续低于阈值。

## 图

`stop_tail.png`：匹配对 A/B 停止后的力与位移。这是动机图，不是 p90 证书。
""",
    )
    print(
        f"[12] pairs={len(pairs)} matched={len(matched)} split={len(splits)}  "
        f"w_bar={w_bar:.2f} N  data={data}",
        flush=True,
    )
    return payload


def _plot_12(rows, pairs: list[dict], visu: Path) -> None:
    t = col(rows, "t_wall_s", "t_mono_s")
    fz = wrench6(rows)[:, 2]
    dx = 1e3 * tool_z_displacement(pose6(rows))
    dt = float(np.nanmedian(np.diff(t[np.isfinite(t)]))) if t.size > 2 else 0.005
    n = max(int(round(0.25 / max(dt, 1e-4))), 8)
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 3.80), sharex=True, constrained_layout=True)
    drawn = False
    for pair, color in zip([p for p in pairs if p.get("matched")], (ACH, MINUS, CMD, "#009E73")):
        for side, ls in (("A", "-"), ("B", "--")):
            i0 = pair[side].get("i0")
            if i0 is None:
                continue
            sl = slice(int(i0), min(int(i0) + n, t.size))
            tt = t[sl] - t[int(i0)]
            axes[0].plot(tt, fz[sl], color=color, lw=1.1, ls=ls)
            axes[1].plot(tt, dx[sl] - dx[int(i0)], color=color, lw=1.1, ls=ls)
            drawn = True
    if not drawn:
        t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
        axes[0].plot(t - t0, fz, color=ACH, lw=1.0)
        axes[1].plot(t - t0, dx, color=MINUS, lw=1.0)
    axes[0].set_ylabel("Fz (N)")
    axes[1].set_ylabel("tool-Z after stop (mm)")
    axes[1].set_xlabel("time after t0 (s)")
    for letter, ax in zip("ab", axes):
        ax.grid(True, alpha=0.28)
        panel_tag(ax, letter)
    save(fig, visu, "stop_tail.png")
    plt.close(fig)


def _settle(
    srv: ContactLogger,
    f_pre: float,
    settle_s: float,
    band: float,
    *,
    eps_dfdt: float,
    quiet_s: float,
) -> bool:
    if not srv.seek_force(f_pre, vel_m_s=0.004, band_n=band, phase="settle"):
        return False
    if not srv.hold_quiet(
        settle_s,
        settle_s + 2.5,
        "settle",
        eps_dfdt=eps_dfdt,
        quiet_s=quiet_s,
    ):
        return False
    srv.reset_proxy()
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=5.50, contact_n=0.40)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--f-preload", type=float, default=1.40)
    p.add_argument("--f-match", type=float, default=2.50)
    p.add_argument("--u-match-mm-s", type=float, default=8.0)
    p.add_argument("--u-burst-mm-s", type=float, default=16.0)
    p.add_argument("--hist-ms", type=float, default=80.0)
    p.add_argument("--approach-s", type=float, default=1.60)
    p.add_argument("--stop-s", type=float, default=0.35)
    p.add_argument("--pair-settle-s", type=float, default=1.20, help="minimum wait between A and B")
    p.add_argument("--eps-dfdt", type=float, default=0.80, help="|dF/dt| N/s to call the pad settled")
    p.add_argument("--quiet-s", type=float, default=0.40)
    p.add_argument("--eps-f", type=float, default=0.25)
    p.add_argument("--eps-x-mm", type=float, default=0.35)
    p.add_argument("--eps-v-mm-s", type=float, default=4.0)
    p.add_argument("--tilt-deg-s", type=float, default=8.0)
    args = p.parse_args()
    when = stamp()
    print(
        f"[PLAN] MOVEJ mid, seek, then {args.pairs} matched-state pairs: "
        f"steady {args.u_match_mm_s:.0f} mm/s vs burst {args.u_burst_mm_s:.0f}→"
        f"{args.u_match_mm_s:.0f} mm/s, same backup, settle ≥{args.pair_settle_s:.1f}s "
        f"and |dF/dt|<{args.eps_dfdt:.2f} N/s  "
        f"force used only for preload  hist/stop are open-loop twist  "
        f"abort F≥{args.abort_n:.1f} N",
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
                hist_ms=args.hist_ms,
                eps_f=args.eps_f,
                eps_x_mm=args.eps_x_mm,
                eps_v_mm_s=args.eps_v_mm_s,
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
    log = data / "tail.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
    )
    u_match = args.u_match_mm_s / 1000.0
    u_burst = args.u_burst_mm_s / 1000.0
    eps_x = args.eps_x_mm / 1000.0
    eps_v = args.eps_v_mm_s / 1000.0
    f_switch = args.f_match - 0.35
    try:
        srv.start_twist()
        if not srv.seek_contact(0.008, contact_n=args.contact_n):
            return 2
        for i in range(int(args.pairs)):
            tag = f"{i:02d}"
            if not _settle(
                srv,
                args.f_preload,
                args.pair_settle_s,
                0.18,
                eps_dfdt=args.eps_dfdt,
                quiet_s=args.quiet_s,
            ):
                return 0 if srv.aborted else 2
            hit = srv.play_until(
                lambda _s: axis_twist(2, u_match),
                f"hist_A{tag}",
                lambda s: math.isfinite(s.last_fz) and s.last_fz >= args.f_match,
                args.approach_s,
            )
            if srv.aborted:
                return 0
            if not (math.isfinite(srv.last_fz) and srv.last_fz > args.contact_n):
                print(f"[12] pair {tag} A never in contact", flush=True)
                continue
            state = (float(srv.last_fz), float(srv.x_proxy), float(srv.last_vz), float(srv.last_twist[2]))
            print(
                f"[12] A{tag} latch  F={state[0]:.2f} N  x={1e3 * state[1]:.2f} mm  "
                f"v={1e3 * state[2]:.1f} mm/s  hit={hit}",
                flush=True,
            )
            if not srv.hold(0.0, args.stop_s, f"stop_A{tag}", check_abort=False):
                return 0 if srv.aborted else 130
            if not _settle(
                srv,
                args.f_preload,
                args.pair_settle_s,
                0.18,
                eps_dfdt=args.eps_dfdt,
                quiet_s=args.quiet_s,
            ):
                return 0 if srv.aborted else 2

            def _twist_b(s, _u_match=u_match, _u_burst=u_burst, _fsw=f_switch):
                if math.isfinite(s.last_fz) and s.last_fz >= _fsw:
                    return axis_twist(2, _u_match)
                return axis_twist(2, _u_burst)

            def _pred_b(
                s,
                _st=state,
                _ef=args.eps_f,
                _ex=eps_x,
                _ev=eps_v,
                _um=u_match,
            ):
                return (
                    math.isfinite(s.last_fz)
                    and abs(s.last_fz - _st[0]) < _ef
                    and abs(s.x_proxy - _st[1]) < _ex
                    and math.isfinite(s.last_vz)
                    and abs(s.last_vz - _st[2]) < _ev
                    and abs(float(s.last_twist[2]) - _um) < 1e-6
                )

            matched = srv.play_until(_twist_b, f"hist_B{tag}", _pred_b, args.approach_s)
            if srv.aborted:
                return 0
            if matched:
                print(
                    f"[12] B{tag} MATCH  F={srv.last_fz:.2f} N  x={1e3 * srv.x_proxy:.2f} mm  "
                    f"v={1e3 * srv.last_vz:.1f} mm/s",
                    flush=True,
                )
                if not srv.hold(0.0, args.stop_s, f"stop_B{tag}", check_abort=False):
                    return 0 if srv.aborted else 130
            else:
                print(f"[12] B{tag} miss — not claiming same state", flush=True)
                srv.hold(0.0, 0.12, f"miss_B{tag}", check_abort=False)
        # Envelope extras, not the matched-state claim.
        if math.isfinite(srv.last_fz) and srv.last_fz > args.contact_n:
            srv.hold(axis_twist(2, -0.006), 0.6, "run_retract")
            srv.hold(0.0, args.stop_s, "env_retract", check_abort=False)
            srv.hold(axis_twist(2, 0.006), 0.35, "run_reverse")
            srv.hold(axis_twist(2, -0.006), args.stop_s, "env_reverse", check_abort=False)
            srv.hold(axis_twist(args.theta_axis, math.radians(args.tilt_deg_s)), 0.35, "run_tilt")
            srv.hold(0.0, args.stop_s, "env_tilt", check_abort=False)
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
                hist_ms=args.hist_ms,
                eps_f=args.eps_f,
                eps_x_mm=args.eps_x_mm,
                eps_v_mm_s=args.eps_v_mm_s,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
