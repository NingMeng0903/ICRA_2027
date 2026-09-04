#!/usr/bin/env python3
"""System time delay of the inner velocity servo (force loop OFF).

What: T0 / Γd from vel_ff → v_ach on SERVO_TWIST steps.
Why: De Stefano μ, CDYOB Γd, and every later certificate sit on this delay.
     This is the robot+inner-loop dead time, not a force-law number.

How: command a known velocity step, cross-correlate command vs achieved.
     Report ms and ticks.  feedback_age is a freshness band, not T0.
     Do not use 3 Hz group delay or yaml system_delay_s=0.055 as T0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import fopdt_fit, load_pair, median_dt, moments, step_edges
from io_csv import write_csv, write_json
from goto_mid import add_movej_args, go_mid_from_args
from paper_fig import ACH, CMD, GUIDE, MINUS, PLUS, mpl, panel_tag, save
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme

LINEAR_MM_S = 15.0


def _onset(edges: list[dict]) -> list[dict]:
    return [e for e in edges if float(e.get("cmd_mm_s") or 0.0) >= 1.0]


def _median_ms(edges: list[dict]) -> float:
    vals = [1e3 * float(e["delay_s"]) for e in edges if math.isfinite(e.get("delay_s", float("nan")))]
    return float(np.median(vals)) if vals else float("nan")


def _pick_zoom(onset: list[dict]) -> dict | None:
    linear = [e for e in onset if float(e["cmd_mm_s"]) + 1e-9 >= LINEAR_MM_S]
    if not linear:
        return onset[0] if onset else None
    prefer = [e for e in linear if float(e["sign"]) > 0 and abs(float(e["cmd_mm_s"]) - 25.0) < 1.0]
    return prefer[0] if prefer else linear[0]


def _plot_01(t, u, y, onset: list[dict], t0_ms: float, visu: Path) -> None:
    plt = mpl()
    t0w = float(t[0]) if t.size else 0.0
    tw = t - t0w

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    axes[0].plot(tw, 1e3 * u, color=CMD, lw=1.05, label="Commanded")
    axes[0].plot(tw, 1e3 * y, color=ACH, lw=1.05, label="Achieved")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Velocity (mm/s)")
    axes[0].set_xlim(tw[0], tw[-1] if tw.size else 1.0)
    axes[0].legend(loc="upper left", ncol=1)
    panel_tag(axes[0], "a")

    edge = _pick_zoom(onset)
    ax = axes[1]
    if edge is not None:
        te = float(edge["t_edge_s"])
        dly = float(edge["delay_s"])
        m = (t >= te - 0.06) & (t <= te + 0.28)
        ax.plot(t[m] - te, 1e3 * u[m], color=CMD, lw=1.15, label="Commanded")
        ax.plot(t[m] - te, 1e3 * y[m], color=ACH, lw=1.15, label="Achieved")
        cmd_mm = float(edge["sign"]) * float(edge["cmd_mm_s"])
        if math.isfinite(dly) and dly > 0:
            ax.axvspan(0.0, dly, color=ACH, alpha=0.12, lw=0)
            y_arr = 0.08 * cmd_mm
            ax.annotate(
                "",
                xy=(dly, y_arr),
                xytext=(0.0, y_arr),
                arrowprops=dict(arrowstyle="<->", color=GUIDE, lw=0.9),
            )
            ax.text(0.5 * dly, y_arr, r"$T_0$", ha="center", va="bottom", color=GUIDE, fontsize=9)
        ax.axvline(0.0, color=GUIDE, ls=":", lw=0.7)
        ax.set_xlim(-0.06, 0.28)
    ax.set_xlabel("Time from command step (s)")
    ax.set_ylabel("Velocity (mm/s)")
    panel_tag(ax, "b")
    for a in axes:
        a.grid(True, alpha=0.28)
    save(fig, visu, "velocity_steps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.50, 2.55), constrained_layout=True)
    usable = [e for e in onset if float(e.get("delay_s") or 0.0) > 1.0e-6]
    pos = [e for e in usable if float(e["sign"]) > 0]
    neg = [e for e in usable if float(e["sign"]) < 0]
    ax.plot(
        [e["cmd_mm_s"] for e in pos],
        [1e3 * e["delay_s"] for e in pos],
        "o",
        ms=5.5,
        color=PLUS,
        label="Positive",
    )
    ax.plot(
        [e["cmd_mm_s"] for e in neg],
        [1e3 * e["delay_s"] for e in neg],
        "s",
        ms=5.0,
        color=MINUS,
        label="Negative",
    )
    if math.isfinite(t0_ms):
        ax.axhline(t0_ms, color=GUIDE, ls="--", lw=0.9, label="Median")
    ax.set_xlabel("Command speed (mm/s)")
    ax.set_ylabel("Delay (ms)")
    ax.set_xticks([8, 15, 25, 40])
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right")
    save(fig, visu, "delay_vs_speed.png")
    plt.close(fig)


def _readme_01(payload: dict) -> str:
    t0 = float(payload.get("T0_step_median_ms") or float("nan"))
    gd = payload.get("Gamma_d_ticks_from_steps")
    age = 1e3 * float((payload.get("feedback_age") or {}).get("p95") or float("nan"))
    when = payload.get("collected_at") or ""
    xc = payload.get("FOPDT_crosscheck") or {}
    return f"""# 01_delay — 内环死区 T0

采集：`{when}` · filter OFF · 力环关 · MOVEJ 中位 · `vel_ff → v_ach`

## 结论

- **T0 = {t0:.1f} ms（Γd ≈ {gd} tick）**。中位只收 15–40 mm/s 的 onset，不含回零、不含 8 mm/s。
- 8 mm/s 互相关会塌成 0 ms；曲线仍是先静约 40 ms 再爬。图上已去掉这些点。
- 步进整段 FOPDT（T0={1e3 * float(xc.get("T0_s") or float("nan")):.0f} ms，Tp={1e3 * float(xc.get("Tp_s") or float("nan")):.0f} ms）**不是 T0**。写模型用 03 的 Bode。
- feedback_age p95 = {age:.1f} ms，是新鲜度，不是植物延迟。
- 关滤波后上升变快（见 02）。本数比开着 50 ms slew 的旧 01（45 ms）干净，仍比 FRF 的 28 ms 多 1–2 tick（互相关会吃一点上升）。

## 图

- `velocity_steps.png` **a** 全列车，**b** 线性档一个 onset，阴影是该步死区。
- `delay_vs_speed.png` 15/25/40 mm/s 围着中位。

不要写 yaml。数据：`DATA/01_delay/`
"""


def analyze(csv_path: Path, *, when: str) -> dict:
    t, u, y, fb, dt_col, _rows = load_pair(csv_path)
    dt = median_dt(t, dt_col)
    edges = step_edges(t, u, y)
    onset = _onset(edges)
    linear = [e for e in onset if float(e["cmd_mm_s"]) + 1e-9 >= LINEAR_MM_S]
    t0_ms = _median_ms(linear) if linear else _median_ms(onset)
    t0 = t0_ms / 1e3 if math.isfinite(t0_ms) else float("nan")
    try:
        ticks, tp, gain = fopdt_fit(u, y, dt)
    except Exception:
        ticks, tp, gain = 0, float("nan"), float("nan")
    age = moments(fb)
    payload = {
        "csv": str(csv_path),
        "pair": "vel_ff → v_ach",
        "dt_s": dt,
        "T0_step_median_s": t0,
        "T0_step_median_ms": t0_ms,
        "T0_all_onset_ms": _median_ms(onset),
        "T0_includes_return_to_zero": False,
        "Gamma_d_ticks_from_steps": int(round(t0 / dt)) if math.isfinite(t0) and dt > 0 else None,
        "FOPDT_crosscheck": {
            "Gamma_d_ticks": ticks,
            "T0_s": ticks * dt,
            "Tp_s": tp,
            "K": gain,
            "note": "Step records under-identify phase.  Use 03_chirp_gv.py for Tn.",
        },
        "feedback_age": age,
        "not_T0": [
            "3 Hz group delay",
            "yaml system_delay_s=0.055",
            "feedback_age alone",
            "whole-record FOPDT on a step train",
        ],
        "edges": edges,
        "collected_at": when,
    }
    data, visu = kind_dirs("01_delay", preserve=csv_path)
    write_json(data / "delay.json", payload)
    write_csv(data / "edges.csv", edges)
    _plot_01(t, u, y, onset, t0_ms, visu)
    write_readme(visu, _readme_01(payload))
    print(
        f"[DELAY] T0={t0_ms:.1f} ms  Γd≈{payload['Gamma_d_ticks_from_steps']} ticks  "
        f"age_p95={1e3 * age.get('p95', float('nan')):.1f} ms  "
        f"onset={len(onset)}  data={data}  visu={visu}",
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="", help="analyze only")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--speeds", default="8,15,25,40")
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--rest-s", type=float, default=0.4)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    speeds = [float(x) for x in args.speeds.split(",") if x.strip()]
    if not args.csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, then SERVO_TWIST steps ±{speeds} mm/s  "
            f"hold={args.hold_s}s  force loop OFF  pair=vel_ff→v_ach",
            flush=True,
        )
    if dry_exit(args):
        return 0
    if args.csv:
        analyze(Path(args.csv), when=when)
        return 0
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("01_delay")
    log = data / "delay.csv"
    srv = ServoLogger(prefix=args.shm_prefix, hz=args.hz, log_csv=log)
    try:
        srv.start_twist()
        if not srv.hold(0.0, args.rest_s, "rest"):
            return 130
        for mm in speeds:
            for sign in (1.0, -1.0):
                cmd = sign * mm / 1000.0
                print(f"[STEP] {cmd:+.4f} m/s", flush=True)
                if not srv.hold(cmd, args.hold_s, f"step_{mm:.0f}"):
                    return 130
                if not srv.hold(0.0, args.rest_s, "rest"):
                    return 130
        srv.tick(0.0, "done")
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
    finally:
        srv.close()
    if log.is_file() and srv.n_rows > 16:
        analyze(log, when=when)
    return 0


if __name__ == "__main__":
    sys.exit(main())
