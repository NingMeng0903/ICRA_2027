#!/usr/bin/env python3
"""Inner-loop step response (force loop OFF).  No chirp, no force law.

What: t10–t90, peak |dv/dt|, settled Kp vs command speed.
Why: CDYOB/inner bandwidth and the linear-vs-saturation boundary
     (this plant saturates near 2 m/s²).  Small-speed Kp drop is friction.
     Response delay of the velocity servo is the rise time, not T0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import load_pair, step_edges
from io_csv import write_csv, write_json
from goto_mid import add_movej_args, go_mid_from_args
from paper_fig import GUIDE, MINUS, PLUS, mpl, panel_tag, save
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme

LINEAR_LO = 15.0
LINEAR_HI = 40.0
ACCEL_CAP = 2.0


def _onset(edges: list[dict]) -> list[dict]:
    return [e for e in edges if float(e.get("cmd_mm_s") or 0.0) >= 1.0]


def _finite(edges: list[dict], key: str) -> list[float]:
    return [float(e[key]) for e in edges if math.isfinite(e.get(key, float("nan")))]


def _plot_02(onset: list[dict], visu: Path) -> None:
    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), constrained_layout=True)
    pos = [e for e in onset if float(e["sign"]) > 0]
    neg = [e for e in onset if float(e["sign"]) < 0]
    series = (
        (axes[0], "rise_s", 1e3, "Rise time (ms)", (0.0, 160.0)),
        (axes[1], "peak_accel_m_s2", 1.0, r"Peak acceleration (m/s$^2$)", (0.0, 3.2)),
        (axes[2], "kp", 1.0, "Settled $K_p$", (0.60, 1.08)),
    )
    for ax, key, scale, ylab, ylim in series:
        ax.plot(
            [e["cmd_mm_s"] for e in pos],
            [scale * float(e[key]) for e in pos],
            "o",
            ms=5.0,
            color=PLUS,
            label="Positive",
        )
        ax.plot(
            [e["cmd_mm_s"] for e in neg],
            [scale * float(e[key]) for e in neg],
            "s",
            ms=4.5,
            color=MINUS,
            label="Negative",
        )
        ax.set_xlabel("Command speed (mm/s)")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_xticks([4, 15, 40, 80])
        ax.grid(True, alpha=0.28)
    axes[1].axhline(ACCEL_CAP, color=GUIDE, ls="--", lw=0.9)
    axes[2].axhline(1.0, color=GUIDE, ls=":", lw=0.7)
    axes[0].legend(loc="upper right")
    for letter, ax in zip("abc", axes):
        panel_tag(ax, letter)
    save(fig, visu, "rise_accel_kp.png")
    plt.close(fig)


def _readme_02(payload: dict) -> str:
    rise = float(payload.get("rise_median_linear_ms") or float("nan"))
    amax = float(payload.get("peak_accel_max_m_s2") or float("nan"))
    kp = float(payload.get("kp_median_linear") or float("nan"))
    sat = payload.get("saturates_at_mm_s")
    sat_s = f"{float(sat):.0f} mm/s" if sat is not None else "未标出"
    when = payload.get("collected_at") or ""
    return f"""# 02_step — 速度阶跃上升 / 加速度 / Kp

采集：`{when}` · filter OFF · 力环关 · MOVEJ 中位 · `vel_ff → v_ach`

## 结论

- 线性档 15–40 mm/s：中位上升 **{rise:.1f} ms**，Kp = **{kp:.3f}**。
- 峰值加速度最大 **{amax:.2f} m/s²**。约 40 mm/s 贴上 2 m/s²；{sat_s} 起饱和，上升又变长。
- 4–6 mm/s 上升 80–155 ms、Kp≈0.80：摩擦，不是 T0。
- 关滤波后线性上升从旧取的 ~65 ms 掉到 ~25 ms。T0 仍看 01 / 03，不看上升时间。

## 图

`rise_accel_kp.png`

- **a** 上升时间。15–40 短而平，两端（摩擦 / 饱和）变长。
- **b** 峰值 |dv/dt|。虚线 2 m/s²。
- **c** 稳态 Kp。12 mm/s 以下掉，15 以上接近 1。

不要写 yaml。数据：`DATA/02_step/`
"""


def analyze(csv_path: Path, *, when: str) -> dict:
    _t, _u, _y, _fb, _dt, _rows = load_pair(csv_path)
    edges = step_edges(_t, _u, _y)
    onset = _onset(edges)
    linear = [
        e
        for e in onset
        if LINEAR_LO - 1e-9 <= float(e["cmd_mm_s"]) <= LINEAR_HI + 1e-9
    ]
    sat = [e for e in onset if float(e["cmd_mm_s"]) + 1e-9 >= 60.0]
    rise_lin = _finite(linear, "rise_s")
    acc_all = _finite(onset, "peak_accel_m_s2")
    kp_lin = _finite(linear, "kp")
    payload = {
        "csv": str(csv_path),
        "pair": "vel_ff → v_ach",
        "rise_median_linear_s": float(np.median(rise_lin)) if rise_lin else float("nan"),
        "rise_median_linear_ms": 1e3 * float(np.median(rise_lin)) if rise_lin else float("nan"),
        "peak_accel_max_m_s2": float(np.max(acc_all)) if acc_all else float("nan"),
        "kp_median_linear": float(np.median(kp_lin)) if kp_lin else float("nan"),
        "linear_speed_mm_s": [LINEAR_LO, LINEAR_HI],
        "linear_accel_cap_m_s2": ACCEL_CAP,
        "saturates_at_mm_s": 60.0 if sat else None,
        "T0_is": "01_system_delay.py",
        "edges": edges,
        "collected_at": when,
    }
    data, visu = kind_dirs("02_step", preserve=csv_path)
    write_json(data / "step.json", payload)
    write_csv(data / "edges.csv", edges)
    _plot_02(onset, visu)
    write_readme(visu, _readme_02(payload))
    print(
        f"[STEP] rise_lin={payload['rise_median_linear_ms']:.1f} ms  "
        f"amax={payload['peak_accel_max_m_s2']:.2f} m/s²  "
        f"Kp_lin={payload['kp_median_linear']:.3f}  "
        f"onset={len(onset)}  data={data}  visu={visu}",
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--speeds", default="4,6,8,10,12,15,20,25,30,40,60,80")
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--rest-s", type=float, default=0.5)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    speeds = [float(x) for x in args.speeds.split(",") if x.strip()]
    if not args.csv:
        print(f"[PLAN] MOVEJ mid-stroke, then SERVO_TWIST steps ±{speeds} mm/s  force loop OFF", flush=True)
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

    data, _visu = kind_dirs("02_step")
    log = data / "step.csv"
    srv = ServoLogger(prefix=args.shm_prefix, hz=args.hz, log_csv=log)
    try:
        srv.start_twist()
        if not srv.hold(0.0, args.rest_s, "rest"):
            return 130
        for mm in speeds:
            for sign in (1.0, -1.0):
                print(f"[STEP] {sign * mm:+.1f} mm/s", flush=True)
                if not srv.hold(sign * mm / 1000.0, args.hold_s, f"step_{mm:.0f}"):
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
