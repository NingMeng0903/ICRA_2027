#!/usr/bin/env python3
"""Inner-loop velocity tracking error (force loop OFF).

What: e = vel_ff − v_ach, RMSE and |e| vs time / speed.
Why: Lee 2024 BEFM: coupled stability needs the proxy–robot velocity
     error small.  That error is a property of the inner velocity servo,
     not of the admittance law.  We do not have joint torque τ, so we
     cannot run Lee's modulator — only measure the error their theorem
     actually depends on.

Drive a single chirp or pass --csv from 03_chirp_gv.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import load_pair, mpl, save_fig
from io_csv import write_json
from goto_mid import add_movej_args, go_mid_from_args
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme

BANDS = (
    ("0p3_1", 0.3, 1.0),
    ("1_3", 1.0, 3.0),
    ("0p3_3", 0.3, 3.0),
    ("3_8", 3.0, 8.0),
)


def _err_stats(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "n": 0,
            "rmse_m_s": float("nan"),
            "p95_abs_m_s": float("nan"),
            "max_abs_m_s": float("nan"),
        }
    return {
        "n": int(e.size),
        "rmse_m_s": float(np.sqrt(np.mean(e * e))),
        "p95_abs_m_s": float(np.percentile(np.abs(e), 95)),
        "max_abs_m_s": float(np.max(np.abs(e))),
    }


def _mm(stats: dict, key: str) -> float:
    return 1e3 * float(stats.get(key) or float("nan"))


def _inst_freq(t: np.ndarray, f0: float, f1: float) -> np.ndarray:
    span = max(float(t[-1] - t[0]), 1e-9)
    return f0 * np.exp(math.log(max(f1, f0 + 1e-6) / max(f0, 1e-6)) / span * (t - t[0]))


def _chirp_mask(rows: list, t: np.ndarray, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    from io_csv import col

    t_raw = col(rows, "t_wall_s", "t_mono_s")
    u_raw = col(rows, "vel_ff", "vel_ff_vz")
    y_raw = col(rows, "v_ach", "vz_achieved_tool")
    ph = np.array([str(r.get("phase") or "") for r in rows])
    mask = np.isfinite(t_raw) & np.isfinite(u_raw) & np.isfinite(y_raw)
    order = np.argsort(t_raw[mask], kind="stable")
    aligned = ph[mask][order]
    if aligned.size != t.size:
        return np.ones(t.size, dtype=bool)
    chirp = aligned == "chirp"
    return chirp if int(np.count_nonzero(chirp)) >= 64 else np.ones(t.size, dtype=bool)


def _readme_05(payload: dict) -> str:
    when = payload.get("collected_at") or ""
    bands = payload.get("bands") or {}
    lo = bands.get("0p3_1") or {}
    mid = bands.get("1_3") or {}
    hand = bands.get("0p3_3") or {}
    return f"""# 05_track — 内环速度跟踪误差

采集：`{when}` · filter OFF · 力环关 · MOVEJ 中位 · 15 mm/s chirp 40 s · `e = vel_ff − v_ach`

## 结论

- 全段 RMSE = {_mm(payload, "rmse_m_s"):.2f} mm/s，p95 |e| = {_mm(payload, "p95_abs_m_s"):.2f}，max = {_mm(payload, "max_abs_m_s"):.2f}。瞬时差，被 T0 相位拉大，不是跟丢。和 03 全段（≈8.2 / 18.5）同一量级，有效重复。
- **0.3–1 Hz（慢扫）：RMSE {_mm(lo, "rmse_m_s"):.2f}，p95 {_mm(lo, "p95_abs_m_s"):.2f}。不大。** 相对 15 mm/s 指令可以当跟得上。
- **1–3 Hz：RMSE {_mm(mid, "rmse_m_s"):.2f}，p95 {_mm(mid, "p95_abs_m_s"):.2f}。偏大。** 28 ms 死区在 3 Hz 已是约 30°，瞬时误差接近指令一半。
- **0.3–3 Hz 合成：RMSE {_mm(hand, "rmse_m_s"):.2f}，p95 {_mm(hand, "p95_abs_m_s"):.2f}。偏大，被 1–3 Hz 拉高。** 不能笼统说「手扫带误差小」。
- 这是已经量到的 T0，再拧内环增益压不下去。Lee 的「代理–机器人速度差小」只在 <1 Hz 成立；把 15 mm/s、0.3–3 Hz 整段当工作区则不成立。
- 没有关节力矩，不能 apply BEFM。不要写 yaml。

## 图

`tracking_error.png`：上指令/实现，下瞬时误差。全段看起来大，是 chirp 扫到 8–10 Hz。

数据：`DATA/05_track/`
"""


def analyze(csv_path: Path, *, when: str, f0: float = 0.2, f1: float = 10.0) -> dict:
    t, u, y, _fb, _dt_col, rows = load_pair(csv_path)
    e = u - y
    chirp = _chirp_mask(rows, t, u, y)
    freq = _inst_freq(t[chirp], f0, f1) if int(np.count_nonzero(chirp)) else np.array([])
    e_c = e[chirp]
    bands = {}
    for name, lo, hi in BANDS:
        if freq.size:
            pick = (freq >= lo) & (freq < hi)
            bands[name] = _err_stats(e_c[pick])
            bands[name]["f_lo_hz"] = lo
            bands[name]["f_hi_hz"] = hi
        else:
            bands[name] = _err_stats(np.array([]))
    payload = {
        "csv": str(csv_path),
        "pair": "e = vel_ff − v_ach",
        **_err_stats(e),
        "bands": bands,
        "lee_tau_available": False,
        "note": "No joint-torque interface — cannot apply BEFM.  This is the tracking error only.",
        "collected_at": when,
    }
    data, visu = kind_dirs("05_track", preserve=csv_path)
    write_json(data / "tracking.json", payload)
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)
    t0 = float(t[0]) if t.size else 0.0
    axes[0].plot(t - t0, 1e3 * u, label="vel_ff")
    axes[0].plot(t - t0, 1e3 * y, label="v_ach")
    axes[1].plot(t - t0, 1e3 * e)
    axes[0].set_ylabel("mm/s")
    axes[1].set_ylabel("error (mm/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_fig(fig, visu, "tracking_error.png")
    plt.close(fig)
    write_readme(visu, _readme_05(payload))
    hand = payload["bands"].get("0p3_3") or {}
    print(
        f"[TRACK] all RMSE={_mm(payload, 'rmse_m_s'):.2f}  "
        f"p95={_mm(payload, 'p95_abs_m_s'):.2f}  "
        f"0.3–3 Hz RMSE={_mm(hand, 'rmse_m_s'):.2f}  "
        f"p95={_mm(hand, 'p95_abs_m_s'):.2f} mm/s  data={data}",
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--amp-mm-s", type=float, default=15.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=10.0)
    p.add_argument("--chirp-s", type=float, default=40.0)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    if not args.csv:
        print(
            f"[PLAN] SERVO_TWIST chirp {args.amp_mm_s:.1f} mm/s for tracking error  "
            "force loop OFF  (MOVEJ mid first)",
            flush=True,
        )
    if dry_exit(args):
        return 0
    if args.csv:
        analyze(Path(args.csv), when=when, f0=args.f0, f1=args.f1)
        return 0
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("05_track")
    log = data / "track.csv"
    srv = ServoLogger(prefix=args.shm_prefix, hz=args.hz, log_csv=log)
    try:
        srv.start_twist()
        if not srv.hold(0.0, 0.4, "rest"):
            return 130
        if not srv.chirp(args.amp_mm_s / 1000.0, args.f0, args.f1, args.chirp_s, "chirp"):
            return 130
        srv.tick(0.0, "done")
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
    finally:
        srv.close()
    if log.is_file() and srv.n_rows > 64:
        analyze(log, when=when, f0=args.f0, f1=args.f1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
