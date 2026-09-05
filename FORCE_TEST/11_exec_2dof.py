#!/usr/bin/env python3
"""Two feedback-DoF execution contract.  Force loop OFF.

What: uz and ωθ (image-normal rotation, default wy) separately, then a
      frequency-split joint multisine.  Air first, then a light preload.
Why: the force correction must map to motion the servo can actually make.
     10 observed Tn on tool-Z only.  This file adds coupling and an
     independent validation split.

Lever ω × r is removed from the linear velocity before any cross-term
is called “dynamic”.  Hard force certificates stay off this round.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_log import ContactLogger, axis_twist
from frf_util import fopdt_from_frf, median_dt, welch_frf
from goto_mid import go_mid_from_args
from id_common import add_contact_args, load_aligned, stash_window_a
from io_csv import col, write_json
from paper_fig import ACH, CMD, MINUS, mpl, panel_tag, save
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import (
    AlignmentError,
    fmt_finite,
    phase_mask,
    pose6,
    tool_z_world,
    twist_ach6,
    twist_cmd6,
)

KIND = "11_exec_2dof"
LEVER_M = 0.0


def _pair(cmd: np.ndarray, ach: np.ndarray, t: np.ndarray, dt_col: np.ndarray, mask: np.ndarray):
    m = mask & np.isfinite(t) & np.isfinite(cmd) & np.isfinite(ach)
    if int(np.count_nonzero(m)) < 64:
        return None
    order = np.argsort(t[m], kind="stable")
    return t[m][order], cmd[m][order], ach[m][order], dt_col[m][order]


def _channel(name: str, t, u, y, dt_col) -> dict:
    dt = median_dt(t, dt_col)
    freq, mag, phase, coh = welch_frf(u, y, dt)
    t0, tp, gain = fopdt_from_frf(freq, mag, phase, coh, f_hi=8.0)
    err = y - u
    return {
        "name": name,
        "T0_s": t0,
        "Tp_s": tp,
        "K": gain,
        "n": int(t.size),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "p95_abs": float(np.percentile(np.abs(err), 95)),
        "freq": freq,
        "mag": mag,
        "phase": phase,
        "coh": coh,
    }


def _lever_corrected_vz(v_lin: np.ndarray, omega: np.ndarray, pose, lever_m: float) -> np.ndarray:
    """vz minus the geometric (ω × r)_z contribution at a fixed tool offset."""

    if abs(float(lever_m)) < 1e-9:
        return v_lin
    n = tool_z_world(pose)
    r = float(lever_m) * n
    cross = np.cross(omega, r)
    return v_lin - np.sum(cross * n, axis=1)


def analyze(csv_path: Path, *, when: str, window_a_csv: str = "", lever_m: float = LEVER_M) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt_col = col(rows, "dt_actual_s")
    cmd = twist_cmd6(rows)
    ach = twist_ach6(rows)
    pose = pose6(rows)
    vz_ach = _lever_corrected_vz(ach[:, 2], ach[:, 3:6], pose, lever_m)
    channels = []
    for phase, u, y, label in (
        ("air_z", cmd[:, 2], vz_ach, "air uz→vz"),
        ("air_th", cmd[:, 4], ach[:, 4], "air ωθ→ωθ"),
        ("contact_z", cmd[:, 2], vz_ach, "preload uz→vz"),
        ("contact_th", cmd[:, 4], ach[:, 4], "preload ωθ→ωθ"),
    ):
        pair = _pair(u, y, t, dt_col, phase_mask(rows, phase))
        if pair is None:
            channels.append({"name": label, "T0_s": float("nan"), "n": 0})
            continue
        channels.append(_channel(label, *pair))
    cross = None
    jmask = phase_mask(rows, "air_joint", "contact_joint")
    cross_pair = _pair(cmd[:, 4], vz_ach, t, dt_col, jmask)
    if cross_pair is not None:
        ct, cu, cy, cdt = cross_pair
        freq, mag, phase, coh = welch_frf(cu, cy, median_dt(ct, cdt))
        band = (freq >= 0.3) & (freq <= 3.0) & (coh >= 0.4)
        cross = {
            "median_mag_0p3_3hz": float(np.median(mag[band])) if np.any(band) else float("nan"),
            "n_band": int(np.count_nonzero(band)),
        }
    payload = {
        "csv": str(csv_path),
        "align": {k: v for k, v in align.items() if k != "reason" or v},
        "lever_m": lever_m,
        "channels": [
            {k: v for k, v in ch.items() if k not in ("freq", "mag", "phase", "coh")}
            for ch in channels
        ],
        "cross_wth_to_vz": cross,
        "collected_at": when,
        "what": "2-DoF execution contract, not a force certificate",
    }
    air_z = next((c for c in channels if c["name"].startswith("air uz")), {})
    cross_small = (
        cross is None
        or not math.isfinite(float(cross.get("median_mag_0p3_3hz") or float("nan")))
        or float(cross["median_mag_0p3_3hz"]) < 0.15
    )
    payload["keep_diagonal"] = bool(cross_small)
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "exec.json", payload)
    _plot_11(channels, visu)
    t0 = 1e3 * float(air_z.get("T0_s") or float("nan"))
    write_readme(
        visu,
        f"""# 11_exec_2dof — 两个反馈自由度的执行合同

采集：`{when}` · filter OFF · 力环关 · Window A 对齐中位 {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms

## 结论

- 空气 uz→vz：**T0 = {fmt_finite(t0, '.1f')} ms**，K = {fmt_finite(float(air_z.get('K') or float('nan')), '.3f')}。对照 03 的 28 ms / 0.96，不要强行统一。
- 杠杆项已按 r = {lever_m:.3f} m 从 vz 扣掉（0 表示 TCP 与接触点重合的工作假设）。
- 交叉 |ωθ → vz| 在 0.3–3 Hz {('小，保留对角模型。' if cross_small else '不可忽略，QP 预测要用 2×2。')}
- 这不是力证书。独立 trial 的轨迹还没被 tube 覆盖之前，不要激活硬 force row。

## 图

`exec_bode.png`：有数据的通道的 |G|。
""",
    )
    print(f"[11] air T0={t0:.1f} ms  diagonal={payload['keep_diagonal']}  data={data}", flush=True)
    return payload


def _plot_11(channels: list[dict], visu: Path) -> None:
    plt = mpl()
    drawn = [c for c in channels if c.get("freq") is not None and np.size(c.get("freq"))]
    if not drawn:
        fig, ax = plt.subplots(figsize=(3.50, 2.20), constrained_layout=True)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$|G|$ (dB)")
        save(fig, visu, "exec_bode.png")
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    colors = (ACH, MINUS, CMD, "#009E73")
    for ch, color in zip(drawn, colors):
        f = ch["freq"]
        use = (f >= 0.18) & (f <= 10.5)
        if not np.any(use):
            continue
        ax.semilogx(
            f[use],
            20.0 * np.log10(np.maximum(ch["mag"][use], 1e-6)),
            color=color,
            lw=1.15,
            label=ch["name"],
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|G|$ (dB)")
    ax.set_xlim(0.18, 10.5)
    ax.grid(True, which="major", alpha=0.28)
    ax.legend(loc="lower left")
    save(fig, visu, "exec_bode.png")
    plt.close(fig)


def _multisine_pair(dt: float, seconds: float, amp_z: float, amp_th: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = int(round(float(seconds) / dt))
    t = np.arange(n) * dt
    seq = np.zeros((n, 6), dtype=float)
    for f in np.arange(0.4, 2.4, 0.4):
        seq[:, 2] += np.sin(2.0 * math.pi * f * t + float(rng.uniform(0, 2 * math.pi)))
    for f in np.arange(0.3, 1.8, 0.5):
        seq[:, 4] += np.sin(2.0 * math.pi * f * t + float(rng.uniform(0, 2 * math.pi)))
    z_peak = float(np.max(np.abs(seq[:, 2]))) or 1.0
    th_peak = float(np.max(np.abs(seq[:, 4]))) or 1.0
    seq[:, 2] *= amp_z / z_peak
    seq[:, 4] *= amp_th / th_peak
    return seq


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=3.50, contact_n=0.40)
    p.add_argument("--amp-mm-s", type=float, default=8.0)
    p.add_argument("--amp-deg-s", type=float, default=6.0)
    p.add_argument("--chirp-s", type=float, default=25.0)
    p.add_argument("--joint-s", type=float, default=20.0)
    p.add_argument("--preload-s", type=float, default=0.6)
    p.add_argument("--skip-contact", action="store_true")
    p.add_argument("--lever-mm", type=float, default=0.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=8.0)
    args = p.parse_args()
    when = stamp()
    amp_z = args.amp_mm_s / 1000.0
    amp_th = math.radians(args.amp_deg_s)
    print(
        f"[PLAN] MOVEJ mid, air uz chirp / ωθ chirp / joint multisine, "
        f"then optional preload repeat  amp {args.amp_mm_s:.1f} mm/s / "
        f"{args.amp_deg_s:.1f} deg/s  force loop OFF",
        flush=True,
    )
    if dry_exit(args):
        return 0
    if args.csv:
        try:
            analyze(Path(args.csv), when=when, window_a_csv=args.window_a_csv, lever_m=args.lever_mm / 1000.0)
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
    log = data / "exec.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
    )
    try:
        srv.start_twist()
        if not srv.hold(0.0, 0.3, "rest"):
            return 130
        if not srv.chirp_axis(2, amp_z, args.f0, args.f1, args.chirp_s, "air_z"):
            return 0 if srv.aborted else 130
        if not srv.hold(0.0, 0.3, "rest"):
            return 130
        if not srv.chirp_axis(args.theta_axis, amp_th, args.f0, min(args.f1, 4.0), args.chirp_s, "air_th"):
            return 0 if srv.aborted else 130
        if not srv.play(_multisine_pair(srv.dt, args.joint_s, amp_z, amp_th, 1), "air_joint"):
            return 0 if srv.aborted else 130
        if not args.skip_contact:
            if not srv.seek_contact(0.008, contact_n=args.contact_n):
                return 2
            if not srv.hold(axis_twist(2, 0.0), args.preload_s, "preload"):
                return 0 if srv.aborted else 130
            if not srv.chirp_axis(2, 0.5 * amp_z, args.f0, 4.0, 0.6 * args.chirp_s, "contact_z"):
                return 0 if srv.aborted else 130
            if not srv.chirp_axis(args.theta_axis, 0.5 * amp_th, args.f0, 3.0, 0.6 * args.chirp_s, "contact_th"):
                return 0 if srv.aborted else 130
            if not srv.play(_multisine_pair(srv.dt, 0.6 * args.joint_s, 0.5 * amp_z, 0.5 * amp_th, 2), "contact_joint"):
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
            analyze(log, when=when, window_a_csv=args.window_a_csv, lever_m=args.lever_mm / 1000.0)
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
