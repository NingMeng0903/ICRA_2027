#!/usr/bin/env python3
"""Two feedback-DoF execution contract.  Force loop OFF.

What: identify the full 2×2 plant
      [vz, ωθ_ach] = G(s) [uz, ωθ] + w
      with frequency-split joint multisine (no shared lines), then a
      residual tube W on an in-take validation split.
Why: QP may only request (uz, ωθ) the servo can actually produce.
     Cross-term “small yes/no” is not the contract.

Contact z uses a displacement-limited chirp so the pad stays preloaded.
Hard force certificates stay off this file.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_log import ContactLogger, axis_twist
from frf_util import fopdt_from_frf, median_dt, peak_accel, welch_frf
from goto_mid import go_mid_from_args
from id_common import add_contact_args, load_aligned, stash_window_a
from id_math import (
    TH_FREQS_HZ,
    Z_FREQS_HZ,
    dump_jsonable,
    finite_channel,
    freq_split_disp_multisine,
    predict_2x2,
    residual_tube,
)
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
G_NAMES = ("zz", "zth", "thz", "thth")


def _pair(cmd: np.ndarray, ach: np.ndarray, t: np.ndarray, dt_col: np.ndarray, mask: np.ndarray):
    m = mask & np.isfinite(t) & np.isfinite(cmd) & np.isfinite(ach)
    if int(np.count_nonzero(m)) < 64:
        return None
    order = np.argsort(t[m], kind="stable")
    return t[m][order], cmd[m][order], ach[m][order], dt_col[m][order]


def _fit_channel(name: str, t, u, y, dt_col) -> dict:
    dt = median_dt(t, dt_col)
    freq, mag, phase, coh = welch_frf(u, y, dt)
    t0, tp, gain = fopdt_from_frf(freq, mag, phase, coh, f_hi=8.0)
    return {
        "name": name,
        "T0_s": t0,
        "Tp_s": tp,
        "K": gain,
        "n": int(t.size),
        "dt_s": dt,
        "peak_accel": peak_accel(y, t),
        "freq": freq,
        "mag": mag,
        "phase": phase,
        "coh": coh,
    }


def _lever_corrected_vz(v_lin: np.ndarray, omega: np.ndarray, pose, lever_m: float) -> np.ndarray:
    if abs(float(lever_m)) < 1e-9:
        return v_lin
    n = tool_z_world(pose)
    r = float(lever_m) * n
    cross = np.cross(omega, r)
    return v_lin - np.sum(cross * n, axis=1)


def _split_idx(n: int, frac: float = 0.70) -> int:
    return max(int(round(frac * n)), 32)


def analyze(
    csv_path: Path,
    *,
    when: str,
    window_a_csv: str = "",
    lever_m: float = LEVER_M,
    theta_axis: int = 4,
    slack_vz: float = 0.001,
    slack_wth: float = 0.02,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt_col = col(rows, "dt_actual_s")
    cmd = twist_cmd6(rows)
    ach = twist_ach6(rows)
    pose = pose6(rows)
    th = int(theta_axis)
    vz_ach = _lever_corrected_vz(ach[:, 2], ach[:, 3:6], pose, lever_m)
    wth_ach = ach[:, th]
    uz = cmd[:, 2]
    wth_u = cmd[:, th]

    # Diagonal + both cross terms from single-axis phases (air / contact).
    jobs = {
        "zz": [("air_z", "contact_z"), uz, vz_ach],
        "thth": [("air_th", "contact_th"), wth_u, wth_ach],
        "zth": [("air_th", "contact_th"), wth_u, vz_ach],
        "thz": [("air_z", "contact_z"), uz, wth_ach],
    }
    G: dict[str, dict] = {}
    drawn = []
    for name, (phases, u, y) in jobs.items():
        pair = _pair(u, y, t, dt_col, phase_mask(rows, *phases))
        if pair is None:
            G[name] = {"name": name, "T0_s": float("nan"), "Tp_s": float("nan"), "K": float("nan"), "n": 0}
            continue
        ch = _fit_channel(name, *pair)
        G[name] = ch
        drawn.append(ch)

    # Time-domain 2×2 residual on joint ID vs val (air_joint / contact_joint / val_joint).
    id_mask = phase_mask(rows, "air_joint", "contact_joint")
    val_mask = phase_mask(rows, "val_joint")
    if not np.any(val_mask):
        # Last 30% of joint as val if the collect did not add a dedicated split.
        j = np.flatnonzero(id_mask)
        if j.size >= 80:
            cut = j[0] + _split_idx(j.size)
            val_mask = np.zeros(id_mask.size, dtype=bool)
            val_mask[cut : j[-1] + 1] = True
            id_mask[cut:] = False

    def _tube_on(mask: np.ndarray) -> tuple[dict, dict, dict]:
        m = mask & np.isfinite(t) & np.isfinite(uz) & np.isfinite(vz_ach)
        if int(np.count_nonzero(m)) < 32:
            empty = residual_tube(np.array([]))
            return empty, empty, {"n": 0, "dt_s": float("nan")}
        dt = median_dt(t[m], dt_col[m])
        g_fit = {k: G[k] for k in G_NAMES}
        vz_hat, th_hat = predict_2x2(uz[m], wth_u[m], dt, g_fit)
        ev = residual_tube(vz_ach[m] - vz_hat, slack=slack_vz)
        ew = residual_tube(wth_ach[m] - th_hat, slack=slack_wth)
        return ev, ew, {"n": int(np.count_nonzero(m)), "dt_s": dt}

    ev_id, ew_id, meta_id = _tube_on(id_mask)
    ev_val, ew_val, meta_val = _tube_on(val_mask)
    val_cov_z = float("nan")
    val_cov_w = float("nan")
    if meta_val.get("n", 0) >= 32 and meta_id.get("n", 0) >= 32:
        dt = float(meta_val["dt_s"])
        m = val_mask & np.isfinite(uz) & np.isfinite(vz_ach)
        vz_hat, th_hat = predict_2x2(uz[m], wth_u[m], dt, G)
        ez = vz_ach[m] - vz_hat
        ew = wth_ach[m] - th_hat
        if ev_id["n"] and math.isfinite(float(ev_id.get("bar") or float("nan"))):
            val_cov_z = float(np.mean(np.abs(ez[np.isfinite(ez)]) <= ev_id["bar"] + 1e-12))
        if ew_id["n"] and math.isfinite(float(ew_id.get("bar") or float("nan"))):
            val_cov_w = float(np.mean(np.abs(ew[np.isfinite(ew)]) <= ew_id["bar"] + 1e-12))

    keep_full = any(finite_channel(G[name]) and abs(float(G[name]["K"])) > 0.08 for name in ("zth", "thz"))
    delays = [float(G[n]["T0_s"]) for n in ("zz", "thth") if math.isfinite(float(G[n].get("T0_s") or float("nan")))]
    payload = {
        "csv": str(csv_path),
        "align": {k: v for k, v in align.items() if k != "reason" or v},
        "lever_m": lever_m,
        "theta_axis": th,
        "G": {name: {k: G[name].get(k) for k in ("T0_s", "Tp_s", "K", "n", "peak_accel")} for name in G_NAMES},
        "z_freqs_hz": list(Z_FREQS_HZ),
        "th_freqs_hz": list(TH_FREQS_HZ),
        "tube": {
            "e_vz": ev_id,
            "e_wth": ew_id,
            "e_vz_bar": ev_id.get("bar"),
            "e_wth_bar": ew_id.get("bar"),
        },
        "tube_val": {"e_vz": ev_val, "e_wth": ew_val, "cover_vz": val_cov_z, "cover_wth": val_cov_w},
        "delay_s": float(np.median(delays)) if delays else float("nan"),
        "peak_accel_z_m_s2": G["zz"].get("peak_accel"),
        "peak_accel_th_rad_s2": G["thth"].get("peak_accel"),
        "keep_full_2x2": bool(keep_full),
        "keep_diagonal": (not keep_full),
        "collected_at": when,
        "what": "execution contract y ∈ Ĝ u ⊕ W, not a force certificate",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "exec.json", dump_jsonable(payload))
    _plot_11(drawn, visu)
    _plot_11_tube(rows, G, id_mask, val_mask, uz, wth_u, vz_ach, wth_ach, dt_col, t, visu)
    t0 = 1e3 * float(G["zz"].get("T0_s") or float("nan"))
    write_readme(
        visu,
        f"""# 11_exec_2dof — 2×2 执行合同

采集：`{when}` · filter OFF · 力环关 · 对齐中位 {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms

## 结论

- \(G_{{zz}}\): T0 = {fmt_finite(t0, '.1f')} ms，K = {fmt_finite(float(G['zz'].get('K') or float('nan')), '.3f')}。
- \(G_{{\\theta\\theta}}\): T0 = {fmt_finite(1e3 * float(G['thth'].get('T0_s') or float('nan')), '.1f')} ms，K = {fmt_finite(float(G['thth'].get('K') or float('nan')), '.3f')}。
- 交叉 \(G_{{z\\theta}}\) K = {fmt_finite(float(G['zth'].get('K') or float('nan')), '.3f')}，\(G_{{\\theta z}}\) K = {fmt_finite(float(G['thz'].get('K') or float('nan')), '.3f')}。
  {'QP 预测用完整 2×2。' if keep_full else '交叉小，对角模型够用；tube 仍按 2×2 残差报。'}
- 训练残差管：ē_v = {fmt_finite(1e3 * float(ev_id.get('bar') or float('nan')), '.2f')} mm/s，ē_ω = {fmt_finite(float(ev_id.get('bar') and ew_id.get('bar') or float('nan')), '.3f')} rad/s（max+|slack|，不是 p90）。
- 同次 val 覆盖 vz {fmt_finite(val_cov_z, '.2f')}，ωθ {fmt_finite(val_cov_w, '.2f')}。硬 force row 要等 15 的独立覆盖。
- 联合激励频率 z {Z_FREQS_HZ} Hz，θ {TH_FREQS_HZ} Hz，无共线。

## 图

`exec_bode.png`：四个通道的 |G|。`exec_tube.png`：2×2 预测残差。
""",
    )
    print(
        f"[11] Gzz T0={t0:.1f} ms  2x2={keep_full}  "
        f"ē_v={1e3 * float(ev_id.get('bar') or float('nan')):.2f} mm/s  data={data}",
        flush=True,
    )
    return payload


def _plot_11(channels: list[dict], visu: Path) -> None:
    plt = mpl()
    drawn = [c for c in channels if c.get("freq") is not None and np.size(c.get("freq"))]
    fig, axes = plt.subplots(2, 2, figsize=(3.50, 3.60), sharex=True, constrained_layout=True)
    labels = {"zz": r"$G_{zz}$", "zth": r"$G_{z\theta}$", "thz": r"$G_{\theta z}$", "thth": r"$G_{\theta\theta}$"}
    order = ["zz", "zth", "thz", "thth"]
    colors = (ACH, MINUS, CMD, "#009E73")
    by = {c["name"]: c for c in drawn}
    for ax, name, color, letter in zip(axes.ravel(), order, colors, "abcd"):
        ch = by.get(name)
        if ch is not None:
            f = ch["freq"]
            use = (f >= 0.18) & (f <= 10.5)
            if np.any(use):
                ax.semilogx(
                    f[use],
                    20.0 * np.log10(np.maximum(ch["mag"][use], 1e-6)),
                    color=color,
                    lw=1.15,
                )
        ax.set_ylabel(r"$|G|$ (dB)")
        ax.grid(True, which="major", alpha=0.28)
        ax.set_xlim(0.18, 10.5)
        panel_tag(ax, letter)
        ax.text(0.98, 0.96, labels[name], transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    save(fig, visu, "exec_bode.png")
    plt.close(fig)


def _plot_11_tube(rows, G, id_mask, val_mask, uz, wth_u, vz_ach, wth_ach, dt_col, t, visu: Path) -> None:
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 3.60), sharex=True, constrained_layout=True)
    m = (id_mask | val_mask) & np.isfinite(t)
    if not np.any(m):
        save(fig, visu, "exec_tube.png")
        plt.close(fig)
        return
    dt = median_dt(t[m], dt_col[m])
    vz_hat, th_hat = predict_2x2(uz, wth_u, dt, G)
    t0 = float(t[m][0])
    tt = t - t0
    axes[0].plot(tt[m], 1e3 * (vz_ach[m] - vz_hat[m]), color=ACH, lw=0.8)
    axes[1].plot(tt[m], wth_ach[m] - th_hat[m], color=MINUS, lw=0.8)
    axes[0].set_ylabel(r"$e_{v_z}$ (mm/s)")
    axes[1].set_ylabel(r"$e_{\omega_\theta}$ (rad/s)")
    axes[1].set_xlabel("time (s)")
    for letter, ax in zip("ab", axes):
        ax.grid(True, alpha=0.28)
        panel_tag(ax, letter)
    save(fig, visu, "exec_tube.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=3.50, contact_n=1.20)
    p.add_argument("--amp-mm-s", type=float, default=8.0, help="air z velocity chirp peak")
    p.add_argument("--amp-deg-s", type=float, default=6.0, help="air θ velocity chirp peak")
    p.add_argument("--ax-mm", type=float, default=0.35, help="contact z displacement amplitude")
    p.add_argument("--ath-deg", type=float, default=0.60, help="contact θ angle amplitude")
    p.add_argument("--air-ax-mm", type=float, default=2.0)
    p.add_argument("--air-ath-deg", type=float, default=2.0)
    p.add_argument("--chirp-s", type=float, default=20.0)
    p.add_argument("--joint-s", type=float, default=18.0)
    p.add_argument("--val-s", type=float, default=10.0)
    p.add_argument("--preload-s", type=float, default=0.6)
    p.add_argument("--skip-contact", action="store_true")
    p.add_argument("--lever-mm", type=float, default=0.0)
    p.add_argument("--f0", type=float, default=0.3)
    p.add_argument("--f1", type=float, default=8.0)
    args = p.parse_args()
    when = stamp()
    amp_z = args.amp_mm_s / 1000.0
    amp_th = math.radians(args.amp_deg_s)
    ax_c = args.ax_mm / 1000.0
    ath_c = math.radians(args.ath_deg)
    ax_a = args.air_ax_mm / 1000.0
    ath_a = math.radians(args.air_ath_deg)
    print(
        f"[PLAN] MOVEJ mid, air uz/ωθ chirp + freq-split joint, "
        f"then preload F≈{args.contact_n:.1f} N with disp-limited 2×2  "
        f"z lines {Z_FREQS_HZ}  θ lines {TH_FREQS_HZ}  force loop OFF",
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
                lever_m=args.lever_mm / 1000.0,
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
        if not srv.play(
            freq_split_disp_multisine(srv.dt, args.joint_s, ax_a, ath_a, theta_axis=args.theta_axis, seed=1),
            "air_joint",
        ):
            return 0 if srv.aborted else 130
        if not args.skip_contact:
            if not srv.seek_contact(0.006, contact_n=args.contact_n):
                return 2
            if not srv.hold(axis_twist(2, 0.0), args.preload_s, "preload"):
                return 0 if srv.aborted else 130
            if not srv.chirp_disp_axis(2, ax_c, args.f0, 6.0, 0.7 * args.chirp_s, "contact_z", min_n=0.25):
                if not (srv.aborted or srv.unloaded):
                    return 130
            if not srv.chirp_disp_axis(
                args.theta_axis, ath_c, args.f0, 4.0, 0.7 * args.chirp_s, "contact_th", min_n=0.25
            ):
                if not (srv.aborted or srv.unloaded):
                    return 130
            if not srv.play(
                freq_split_disp_multisine(
                    srv.dt, 0.7 * args.joint_s, ax_c, ath_c, theta_axis=args.theta_axis, seed=2
                ),
                "contact_joint",
            ):
                if not (srv.aborted or srv.unloaded):
                    return 130
            if not srv.play(
                freq_split_disp_multisine(
                    srv.dt, args.val_s, ax_c, ath_c, theta_axis=args.theta_axis, seed=11
                ),
                "val_joint",
            ):
                if not (srv.aborted or srv.unloaded):
                    return 130
            srv.retract_z(0.008, 2.5, 0.30)
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
                lever_m=args.lever_mm / 1000.0,
                theta_axis=args.theta_axis,
            )
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
