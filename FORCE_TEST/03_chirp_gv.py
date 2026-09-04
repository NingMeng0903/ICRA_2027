#!/usr/bin/env python3
"""Inner-loop velocity FRF Gv (force loop OFF).  One amplitude, one sweep.

What: nonparametric Gv = v_ach / vel_ff, plus FOPDT (K, T0, Tp) and coherence.
Why: CDYOB nominal model Cn / Tn.  Sweep to 10 Hz; treat γ²≥0.6 as valid
     (usually 1–8 Hz).  8 Hz is enough for 0.3–3 Hz hand-scan.

Not: Ya, Dimeas filter, force-law Bode.  Those mix the outer law into Gv.
     Run once per amplitude (8 / 15 / 25 mm/s) if you want amplitude curves.
     Time-domain xcorr on the chirp is not T0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import fopdt_fit, fopdt_from_frf, median_dt, welch_frf, xcorr_delay_s
from io_csv import col, load_rows, write_json
from goto_mid import add_movej_args, go_mid_from_args
from paper_fig import ACH, CMD, GUIDE, log_freq_ticks, mpl, panel_tag, save
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme


def _chirp_pair(csv_path: Path):
    rows = load_rows(csv_path)
    chirp = [r for r in rows if (r.get("phase") or "") == "chirp"]
    use = chirp if len(chirp) > 64 else rows
    t = col(use, "t_mono_s", "t_wall_s")
    u = col(use, "vel_ff", "vel_ff_vz")
    y = col(use, "v_ach", "vz_achieved_tool")
    dt_col = col(use, "dt_actual_s")
    mask = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    order = np.argsort(t[mask], kind="stable")
    return t[mask][order], u[mask][order], y[mask][order], dt_col[mask][order]


def _fopdt_curve(freq, t0, tp, gain):
    w = 2.0 * math.pi * np.asarray(freq, dtype=float)
    den = 1.0 + 1j * w * float(tp)
    g = np.exp(-1j * w * float(t0)) * float(gain) / den
    return np.abs(g), np.angle(g)


def _plot_03(freq, mag, phase, coh, *, t0, tp, gain, f1, visu: Path) -> None:
    plt = mpl()
    fig, axes = plt.subplots(3, 1, figsize=(3.50, 5.60), sharex=True, constrained_layout=True)
    cut = 8.0
    f_draw = min(float(f1) + 0.4, 10.0)
    use = freq <= f_draw + 1e-12 if freq.size else np.zeros(0, dtype=bool)
    if np.any(use):
        f = freq[use]
        g = mag[use]
        ph = np.degrees(np.unwrap(phase[use]))
        c = coh[use]
        fm, phm = _fopdt_curve(f, t0, tp, gain)
        phm_deg = np.degrees(phm)
        axes[0].semilogx(f, 20.0 * np.log10(np.maximum(g, 1e-6)), color=ACH, lw=1.15)
        axes[1].semilogx(f, ph, color=ACH, lw=1.15)
        if math.isfinite(t0) and math.isfinite(tp):
            axes[0].semilogx(f, 20.0 * np.log10(np.maximum(fm, 1e-6)), color=CMD, lw=0.9, ls="--")
            axes[1].semilogx(f, phm_deg, color=CMD, lw=0.9, ls="--")
        axes[2].semilogx(f, c, color=ACH, lw=1.15)
        band = f <= cut + 1e-12
        mag_db = 20.0 * np.log10(np.maximum(g, 1e-6))
        mag_s = mag_db[band] if np.any(band) else mag_db
        ph_s = ph[band] if np.any(band) else ph
        pm_s = phm_deg[band] if np.any(band) else phm_deg
        axes[0].set_ylim(float(np.min(mag_s)) - 3.0, float(np.max(mag_s)) + 3.0)
        plo = float(min(np.min(ph_s), np.min(pm_s))) - 15.0
        phi = float(max(np.max(ph_s), np.max(pm_s))) + 15.0
        axes[1].set_ylim(max(plo, -200.0), min(phi, 20.0))
    axes[0].set_ylabel(r"$|G_v|$ (dB)")
    axes[1].set_ylabel("Phase (deg)")
    axes[2].set_ylabel(r"Coherence $\gamma^2$")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylim(0.0, 1.05)
    x_left, x_right = 0.18, 10.5
    for letter, ax in zip("abc", axes):
        ax.grid(True, which="major", alpha=0.28)
        ax.set_xlim(x_left, x_right)
        log_freq_ticks(ax, f_lo=x_left, f_hi=x_right, cutoff_hz=cut)
        panel_tag(ax, letter)
    y_top = axes[0].get_ylim()[1]
    axes[0].text(
        cut,
        y_top,
        "8 Hz cut-off",
        color=GUIDE,
        fontsize=8,
        ha="center",
        va="bottom",
    )
    save(fig, visu, "gv_bode.png")
    plt.close(fig)


def _readme_03(payload: dict, *, prefix: str) -> str:
    t0 = 1e3 * float(payload.get("T0_s") or float("nan"))
    tp = 1e3 * float(payload.get("Tp_s") or float("nan"))
    k = float(payload.get("K") or float("nan"))
    flo = float(payload.get("valid_f_lo_hz") or float("nan"))
    fhi = float(payload.get("valid_f_hi_hz") or float("nan"))
    xc = 1e3 * float(payload.get("time_domain_xcorr_s") or float("nan"))
    when = payload.get("collected_at") or ""
    amp = float(payload.get("amp_mm_s") or float("nan"))
    return f"""# {prefix} — 内环速度 FRF

采集：`{when}` · filter OFF · 力环关 · `{amp:.0f}` mm/s chirp · `vel_ff → v_ach`

## 结论

- **T0 = {t0:.1f} ms，Tp = {tp:.1f} ms，K = {k:.3f}**（FRF 拟合，γ²≥0.6，≤8 Hz）。
- 有效带 **{flo:.1f}–{fhi:.1f} Hz**（只计扫频以内）。手扫 0.3–3 Hz 足够。
- 时域 xcorr = {xc:.0f} ms **不是 T0**。
- 15 mm/s 已够：0.3–8 Hz 相干约 0.98。再加大激励会在高频顶加速度饱和，帮不上忙。
- 不要用这组数覆盖 yaml，除非明确要写。8 月 29 日空气辨识仍是 T0=35 ms / Tp=12 ms。本轮相位更贴 28+14。

## 图

`gv_bode.png`：对数频率，刻度 0.2 / 0.5 / 1 / 2 / 5 / 8 / 10 Hz。曲线可画出 8 Hz 一点；拟合和有效带只到 8 Hz。顶上 “8 Hz cut-off” 标截止，图上不再画阈值虚线。

- **a** — |Gv|（实线实测，虚线 FOPDT）
- **b** — 相位
- **c** — 相干。顶上写 “8 Hz cut-off”，图上不画阈值虚线。

数据：`DATA/{prefix}/`
"""


def analyze(
    csv_path: Path,
    *,
    when: str,
    f1: float,
    amp_mm_s: float,
    prefix: str = "03_chirp",
) -> dict:
    t, u, y, dt_col = _chirp_pair(csv_path)
    dt = median_dt(t, dt_col)
    freq, mag, phase, coh = welch_frf(u, y, dt)
    t0, tp, gain = fopdt_from_frf(freq, mag, phase, coh, f_hi=min(8.0, f1))
    ticks_td, tp_td, gain_td = fopdt_fit(u, y, dt)
    xc = xcorr_delay_s(u, y, dt)
    ok = (coh >= 0.6) & (freq >= 0.2) & (freq <= min(8.0, float(f1)) + 1e-9)
    flo = float(np.min(freq[ok])) if np.any(ok) else float("nan")
    fhi = float(np.max(freq[ok])) if np.any(ok) else float("nan")
    low = ok & (freq >= 0.3) & (freq <= 1.0)
    k_lf = float(np.median(mag[low])) if np.any(low) else float("nan")
    payload = {
        "csv": str(csv_path),
        "pair": "vel_ff → v_ach",
        "amp_mm_s": amp_mm_s,
        "T0_s": t0,
        "Tp_s": tp,
        "K": gain,
        "K_lf_0p3_1hz": k_lf,
        "Gamma_d_ticks": int(round(t0 / dt)) if math.isfinite(t0) and dt > 0 else None,
        "valid_f_lo_hz": flo,
        "valid_f_hi_hz": fhi,
        "use_band": "gamma2>=0.6",
        "scan_0p3_3hz_enough": bool(math.isfinite(fhi) and fhi >= 3.0),
        "chatter_15_20hz_enough": False,
        "time_domain_xcorr_s": xc,
        "time_domain_xcorr_is_T0": False,
        "time_domain_FOPDT": {
            "T0_s": ticks_td * dt,
            "Tp_s": tp_td,
            "K": gain_td,
            "note": "Chirp xcorr locks T0 high.  Use the FRF fit.",
        },
        "collected_at": when,
    }
    data, visu = kind_dirs(prefix, preserve=csv_path)
    write_json(data / "gv.json", payload)
    _plot_03(freq, mag, phase, coh, t0=t0, tp=tp, gain=gain, f1=f1, visu=visu)
    write_readme(visu, _readme_03(payload, prefix=prefix))
    print(
        f"[GV] T0={1e3 * t0:.1f} ms  Tp={1e3 * tp:.1f} ms  K={gain:.3f}  "
        f"K_lf={k_lf:.3f}  valid {flo:.1f}–{fhi:.1f} Hz  "
        f"xcorr={1e3 * xc:.0f} ms is not T0  data={data}  visu={visu}",
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
    p.add_argument("--chirp-s", type=float, default=60.0)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    if not args.csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, then SERVO_TWIST chirp {args.amp_mm_s:.1f} mm/s  "
            f"{args.f0:.2f}–{args.f1:.1f} Hz  {args.chirp_s:.0f}s  force loop OFF",
            flush=True,
        )
    if dry_exit(args):
        return 0
    if args.csv:
        analyze(Path(args.csv), when=when, f1=args.f1, amp_mm_s=args.amp_mm_s)
        return 0
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("03_chirp")
    log = data / "chirp.csv"
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
        analyze(log, when=when, f1=args.f1, amp_mm_s=args.amp_mm_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
