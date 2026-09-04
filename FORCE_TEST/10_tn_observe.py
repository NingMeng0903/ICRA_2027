#!/usr/bin/env python3
"""Observe-only first term of PROB (10): r = T_n^{-1} V_m − V_i.

Samuel eq. (10):  P̂ert = Q[Tn^{-1} Vm − Vi] + N1 Fm − N2 Vm

What this script does
  Fit Γd + Tn from vel_ff → v_ach (same pair as 01/03/09).
  Predict Vm from Vi with that FOPDT (delay is never inverted).
  The residual Vm − Tn Γd Vi is the inner-loop model error / disturbance
  estimate.  That is the observation method for *inner dynamics*.

What this script does not do
  Does not turn CDYOB on.  Does not write pert into the command.
  Does not identify Q, N1, N2.  Those are designed filters:
    Q = ωQ/(s+ωQ)     properness + noise  (paper plots 15 Hz; not a plant)
    N1, N2            built from A, Pn, Cn, Rn  (outer law + payload)
  In air Fm≈0 so N1 Fm is empty; N2 needs a payload model Pn=1/(Mn s).
  Cn and Rn cannot be split on this black-box servo — the paper's own
  CDYOB equivalent uses Tn only: N1 ~ Q A (Tn^{-1}−1).

Optional --q-hz only low-passes the residual for display (makes Tn^{-1}
proper).  It is not an identified number.  Default off.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import fopdt_fit, load_pair, median_dt, mpl, save_fig
from io_csv import write_json
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme


def _simulate(u: np.ndarray, *, delay_ticks: int, tp_s: float, gain: float, dt: float) -> np.ndarray:
    add_playground()
    from peirastic.apps.identify_plant import _simulate_fopdt_k

    return _simulate_fopdt_k(
        u, delay_steps=max(int(delay_ticks), 0), tp_s=tp_s, gain=gain, dt=dt
    )


def _lpf(x: np.ndarray, dt: float, hz: float) -> np.ndarray:
    if hz <= 0.0:
        return x
    a = 1.0 - math.exp(-2.0 * math.pi * hz * dt)
    y = np.zeros_like(x)
    acc = 0.0
    for i, xi in enumerate(x):
        acc = acc + a * (float(xi) - acc)
        y[i] = acc
    return y


def analyze(
    csv_path: Path,
    *,
    when: str,
    q_hz: float = 0.0,
    t0_s: float | None = None,
    tp_s: float | None = None,
    gain: float | None = None,
) -> dict:
    t, u, y, _fb, dt_col, _rows = load_pair(csv_path)
    dt = median_dt(t, dt_col)
    ticks, tp_fit, k_fit = fopdt_fit(u, y, dt)
    t0 = float(t0_s) if t0_s is not None else ticks * dt
    tp = float(tp_s) if tp_s is not None else tp_fit
    k = float(gain) if gain is not None else k_fit
    delay = int(round(t0 / dt)) if dt > 0 else ticks
    pred = _simulate(u, delay_ticks=delay, tp_s=tp, gain=k, dt=dt)
    err = y - pred
    finite = err[np.isfinite(err)]
    rmse = float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")
    # First term of (10), delay not inverted: Tn^{-1} ≈ (1 + Tp d/dt)/K on Vm.
    dvm = np.concatenate(([0.0], np.diff(y) / dt))
    inv = (y + tp * dvm) / max(k, 1e-6)
    r_raw = inv - u
    r_show = _lpf(r_raw, dt, q_hz) if q_hz > 0.0 else r_raw
    payload = {
        "csv": str(csv_path),
        "paper": "Samuel 2024 eq. (10) first term only",
        "pair": "Vi=vel_ff, Vm=v_ach",
        "Gamma_d_ticks": delay,
        "T0_s": t0,
        "Tn": "K / (Tp s + 1) after Γd; Γd never inverted",
        "Tp_s": tp,
        "K": k,
        "rmse_vm_minus_pred_m_s": rmse,
        "q_hz_display_only": q_hz,
        "q_is_plant_parameter": False,
        "n1_n2_identified": False,
        "cdyob_applied": False,
        "note": (
            "Small RMSE in air ⇒ Tn explains the inner loop.  "
            "That is the V/Vr|Q=1=Tn check.  N1/N2 stay un-identified."
        ),
        "collected_at": when,
    }
    data, visu = kind_dirs("10_tn", preserve=csv_path)
    write_json(data / "tn_observe.json", payload)
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True)
    t0w = float(t[0]) if t.size else 0.0
    axes[0].plot(t - t0w, 1e3 * u, label="Vi")
    axes[0].plot(t - t0w, 1e3 * y, label="Vm")
    axes[0].plot(t - t0w, 1e3 * pred, label="Tn Γd Vi", lw=1.0)
    axes[1].plot(t - t0w, 1e3 * err, label="Vm − pred")
    if q_hz > 0.0:
        axes[1].plot(t - t0w, 1e3 * r_show, label=f"Q(Tn⁻¹Vm−Vi) Q={q_hz:.1f}Hz", alpha=0.8)
    axes[0].set_ylabel("mm/s")
    axes[1].set_ylabel("residual (mm/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_fig(fig, visu, "tn_residual.png")
    plt.close(fig)
    write_readme(
        visu,
        f"""# 10_tn — 只观察 PROB (10) 第一项

采集：`{when}` · 对已有 CSV 观察 · 不开 CDYOB

## 结论

- 模型 **T0 = {1e3 * t0:.1f} ms，Tp = {1e3 * tp:.1f} ms，K = {k:.3f}**。Γd = {delay} tick，不反演延迟。
- **Vm − Tn Γd Vi 的 RMSE = {1e3 * rmse:.2f} mm/s**。空气里小，说明 Tn 能解释内环。
- 没有辨识 Q / N1 / N2，也没有 apply。

数据：`DATA/10_tn/`
""",
    )
    print(
        f"[TN] T0={1e3 * t0:.1f} ms  Tp={1e3 * tp:.1f} ms  K={k:.3f}  "
        f"RMSE={1e3 * rmse:.2f} mm/s  data={data}",
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", required=True, help="03/09 SERVO_TWIST CSV (force off)")
    p.add_argument("--t0", type=float, default=None, help="override T0 [s]")
    p.add_argument("--tp", type=float, default=None, help="override Tp [s]")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--q-hz",
        type=float,
        default=0.0,
        help="display LPF on the DOB residual; not an identified parameter",
    )
    args = p.parse_args()
    if dry_exit(args):
        return 0
    analyze(
        Path(args.csv),
        when=stamp(),
        q_hz=float(args.q_hz),
        t0_s=args.t0,
        tp_s=args.tp,
        gain=args.k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
