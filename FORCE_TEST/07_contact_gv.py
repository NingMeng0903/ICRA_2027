#!/usr/bin/env python3
"""Same inner Gv, already in light contact.  Force loop OFF.

What: air then contact, both displacement-limited.  Air is two-sided
      (x = Ax sin φ, Ax ≈ 0.4 mm) at the mid-stroke TCP, which must
      already sit 10–25 mm above the pad.  Approach is two-stage and
      travel-capped: fast until a kiss, then slow to ~1.2 N.  Contact
      chirp is one-sided (x = Ax (1 − cos φ) ∈ [0, 2Ax]).
Why: a two-sided ±Ax chirp on a just-kissed slope unloads on the
     first negative half-cycle.  A 5 mm/s / 0.2 Hz velocity chirp
     walks ~4 mm.  This file asks whether 03's Tn still holds, and
     returns Ev = v_ach − Ĝ_air u.

Not a force Bode.  Hard first-touch is out of scope.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_log import ContactLogger
from frf_util import fopdt_from_frf, median_dt, welch_frf
from goto_mid import go_mid_from_args
from id_common import add_contact_args, load_aligned, load_kind_json, stash_window_a
from id_math import dump_jsonable, fopdt_predict, residual_tube
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
)

KIND = "07_contact_gv"


def _pair(cmd, ach, t, dt_col, mask):
    m = mask & np.isfinite(t) & np.isfinite(cmd) & np.isfinite(ach)
    if int(np.count_nonzero(m)) < 64:
        return None
    order = np.argsort(t[m], kind="stable")
    return t[m][order], cmd[m][order], ach[m][order], dt_col[m][order]


def _channel(name: str, t, u, y, dt_col) -> dict:
    dt = median_dt(t, dt_col)
    freq, mag, phase, coh = welch_frf(u, y, dt)
    t0, tp, gain = fopdt_from_frf(freq, mag, phase, coh, f_hi=8.0)
    yhat = fopdt_predict(u, dt, t0, tp, gain)
    err = y - yhat
    tube = residual_tube(err)
    return {
        "name": name,
        "T0_s": t0,
        "Tp_s": tp,
        "K": gain,
        "n": int(t.size),
        "dt_s": dt,
        "rmse": float(np.sqrt(np.mean(np.square(err[np.isfinite(err)]))))
        if np.isfinite(err).any()
        else float("nan"),
        "tube": tube,
        "freq": freq,
        "mag": mag,
        "phase": phase,
        "coh": coh,
        "u": u,
        "y": y,
        "yhat": yhat,
        "t": t,
    }


def analyze(csv_path: Path, *, when: str, window_a_csv: str = "", ax_mm: float = 0.4) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt_col = col(rows, "dt_actual_s")
    uz = twist_cmd6(rows)[:, 2]
    vz = twist_ach6(rows)[:, 2]
    dx = tool_z_displacement(pose6(rows))
    channels = []
    for phase, label in (
        ("air_chirp", "air"),
        ("contact_chirp", "contact"),
        ("contact_ms", "contact_ms"),
    ):
        pair = _pair(uz, vz, t, dt_col, phase_mask(rows, phase))
        if pair is None:
            channels.append({"name": label, "T0_s": float("nan"), "n": 0})
            continue
        channels.append(_channel(label, *pair))
    air = next((c for c in channels if c["name"] == "air"), {})
    contact = next((c for c in channels if c["name"] == "contact"), {})
    ev = {"n": 0, "p95": float("nan"), "max_abs": float("nan"), "bar": float("nan")}
    if air.get("n", 0) >= 64 and contact.get("n", 0) >= 64:
        yhat_c = fopdt_predict(
            contact["u"],
            float(contact.get("dt_s") or 0.005),
            float(air.get("T0_s", float("nan"))),
            float(air.get("Tp_s", float("nan"))),
            float(air.get("K", float("nan"))),
        )
        ev = residual_tube(contact["y"] - yhat_c)
    pose_amp = {}
    for phase, key in (("air_chirp", "air"), ("contact_chirp", "contact")):
        m = phase_mask(rows, phase)
        if np.any(m) and np.isfinite(dx[m]).sum() >= 4:
            xx = dx[m]
            xx = xx[np.isfinite(xx)]
            pose_amp[key] = {
                "peak_mm": 1e3 * float(np.max(np.abs(xx - xx[0]))),
                "p95_mm": 1e3 * float(np.percentile(np.abs(xx - np.median(xx)), 95)),
            }
        else:
            pose_amp[key] = {"peak_mm": float("nan"), "p95_mm": float("nan")}
    air_ref = load_kind_json("03_chirp", "gv.json") or {}
    payload = {
        "csv": str(csv_path),
        "align": {k: v for k, v in align.items() if k != "reason" or v},
        "ax_mm": ax_mm,
        "air": {k: air.get(k) for k in ("T0_s", "Tp_s", "K", "n", "rmse", "tube")},
        "contact": {k: contact.get(k) for k in ("T0_s", "Tp_s", "K", "n", "rmse", "tube")},
        "Ev_contact_vs_air_model": ev,
        "pose_amp_mm": pose_amp,
        "air_03_T0_s": air_ref.get("T0_s"),
        "collected_at": when,
        "what": "contact vs air Gv under bounded indentation, not a force loop",
    }
    data, visu = kind_dirs(KIND, preserve=csv_path)
    write_json(data / "gv.json", dump_jsonable(payload))
    _plot_07(channels, visu)
    t0_a = 1e3 * float(air.get("T0_s") or float("nan"))
    t0_c = 1e3 * float(contact.get("T0_s") or float("nan"))
    ax_c = pose_amp.get("contact", {}).get("peak_mm", float("nan"))
    write_readme(
        visu,
        f"""# 07_contact_gv — displacement-limited contact Gv

Take `{when}` · filter OFF · force loop OFF · Window A median gap {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms · clock resets {align.get('wa_clock_resets', 0)} (used latest SERVO slice)

## Verdict

- Air **T0 = {fmt_finite(t0_a, '.1f')} ms**, Tp = {fmt_finite(1e3 * float(air.get('Tp_s') or float('nan')), '.1f')} ms, K = {fmt_finite(float(air.get('K') or float('nan')), '.3f')}.
- Contact **T0 = {fmt_finite(t0_c, '.1f')} ms**, Tp = {fmt_finite(1e3 * float(contact.get('Tp_s') or float('nan')), '.1f')} ms, K = {fmt_finite(float(contact.get('K') or float('nan')), '.3f')}.
- Residual vs the air model \\(\\mathcal{{E}}_v\\): p95 = {fmt_finite(1e3 * float(ev.get('p95') or float('nan')), '.2f')} mm/s, max = {fmt_finite(1e3 * float(ev.get('max_abs') or float('nan')), '.2f')} mm/s. This set matters more than a single T0.
- Contact pose offset from the start of that chirp: {fmt_finite(ax_c, '.2f')} mm (commanded Ax = {ax_mm:.2f} mm; this is not peak-to-peak). If it is near 4 mm the excitation is still too large.
- 03 air T0 = {fmt_finite(1e3 * float(air_ref.get('T0_s') or float('nan')), '.1f')} ms. Do not read this as a force Bode.

## Figure

`gv_compare.png`: air / contact \\(|G_v|\\).
""",
    )
    print(
        f"[07] air T0={t0_a:.1f} ms  contact T0={t0_c:.1f} ms  "
        f"Ev_p95={1e3 * float(ev.get('p95') or float('nan')):.2f} mm/s  data={data}",
        flush=True,
    )
    return payload


def _plot_07(channels: list[dict], visu: Path) -> None:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    colors = {"air": ACH, "contact": MINUS, "contact_ms": CMD}
    for ch in channels:
        freq = ch.get("freq")
        if freq is None or not np.size(freq):
            continue
        f = freq
        use = (f >= 0.18) & (f <= 10.5)
        if not np.any(use):
            continue
        ax.semilogx(
            f[use],
            20.0 * np.log10(np.maximum(ch["mag"][use], 1e-6)),
            color=colors.get(ch["name"], ACH),
            lw=1.15,
            label=ch["name"],
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|G_v|$ (dB)")
    ax.set_xlim(0.18, 10.5)
    ax.grid(True, which="major", alpha=0.28)
    ax.legend(loc="lower left")
    save(fig, visu, "gv_compare.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_contact_args(p, abort_n=3.50, contact_n=1.20)
    p.add_argument("--ax-mm", type=float, default=0.40, help="air chirp amplitude")
    p.add_argument(
        "--contact-ax-mm",
        type=float,
        default=0.20,
        help="contact chirp amplitude; must stay well below seated indentation",
    )
    p.add_argument(
        "--seat-n",
        type=float,
        default=2.00,
        help="unused; old 2 N slam unloaded the slope — kept so old flags do not fail",
    )
    p.add_argument(
        "--seat-mm",
        type=float,
        default=0.80,
        help="unused displacement seat; kept so old flags do not fail",
    )
    p.add_argument("--seat-s", type=float, default=1.60)
    p.add_argument("--f0", type=float, default=0.3)
    p.add_argument("--f1", type=float, default=8.0)
    p.add_argument("--chirp-s", type=float, default=25.0)
    p.add_argument("--ms-s", type=float, default=12.0)
    p.add_argument("--skip-air", action="store_true")
    p.add_argument("--skip-ms", action="store_true", help="ignored; contact multisine is off unless --do-ms")
    p.add_argument(
        "--do-ms",
        action="store_true",
        help="optional two-sided contact multisine after the one-sided chirp",
    )
    p.add_argument("--unload-n", type=float, default=0.25)
    p.add_argument(
        "--kiss-n",
        type=float,
        default=0.40,
        help="stop the fast approach here, then creep to --contact-n",
    )
    p.add_argument("--seek-mm-s", type=float, default=8.0, help="fast tool-Z approach until --kiss-n")
    p.add_argument("--seek-s", type=float, default=5.0, help="max fast-approach time (travel cap is --seek-max-mm)")
    p.add_argument(
        "--seek-max-mm",
        type=float,
        default=25.0,
        help="refuse a long skate: mid-stroke TCP must already be this close to the pad",
    )
    p.add_argument("--slow-mm-s", type=float, default=2.0, help="creep speed from kiss to --contact-n")
    p.add_argument("--slow-s", type=float, default=4.0, help="max creep time after the kiss")
    args = p.parse_args()
    when = stamp()
    ax = args.ax_mm / 1000.0
    ax_c = args.contact_ax_mm / 1000.0
    print(
        f"[PLAN] MOVEJ mid (TCP must already be ≤{args.seek_max_mm:.0f} mm above the pad), "
        f"air disp-chirp Ax={args.ax_mm:.2f} mm "
        f"{args.f0:.1f}–{args.f1:.1f} Hz, then +Z {args.seek_mm_s:.1f} mm/s "
        f"≤{args.seek_max_mm:.0f} mm until F≈{args.kiss_n:.2f} N, creep "
        f"{args.slow_mm_s:.1f} mm/s to F≈{args.contact_n:.2f} N, "
        f"one-sided contact Ax={args.contact_ax_mm:.2f} mm  "
        f"secondary={args.secondary}  force loop OFF  abort F≥{args.abort_n:.1f} N  "
        "compare T0/Tp/K and Ev, not a force Bode",
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
                ax_mm=args.contact_ax_mm,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        return 0
    if not args.window_a_csv:
        print("[ERR] --window-a-csv is required so Ev uses achieved twist / pose", flush=True)
        return 2
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    data, _visu = kind_dirs(KIND)
    log = data / "contact_gv.csv"
    srv = ContactLogger(
        prefix=args.shm_prefix,
        hz=args.hz,
        log_csv=log,
        abort_n=args.abort_n,
        theta_axis=args.theta_axis,
        scan_axis=args.scan_axis,
        secondary=args.secondary,
    )
    try:
        srv.start_twist()
        if not args.skip_air:
            if not srv.chirp_disp_axis(2, ax, args.f0, args.f1, args.chirp_s, "air_chirp"):
                return 0 if srv.aborted else 130
            if not srv.hold(0.0, 0.3, "rest"):
                return 130
        if not srv.seek_contact(
            args.seek_mm_s / 1000.0,
            contact_n=args.kiss_n,
            seconds=args.seek_s,
            phase="seek",
            max_travel_m=args.seek_max_mm / 1000.0,
        ):
            return 2
        if math.isfinite(srv.last_fz) and srv.last_fz > args.contact_n:
            print(
                f"[KISS] Fz={srv.last_fz:.2f} N already ≥ {args.contact_n:.2f} N — skip creep",
                flush=True,
            )
        elif not srv.seek_contact(
            args.slow_mm_s / 1000.0,
            contact_n=args.contact_n,
            seconds=args.slow_s,
            phase="seat",
        ):
            return 2
        if not (math.isfinite(srv.last_fz) and srv.last_fz >= 0.80 * args.contact_n):
            print(
                f"[ERR] Fz={srv.last_fz:.2f} N after approach — probe slid off. "
                "Flatten the phantom or align tool-Z; not starting the contact chirp.",
                flush=True,
            )
            srv.retract_z(0.008, 2.5, 0.30)
            srv.tick(0.0, "done", check_abort=False)
            return 2
        print(
            f"[CONTACT-CHIRP] one-sided Ax={args.contact_ax_mm:.2f} mm  "
            f"x∈[0, {2.0 * args.contact_ax_mm:.2f}] mm from this pose  "
            f"F={srv.last_fz:.2f} N",
            flush=True,
        )
        if not srv.chirp_disp_axis(
            2,
            ax_c,
            args.f0,
            args.f1,
            args.chirp_s,
            "contact_chirp",
            min_n=args.unload_n,
            onesided=True,
        ):
            if srv.unloaded:
                print("[07] contact lost during chirp — still analyzing", flush=True)
            elif srv.aborted:
                pass
            else:
                return 130
        if args.do_ms and not srv.unloaded:
            from id_math import disp_multisine

            v_ms, _ = disp_multisine(
                srv.dt, args.ms_s, ax_c, (0.5, 1.1, 1.9, 3.1, 4.7), seed=7
            )
            seq = np.zeros((v_ms.size, 6), dtype=float)
            seq[:, 2] = v_ms
            if not srv.play(seq, "contact_ms"):
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
            analyze(log, when=when, window_a_csv=args.window_a_csv, ax_mm=args.contact_ax_mm)
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
