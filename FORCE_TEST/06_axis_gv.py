#!/usr/bin/env python3
"""Per-axis inner Gv (force loop OFF).  Ma 2015 macro/mini split.

What: the same chirp on tool X, Y, or Z.  Fit T0/Tp/K per axis.
Why: Ma 2015 mid-ranging needs the slow macro vs fast mini time constants.
     On this 8-DoF, tool-Z usually mixes rail (macro) + arm (mini); X/Y are
     mostly arm.  Different T0/Tp across axes is the measurable split.
     This is still the velocity servo, not a force law.

MotionBus only publishes achieved Z.  For X/Y pass Window A's --log-csv
and we use twist_requested_* / twist_achieved_*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import fopdt_from_frf, median_dt, mpl, welch_frf
from io_csv import col, load_rows, write_json
from goto_mid import add_movej_args, go_mid_from_args
from paper_fig import ACH, GUIDE, log_freq_ticks, save
import paths
from paths import add_playground, dry_exit, stamp, write_readme

AXIS = {"x": 0, "y": 1, "z": 2}
NAME = {0: "x", 1: "y", 2: "z"}


def axis_dirs(letter: str, *, preserve: Path | None = None) -> tuple[Path, Path]:
    """Use one immutable run per axis while keeping the fixed latest name."""
    return paths.kind_dirs(f"06_axis_{str(letter)}", preserve=preserve)


def _write_index(*, collect: bool = False) -> None:
    rows = []
    for letter in "xyz":
        js = paths.resolve_kind_dir(f"06_axis_{letter}") / "axis.json"
        if js.is_file():
            import json

            p = json.loads(js.read_text(encoding="utf-8"))
            t0 = 1e3 * float(p.get("T0_s") or float("nan"))
            tp = 1e3 * float(p.get("Tp_s") or float("nan"))
            k = float(p.get("K") or float("nan"))
            when = p.get("collected_at") or ""
            rows.append(f"| {letter.upper()} | `{when}` | {t0:.1f} ms | {tp:.1f} ms | {k:.3f} |")
        else:
            rows.append(f"| {letter.upper()} | 还没采 | | | |")
    write_readme(
        paths.run_visu_dir("06_axis", collect=collect),
        f"""# 06_axis — 分轴 Gv

图和分析在平行子目录：`x/`、`y/`、`z/`。不要扫姿态，也不要扫斜向混合。

| 轴 | 采集 | T0 | Tp | K |
|---|---|---|---|---|
{chr(10).join(rows)}

Z 混轨道+臂，X/Y 多半是臂。不要写 yaml。
""",
    )


def _finite_pair(t, u, y, dt_col):
    mask = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 8:
        raise ValueError("no finite command/achieved velocity pair")
    order = np.argsort(t[mask], kind="stable")
    return t[mask][order], u[mask][order], y[mask][order], dt_col[mask][order]


def _pair_from_window_a(path: Path, axis: int):
    rows = load_rows(path)
    servo = [r for r in rows if str(r.get("phase") or "") == "servo_twist"]
    use = servo if len(servo) > 64 else rows
    letter = "xyz"[axis]
    return _finite_pair(
        col(use, "t_wall_s"),
        col(use, f"twist_requested_v{letter}", f"twist_v{letter}", f"vel_ff_v{letter}"),
        col(use, f"twist_achieved_v{letter}"),
        col(use, "dt_actual_s"),
    )


def _pair_from_servo(path: Path):
    rows = load_rows(path)
    chirp = [r for r in rows if str(r.get("phase") or "").startswith("chirp")]
    use = chirp if len(chirp) > 64 else rows
    return _finite_pair(
        col(use, "t_wall_s", "t_mono_s"),
        col(use, "vel_ff", "vel_ff_vz"),
        col(use, "v_ach", "vz_achieved_tool"),
        col(use, "dt_actual_s"),
    )


def _plot_06(freq, mag, *, f1: float, visu: Path, name: str) -> None:
    """Magnitude only to a bit past 8 Hz. Bins after the chirp are not Gv."""

    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.50, 2.20), constrained_layout=True)
    cut = 8.0
    f_draw = min(float(f1) + 0.4, 10.0)
    use = freq <= f_draw + 1e-12 if freq.size else np.zeros(0, dtype=bool)
    if np.any(use):
        f = freq[use]
        db = 20.0 * np.log10(np.maximum(mag[use], 1e-6))
        ax.semilogx(f, db, color=ACH, lw=1.15)
        band = f <= cut + 1e-12
        shown = db[band] if np.any(band) else db
        ax.set_ylim(float(np.min(shown)) - 3.0, float(np.max(shown)) + 3.0)
    ax.set_ylabel(r"$|G_v|$ (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0.18, 10.5)
    ax.grid(True, which="major", alpha=0.28)
    log_freq_ticks(ax, f_lo=0.18, f_hi=10.5, cutoff_hz=cut)
    ax.text(cut, ax.get_ylim()[1], "8 Hz cut-off", color=GUIDE, fontsize=8, ha="center", va="bottom")
    save(fig, visu, name)
    plt.close(fig)


def analyze(
    csv_path: Path,
    *,
    when: str,
    axis: int,
    window_a: Path | None,
    f1: float,
    collect: bool = False,
) -> dict:
    src_path = Path(window_a) if window_a is not None else Path(csv_path)
    from_03 = "03_chirp" in str(src_path)
    if window_a is not None:
        t, u, y, dt_col = _pair_from_window_a(window_a, axis)
        pair = f"twist_requested_v{NAME[axis]} → twist_achieved_v{NAME[axis]}"
    else:
        t, u, y, dt_col = _pair_from_servo(csv_path)
        pair = "vel_ff → v_ach"
    dt = median_dt(t, dt_col)
    freq, mag, phase, coh = welch_frf(u, y, dt)
    t0, tp, gain = fopdt_from_frf(freq, mag, phase, coh, f_hi=min(8.0, f1))
    ticks = int(round(t0 / dt)) if dt > 0 and np.isfinite(t0) else None
    payload = {
        "csv": str(src_path),
        "axis": NAME[axis],
        "pair": pair,
        "T0_s": t0,
        "Tp_s": tp,
        "K": gain,
        "Gamma_d_ticks": ticks,
        "from_03_chirp": from_03,
        "collected_at": when,
    }
    data, visu = axis_dirs(NAME[axis], preserve=csv_path)
    write_json(data / "axis.json", payload)
    _plot_06(freq, mag, f1=f1, visu=visu, name=f"gv_{NAME[axis]}.png")
    src_note = "同一趟 03 chirp，不是新采" if from_03 else "本轴新采"
    write_readme(
        visu,
        f"""# 06_axis / {NAME[axis]} — 分轴 Gv

采集：`{when}` · filter OFF · 力环关 · 轴 {NAME[axis]} · {src_note} · `{pair}`

## 结论

- **T0 = {1e3 * float(t0):.1f} ms，Tp = {1e3 * float(tp):.1f} ms，K = {float(gain):.3f}**（FRF，γ²≥0.6，≤8 Hz）。
- 图只画到约 10 Hz。8 Hz 以后不是植物。
- 不要写 yaml。数据：`DATA/06_axis_{NAME[axis]}/`
""",
    )
    _write_index(collect=collect)
    print(
        f"[AXIS {NAME[axis]}] T0={1e3 * float(t0):.1f} ms  Tp={1e3 * float(tp):.1f} ms  "
        f"K={float(gain):.3f}  data={data}",
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="")
    p.add_argument("--window-a-csv", default="")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--amp-mm-s", type=float, default=15.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=10.0)
    p.add_argument("--chirp-s", type=float, default=45.0)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    axis = AXIS[args.axis]
    if not args.csv and not args.window_a_csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, then SERVO_TWIST {args.axis}-chirp "
            f"{args.amp_mm_s:.1f} mm/s  {args.f0:.2f}–{args.f1:.1f} Hz  force loop OFF",
            flush=True,
        )
        if args.axis != "z":
            print(
                "[PLAN] X/Y achieved velocity is not on MotionBus — "
                "start Window A with --log-csv and pass it to --window-a-csv after the run",
                flush=True,
            )
    if dry_exit(args):
        return 0
    if args.csv or args.window_a_csv:
        analyze(
            Path(args.csv) if args.csv else Path(args.window_a_csv),
            when=when,
            axis=axis,
            window_a=Path(args.window_a_csv) if args.window_a_csv else None,
            f1=args.f1,
            collect=False,
        )
        return 0
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    from servo_log import ServoLogger

    data, _visu = axis_dirs(args.axis)
    log = data / "axis.csv"
    srv = ServoLogger(prefix=args.shm_prefix, hz=args.hz, log_csv=log, axis=axis)
    try:
        srv.start_twist()
        if not srv.hold(0.0, 0.4, "rest"):
            return 130
        if not srv.chirp(args.amp_mm_s / 1000.0, args.f0, args.f1, args.chirp_s, f"chirp_{args.axis}"):
            return 130
        srv.tick(0.0, "done")
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
    finally:
        srv.close()
    wa = Path(args.window_a_csv) if args.window_a_csv else None
    if axis != 2 and wa is None:
        print(
            "[ERR] X/Y achieved velocity is not on MotionBus. "
            "This run drove the chirp; re-analyze with --window-a-csv "
            "from Window A --log-csv. Do not use axis.csv for X/Y.",
            flush=True,
        )
        return 2
    if log.is_file() and srv.n_rows > 64:
        analyze(log, when=when, axis=axis, window_a=wa, f1=args.f1, collect=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
