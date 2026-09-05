#!/usr/bin/env python3
"""Open-loop environment stiffness Ke.  Force loop OFF.

What: seek, then a constant +press velocity.  Ke ≈ ΔF / Δx_toolz.
Why: later force work needs a contact stiffness number.  This is the
     pad/tissue, not an inner-loop parameter.  Do not start hybrid.

Δx is the Window A TCP pose projected on the first-press tool-Z axis.
Command-velocity integration is not Ke — it folds the 28–40 ms delay
into the stiffness.

The paper number is not a single secant.  Analysis returns the local
Ke(F) envelope on loading and unloading inside the work band
F ∈ [f_lo, f_hi] (default 2–5 N, aligned with --target-n), and
ᾱKe = max |Ke_local|.  work_band_reached only if the raw loading
Fz spans [f_lo, f_hi] to within ε_F and in-band local Ke exists.
A single sample inside the band is not coverage.  --f-hi must not
exceed --target-n.

After the press: controlled unload at the same speed, then retract
off the pad and MOVEJ mid-stroke.  DATA/08_ke/ keeps only the latest
take.  Each row is appended to DATA/ke_sites.csv and
DATA/ke_envelope.csv (wipe does not touch those).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goto_mid import add_movej_args, go_mid, go_mid_from_args
from id_common import load_aligned, stash_window_a
from id_math import dump_jsonable, local_stiffness, stiffness_envelope, work_band_spanned
from io_csv import col, write_json
from paper_fig import ACH, MINUS, mpl, panel_tag, save
from paths import DATA, add_playground, dry_exit, kind_dirs, stamp, write_readme
from window_a import AlignmentError, add_window_a_arg, fmt_finite, phase_mask, tool_z_displacement

SITES_CSV = DATA / "ke_sites.csv"
ENVELOPE_CSV = DATA / "ke_envelope.csv"
SITE_FIELDS = (
    "collected_at",
    "site",
    "ke_n_m",
    "fz0_n",
    "fz1_n",
    "dF_n",
    "dx_mm",
    "dx_source",
    "press_mm_s",
    "press_s",
    "aborted",
    "align_gap_ms",
)
ENV_FIELDS = SITE_FIELDS + (
    "ke_bar_n_m",
    "ke_min_n_m",
    "ke_load_bar_n_m",
    "ke_unload_bar_n_m",
    "f_lo_n",
    "f_hi_n",
    "work_band_reached",
    "f_valid_min",
    "f_valid_max",
    "n_local",
)
TAU_L_S = 0.040
V_E_EXAMPLE_M_S = 0.010


def append_site_row(row: dict) -> Path:
    """Append one Ke/site row.  Lives outside DATA/08_ke/ so wipe keeps it."""
    DATA.mkdir(parents=True, exist_ok=True)
    fresh = not SITES_CSV.is_file()
    with SITES_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SITE_FIELDS))
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SITE_FIELDS})
    print(f"[SITES] {row.get('site') or '(no --site)'}  Ke={row.get('ke_n_m')}  {SITES_CSV}", flush=True)
    return SITES_CSV


def append_envelope_row(row: dict) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    fresh = not ENVELOPE_CSV.is_file()
    with ENVELOPE_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ENV_FIELDS))
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in ENV_FIELDS})
    return ENVELOPE_CSV


def _press_slice(rows, fz: np.ndarray) -> np.ndarray:
    press = phase_mask(rows, "press")
    if int(np.count_nonzero(press)) >= 4:
        return press
    return np.isfinite(fz) & (fz > 0.4)


def _secant(x: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    ok = np.isfinite(x) & np.isfinite(f)
    if int(np.count_nonzero(ok)) < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    fz0 = float(f[ok][0])
    fz1 = float(f[ok][-1])
    dxp = float(x[ok][-1] - x[ok][0])
    ke = float((fz1 - fz0) / dxp) if abs(dxp) > 1e-6 else float("nan")
    return fz0, fz1, dxp, ke


def _limb_envelope(
    x: np.ndarray,
    f: np.ndarray,
    mask: np.ndarray,
    f_lo: float,
    f_hi: float,
    *,
    eps_f: float = 0.15,
) -> dict:
    if int(np.count_nonzero(mask)) < 8:
        empty = stiffness_envelope(np.array([]), np.array([]), f_lo, f_hi, eps_f=eps_f)
        empty["secant"] = float("nan")
        empty["f_limb_min"] = float("nan")
        empty["f_limb_max"] = float("nan")
        return empty
    fm, ke = local_stiffness(x[mask], f[mask])
    env = stiffness_envelope(fm, ke, f_lo, f_hi, eps_f=eps_f)
    span = work_band_spanned(f[mask], f_lo, f_hi, eps_f=eps_f)
    env["f_limb_min"] = span["f_min"]
    env["f_limb_max"] = span["f_max"]
    # Coverage is the force that was actually applied, not a window-mean
    # that sits inside the peak.  Still require in-band local Ke samples.
    env["work_band_reached"] = bool(span["work_band_reached"] and int(env.get("n") or 0) >= 4)
    _fz0, _fz1, _dx, sec = _secant(x[mask], f[mask])
    env["secant"] = sec
    env["f_center"] = fm
    env["ke_local"] = ke
    return env


def analyze(
    csv_path: Path,
    *,
    when: str,
    site: str = "",
    press_mm_s: float | None = None,
    press_s: float | None = None,
    aborted: bool = False,
    record_site: bool = True,
    window_a_csv: str = "",
    f_lo: float = 2.0,
    f_hi: float = 5.0,
    eps_f: float = 0.15,
) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    fz = col(rows, "fz", "fz_raw_comp")
    pose_dx = tool_z_displacement(
        np.column_stack(
            [
                col(rows, "pose_x", "pose_meas_x"),
                col(rows, "pose_y", "pose_meas_y"),
                col(rows, "pose_z", "pose_meas_z"),
                col(rows, "pose_rx", "pose_meas_rx"),
                col(rows, "pose_ry", "pose_meas_ry"),
                col(rows, "pose_rz", "pose_meas_rz"),
            ]
        )
    )
    press = _press_slice(rows, fz)
    unload = phase_mask(rows, "unload")
    fz0, fz1, dxp, ke = _secant(pose_dx[press], fz[press]) if np.any(press) else (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )
    load_env = _limb_envelope(pose_dx, fz, press, f_lo, f_hi, eps_f=eps_f)
    unload_env = _limb_envelope(pose_dx, fz, unload, f_lo, f_hi, eps_f=eps_f)
    bars = [
        v
        for v in (load_env.get("ke_bar"), unload_env.get("ke_bar"))
        if math.isfinite(float(v or float("nan")))
    ]
    ke_bar = float(max(bars)) if bars else float("nan")
    mins = [
        v
        for v in (load_env.get("ke_min"), unload_env.get("ke_min"))
        if math.isfinite(float(v or float("nan")))
    ]
    ke_min = float(min(mins)) if mins else float("nan")
    band_ok = bool(load_env.get("work_band_reached"))
    bound_n = (
        2.0 * float(ke_bar) * V_E_EXAMPLE_M_S * TAU_L_S
        if math.isfinite(ke_bar)
        else float("nan")
    )
    payload = {
        "csv": str(csv_path),
        "window_a_csv": align.get("window_a_csv"),
        "site": site,
        "ke_n_m": ke,
        "ke_secant_n_m": ke,
        "ke_bar_n_m": ke_bar,
        "ke_min_n_m": ke_min,
        "ke_load": {
            k: load_env.get(k)
            for k in (
                "n",
                "ke_min",
                "ke_max",
                "ke_bar",
                "ke_median",
                "work_band_reached",
                "secant",
                "f_valid_min",
                "f_valid_max",
                "f_limb_min",
                "f_limb_max",
            )
        },
        "ke_unload": {
            k: unload_env.get(k)
            for k in (
                "n",
                "ke_min",
                "ke_max",
                "ke_bar",
                "ke_median",
                "work_band_reached",
                "secant",
                "f_valid_min",
                "f_valid_max",
                "f_limb_min",
                "f_limb_max",
            )
        },
        "f_lo_n": f_lo,
        "f_hi_n": f_hi,
        "eps_f_n": eps_f,
        "work_band_reached": band_ok,
        "fz0_n": fz0,
        "fz1_n": fz1,
        "dF_n": fz1 - fz0 if math.isfinite(fz0) and math.isfinite(fz1) else float("nan"),
        "dx_mm": 1e3 * dxp if math.isfinite(dxp) else float("nan"),
        "dx_source": "pose_tool_z",
        "press_mm_s": press_mm_s,
        "press_s": press_s,
        "aborted": bool(aborted),
        "align": align,
        "example_delay_bound_n": bound_n,
        "example_bound_note": "2 * Ke_bar * 10 mm/s * 40 ms; uses worst-case local Ke, not the secant",
        "what": "stiffness envelope from actual pose; secant is not the paper number",
        "collected_at": when,
        "sites_csv": str(SITES_CSV),
        "envelope_csv": str(ENVELOPE_CSV),
    }
    data, visu = kind_dirs("08_ke", preserve=csv_path)
    write_json(data / "ke.json", dump_jsonable(payload))
    _plot_08(t, fz, pose_dx, press, unload, load_env, f_lo, f_hi, visu)
    site_txt = site if site else "（未标部位，用 --site）"
    hard = math.isfinite(ke_bar) and ke_bar >= 500.0
    band_note = (
        f"加载段覆盖 [{fmt_finite(load_env.get('f_valid_min'), '.2f')},"
        f"{fmt_finite(load_env.get('f_valid_max'), '.2f')}] N，"
        f"声称工作带 [{f_lo:.1f},{f_hi:.1f}] N：ᾱKe = {fmt_finite(ke_bar, '.1f')} N/m，"
        f"下缘 {fmt_finite(ke_min, '.1f')} N/m。"
        if band_ok
        else (
            f"加载段只覆盖 [{fmt_finite(load_env.get('f_valid_min'), '.2f')},"
            f"{fmt_finite(load_env.get('f_valid_max'), '.2f')}] N，"
            f"没有跨满 [{f_lo:.1f},{f_hi:.1f}] N。这一拍不能写成工作带 ᾱKe。"
        )
    )
    write_readme(
        visu,
        f"""# 08_ke — 开环环境刚度包络

采集：`{when}` · filter OFF · 力环关 · 开环下压/卸载 · 部位 **{site_txt}**

## 结论

- 单次 secant Ke ≈ {fmt_finite(ke, '.1f')} N/m（ΔF / 实际 tool-Z，press 段）。**论文用的是包络，不是这个数。**
- {band_note}
- 加载 ᾱKe = {fmt_finite(load_env.get('ke_bar'), '.1f')}，卸载 ᾱKe = {fmt_finite(unload_env.get('ke_bar'), '.1f')} N/m。
- 位移来源是 Window A 位姿，**不是**命令积分。对齐中位间隙 {fmt_finite(align.get('gap_median_ms', float('nan')), '.1f')} ms。
- 量纲示例 2 ᾱKe ve τℓ（ve=10 mm/s，τℓ=40 ms）≈ {fmt_finite(bound_n, '.2f')} N。{'硬垫，C1 下界有机会紧。' if hard else '若这是软垫，主实验必须另采硬垫才谈不可实现。'}
- 矩阵行追加在 `{ENVELOPE_CSV.name}`，wipe 不删。同一部位至少再换 2–3 个速度。

不要开混合。不要写进 yaml，除非明确要写。
""",
    )
    site_row = {
        "collected_at": when,
        "site": site,
        "ke_n_m": f"{ke:.3f}" if math.isfinite(ke) else "",
        "fz0_n": f"{fz0:.3f}" if math.isfinite(fz0) else "",
        "fz1_n": f"{fz1:.3f}" if math.isfinite(fz1) else "",
        "dF_n": f"{fz1 - fz0:.3f}" if math.isfinite(fz0) and math.isfinite(fz1) else "",
        "dx_mm": f"{1e3 * dxp:.3f}" if math.isfinite(dxp) else "",
        "dx_source": "pose_tool_z",
        "press_mm_s": "" if press_mm_s is None else f"{press_mm_s:.3f}",
        "press_s": "" if press_s is None else f"{press_s:.3f}",
        "aborted": "1" if aborted else "0",
        "align_gap_ms": f"{float(align.get('gap_median_ms', float('nan'))):.2f}",
        "ke_bar_n_m": f"{ke_bar:.3f}" if math.isfinite(ke_bar) else "",
        "ke_min_n_m": f"{ke_min:.3f}" if math.isfinite(ke_min) else "",
        "ke_load_bar_n_m": f"{float(load_env.get('ke_bar') or float('nan')):.3f}"
        if math.isfinite(float(load_env.get("ke_bar") or float("nan")))
        else "",
        "ke_unload_bar_n_m": f"{float(unload_env.get('ke_bar') or float('nan')):.3f}"
        if math.isfinite(float(unload_env.get("ke_bar") or float("nan")))
        else "",
        "f_lo_n": f"{f_lo:.2f}",
        "f_hi_n": f"{f_hi:.2f}",
        "work_band_reached": "1" if band_ok else "0",
        "f_valid_min": f"{float(load_env.get('f_valid_min') or float('nan')):.3f}"
        if math.isfinite(float(load_env.get("f_valid_min") or float("nan")))
        else "",
        "f_valid_max": f"{float(load_env.get('f_valid_max') or float('nan')):.3f}"
        if math.isfinite(float(load_env.get("f_valid_max") or float("nan")))
        else "",
        "n_local": str(int(load_env.get("n") or 0) + int(unload_env.get("n") or 0)),
    }
    if record_site:
        append_site_row(site_row)
        append_envelope_row(site_row)
    print(
        f"[KE] secant={ke:.1f}  bar={ke_bar:.1f} N/m  band={band_ok}  "
        f"site={site_txt}  data={data}",
        flush=True,
    )
    return payload


def _plot_08(t, fz, pose_dx, press, unload, load_env, f_lo, f_hi, visu: Path) -> None:
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(3.50, 3.80), sharex=True, constrained_layout=True)
    t0 = float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else 0.0
    tt = t - t0
    axes[0].plot(tt, fz, color=ACH, lw=1.15)
    axes[1].plot(tt, 1e3 * pose_dx, color=MINUS, lw=1.15)
    if np.any(press):
        axes[0].axvspan(float(tt[press][0]), float(tt[press][-1]), color=ACH, alpha=0.08)
        axes[1].axvspan(float(tt[press][0]), float(tt[press][-1]), color=MINUS, alpha=0.08)
    if np.any(unload):
        axes[0].axvspan(float(tt[unload][0]), float(tt[unload][-1]), color=MINUS, alpha=0.08)
    axes[0].set_ylabel("Fz (N)")
    axes[1].set_ylabel("tool-Z travel (mm)")
    axes[1].set_xlabel("time (s)")
    for letter, ax in zip("ab", axes):
        ax.grid(True, alpha=0.28)
        panel_tag(ax, letter)
    save(fig, visu, "press.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.50, 2.40), constrained_layout=True)
    fm = load_env.get("f_center")
    ke_l = load_env.get("ke_local")
    if fm is not None and ke_l is not None:
        ok = np.isfinite(fm) & np.isfinite(ke_l)
        if np.any(ok):
            ax.plot(fm[ok], ke_l[ok], color=ACH, lw=1.0)
    ax.axvspan(f_lo, f_hi, color=MINUS, alpha=0.08)
    ax.set_xlabel("Fz (N)")
    ax.set_ylabel("local Ke (N/m)")
    ax.grid(True, alpha=0.28)
    save(fig, visu, "ke_envelope.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="", help="analyze an existing press CSV")
    p.add_argument("--site", default="", help="body site or pad name; appended to DATA/ke_sites.csv")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--seek-mm-s", type=float, default=10.0)
    p.add_argument("--press-mm-s", type=float, default=3.0)
    p.add_argument("--press-s", type=float, default=5.0)
    p.add_argument("--target-n", type=float, default=5.0, help="stop press at this F if reached first")
    p.add_argument("--f-lo", type=float, default=2.0)
    p.add_argument("--f-hi", type=float, default=5.0, help="must match --target-n; coverage requires the press to span this band")
    p.add_argument("--eps-f", type=float, default=0.15)
    p.add_argument("--unload-s", type=float, default=4.0)
    p.add_argument("--retract-mm-s", type=float, default=8.0)
    p.add_argument("--retract-s", type=float, default=2.5)
    p.add_argument("--contact-n", type=float, default=0.40)
    p.add_argument("--abort-n", type=float, default=6.50)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument(
        "--skip-return-movej",
        action="store_true",
        help="do not MOVEJ mid-stroke after the press",
    )
    p.add_argument("--dry-run", action="store_true")
    add_window_a_arg(p)
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    site = str(args.site).strip()
    if not args.csv:
        print(
            f"[PLAN] MOVEJ mid-stroke, seek + press {args.press_mm_s:.1f} mm/s "
            f"until F≈{args.target_n:.1f} N or {args.press_s:.1f}s, same-speed unload, "
            f"retract −Z, MOVEJ mid  site={site or '(tell me after)'}  "
            f"band [{args.f_lo:.1f},{args.f_hi:.1f}] N  abort F≥{args.abort_n:.1f} N  "
            f"repeat each site at 1.5, 3, 6 mm/s  "
            f"force loop OFF  Δx=Window A pose  log→{ENVELOPE_CSV.name}",
            flush=True,
        )
        print("[PLAN] Window A must be started with --log-csv; pass --window-a-csv", flush=True)
    if args.f_hi > args.target_n + args.eps_f and not args.csv:
        print(
            f"[ERR] work band [{args.f_lo:.1f},{args.f_hi:.1f}] N cannot be covered "
            f"when --target-n={args.target_n:.1f} N (need target ≥ f-hi − eps-f). "
            "Either lower --f-hi to 5 or raise --target-n (and abort).",
            flush=True,
        )
        return 2
    if dry_exit(args):
        return 0
    if args.csv:
        try:
            analyze(
                Path(args.csv),
                when=when,
                site=site,
                press_mm_s=args.press_mm_s,
                press_s=args.press_s,
                window_a_csv=args.window_a_csv,
                f_lo=args.f_lo,
                f_hi=args.f_hi,
                eps_f=args.eps_f,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        return 0
    if not args.window_a_csv:
        print("[ERR] --window-a-csv is required so Ke uses actual pose", flush=True)
        return 2
    rc_m = go_mid_from_args(args)
    if rc_m:
        return rc_m
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("08_ke")
    log = data / "press.csv"
    srv = ServoLogger(
        prefix=args.shm_prefix, hz=args.hz, log_csv=log, abort_n=args.abort_n
    )
    rc = 0
    latched = False
    estop = False
    try:
        srv.start_twist()
        t_seek = time.monotonic()
        while time.monotonic() - t_seek < 12.0:
            if not srv.tick(args.seek_mm_s / 1000.0, "seek"):
                estop = not srv.aborted
                rc = 130
                break
            if math.isfinite(srv.last_fz) and srv.last_fz > args.contact_n:
                latched = True
                print(f"[CONTACT] Fz={srv.last_fz:.2f} N — press", flush=True)
                break
            time.sleep(srv.dt)
        if rc == 0 and not latched:
            print("[ERR] no contact", flush=True)
            rc = 2
        if latched and rc == 0:
            t_press = time.monotonic()
            while time.monotonic() - t_press < args.press_s:
                if not srv.tick(args.press_mm_s / 1000.0, "press"):
                    estop = not srv.aborted
                    rc = 0 if srv.aborted else 130
                    break
                if math.isfinite(srv.last_fz) and srv.last_fz >= args.target_n:
                    print(f"[PRESS] Fz={srv.last_fz:.2f} N ≥ {args.target_n:.1f} N", flush=True)
                    break
                time.sleep(srv.dt)
        if latched and not estop and rc == 0:
            t_un = time.monotonic()
            while time.monotonic() - t_un < args.unload_s:
                if not srv.tick(-abs(args.press_mm_s) / 1000.0, "unload", check_abort=False):
                    estop = True
                    rc = 130
                    break
                if math.isfinite(srv.last_fz) and srv.last_fz <= args.f_lo:
                    print(f"[UNLOAD] Fz={srv.last_fz:.2f} N ≤ {args.f_lo:.1f} N", flush=True)
                    break
                time.sleep(srv.dt)
        if latched and not estop:
            srv.retract_z(
                args.retract_mm_s / 1000.0,
                args.retract_s,
                args.contact_n,
            )
        srv.tick(0.0, "done", check_abort=False)
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
        rc = 130
    finally:
        srv.close()
    if not args.skip_return_movej and not estop:
        print("[MOVEJ] return mid-stroke after press", flush=True)
        rc_back = go_mid(
            prefix=str(args.shm_prefix),
            skip=False,
            v=float(args.movej_v),
            settle_s=0.0,
        )
        if rc_back and rc == 0:
            rc = rc_back
    elif estop:
        print("[MOVEJ] skipped (ESTOP)", flush=True)
    try:
        stash_window_a(args.window_a_csv, data)
    except AlignmentError as exc:
        print(f"[ERR] {exc}", flush=True)
        return 2
    if log.is_file() and srv.n_rows > 16:
        try:
            analyze(
                log,
                when=when,
                site=site,
                press_mm_s=args.press_mm_s,
                press_s=args.press_s,
                aborted=bool(srv.aborted),
                window_a_csv=args.window_a_csv,
                f_lo=args.f_lo,
                f_hi=args.f_hi,
                eps_f=args.eps_f,
            )
        except AlignmentError as exc:
            print(f"[ERR] {exc}", flush=True)
            return 2
        if not site:
            print("[SITES] no --site; tell me the site and I will fill that row", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
