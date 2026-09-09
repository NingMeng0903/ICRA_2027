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
import json
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
    shared_window_a_fields,
    transverse_error_report,
    tool_z_displacement,
    twist_ach6,
    twist_external_cmd6,
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


def _after_phase_mask(rows: list[dict], phase: str) -> np.ndarray:
    """Rows after a phase, used only as a measured tail check."""
    mask = phase_mask(rows, phase)
    out = np.zeros(mask.size, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size and int(idx[-1]) + 1 < out.size:
        out[int(idx[-1]) + 1 :] = True
    return out


def _run_meta(rows: list[dict]) -> dict:
    out = {}
    for key in (
        "run_id",
        "dof_requested",
        "dof_active",
        "dof_config_source",
        "controller_api_version",
        "mode_name",
        "mode_request_seq",
        "mode_install_seq",
        "mode_install_status",
    ):
        vals = [str(row.get(key) or "").strip() for row in rows]
        vals = [v for v in vals if v]
        out[key] = vals[0] if vals else None
    return out


def _command_provenance(rows: list[dict]) -> tuple[str, str]:
    """Return the frame/source of the external command after alignment."""
    frames = {
        str(row.get("command_frame") or "").strip().lower()
        for row in rows
        if str(row.get("command_frame") or "").strip()
    }
    sources = {
        str(row.get("command_source") or "").strip()
        for row in rows
        if str(row.get("command_source") or "").strip()
    }
    if not frames:
        frame = "tool"
    elif len(frames) == 1:
        frame = sorted(frames)[0]
    else:
        frame = "mixed"
    source = sorted(sources)[0] if len(sources) == 1 else "contact_log:v_cmd_*"
    return frame, source


def _nan_numeric_tree(value) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if key == "n":
            continue
        if isinstance(item, dict):
            _nan_numeric_tree(item)
        elif isinstance(item, (int, float, np.integer, np.floating)):
            value[key] = float("nan")


def _tail_report(
    pose: np.ndarray,
    twist: np.ndarray,
    mask: np.ndarray,
    *,
    t: np.ndarray | None = None,
    command: np.ndarray | None = None,
    command_frame: str = "tool",
) -> dict:
    report = transverse_error_report(
        pose,
        twist,
        mask,
        t=t,
        command=command,
        command_frame=command_frame,
    )
    if int(report.get("n", 0)) < 4:
        _nan_numeric_tree(report)
    return report


def _stats(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
        }
    absolute = np.abs(finite)
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _numeric_field(rows: list[dict], names: tuple[str, ...]) -> tuple[np.ndarray, str | None]:
    """Read the first present scalar column; old CSVs become all-NaN."""
    for name in names:
        if any(row.get(name) not in (None, "") for row in rows):
            return col(rows, name), name
    return np.full(len(rows), np.nan, dtype=float), None


def _json_component(rows: list[dict], name: str, index: int) -> tuple[np.ndarray, str | None]:
    out = np.full(len(rows), np.nan, dtype=float)
    present = False
    for i, row in enumerate(rows):
        raw = row.get(name)
        if raw in (None, ""):
            continue
        present = True
        try:
            value = json.loads(str(raw))
            if len(value) > int(index):
                out[i] = float(value[int(index)])
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            continue
    return out, (f"{name}[{int(index)}]" if present else None)


def _qp_attribution(rows: list[dict], mask: np.ndarray) -> dict:
    """Compare model-side Y attribution with measured Window-A ``vy``.

    The JSON QPIK columns are solver/model quantities.  A missing legacy
    scalar column remains NaN; no physical TCP value is synthesized from a
    model field.  ``slack_norm`` is retained separately because it mixes
    task units and is not a Y-velocity bound.
    """
    actual, actual_source = _numeric_field(rows, ("twist_achieved_vy",))
    requested, requested_source = _numeric_field(
        rows, ("v_cmd_vy", "twist_requested_vy", "twist_vy")
    )
    task, task_source = _numeric_field(rows, ("task_residual_y", "qpik_task_residual_y"))
    rail, rail_source = _numeric_field(rows, ("rail_exec_contrib_y", "qpik_rail_contrib_y"))
    arm, arm_source = _numeric_field(rows, ("arm_contrib_y", "qpik_arm_contrib_y"))
    # Keep the requested scalar names NaN when an old CSV lacks them.  The
    # newer JSON fields are reported separately and may still support an
    # explicitly labelled model-vs-measurement comparison.
    model_task, model_task_source = _json_component(rows, "qpik_protected_residual_json", 1)
    model_rail, model_rail_source = _json_component(rows, "qpik_rail_xy_contribution_json", 1)
    model_arm, model_arm_source = _json_component(rows, "qpik_arm_xy_contribution_json", 1)
    last_slack, last_slack_source = _numeric_field(rows, ("last_slack",))
    slack_norm, slack_source = _numeric_field(rows, ("slack_norm",))

    use = np.asarray(mask, dtype=bool)
    def select(value: np.ndarray) -> np.ndarray:
        return value[use] if value.size == use.size else np.full(0, np.nan)

    if task_source is not None:
        model_task = task.copy()
        model_task_source = task_source
    if rail_source is not None:
        model_rail = rail.copy()
        model_rail_source = rail_source
    if arm_source is not None:
        model_arm = arm.copy()
        model_arm_source = arm_source
    model_total = model_rail + model_arm
    predicted = np.full(len(rows), np.nan, dtype=float)
    model_ok = np.isfinite(model_total)
    predicted[model_ok] = model_total[model_ok]
    residual_ok = ~model_ok & np.isfinite(requested) & np.isfinite(model_task)
    predicted[residual_ok] = requested[residual_ok] - model_task[residual_ok]
    difference = actual - predicted
    return {
        "actual_vy_m_s": _stats(select(actual)),
        "requested_vy_m_s": _stats(select(requested)),
        "task_residual_y_m_s": _stats(select(task)),
        "rail_exec_contrib_y_m_s": _stats(select(rail)),
        "arm_contrib_y_m_s": _stats(select(arm)),
        "model_protected_residual_y_m_s": _stats(select(model_task)),
        "model_rail_exec_contrib_y_m_s": _stats(select(model_rail)),
        "model_arm_contrib_y_m_s": _stats(select(model_arm)),
        "model_total_y_m_s": _stats(select(predicted)),
        "actual_minus_model_y_m_s": _stats(select(difference)),
        "last_slack": _stats(select(last_slack)),
        "slack_norm": _stats(select(slack_norm)),
        "sources": {
            "actual_vy": actual_source,
            "requested_vy": requested_source,
            "task_residual_y": task_source,
            "rail_exec_contrib_y": rail_source,
            "arm_contrib_y": arm_source,
            "last_slack": last_slack_source,
            "slack_norm": slack_source,
            "model_protected_residual_y": model_task_source,
            "model_rail_exec_contrib_y": model_rail_source,
            "model_arm_contrib_y": model_arm_source,
        },
        "model_fields_are_physical_measurements": False,
    }


def _shared_window_a_report(rows: list[dict]) -> dict:
    """Summarize append-only controller fields without changing old CSVs."""
    fields = shared_window_a_fields(rows)
    text_fields = {"task_pause_reason", "execution_model_hash"}
    report = {
        "present": {
            name: any(value not in (None, "") for value in values)
            for name, values in fields.items()
        },
        "numeric": {},
        "task_pause_reason_values": sorted(
            {str(value) for value in fields["task_pause_reason"] if value not in (None, "")}
        ),
        "execution_model_hash_values": sorted(
            {str(value) for value in fields["execution_model_hash"] if value not in (None, "")}
        ),
    }
    for name, values in fields.items():
        if name in text_fields:
            continue
        report["numeric"][name] = _stats(col(rows, name))
    actual_vy = col(rows, "twist_achieved_vy")
    predicted_vy = col(rows, "execution_predicted_vy")
    report["execution_predicted_vy_minus_actual_m_s"] = _stats(predicted_vy - actual_vy)
    return report


def analyze(csv_path: Path, *, when: str, window_a_csv: str = "", ax_mm: float = 0.4) -> dict:
    rows, align = load_aligned(Path(csv_path), window_a_csv or None)
    t = col(rows, "t_wall_s", "t_mono_s")
    dt_col = col(rows, "dt_actual_s")
    # ContactLogger v_cmd_* is a TOOL-frame command.  Window-A v_cmd_* and
    # twist_requested_* remain BASE-frame controller telemetry; merge_logs
    # keeps the external command under command_v_cmd_* to avoid frame reuse.
    command = twist_external_cmd6(rows)
    command_frame, command_source = _command_provenance(rows)
    uz = command[:, 2]
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
    transverse = {
        "air": transverse_error_report(
            pose6(rows),
            twist_ach6(rows),
            phase_mask(rows, "air_chirp"),
            t=t,
            command=command,
            command_frame=command_frame,
        ),
        "contact": transverse_error_report(
            pose6(rows),
            twist_ach6(rows),
            phase_mask(rows, "contact_chirp"),
            t=t,
            command=command,
            command_frame=command_frame,
        ),
        "contact_tail": _tail_report(
            pose6(rows),
            twist_ach6(rows),
            _after_phase_mask(rows, "contact_chirp"),
            t=t,
            command=command,
            command_frame=command_frame,
        ),
    }
    qp_attribution = _qp_attribution(rows, phase_mask(rows, "contact_chirp"))
    shared_fields_report = _shared_window_a_report(rows)
    air_ref = load_kind_json("03_chirp", "gv.json") or {}
    payload = {
        "csv": str(csv_path),
        "align": {k: v for k, v in align.items() if k != "reason" or v},
        "ax_mm": ax_mm,
        "air": {k: air.get(k) for k in ("T0_s", "Tp_s", "K", "n", "rmse", "tube")},
        "contact": {k: contact.get(k) for k in ("T0_s", "Tp_s", "K", "n", "rmse", "tube")},
        "Ev_contact_vs_air_model": ev,
        "pose_amp_mm": pose_amp,
        "command_frame": command_frame,
        "command_source": command_source,
        "window_a_command_frame": "base",
        "window_a_command_source": "window_a:v_cmd_*/twist_requested_*",
        "transverse_error": transverse,
        # Preserve the historical key for consumers; nested
        # ``initial_tool_z_projection`` now labels that diagnostic explicitly.
        "transverse_error_after_tool_z": transverse,
        "qp_attribution_contact": qp_attribution,
        "window_a_shared_fields": shared_fields_report,
        "run_metadata": _run_meta(rows),
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
- Command-relative contact transverse world-Y velocity p95 = {fmt_finite(1e3 * float(transverse['contact']['command_relative']['world_y_velocity_abs_m_s']['p95']), '.2f')} mm/s and displacement p95 = {fmt_finite(1e3 * float(transverse['contact']['command_relative']['world_y_displacement_abs_m']['p95']), '.2f')} mm. The command is transformed by the measured pose at every row and integrated with Window-A timestamps.
- Requested command world-Y displacement p95 = {fmt_finite(1e3 * float(transverse['contact']['command_relative']['requested_world_y_displacement_m']['p95_abs']), '.4f')} mm; this reports the commanded trajectory separately from the measured tracking error.
- Command provenance: `{command_frame}` frame from `{command_source}`; Window-A controller command columns are retained separately as BASE-frame telemetry. The external command is aligned to each measured Window-A pose before the frame transform.
- Legacy diagnostic after subtracting the **initial** tool-Z projection: contact transverse world-Y velocity p95 = {fmt_finite(1e3 * float(transverse['contact']['initial_tool_z_projection']['world_y_velocity_abs_m_s']['p95']), '.2f')} mm/s and displacement p95 = {fmt_finite(1e3 * float(transverse['contact']['initial_tool_z_projection']['world_y_displacement_abs_m']['p95']), '.2f')} mm.
- Post-contact tail rows = {transverse['contact_tail']['n']}; tail metrics are NaN when Window A has too few finite rows.
- QPIK contact attribution keeps actual Window-A `vy` separate from model columns: actual-minus-model p95 = {fmt_finite(1e3 * float(qp_attribution['actual_minus_model_y_m_s']['p95_abs']), '.2f')} mm/s. Missing legacy `task_residual_y` / `rail_exec_contrib_y` / `arm_contrib_y` / `last_slack` stay NaN; `slack_norm` is a mixed solver norm, not a Y-velocity bound.
- Append-only execution/model fields (task progress/pause, rail command ACK sequence, observer hash/predicted twist, and task/rail/arm model components) are preserved when present; execution predicted-vy minus measured-vy p95 = {fmt_finite(1e3 * float(shared_fields_report['execution_predicted_vy_minus_actual_m_s']['p95_abs']), '.2f')} mm/s. Old Window A files report NaN/empty provenance.
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
        f"dof={args.dof}  force loop OFF  abort F≥{args.abort_n:.1f} N  "
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
