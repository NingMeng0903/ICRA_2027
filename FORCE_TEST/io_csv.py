"""Load air-campaign or Window A hybrid CSVs with column aliases."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("t_wall_s", "t_ref_s", "t_mono_s"),
    "dt": ("dt_actual_s",),
    "vel_ff_vz": ("vel_ff_vz", "v_cmd_z"),
    "v_cmd": ("v_force_z", "vel_ff_vz", "v_cmd_z", "twist_requested_vz"),
    "v_meas": ("vz_achieved_tool", "twist_achieved_vz", "v_tcp_z"),
    "fz": ("fz", "f_ext_z", "fz_raw_comp"),
    "fstar": ("f_des_z_eff", "f_star", "desired_z"),
    "feedback_age": ("feedback_age_s",),
    "sensor_age": ("sensor_age_s",),
    "contact": ("contact_present", "physical_contact_state", "force_task_latched"),
    "loss": ("physical_contact_loss_event",),
    "acquire": ("physical_contact_acquire_event",),
    "reacquire": ("physical_contact_reacquire_event",),
    "ke_est": ("ke_est", "ke_hat"),
    "pose_z": ("pose_z", "pose_meas_z"),
    "phase": ("phase",),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def col(rows: list[dict[str, str]], *keys: str) -> np.ndarray:
    names = keys if keys else ()
    out = np.full(len(rows), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for key in names:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[i] = value
                break
    return out


def col_alias(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return col(rows, *ALIASES[name])


def phases(rows: list[dict[str, str]]) -> list[str]:
    return [str(row.get("phase") or "") for row in rows]


def moments(arr: np.ndarray) -> dict[str, float]:
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "p05": float("nan"),
                "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def bool_col(rows: list[dict[str, str]], *keys: str) -> np.ndarray:
    out = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows):
        for key in keys:
            raw = str(row.get(key) or "").strip().lower()
            if raw in ("1", "true", "yes"):
                out[i] = 1.0
                break
            if raw in ("0", "false", "no", ""):
                continue
            try:
                out[i] = 1.0 if float(raw) > 0.5 else 0.0
                break
            except (TypeError, ValueError):
                continue
    return out


def write_json(path: Path, payload: dict) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
