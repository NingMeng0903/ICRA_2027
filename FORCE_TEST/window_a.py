"""Window A log contract for contact ID (07/08, 11–15).

MotionBus only publishes tool-Z.  Achieved 6-D twist, pose, and wrench
live on Window A's --log-csv.  Analyze fails instead of inventing those
columns from the command file.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from id_math import so3_log_vee
from io_csv import col, load_rows, moments
from paths import write_readme

TWIST_LETTERS = ("vx", "vy", "vz", "wx", "wy", "wz")
POSE_KEYS = ("pose_x", "pose_y", "pose_z", "pose_rx", "pose_ry", "pose_rz")
POSE_MEAS_KEYS = (
    "pose_meas_x",
    "pose_meas_y",
    "pose_meas_z",
    "pose_meas_rx",
    "pose_meas_ry",
    "pose_meas_rz",
)
WRENCH_F = ("fx", "fy", "fz")
WRENCH_T = ("tx", "ty", "tz")
WRENCH_F_RAW = ("fx_raw_comp", "fy_raw_comp", "fz_raw_comp")
WRENCH_T_RAW = ("tx_raw_comp", "ty_raw_comp", "tz_raw_comp")


class AlignmentError(ValueError):
    """Command log and Window A cannot rebuild the sent queue."""


def add_window_a_arg(parser) -> None:
    parser.add_argument(
        "--window-a-csv",
        default="",
        help="Window A --log-csv path (copied into DATA/<kind>/window_a.csv)",
    )


def resolve_window_a(
    window_a_csv: str | Path | None,
    data_dir: Path | None = None,
) -> Path | None:
    if window_a_csv:
        path = Path(window_a_csv)
        if path.is_file():
            return path
    if data_dir is not None:
        sibling = Path(data_dir) / "window_a.csv"
        if sibling.is_file():
            return sibling
    return None


def copy_window_a(src: Path, data_dir: Path) -> Path:
    dest = Path(data_dir) / "window_a.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src).resolve()
    dest_r = dest.resolve()
    if src != dest_r:
        shutil.copy2(src, dest)
    print(f"[WINDOW-A] {dest}", flush=True)
    return dest


def require_window_a(
    window_a_csv: str | Path | None,
    data_dir: Path | None = None,
) -> Path:
    path = resolve_window_a(window_a_csv, data_dir)
    if path is None:
        raise AlignmentError(
            "Window A CSV missing. Start Window A with --log-csv and pass "
            "--window-a-csv. Do not analyze command-only logs for 07/08/11–15."
        )
    return path


def _nearest_indices(src: np.ndarray, query: np.ndarray) -> np.ndarray:
    order = np.argsort(src, kind="stable")
    s = src[order]
    pos = np.searchsorted(s, query)
    pos = np.clip(pos, 0, s.size - 1)
    left = np.clip(pos - 1, 0, s.size - 1)
    pick_right = np.abs(s[pos] - query) < np.abs(s[left] - query)
    idx = np.where(pick_right, pos, left)
    return order[idx]


def alignment_report(cmd_t: np.ndarray, wa_t: np.ndarray) -> dict:
    if cmd_t.size == 0 or wa_t.size == 0:
        return {
            "n_cmd": int(cmd_t.size),
            "n_window_a": int(wa_t.size),
            "n_matched": 0,
            "gap_median_ms": float("nan"),
            "gap_p95_ms": float("nan"),
            "gap_max_ms": float("nan"),
            "aligned": False,
            "reason": "empty command or Window A time",
        }
    idx = _nearest_indices(wa_t, cmd_t)
    gap = np.abs(cmd_t - wa_t[idx])
    med = float(np.median(gap))
    p95 = float(np.percentile(gap, 95))
    mx = float(np.max(gap))
    ok = med <= 0.020 and p95 <= 0.040 and int(cmd_t.size) >= 16
    return {
        "n_cmd": int(cmd_t.size),
        "n_window_a": int(wa_t.size),
        "n_matched": int(cmd_t.size),
        "gap_median_ms": 1e3 * med,
        "gap_p95_ms": 1e3 * p95,
        "gap_max_ms": 1e3 * mx,
        "aligned": bool(ok),
        "reason": "" if ok else "timestamp gap above 20 ms median / 40 ms p95",
    }


def merge_logs(cmd_path: Path, window_a_path: Path) -> tuple[list[dict], dict]:
    """Join command ticks to the nearest Window A row on t_wall_s."""

    cmd_rows = load_rows(cmd_path)
    wa_rows = load_rows(window_a_path)
    cmd_t = col(cmd_rows, "t_wall_s", "t_mono_s")
    wa_t = col(wa_rows, "t_wall_s", "t_ref_s")
    finite_c = np.isfinite(cmd_t)
    finite_w = np.isfinite(wa_t)
    if int(np.count_nonzero(finite_c)) < 16 or int(np.count_nonzero(finite_w)) < 16:
        raise AlignmentError(
            f"{cmd_path} or {window_a_path} has too few finite timestamps"
        )
    report = alignment_report(cmd_t[finite_c], wa_t[finite_w])
    if not report["aligned"]:
        raise AlignmentError(
            f"cannot rebuild the sent queue: {report['reason']} "
            f"(median {report['gap_median_ms']:.1f} ms)"
        )
    wa_idx_all = _nearest_indices(wa_t, cmd_t)
    merged: list[dict] = []
    for i, cmd in enumerate(cmd_rows):
        row = dict(wa_rows[int(wa_idx_all[i])])
        row["phase"] = str(cmd.get("phase") or row.get("phase") or "")
        for letter in TWIST_LETTERS:
            key = f"v_cmd_{letter}"
            if cmd.get(key) not in (None, ""):
                row[key] = cmd[key]
        if cmd.get("t_mono_s") not in (None, ""):
            row["t_mono_s"] = cmd["t_mono_s"]
        if cmd.get("motion_seq") not in (None, ""):
            row["cmd_motion_seq"] = cmd["motion_seq"]
        merged.append(row)
    report["torque_present"] = has_torque(merged)
    report["cmd_csv"] = str(cmd_path)
    report["window_a_csv"] = str(window_a_path)
    return merged, report


def has_torque(rows: list[dict]) -> bool:
    tau = col(rows, *WRENCH_T)
    return bool(np.isfinite(tau).any())


def pose6(rows: list[dict]) -> np.ndarray:
    out = np.column_stack([col(rows, key, meas) for key, meas in zip(POSE_KEYS, POSE_MEAS_KEYS)])
    return out


def wrench6(rows: list[dict]) -> np.ndarray:
    force = np.column_stack([col(rows, key) for key in WRENCH_F])
    torque = np.column_stack(
        [col(rows, key, raw) for key, raw in zip(WRENCH_T, WRENCH_T_RAW)]
    )
    return np.column_stack([force, torque])


def twist_cmd6(rows: list[dict]) -> np.ndarray:
    cols = []
    for letter in TWIST_LETTERS:
        cols.append(
            col(
                rows,
                f"v_cmd_{letter}",
                f"twist_requested_{letter}",
                f"twist_{letter}",
                f"vel_ff_{letter}",
            )
        )
    return np.column_stack(cols)


def twist_ach6(rows: list[dict]) -> np.ndarray:
    cols = []
    for letter in TWIST_LETTERS:
        keys = [f"twist_achieved_{letter}"]
        if letter == "vz":
            keys.extend(("vz_achieved_tool", "v_ach"))
        cols.append(col(rows, *keys))
    return np.column_stack(cols)


def rot_xyz(rx: np.ndarray, ry: np.ndarray, rz: np.ndarray) -> np.ndarray:
    """Base-from-tool rotation. Extrinsic 'xyz' (Rz @ Ry @ Rx), same as scipy."""

    eul = np.column_stack([rx, ry, rz])
    out = np.full((eul.shape[0], 3, 3), np.nan, dtype=float)
    ok = np.isfinite(eul).all(axis=1)
    if not np.any(ok):
        return out
    cx, sx = np.cos(rx[ok]), np.sin(rx[ok])
    cy, sy = np.cos(ry[ok]), np.sin(ry[ok])
    cz, sz = np.cos(rz[ok]), np.sin(rz[ok])
    r = np.zeros((int(np.count_nonzero(ok)), 3, 3), dtype=float)
    r[:, 0, 0] = cy * cz
    r[:, 0, 1] = sx * sy * cz - cx * sz
    r[:, 0, 2] = cx * sy * cz + sx * sz
    r[:, 1, 0] = cy * sz
    r[:, 1, 1] = sx * sy * sz + cx * cz
    r[:, 1, 2] = cx * sy * sz - sx * cz
    r[:, 2, 0] = -sy
    r[:, 2, 1] = sx * cy
    r[:, 2, 2] = cx * cy
    out[ok] = r
    return out


def tool_z_world(pose: np.ndarray) -> np.ndarray:
    r = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    return r[:, :, 2]


def tool_axis_world(pose: np.ndarray, axis: int) -> np.ndarray:
    r = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    return r[:, :, int(axis)]


def tool_z_displacement(pose: np.ndarray) -> np.ndarray:
    """Signed tool-Z travel from the first finite pose, using that sample's axis."""

    n0 = tool_z_world(pose)
    p = pose[:, :3]
    out = np.full(pose.shape[0], np.nan, dtype=float)
    ok = np.isfinite(p).all(axis=1) & np.isfinite(n0).all(axis=1)
    if not np.any(ok):
        return out
    i0 = int(np.flatnonzero(ok)[0])
    axis = n0[i0]
    out[ok] = (p[ok] - p[i0]) @ axis
    return out


def axis_unit(axis: int) -> np.ndarray:
    e = np.zeros(3, dtype=float)
    e[int(axis) % 3] = 1.0
    return e


def relative_axis_angle(pose: np.ndarray, axis: int, *, i_ref: int | None = None) -> np.ndarray:
    """θ = Log(R_ref^T R)^∨ · e_axis.  Not a global Euler angle."""

    pose = np.asarray(pose, dtype=float)
    n = int(pose.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    R = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    ok = np.isfinite(R).all(axis=(1, 2))
    if not np.any(ok):
        return out
    i0 = int(i_ref) if i_ref is not None else int(np.flatnonzero(ok)[0])
    if not ok[i0]:
        return out
    e = axis_unit(axis)
    Rref_t = R[i0].T
    for i in np.flatnonzero(ok):
        out[int(i)] = float(np.dot(so3_log_vee(Rref_t @ R[int(i)]), e))
    return out


def pose_body_twist(pose: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Tool-frame twist from pose: v = R^T ṗ, ω = (Log(R_k^T R_{k+1}))^∨ / Δt.

    Euler-angle differences are not angular velocity and are not used here.
    """

    pose = np.asarray(pose, dtype=float)
    dt = np.asarray(dt, dtype=float).reshape(-1)
    n = int(pose.shape[0])
    out = np.zeros((n, 6), dtype=float)
    if n < 2:
        return out
    R = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    for i in range(n - 1):
        dti = float(dt[i + 1]) if i + 1 < dt.size else float("nan")
        if not (math.isfinite(dti) and dti > 1e-6):
            continue
        if not (np.isfinite(R[i]).all() and np.isfinite(R[i + 1]).all()):
            continue
        if not (np.isfinite(pose[i, :3]).all() and np.isfinite(pose[i + 1, :3]).all()):
            continue
        v_world = (pose[i + 1, :3] - pose[i, :3]) / dti
        out[i + 1, :3] = R[i].T @ v_world
        out[i + 1, 3:6] = so3_log_vee(R[i].T @ R[i + 1]) / dti
    if n > 1:
        out[0] = out[1]
    return out


def integrate_along_axis(pose: np.ndarray, axis: int) -> np.ndarray:
    n = tool_axis_world(pose, axis)
    p = pose[:, :3]
    out = np.zeros(pose.shape[0], dtype=float)
    dp = np.diff(p, axis=0)
    step = np.sum(dp * n[1:], axis=1)
    step = np.where(np.isfinite(step), step, 0.0)
    out[1:] = np.cumsum(step)
    return out


def phase_mask(rows: list[dict], *prefixes: str) -> np.ndarray:
    names = [str(row.get("phase") or "") for row in rows]
    if not prefixes:
        return np.ones(len(names), dtype=bool)
    return np.array(
        [any(name.startswith(pref) for pref in prefixes) for name in names],
        dtype=bool,
    )


def nan_stats(arr: np.ndarray) -> dict[str, float]:
    return moments(arr)


def write_stub(visu: Path, title: str, body: str) -> Path:
    return write_readme(visu, f"# {title}\n\n{body.strip()}\n")


def fmt_finite(value: float, spec: str = ".3f") -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return format(float(value), spec)
