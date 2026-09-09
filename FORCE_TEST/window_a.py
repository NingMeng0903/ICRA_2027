"""Window A log contract for contact ID (07/08, 11–15).

MotionBus only publishes tool-Z.  Achieved 6-D twist, pose, and wrench
live on Window A's --log-csv.  Analyze fails instead of inventing those
columns from the command file.
"""

from __future__ import annotations

import math
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from id_math import so3_log_vee
from io_csv import col, load_rows, moments
from paths import write_readme

TWIST_LETTERS = ("vx", "vy", "vz", "wx", "wy", "wz")
WINDOW_A_SHARED_FIELDS = (
    "task_progress",
    "task_paused",
    "task_pause_reason",
    "rail_base_shaped",
    "rail_base_raw",
    "rail_post_committed",
    "rail_total_committed",
    "rail_pi_xi",
    "rail_d_ref",
    "rail_ref_acceleration",
    "rail_preview_residual",
    "rail_base_committed",
    "rail_commit_authority",
    "rail_command_rx_seq",
    "rail_command_processed_seq",
    "rail_command_written_seq",
    "rail_drive_write_seq",
    "rail_command_write_mono_s",
    "execution_model_hash",
    "execution_observer_validated",
    "execution_predicted_vx",
    "execution_predicted_vy",
    "execution_predicted_vz",
    "execution_predicted_wx",
    "execution_predicted_wy",
    "execution_predicted_wz",
    "task_requested_vx",
    "task_requested_vy",
    "task_requested_vz",
    "task_requested_wx",
    "task_requested_wy",
    "task_requested_wz",
    "task_accepted_vx",
    "task_accepted_vy",
    "task_accepted_vz",
    "task_accepted_wx",
    "task_accepted_wy",
    "task_accepted_wz",
    "task_model_vx",
    "task_model_vy",
    "task_model_vz",
    "task_model_wx",
    "task_model_wy",
    "task_model_wz",
    "rail_model_vx",
    "rail_model_vy",
    "rail_model_vz",
    "rail_model_wx",
    "rail_model_wy",
    "rail_model_wz",
    "arm_model_vx",
    "arm_model_vy",
    "arm_model_vz",
    "arm_model_wx",
    "arm_model_wy",
    "arm_model_wz",
)
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


def file_sha256(path: str | Path) -> str:
    """Hash a source log in chunks for reproducible alignment provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_window_a_fields(rows: list[dict]) -> dict[str, list]:
    """Select the append-only Window-A fields shared with the controller.

    Older logs simply return an empty string for a field introduced later;
    callers should pass numeric values through :func:`io_csv.col`, which
    turns those gaps into NaN.  Keeping this list explicit prevents a newer
    Window-A logger from being silently ignored by ICRA analyses.
    """
    return {
        name: [row.get(name, "") for row in rows]
        for name in WINDOW_A_SHARED_FIELDS
    }


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
    provenance = {
        "source_path": str(src),
        "source_sha256": file_sha256(src),
        "copy_path": str(dest),
        "copy_sha256": file_sha256(dest),
        "copied_at": datetime.now().isoformat(timespec="seconds"),
    }
    (dest.parent / "window_a_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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


def monotonic_time_segments(t: np.ndarray, *, rewind_s: float = 0.25) -> list[tuple[int, int]]:
    """Split a log where t_wall_s restarts at 0 on every MOVEJ / SERVO mode.

    Window A writes phase-relative t_wall_s.  One --log-csv therefore contains
    several overlapping [0, T] clocks.  Nearest-neighbour on the whole file
    mixes an old mid-stroke pose into a later take and looks like a 15 cm jump.
    """

    t = np.asarray(t, dtype=float).reshape(-1)
    cuts = [0]
    if t.size >= 2:
        dt = np.diff(t)
        for i in np.flatnonzero(np.isfinite(dt) & (dt < -float(rewind_s))):
            cuts.append(int(i) + 1)
    cuts.append(int(t.size))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _pick_time_segment(wa_t: np.ndarray, cmd_t: np.ndarray) -> tuple[slice, dict]:
    """Use the later rewind-free slice that actually aligns with the command log."""

    cmd_t = np.asarray(cmd_t, dtype=float)
    cmd_t = cmd_t[np.isfinite(cmd_t)]
    segs = monotonic_time_segments(wa_t)
    ranked: list[tuple[float, int, slice, dict]] = []
    for i, (a, b) in enumerate(segs):
        if b - a < 16:
            continue
        sl = slice(a, b)
        report = alignment_report(cmd_t, wa_t[sl])
        if not report["aligned"]:
            continue
        ranked.append((float(report["gap_median_ms"]), i, sl, report))
    if not ranked:
        raise AlignmentError(
            "Window A t_wall_s restarts on every mode change; no single "
            "segment lines up with the command log. Restart Window A --log-csv "
            "for this take, or pass a CSV that covers only this SERVO_TWIST."
        )
    # The current take is the last SERVO slice.  An older long slice can
    # have a slightly tighter gap and still be the wrong experiment.
    ranked.sort(key=lambda item: (-item[1], item[0]))
    _gap, idx, sl, report = ranked[0]
    report["wa_segments"] = len(segs)
    report["wa_segment_index"] = idx
    report["wa_segment_rows"] = int(sl.stop - sl.start)
    report["wa_clock_resets"] = max(len(segs) - 1, 0)
    return sl, report


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
    sl, report = _pick_time_segment(wa_t[finite_w], cmd_t[finite_c])
    # finite_w may drop rows; map slice back onto the original WA rows.
    wa_keep = np.flatnonzero(finite_w)[sl]
    wa_t_use = wa_t[wa_keep]
    wa_idx_all = wa_keep[_nearest_indices(wa_t_use, cmd_t)]
    merged: list[dict] = []
    for i, cmd in enumerate(cmd_rows):
        row = dict(wa_rows[int(wa_idx_all[i])])  # original-row index
        row["phase"] = str(cmd.get("phase") or row.get("phase") or "")
        # ContactLogger's v_cmd_* is the tool-frame command sent by the
        # experiment.  Window A also has v_cmd_* columns, but those are the
        # controller's base-frame command.  Keep both namespaces: replacing
        # Window A's columns here silently changes the frame of an existing
        # field and makes a later analysis double-rotate it.
        command_frame = str(cmd.get("command_frame") or "tool").strip().lower()
        row["command_frame"] = command_frame
        row["command_source"] = str(
            cmd.get("command_source") or f"{Path(cmd_path).name}:v_cmd_*"
        )
        for letter in TWIST_LETTERS:
            key = f"v_cmd_{letter}"
            if cmd.get(key) not in (None, ""):
                row[f"command_{key}"] = cmd[key]
        if cmd.get("t_mono_s") not in (None, ""):
            row["t_mono_s"] = cmd["t_mono_s"]
        if cmd.get("motion_seq") not in (None, ""):
            row["cmd_motion_seq"] = cmd["motion_seq"]
        # Both hardware loggers use the same install handshake.  Carry its
        # request/install provenance through the alignment so an analyzed
        # Window-A copy can prove that every recorded tick followed the live
        # SERVO_TWIST install ACK.
        for name in (
            "mode_name",
            "mode_request_seq",
            "mode_install_seq",
            "mode_install_status",
        ):
            if cmd.get(name) not in (None, ""):
                row[name] = cmd[name]
        merged.append(row)
    # Keep the append-only fields addressable even when this is an old
    # Window-A CSV whose header predates the root controller additions.
    for row in merged:
        for name in WINDOW_A_SHARED_FIELDS:
            row.setdefault(name, "")
    report["torque_present"] = has_torque(merged)
    report["cmd_csv"] = str(cmd_path)
    report["window_a_csv"] = str(window_a_path)
    command_frames = {
        str(cmd.get("command_frame") or "tool").strip().lower()
        for cmd in cmd_rows
    }
    report["command_frame"] = (
        next(iter(command_frames)) if len(command_frames) == 1 else "mixed"
    )
    report["command_source"] = f"{Path(cmd_path).name}:v_cmd_*"
    report["window_a_command_frame"] = "base"
    report["window_a_command_source"] = (
        f"{Path(window_a_path).name}:v_cmd_*/twist_requested_*"
    )
    report["cmd_sha256"] = file_sha256(cmd_path)
    report["window_a_sha256"] = file_sha256(window_a_path)
    provenance_path = Path(window_a_path).parent / "window_a_provenance.json"
    report["window_a_provenance_json"] = str(provenance_path) if provenance_path.is_file() else None
    report["window_a_source_sha256"] = None
    report["window_a_source_path"] = None
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            report["window_a_source_sha256"] = provenance.get("source_sha256")
            report["window_a_source_path"] = provenance.get("source_path")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return merged, report


def has_torque(rows: list[dict]) -> bool:
    tau = col(rows, *WRENCH_T)
    return bool(np.isfinite(tau).any())


def pose6(rows: list[dict]) -> np.ndarray:
    """Measured TCP if present; pose_* is often a copy of pose_meas_*."""

    out = np.column_stack(
        [col(rows, meas, key) for key, meas in zip(POSE_KEYS, POSE_MEAS_KEYS)]
    )
    return out


def wrench6(rows: list[dict]) -> np.ndarray:
    force = np.column_stack([col(rows, key) for key in WRENCH_F])
    torque = np.column_stack(
        [col(rows, key, raw) for key, raw in zip(WRENCH_T, WRENCH_T_RAW)]
    )
    return np.column_stack([force, torque])


def twist_external_cmd6(rows: list[dict]) -> np.ndarray:
    """Read the experiment command, which is conventionally tool-frame.

    ``merge_logs`` stores command-CSV values under ``command_v_cmd_*`` so
    that Window A's base-frame ``v_cmd_*`` remains untouched.  The fallback
    keeps direct analysis of an old command-only CSV working.
    """
    # Once a merge has marked the row, an absent command_v_cmd_* value is a
    # missing external command.  Falling back to Window A's v_cmd_* in that
    # case would reinterpret a BASE value as TOOL and double-rotate it.
    merged_rows = any("command_frame" in row for row in rows)
    cols = []
    for letter in TWIST_LETTERS:
        keys = [f"command_v_cmd_{letter}"]
        if not merged_rows:
            keys.append(f"v_cmd_{letter}")
        cols.append(col(rows, *keys))
    return np.column_stack(cols)


def twist_window_a_cmd6(rows: list[dict]) -> np.ndarray:
    """Read Window A's controller command, already expressed in BASE."""
    cols = []
    for letter in TWIST_LETTERS:
        cols.append(
            col(
                rows,
                f"v_cmd_{letter}",
                f"twist_requested_{letter}",
                f"twist_{letter}",
                f"path_twist_{letter}",
                f"vel_ff_{letter}",
            )
        )
    return np.column_stack(cols)


# Existing callers use this name for the external experiment command.  Keep
# the alias while making the frame choice explicit at the call site.
def twist_cmd6(rows: list[dict]) -> np.ndarray:
    return twist_external_cmd6(rows)


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


def transverse_tool_z_displacement(
    pose: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Position residual after removing the initial tool-Z projection.

    The returned vectors are in world coordinates.  This is a geometric
    residual, not command integration: a pure motion along the measured
    initial tool-Z axis contributes zero even when that axis has a world X/Y
    component.  Missing pose rows stay NaN so an old Window A file cannot
    silently turn into a zero error.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2:
        return np.full((0, 3), np.nan, dtype=float)
    out = np.full((pose.shape[0], 3), np.nan, dtype=float)
    if pose.shape[1] < 6:
        return out
    axis = tool_z_world(pose)
    ok = np.isfinite(pose[:, :3]).all(axis=1) & np.isfinite(axis).all(axis=1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.size != pose.shape[0]:
            return out
        ok &= mask_arr
    if not np.any(ok):
        return out
    i0 = int(np.flatnonzero(ok)[0])
    p0 = pose[i0, :3]
    n0 = axis[i0]
    dp = pose[:, :3] - p0
    s = dp @ n0
    out[ok] = dp[ok] - s[ok, None] * n0
    return out


def transverse_tool_z_velocity(
    pose: np.ndarray,
    twist: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Achieved linear velocity after removing its tool-Z projection."""
    pose = np.asarray(pose, dtype=float)
    twist = np.asarray(twist, dtype=float)
    if pose.ndim != 2:
        return np.full((0, 3), np.nan, dtype=float)
    out = np.full((pose.shape[0], 3), np.nan, dtype=float)
    if (
        pose.shape[1] < 6
        or twist.ndim != 2
        or twist.shape[0] != pose.shape[0]
        or twist.shape[1] < 3
    ):
        return out
    axis = tool_z_world(pose)
    ok = np.isfinite(axis).all(axis=1) & np.isfinite(twist[:, :3]).all(axis=1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.size != pose.shape[0]:
            return out
        ok &= mask_arr
    if not np.any(ok):
        return out
    v = twist[:, :3]
    projection = np.einsum("ij,ij->i", v[ok], axis[ok])
    out[ok] = v[ok] - projection[:, None] * axis[ok]
    return out


def command_relative_transverse_error(
    pose: np.ndarray,
    twist_actual: np.ndarray,
    twist_command: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    *,
    command_frame: str = "tool",
) -> dict[str, np.ndarray | str]:
    """Compare measured TCP motion with the requested trajectory.

    Window-A ``twist_achieved_*`` is in the base/world frame, while the
    command CSV sent by FORCE_TEST is a tool-frame twist.  Transform the
    command with the *measured orientation at every row*, integrate it using
    the measured timestamps, then project both position and velocity error
    onto the transverse plane of that same row's measured tool-Z axis.

    This is the command-relative metric.  It is separate from the legacy
    initial-axis projection because the latter cannot distinguish a changing
    tool direction from an unrequested TCP displacement.
    """
    pose = np.asarray(pose, dtype=float)
    actual = np.asarray(twist_actual, dtype=float)
    command = np.asarray(twist_command, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(pose.shape[0]) if pose.ndim == 2 else 0
    out: dict[str, np.ndarray | str] = {
        "position_error": np.full((n, 3), np.nan, dtype=float),
        "velocity_error": np.full((n, 3), np.nan, dtype=float),
        "requested_position": np.full((n, 3), np.nan, dtype=float),
        "command_base_velocity": np.full((n, 3), np.nan, dtype=float),
        "command_frame": str(command_frame),
    }
    if command_frame not in ("tool", "base"):
        raise ValueError(f"command_frame must be 'tool' or 'base', got {command_frame!r}")
    if (
        pose.ndim != 2
        or pose.shape[1] < 6
        or actual.ndim != 2
        or actual.shape[0] != n
        or actual.shape[1] < 3
        or command.ndim != 2
        or command.shape[0] != n
        or command.shape[1] < 3
        or t.size != n
        or mask.size != n
    ):
        return out
    rotation = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    tool_z = rotation[:, :, 2]
    command_tool = command[:, :3]
    if command_frame == "tool":
        command_base = np.einsum("nij,nj->ni", rotation, command_tool)
    else:
        command_base = command_tool.copy()
    out["command_base_velocity"] = command_base
    position_error = out["position_error"]
    velocity_error = out["velocity_error"]
    requested_position = out["requested_position"]
    assert isinstance(position_error, np.ndarray)
    assert isinstance(velocity_error, np.ndarray)
    assert isinstance(requested_position, np.ndarray)
    finite = (
        mask
        & np.isfinite(pose[:, :3]).all(axis=1)
        & np.isfinite(tool_z).all(axis=1)
        & np.isfinite(actual[:, :3]).all(axis=1)
        & np.isfinite(command_base).all(axis=1)
        & np.isfinite(t)
    )
    indices = np.flatnonzero(finite)
    if indices.size == 0:
        return out
    previous: int | None = None
    requested = np.full(3, np.nan, dtype=float)
    for index_raw in indices:
        index = int(index_raw)
        if previous is None or index != previous + 1:
            # Each selected contiguous phase gets its own zero-error origin.
            requested = pose[index, :3].copy()
        else:
            dt = float(t[index] - t[previous])
            if not np.isfinite(dt) or dt < 0.0:
                requested = pose[index, :3].copy()
            elif dt > 0.0:
                requested = requested + 0.5 * (
                    command_base[previous] + command_base[index]
                ) * dt
            # A duplicate timestamp is a second sample at the same instant;
            # it contributes no distance.  Resetting to the measured pose
            # here makes the requested trajectory absorb any real drift.
        requested_position[index] = requested
        n_axis = tool_z[index]
        position = pose[index, :3] - requested
        velocity = actual[index, :3] - command_base[index]
        position_error[index] = position - np.dot(position, n_axis) * n_axis
        velocity_error[index] = velocity - np.dot(velocity, n_axis) * n_axis
        previous = index
    return out


def _empty_transverse_stats() -> dict[str, float | int]:
    return {"n": 0, "mean": float("nan"), "p95": float("nan"), "max": float("nan")}


def _transverse_stats(vectors: np.ndarray, *, axis: int = 1) -> dict:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] < 3:
        return {
            "n": 0,
            "world_y_m_s": _empty_transverse_stats(),
            "world_y_abs_m_s": _empty_transverse_stats(),
            "horizontal_m_s": _empty_transverse_stats(),
            "horizontal_abs_m_s": _empty_transverse_stats(),
        }
    y = vectors[:, int(axis)]
    horizontal = np.linalg.norm(vectors[:, :2], axis=1)

    def moments(value: np.ndarray, *, absolute: bool = False) -> dict:
        finite = np.asarray(value, dtype=float)
        finite = finite[np.isfinite(finite)]
        if absolute:
            finite = np.abs(finite)
        if finite.size == 0:
            return _empty_transverse_stats()
        return {
            "n": int(finite.size),
            "mean": float(np.mean(finite)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }

    return {
        "n": int(np.isfinite(y).sum()),
        "world_y_m_s": moments(y),
        "world_y_abs_m_s": moments(y, absolute=True),
        "horizontal_m_s": moments(horizontal),
        "horizontal_abs_m_s": moments(horizontal, absolute=True),
    }


def _position_transverse_stats(vectors: np.ndarray, *, axis: int = 1) -> dict:
    report = _transverse_stats(vectors, axis=axis)
    return {
        "n": report["n"],
        "world_y_displacement_m": report["world_y_m_s"],
        "world_y_displacement_abs_m": report["world_y_abs_m_s"],
        "horizontal_displacement_m": report["horizontal_m_s"],
        "horizontal_displacement_abs_m": report["horizontal_abs_m_s"],
    }


def _velocity_transverse_stats(vectors: np.ndarray, *, axis: int = 1) -> dict:
    report = _transverse_stats(vectors, axis=axis)
    return {
        "n": report["n"],
        "world_y_velocity_m_s": report["world_y_m_s"],
        "world_y_velocity_abs_m_s": report["world_y_abs_m_s"],
        "horizontal_velocity_m_s": report["horizontal_m_s"],
        "horizontal_velocity_abs_m_s": report["horizontal_abs_m_s"],
    }


def _scalar_stats(values: np.ndarray) -> dict[str, float | int]:
    """Compact scalar stats for the requested command trajectory."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "p95_abs": float("nan"), "max_abs": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p95_abs": float(np.percentile(np.abs(finite), 95)),
        "max_abs": float(np.max(np.abs(finite))),
    }


def transverse_error_report(
    pose: np.ndarray,
    twist: np.ndarray,
    mask: np.ndarray,
    *,
    t: np.ndarray | None = None,
    command: np.ndarray | None = None,
    command_frame: str = "tool",
) -> dict:
    """Report legacy initial-axis and command-relative transverse errors."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if pose.ndim != 2 or twist.ndim != 2 or mask.size != pose.shape[0]:
        legacy = {
            "n": 0,
            **_velocity_transverse_stats(np.full((0, 3), np.nan)),
            **_position_transverse_stats(np.full((0, 3), np.nan)),
        }
        return {
            **legacy,
            "initial_tool_z_projection": legacy,
            "command_relative": {"n": 0, "command_frame": str(command_frame)},
        }
    dp = transverse_tool_z_displacement(pose, mask=mask)
    vv = transverse_tool_z_velocity(pose, twist, mask=mask)
    legacy = {
        "n": int(np.count_nonzero(mask)),
        **_velocity_transverse_stats(vv[:, :3]),
        **_position_transverse_stats(dp[:, :3]),
    }
    report = {
        **legacy,
        "initial_tool_z_projection": legacy,
        "command_relative": {"n": 0, "command_frame": str(command_frame)},
    }
    if t is not None and command is not None:
        relative = command_relative_transverse_error(
            pose,
            twist,
            command,
            t,
            mask,
            command_frame=command_frame,
        )
        position_error = relative["position_error"]
        velocity_error = relative["velocity_error"]
        assert isinstance(position_error, np.ndarray)
        assert isinstance(velocity_error, np.ndarray)
        command_report = {
            "command_frame": str(relative["command_frame"]),
            **_velocity_transverse_stats(velocity_error),
            **_position_transverse_stats(position_error),
        }
        requested_position = relative["requested_position"]
        command_base_velocity = relative["command_base_velocity"]
        assert isinstance(requested_position, np.ndarray)
        assert isinstance(command_base_velocity, np.ndarray)
        requested_ok = (
            mask
            & np.isfinite(requested_position).all(axis=1)
            & np.isfinite(command_base_velocity).all(axis=1)
        )
        requested_indices = np.flatnonzero(requested_ok)
        if requested_indices.size:
            requested_delta = requested_position - requested_position[requested_indices[0]]
            command_report["requested_world_y_displacement_m"] = _scalar_stats(
                requested_delta[requested_ok, 1]
            )
            command_report["requested_world_y_velocity_m_s"] = _scalar_stats(
                command_base_velocity[requested_ok, 1]
            )
        else:
            command_report["requested_world_y_displacement_m"] = _scalar_stats(np.array([]))
            command_report["requested_world_y_velocity_m_s"] = _scalar_stats(np.array([]))
        report["command_relative"] = command_report
    return report


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
