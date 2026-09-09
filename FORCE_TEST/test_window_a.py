#!/usr/bin/env python3
"""Offline checks for the Window A merge and pose-Ke contract.  No hardware."""

from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_csv import write_csv
import paths
from mode_handshake import (
    ModeInstallError,
    ModeInstallTimeout,
    wait_for_mode_install,
)
from window_a import (
    AlignmentError,
    alignment_report,
    copy_window_a,
    file_sha256,
    WINDOW_A_SHARED_FIELDS,
    command_relative_transverse_error,
    merge_logs,
    monotonic_time_segments,
    pose6,
    rot_xyz,
    shared_window_a_fields,
    transverse_error_report,
    tool_z_displacement,
    twist_external_cmd6,
    twist_window_a_cmd6,
)


def _write(path: Path, rows: list[dict]) -> Path:
    write_csv(path, rows)
    return path


def test_alignment_and_ke() -> None:
    rng = np.random.default_rng(0)
    n = 400
    t = 0.005 * np.arange(n)
    z = 0.001 * np.arange(n)
    fz = 0.5 + 800.0 * (z - z[80])
    fz[:80] = 0.2
    cmd_rows = []
    wa_rows = []
    for i in range(n):
        phase = "seek" if i < 80 else ("press" if i < 300 else "retract")
        cmd_rows.append(
            {
                "t_wall_s": f"{t[i]:.6f}",
                "t_mono_s": f"{t[i]:.6f}",
                "dt_actual_s": "0.005",
                "phase": phase,
                "v_cmd_vz": "0.003" if phase == "press" else "0.000",
                "v_cmd_vx": "0",
                "v_cmd_vy": "0",
                "v_cmd_wx": "0",
                "v_cmd_wy": "0",
                "v_cmd_wz": "0",
            }
        )
        wa_rows.append(
            {
                "t_wall_s": f"{t[i] + 0.001:.6f}",
                "dt_actual_s": "0.005",
                "phase": "servo_twist",
                "pose_x": "0",
                "pose_y": "0",
                "pose_z": f"{z[i]:.6f}",
                "pose_rx": "0",
                "pose_ry": "0",
                "pose_rz": "0",
                "fx": "0",
                "fy": "0",
                "fz": f"{fz[i]:.4f}",
                "tx": "0",
                "ty": f"{0.02 * math.sin(0.1 * i):.4f}",
                "tz": "0",
                "twist_achieved_vx": "0",
                "twist_achieved_vy": "0",
                "twist_achieved_vz": "0.003" if phase == "press" else "0",
                "twist_achieved_wx": "0",
                "twist_achieved_wy": "0",
                "twist_achieved_wz": "0",
            }
        )
    # Every controller append-only field must survive the join, including
    # text/hash fields.  Older rows are covered separately below.
    for index, name in enumerate(WINDOW_A_SHARED_FIELDS):
        wa_rows[0][name] = f"field-{index}"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cmd = _write(root / "cmd.csv", cmd_rows)
        wa = _write(root / "window_a.csv", wa_rows)
        rows, report = merge_logs(cmd, wa)
        assert report["aligned"], report
        assert report["torque_present"]
        assert len(report["cmd_sha256"]) == 64
        assert len(report["window_a_sha256"]) == 64
        assert all(name in rows[0] for name in WINDOW_A_SHARED_FIELDS)
        assert all(rows[0][name] == f"field-{index}" for index, name in enumerate(WINDOW_A_SHARED_FIELDS))
        copied = copy_window_a(wa, root / "copied")
        _merged_copy, copied_report = merge_logs(cmd, copied)
        assert copied_report["window_a_source_sha256"] == file_sha256(wa)
        from window_a import pose6

        dx = tool_z_displacement(pose6(rows))
        press = np.array([r["phase"] == "press" for r in rows])
        d_pose = float(dx[press][-1] - dx[press][0])
        d_cmd = 0.003 * 0.005 * int(np.count_nonzero(press))
        assert abs(d_pose - (z[299] - z[80])) < 1e-6
        # Command integral over the same ticks is close here; the contract is
        # that analyze must use d_pose, not d_cmd, when they later diverge.
        assert math.isfinite(d_cmd)

        bad_wa = []
        for row in wa_rows:
            item = dict(row)
            item["t_wall_s"] = f"{float(row['t_wall_s']) + 3600.0:.6f}"
            bad_wa.append(item)
        bad = _write(root / "bad.csv", bad_wa)
        try:
            merge_logs(cmd, bad)
        except AlignmentError:
            pass
        else:
            raise AssertionError("wide timestamp gap must fail")
    print("[OK] alignment + pose Δx", flush=True)
    del rng


def test_shared_window_a_fields_backward_compat() -> None:
    old = [{"phase": "servo_twist"}, {"phase": "done", "task_progress": "0.5"}]
    selected = shared_window_a_fields(old)
    assert selected["task_progress"] == ["", "0.5"]
    assert selected["execution_model_hash"] == ["", ""]
    assert len(selected) == len(WINDOW_A_SHARED_FIELDS) == 56
    for prefix in ("task_requested", "task_accepted", "task_model", "rail_model", "arm_model"):
        assert all(f"{prefix}_{axis}" in selected for axis in ("vx", "vy", "vz", "wx", "wy", "wz"))
    print("[OK] Window-A append-only fields + old CSV compatibility", flush=True)


def test_rewind_clock_uses_latest_segment() -> None:
    """Window A t_wall_s restarts; do not mix an old mid-stroke into this take."""

    n = 80
    dt = 0.005
    old = []
    new = []
    cmd_rows = []
    for i in range(n):
        t = dt * i
        old.append(
            {
                "t_wall_s": f"{t:.6f}",
                "dt_actual_s": "0.005",
                "phase": "servo_twist",
                "pose_x": "0.347",
                "pose_y": "0.252",
                "pose_z": "0.281",
                "pose_rx": "0",
                "pose_ry": "0",
                "pose_rz": "0",
                "fx": "0",
                "fy": "0",
                "fz": "0.1",
                "tx": "0",
                "ty": "0",
                "tz": "0",
                "twist_achieved_vz": "0.001",
            }
        )
        new.append(
            {
                "t_wall_s": f"{t:.6f}",
                "dt_actual_s": "0.005",
                "phase": "servo_twist",
                "pose_x": "0.198",
                "pose_y": "0.244",
                "pose_z": "0.261",
                "pose_rx": "0",
                "pose_ry": "0",
                "pose_rz": "0",
                "fx": "0",
                "fy": "0",
                "fz": "1.2",
                "tx": "0",
                "ty": "0",
                "tz": "0",
                "twist_achieved_vz": "0.001",
            }
        )
        cmd_rows.append(
            {
                "t_wall_s": f"{t:.6f}",
                "t_mono_s": f"{t:.6f}",
                "dt_actual_s": "0.005",
                "phase": "air_chirp",
                "v_cmd_vz": "0.001",
                "v_cmd_vx": "0",
                "v_cmd_vy": "0",
                "v_cmd_wx": "0",
                "v_cmd_wy": "0",
                "v_cmd_wz": "0",
            }
        )
    wa_t = np.array([float(r["t_wall_s"]) for r in old + new])
    segs = monotonic_time_segments(wa_t)
    assert len(segs) == 2
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cmd = _write(root / "cmd.csv", cmd_rows)
        wa = _write(root / "wa.csv", old + new)
        rows, report = merge_logs(cmd, wa)
        assert report["aligned"], report
        assert report["wa_clock_resets"] == 1
        xyz = pose6(rows)
        assert abs(float(np.median(xyz[:, 0])) - 0.198) < 1e-9
        assert abs(float(np.max(xyz[:, 0]) - np.min(xyz[:, 0]))) < 1e-9
    print("[OK] rewind clock uses latest Window A segment", flush=True)


def test_transverse_error_removes_tool_z_geometry() -> None:
    pose = np.zeros((8, 6), dtype=float)
    pose[:, 3] = 0.10  # initial tool-Z has a small world-Y component
    pose[:, 2] = np.linspace(0.0, 0.007, 8)
    pose[:, 1] = np.linspace(0.0, 0.0007, 8)  # pure tool-Z geometry
    pose[4:, 1] += 0.001 * np.arange(4)  # plus a real transverse drift
    twist = np.zeros((8, 6), dtype=float)
    twist[:, 2] = 0.001
    twist[:, 1] = 0.0002  # includes a small world-Y residual
    report = transverse_error_report(pose, twist, np.ones(8, dtype=bool))
    assert report["n"] == 8
    assert float(report["world_y_displacement_abs_m"]["p95"]) > 1e-3
    assert float(report["horizontal_velocity_abs_m_s"]["p95"]) > 0.0
    assert report["command_relative"]["n"] == 0
    print("[OK] transverse error removes tool-Z projection", flush=True)


def test_command_relative_transverse_error_uses_each_pose() -> None:
    n = 12
    dt = 0.01
    t = dt * np.arange(n)
    pose = np.zeros((n, 6), dtype=float)
    pose[:, 4] = np.linspace(0.0, 0.35, n)
    command = np.zeros((n, 6), dtype=float)
    command[:, 2] = 0.02  # tool-Z request; its base direction changes with ry
    rotation = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    command_base = np.einsum("nij,nj->ni", rotation, command[:, :3])
    actual = np.zeros((n, 6), dtype=float)
    actual[:, :3] = command_base
    actual[:, 1] += 0.003  # an unrequested world-Y drift
    for i in range(1, n):
        pose[i, :3] = pose[i - 1, :3] + 0.5 * (actual[i - 1, :3] + actual[i, :3]) * dt
    mask = np.ones(n, dtype=bool)
    direct = command_relative_transverse_error(pose, actual, command, t, mask)
    report = transverse_error_report(
        pose,
        actual,
        mask,
        t=t,
        command=command,
    )
    velocity_error = direct["velocity_error"]
    assert isinstance(velocity_error, np.ndarray)
    assert np.isfinite(velocity_error).all()
    assert np.isclose(np.percentile(np.abs(velocity_error[:, 1]), 95), 0.003, atol=1e-9)
    command_report = report["command_relative"]
    assert command_report["command_frame"] == "tool"
    assert np.isclose(command_report["world_y_velocity_abs_m_s"]["p95"], 0.003, atol=1e-9)
    print("[OK] command-relative transverse error uses per-row pose", flush=True)


def test_command_frame_selection_and_no_double_rotation() -> None:
    """Keep BASE Window-A commands separate from TOOL experiment commands."""
    n = 32
    dt = 0.01
    t = dt * np.arange(n)
    pose = np.zeros((n, 6), dtype=float)
    pose[:, 3] = np.linspace(0.15, 0.30, n)
    tool_command = np.zeros((n, 6), dtype=float)
    tool_command[:, 2] = 0.02
    rotation = rot_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    base_command = np.zeros((n, 6), dtype=float)
    base_command[:, :3] = np.einsum(
        "nij,nj->ni", rotation, tool_command[:, :3]
    )
    actual = base_command.copy()
    mask = np.ones(n, dtype=bool)

    # One transform is correct for a TOOL command; BASE data must bypass it.
    tool_report = command_relative_transverse_error(
        pose, actual, tool_command, t, mask, command_frame="tool"
    )
    base_report = command_relative_transverse_error(
        pose, actual, base_command, t, mask, command_frame="base"
    )
    double_report = command_relative_transverse_error(
        pose, actual, base_command, t, mask, command_frame="tool"
    )
    for report in (tool_report, base_report):
        velocity_error = report["velocity_error"]
        assert isinstance(velocity_error, np.ndarray)
        assert float(np.nanmax(np.abs(velocity_error[:, :3]))) < 1e-10
    velocity_error = double_report["velocity_error"]
    assert isinstance(velocity_error, np.ndarray)
    assert float(np.nanmax(np.abs(velocity_error[:, :3]))) > 1e-4

    # A merged row exposes both command namespaces and records the frame.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cmd_rows = []
        wa_rows = []
        for i in range(n):
            cmd_rows.append(
                {
                    "t_wall_s": f"{t[i]:.6f}",
                    "t_mono_s": f"{t[i]:.6f}",
                    "phase": "contact_chirp",
                    "v_cmd_vx": "0",
                    "v_cmd_vy": "0",
                    "v_cmd_vz": "0.020",
                    "v_cmd_wx": "0",
                    "v_cmd_wy": "0",
                    "v_cmd_wz": "0",
                }
            )
            wa_rows.append(
                {
                    "t_wall_s": f"{t[i] + 0.001:.6f}",
                    "pose_x": f"{pose[i, 0]:.6f}",
                    "pose_y": f"{pose[i, 1]:.6f}",
                    "pose_z": f"{pose[i, 2]:.6f}",
                    "pose_rx": f"{pose[i, 3]:.6f}",
                    "pose_ry": f"{pose[i, 4]:.6f}",
                    "pose_rz": f"{pose[i, 5]:.6f}",
                    "v_cmd_vx": f"{base_command[i, 0]:.6f}",
                    "v_cmd_vy": f"{base_command[i, 1]:.6f}",
                    "v_cmd_vz": f"{base_command[i, 2]:.6f}",
                }
            )
        cmd = _write(root / "cmd.csv", cmd_rows)
        wa = _write(root / "window_a.csv", wa_rows)
        rows, report = merge_logs(cmd, wa)
        assert report["command_frame"] == "tool"
        assert rows[0]["command_frame"] == "tool"
        assert rows[0]["command_v_cmd_vz"] == "0.020"
        assert rows[0]["v_cmd_vz"] == f"{base_command[0, 2]:.6f}"
        selected_external = twist_external_cmd6(rows)
        selected_base = twist_window_a_cmd6(rows)
        assert np.isclose(selected_external[0, 2], 0.020)
        assert np.isclose(selected_base[0, 2], base_command[0, 2], atol=1e-6)
    print("[OK] command frame provenance and double-rotation guard", flush=True)


def test_duplicate_timestamp_does_not_absorb_measured_drift() -> None:
    """A zero-time duplicate contributes no request displacement."""
    n = 4
    pose = np.zeros((n, 6), dtype=float)
    pose[:, 2] = np.arange(n, dtype=float) * 0.001
    command = np.zeros((n, 6), dtype=float)
    command[:, 2] = 0.01
    actual = command.copy()
    t = np.array([0.0, 0.1, 0.1, 0.2])
    result = command_relative_transverse_error(
        pose,
        actual,
        command,
        t,
        np.ones(n, dtype=bool),
        command_frame="base",
    )
    requested = result["requested_position"]
    assert isinstance(requested, np.ndarray)
    # Only the two positive 0.1 s intervals advance the requested path.
    assert np.isclose((requested[-1, 2] - requested[0, 2]), 0.002)
    print("[OK] duplicate timestamp does not absorb drift", flush=True)


def test_run_dirs_preserve_old_runs_in_temp() -> None:
    old_data, old_visu = paths.DATA, paths.VISU
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, visu_root = root / "DATA", root / "VISU"
            (data_root / "07_contact_gv").mkdir(parents=True)
            (visu_root / "07_contact_gv").mkdir(parents=True)
            (data_root / "07_contact_gv" / "old.csv").write_text("old\n", encoding="utf-8")
            paths.configure_roots(data=data_root, visu=visu_root)
            paths.set_active_run_id("r1")
            data1, visu1 = paths.kind_dirs("07_contact_gv")
            assert data1.name == "r1" and visu1.name == "r1"
            assert (data_root / "07_contact_gv").is_symlink()
            assert (data_root / "runs" / "07_contact_gv" / "legacy_r1" / "old.csv").is_file()
            paths.set_active_run_id("r2")
            data2, _visu2 = paths.kind_dirs("07_contact_gv")
            assert data2.name == "r2"
            assert (data_root / "runs" / "07_contact_gv" / "r1").is_dir()
            assert (data_root / "07_contact_gv").resolve() == data2.resolve()
            # A second collect with the same explicit id gets a fresh suffix;
            # only analysis of a managed CSV is allowed to reuse an id.
            paths.set_active_run_id("r2")
            data3, _visu3 = paths.kind_dirs("07_contact_gv")
            assert data3.name == "r2_1"
            assert (data_root / "runs" / "07_contact_gv" / "r2").is_dir()
            assert (data_root / "07_contact_gv").resolve() == data3.resolve()

            legacy_kind = "08_ke"
            legacy = data_root / legacy_kind
            legacy.mkdir(parents=True)
            source = legacy / "press.csv"
            source.write_text("legacy\n", encoding="utf-8")
            paths.set_active_run_id("analysis")
            analyzed, _ = paths.kind_dirs(legacy_kind, preserve=source)
            assert source.is_file()
            assert analyzed.parent == data_root / "runs" / legacy_kind
            assert legacy.is_dir() and not legacy.is_symlink()
            # An external/legacy CSV must not reuse an occupied analysis id.
            paths.set_active_run_id("analysis")
            analyzed_again, _ = paths.kind_dirs(legacy_kind, preserve=source)
            assert analyzed_again.name == "analysis_1"
            assert analyzed_again != analyzed

            index_legacy = visu_root / "06_axis"
            index_legacy.mkdir(parents=True)
            (index_legacy / "README.md").write_text("old\n", encoding="utf-8")
            paths.set_active_run_id("index")
            offline_index = paths.run_visu_dir("06_axis", collect=False)
            assert index_legacy.is_dir() and not index_legacy.is_symlink()
            assert offline_index.is_dir()
            collected_index = paths.run_visu_dir("06_axis", collect=True)
            assert index_legacy.is_symlink()
            assert collected_index.is_dir()
            assert collected_index.name == "index_1"
    finally:
        paths.configure_roots(data=old_data, visu=old_visu)
    print("[OK] run-id directories preserve old runs", flush=True)


def test_dof_session_handshake() -> None:
    import dof
    from goto_mid import stop_for_dof_boundary

    class FakeArm:
        controller_api_version = "fake"

        def __init__(self) -> None:
            self.active = 8
            self.calls = []

        def set_dof(self, value, *, after_current=True, block=0):
            self.calls.append((value, after_current, block))
            self.active = int(value)
            return 0

        def get_dof(self):
            return 0, self.active

    class CurrentFacade:
        """Legacy facade: set_dof(dof, *, block=1), without after_current."""

        def __init__(self) -> None:
            self.active = 8
            self.calls = []

        def set_dof(self, value, *, block=0):
            self.calls.append((value, block))
            self.active = int(value)
            return 0

        def get_dof(self):
            return 0, self.active

    class PendingServo:
        """A continuous SERVO_TWIST boundary that times out explicitly."""

        mode = "SERVO_TWIST"

        def set_dof(self, value, *, after_current=True, block=0):
            self.request = (value, after_current, block)
            return -5  # ERR_TIMEOUT; no boundary/ACK was observed

        def get_dof(self):
            return 0, 8

    class StaleActive:
        """Setter returns OK but the boundary has not committed yet."""

        def set_dof(self, value, *, after_current=True, block=0):
            return 0

        def get_dof(self):
            return 0, 8

    class FailedRead:
        def set_dof(self, value, *, after_current=True, block=0):
            return 0

        def get_dof(self):
            return -5, 8

    class NonFiniteRead:
        def set_dof(self, value, *, after_current=True, block=0):
            return 0

        def get_dof(self):
            return 0, float("nan")

    class BoundaryStop:
        def __init__(self, result=0):
            self.events = []
            self.result = result

        def set_arm_stop(self):
            self.events.append("stop")
            return self.result

    previous = dict(dof._LAST)
    try:
        arm = FakeArm()
        dof.set_session_dof(arm, 8)
        dof.set_session_dof(arm, 8)
        assert arm.calls == [(8, True, 1)]
        assert dof.current_metadata()["dof_active"] == 8
        current = CurrentFacade()
        dof.set_session_dof(current, 7)
        assert current.calls == [(7, 1)]
        assert dof.current_metadata()["dof_active"] == 7

        boundary = BoundaryStop()
        assert stop_for_dof_boundary(boundary) == 0
        assert boundary.events == ["stop"]
        failed_boundary = BoundaryStop(-5)
        try:
            stop_for_dof_boundary(failed_boundary)
        except RuntimeError:
            pass
        else:
            raise AssertionError("DOF initialization must report a failed STOP boundary")

        for facade, label in (
            (PendingServo(), "continuous SERVO_TWIST boundary timeout"),
            (StaleActive(), "uncommitted active DOF"),
            (FailedRead(), "get_dof status"),
            (NonFiniteRead(), "non-finite active DOF"),
        ):
            dof._LAST = {
                "dof_requested": 8,
                "dof_active": float("nan"),
                "dof_config_source": "test",
                "controller_api_version": "test",
            }
            try:
                dof.set_session_dof(facade, 7)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{label} must not be accepted without an ACK")
    finally:
        dof._LAST = previous
    print("[OK] session DOF handshake/inheritance + current facade", flush=True)


def test_shared_mode_install_handshake() -> None:
    """A mailbox ACK must not authorize a logger tick before install_seq."""

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += float(seconds)

    class FakeMode:
        name = "SERVO_TWIST"

        def __int__(self) -> int:
            return 1

    class FakeRequest:
        mode = FakeMode()

    class DelayedInstall:
        def __init__(self) -> None:
            self.snapshots = [
                {"ack_seq": 41, "install_seq": 0, "status": 1, "mode": 0},
                {"ack_seq": 41, "install_seq": 41, "status": 1, "mode": 1},
            ]
            self.set_calls = 0
            self.snapshot_calls = 0

        def set_mode(self, request) -> int:
            self.set_calls += 1
            assert request.mode is not None
            return 41

        def snapshot(self) -> dict:
            self.snapshot_calls += 1
            return self.snapshots.pop(0)

    clock = FakeClock()
    delayed = DelayedInstall()
    ack = wait_for_mode_install(
        delayed,
        FakeRequest(),
        timeout_s=0.2,
        poll_s=0.01,
        clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    assert ack.request_seq == 41
    assert ack.install_seq == 41
    assert delayed.set_calls == 1
    assert delayed.snapshot_calls == 2, "ack_seq alone must not authorize the first tick"

    class FaultClient:
        def __init__(self, snapshot: dict) -> None:
            self._snapshot = snapshot

        def set_mode(self, request) -> int:
            return 9

        def snapshot(self) -> dict:
            return dict(self._snapshot)

    for snapshot, label in (
        ({"ack_seq": 10, "install_seq": 10, "status": 1, "mode": 1}, "superseded"),
        ({"ack_seq": 9, "status": 1, "mode": 1}, "missing install_seq"),
        ({"ack_seq": 9, "install_seq": -1, "status": 1, "mode": 1}, "negative install_seq"),
        ({"ack_seq": 9, "install_seq": 9, "status": 1}, "missing live mode"),
        ({"ack_seq": 9, "install_seq": 9, "mode": 1}, "missing status"),
        ({"install_seq": 9, "status": 3, "mode": 1}, "controller error without ack"),
        ({"ack_seq": 9, "install_seq": 9, "status": 1, "mode": 1, "t_mono": -2.0}, "stale snapshot"),
    ):
        try:
            wait_for_mode_install(FaultClient(snapshot), FakeRequest(), timeout_s=0.05)
        except ModeInstallError as exc:
            detail = str(exc)
            assert label in detail or "install_seq" in detail or "stale" in detail
        else:
            raise AssertionError(f"{label} must fail before a tick is authorized")

    timeout_clock = FakeClock()
    try:
        wait_for_mode_install(
            FaultClient({"ack_seq": 9, "install_seq": 0, "status": 1, "mode": 1}),
            FakeRequest(),
            timeout_s=0.025,
            poll_s=0.01,
            clock=timeout_clock.monotonic,
            sleeper=timeout_clock.sleep,
        )
    except ModeInstallTimeout as exc:
        assert "request_seq=9" in str(exc)
        assert "install_seq=0" in str(exc)
    else:
        raise AssertionError("missing install progress must time out clearly")
    print("[OK] shared mode install ACK handshake + faults", flush=True)


def test_run_provenance_captures_dirty_source_and_config() -> None:
    """A run keeps dirty source/config bytes even after the checkout changes."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = root / "fake_repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "ICRA test"], check=True)
        source = repo / "controller.py"
        source.write_text("VERSION = 1\n", encoding="utf-8")
        config = repo / "controller.yaml"
        config.write_text("gain: 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "controller.py", "controller.yaml"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
        source.write_text("VERSION = 2\n", encoding="utf-8")
        untracked = repo / "new_task.py"
        untracked.write_text("TASK = 'new'\n", encoding="utf-8")

        data = root / "run"
        manifest = paths.capture_run_provenance(
            data,
            kind="07_contact_gv",
            run_id="provenance-test",
            repo_roots={"fake": repo},
            config_paths=[config],
            argv=["07_contact_gv.py", "--dof", "8", "--config", str(config)],
        )
        repo_meta = manifest["repositories"]["fake"]
        assert len(str(repo_meta["head"])) == 40
        assert repo_meta["dirty"]
        source_files = {item["path"]: item for item in repo_meta["source_files"]}
        assert {"controller.py", "new_task.py"}.issubset(source_files)
        source_snapshot = data / "source_snapshot" / source_files["controller.py"]["snapshot"]
        assert source_snapshot.read_text(encoding="utf-8") == "VERSION = 2\n"
        config_meta = next(item for item in manifest["configs"] if item["path"] == str(config.resolve()))
        config_snapshot = data / "source_snapshot" / config_meta["snapshot"]
        config.write_text("gain: 99\n", encoding="utf-8")
        assert config_snapshot.read_text(encoding="utf-8") == "gain: 1\n"
        saved = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
        assert saved["command"]["argv"][2] == "8"
        assert paths.capture_run_provenance(
            data,
            kind="07_contact_gv",
            run_id="provenance-test",
            repo_roots={"fake": repo},
            config_paths=[config],
            argv=["mutated"],
        )["command"]["argv"][0] == "07_contact_gv.py"
    print("[OK] immutable dirty-source/config run provenance", flush=True)


def test_dry_scripts() -> None:
    import subprocess

    here = Path(__file__).resolve().parent
    scripts = [
        "01_system_delay.py", "02_step_response.py", "03_chirp_gv.py",
        "04_timing_jitter.py", "05_tracking_error.py", "06_axis_gv.py",
        "07_contact_gv.py", "08_env_ke.py", "09_multisine_gv.py",
        "10_tn_observe.py", "11_exec_2dof.py", "12_stop_tail.py",
        "13_contact_hs.py", "14_port_energy.py", "15_holdout.py",
    ]
    for dof in (7, 8):
        for name in scripts:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(here / name),
                    "--dry-run",
                    "--dof",
                    str(dof),
                    "--run-id",
                    f"test_dry_{dof}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise SystemExit(f"{name} --dof {dof} dry-run failed:\n{proc.stdout}\n{proc.stderr}")
            if "[DRY]" not in proc.stdout:
                raise SystemExit(f"{name} --dof {dof} dry-run printed no [DRY]")
    print("[OK] dry-run 01–15 --dof 7/8", flush=True)


if __name__ == "__main__":
    test_alignment_and_ke()
    test_shared_window_a_fields_backward_compat()
    test_rewind_clock_uses_latest_segment()
    test_transverse_error_removes_tool_z_geometry()
    test_command_relative_transverse_error_uses_each_pose()
    test_command_frame_selection_and_no_double_rotation()
    test_duplicate_timestamp_does_not_absorb_measured_drift()
    test_run_dirs_preserve_old_runs_in_temp()
    test_dof_session_handshake()
    test_shared_mode_install_handshake()
    test_run_provenance_captures_dirty_source_and_config()
    test_dry_scripts()
