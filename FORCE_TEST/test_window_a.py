#!/usr/bin/env python3
"""Offline checks for the Window A merge and pose-Ke contract.  No hardware."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_csv import write_csv
from window_a import (
    AlignmentError,
    alignment_report,
    merge_logs,
    tool_z_displacement,
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
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cmd = _write(root / "cmd.csv", cmd_rows)
        wa = _write(root / "window_a.csv", wa_rows)
        rows, report = merge_logs(cmd, wa)
        assert report["aligned"], report
        assert report["torque_present"]
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


def test_dry_scripts() -> None:
    import subprocess

    here = Path(__file__).resolve().parent
    scripts = [
        "07_contact_gv.py",
        "08_env_ke.py",
        "11_exec_2dof.py",
        "12_stop_tail.py",
        "13_contact_hs.py",
        "14_port_energy.py",
    ]
    for name in scripts:
        proc = subprocess.run(
            [sys.executable, str(here / name), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"{name} dry-run failed:\n{proc.stdout}\n{proc.stderr}")
        if "[DRY]" not in proc.stdout:
            raise SystemExit(f"{name} dry-run printed no [DRY]")
    print("[OK] dry-run 07/08/11–14", flush=True)


if __name__ == "__main__":
    test_alignment_and_ke()
    test_dry_scripts()
