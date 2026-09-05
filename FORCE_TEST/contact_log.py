"""6-D SERVO_TWIST logger.  Force loop stays off.

Writes the command we sent.  Achieved twist, pose, and wrench come from
Window A --log-csv; copy that file into DATA/<kind>/window_a.csv.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

from window_a import TWIST_LETTERS

FIELDS = (
    "t_wall_s",
    "t_mono_s",
    "dt_actual_s",
    "phase",
    "v_cmd_vx",
    "v_cmd_vy",
    "v_cmd_vz",
    "v_cmd_wx",
    "v_cmd_wy",
    "v_cmd_wz",
    "fz",
    "feedback_age_s",
    "motion_seq",
)

# Image-plane tilt: e_θ = e_l × e_z.  With image in tool XZ, that is tool −Y.
THETA_AXIS = 4
SCAN_AXIS = 0


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def axis_twist(axis: int, value: float) -> np.ndarray:
    tw = np.zeros(6, dtype=float)
    tw[int(axis)] = float(value)
    return tw


class ContactLogger:
    """Send a 6-D twist and log the command plus the Fz snapshot."""

    def __init__(
        self,
        *,
        prefix: str,
        hz: float,
        log_csv: Path,
        abort_n: float | None = None,
        theta_axis: int = THETA_AXIS,
        scan_axis: int = SCAN_AXIS,
    ) -> None:
        from peirastic.core.ipc import CommandClient, MotionBus, TwistBus

        self.hz = float(hz)
        self.dt = 1.0 / max(self.hz, 1.0)
        self.abort_n = abort_n
        self.theta_axis = int(theta_axis)
        self.scan_axis = int(scan_axis)
        self.log_csv = Path(log_csv)
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        self.client = CommandClient(prefix=prefix)
        self.bus = TwistBus(prefix=prefix, create=False)
        try:
            self.motion = MotionBus(prefix=prefix, create=False)
        except Exception as exc:
            self.bus.close()
            self.client.close()
            raise RuntimeError(
                f"MotionBus missing ({exc}). Window A must be running this build."
            ) from exc
        self.handle = self.log_csv.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(FIELDS))
        self.writer.writeheader()
        self.last_wall = float("nan")
        self.n_rows = 0
        self.last_fz = float("nan")
        self.last_twist = np.zeros(6, dtype=float)
        self.aborted = False

    def start_twist(self) -> None:
        from peirastic.core.modes import Mode, ModeRequest

        self.client.set_mode(ModeRequest(Mode.SERVO_TWIST, {"filter": False}))
        print(
            f"[MODE] SERVO_TWIST  filter OFF  6-D cmd  force loop OFF  "
            f"log={self.log_csv}",
            flush=True,
        )

    def zero(self) -> None:
        self.bus.write(np.zeros(6, dtype=float), hz=self.hz, connected=True)

    def tick(
        self,
        twist: np.ndarray | float,
        phase: str,
        *,
        check_abort: bool = True,
        axis: int | None = None,
    ) -> bool:
        from peirastic.core.ipc import Status

        if np.isscalar(twist):
            tw = axis_twist(2 if axis is None else int(axis), float(twist))
        else:
            tw = np.asarray(twist, dtype=float).reshape(-1)
            if tw.size != 6:
                raise ValueError(f"twist must be 6-D, got {tw.size}")
        self.last_twist = tw.copy()
        self.bus.write(tw, hz=self.hz, connected=True)
        tel = self.client.snapshot()
        if int(tel["status"]) == int(Status.ESTOP):
            print("[ESTOP] " + str(tel["msg"]), flush=True)
            return False
        row_m = self.motion.read()
        t_wall = float(row_m.get("t_wall_s", float("nan")))
        if not math.isfinite(t_wall):
            t_wall = time.time()
        dt_act = (
            max(t_wall - self.last_wall, 1e-6)
            if math.isfinite(self.last_wall)
            else self.dt
        )
        self.last_wall = t_wall
        fz = float(tel.get("f_ext_z", float("nan")))
        self.last_fz = fz
        rec = {
            "t_wall_s": _fmt(t_wall),
            "t_mono_s": _fmt(time.monotonic()),
            "dt_actual_s": _fmt(dt_act),
            "phase": phase,
            "fz": _fmt(fz),
            "feedback_age_s": _fmt(float(row_m.get("feedback_age_s", float("nan")))),
            "motion_seq": str(int(row_m.get("seq", 0))),
        }
        for letter, value in zip(TWIST_LETTERS, tw):
            rec[f"v_cmd_{letter}"] = _fmt(value)
        self.writer.writerow(rec)
        self.n_rows += 1
        if self.n_rows % 200 == 0:
            self.handle.flush()
        if (
            check_abort
            and self.abort_n is not None
            and math.isfinite(fz)
            and fz >= float(self.abort_n)
        ):
            self.aborted = True
            print(f"[ABORT] Fz={fz:.2f} N ≥ {self.abort_n:.1f} N", flush=True)
            return False
        return True

    def hold(
        self,
        twist: np.ndarray | float,
        seconds: float,
        phase: str,
        *,
        axis: int | None = None,
        check_abort: bool = True,
    ) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(twist, phase, check_abort=check_abort, axis=axis):
                return False
            time.sleep(self.dt)
        return True

    def chirp_axis(
        self,
        axis: int,
        amp: float,
        f0: float,
        f1: float,
        seconds: float,
        phase: str,
    ) -> bool:
        f0 = max(float(f0), 1e-3)
        f1 = max(float(f1), f0 + 1e-3)
        k = math.log(f1 / f0) / max(float(seconds), 1e-6)
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= float(seconds):
                break
            ang = 2.0 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
            if not self.tick(axis_twist(axis, float(amp) * math.sin(ang)), phase):
                return False
            time.sleep(self.dt)
        return True

    def play(self, twists: np.ndarray, phase: str) -> bool:
        seq = np.asarray(twists, dtype=float)
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        for row in seq:
            if row.size == 1:
                tw = axis_twist(2, float(row[0]))
            elif row.size == 6:
                tw = row
            else:
                raise ValueError("play() wants N or N×6")
            if not self.tick(tw, phase):
                return False
            time.sleep(self.dt)
        return True

    def seek_contact(
        self,
        vel_m_s: float,
        *,
        contact_n: float,
        seconds: float = 12.0,
        phase: str = "seek",
    ) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(axis_twist(2, float(vel_m_s)), phase):
                return False
            if math.isfinite(self.last_fz) and self.last_fz > float(contact_n):
                print(f"[CONTACT] Fz={self.last_fz:.2f} N", flush=True)
                return True
            time.sleep(self.dt)
        print("[ERR] no contact", flush=True)
        return False

    def retract_z(
        self,
        vel_m_s: float,
        seconds: float,
        below_n: float,
        phase: str = "retract",
    ) -> bool:
        t0 = time.monotonic()
        speed = -abs(float(vel_m_s))
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(axis_twist(2, speed), phase, check_abort=False):
                return False
            if math.isfinite(self.last_fz) and self.last_fz < float(below_n):
                print(f"[RETRACT] Fz={self.last_fz:.2f} N < {below_n:.2f} N", flush=True)
                return True
            time.sleep(self.dt)
        print(f"[RETRACT] timed out  Fz={self.last_fz:.2f} N", flush=True)
        return True

    def close(self) -> None:
        try:
            self.zero()
        except Exception:
            pass
        self.handle.close()
        self.bus.close()
        self.client.close()
        self.motion.close()
