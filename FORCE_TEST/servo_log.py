"""SERVO_TWIST logger.  Force loop stays off.  Window A only servos."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

from mode_handshake import install_servo_twist, mode_provenance, ModeInstallAck

FIELDS = (
    "t_wall_s",
    "t_mono_s",
    "dt_actual_s",
    "phase",
    "command_frame",
    "mode_name",
    "mode_request_seq",
    "mode_install_seq",
    "mode_install_status",
    "axis",
    "vel_ff",
    "vel_ff_vz",
    "v_cmd_z",
    "v_ach",
    "vz_achieved_tool",
    "fz",
    "feedback_age_s",
    "sensor_age_s",
    "a_tcp_z_plus",
    "motion_seq",
    "run_id",
    "dof_requested",
    "dof_active",
    "dof_config_source",
    "controller_api_version",
)


def _fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


class ServoLogger:
    """Write the command we send + MotionBus achieved velocity + Fz snapshot."""

    def __init__(
        self,
        *,
        prefix: str,
        hz: float,
        log_csv: Path,
        axis: int = 2,
        abort_n: float | None = None,
    ) -> None:
        from peirastic.core.ipc import CommandClient, MotionBus, TwistBus

        self.hz = float(hz)
        self.dt = 1.0 / max(self.hz, 1.0)
        self.axis = int(axis)
        self.abort_n = abort_n
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
        self.aborted = False
        self.mode_install: ModeInstallAck | None = None
        self._mode_ready = False
        from dof import current_metadata
        from paths import active_run_id

        self.run_id = active_run_id()
        self.dof_meta = current_metadata()

    def start_twist(self) -> None:
        self._mode_ready = False
        self.mode_install = install_servo_twist(self.client, filter_enabled=False)
        self._mode_ready = True
        print(
            f"[MODE] SERVO_TWIST  filter OFF  dof={self.dof_meta.get('dof_active')}  "
            f"axis={self.axis}  force loop OFF  request_seq={self.mode_install.request_seq}  "
            f"install_seq={self.mode_install.install_seq}  log={self.log_csv}",
            flush=True,
        )

    def _require_mode_ready(self) -> None:
        if not self._mode_ready or self.mode_install is None:
            raise RuntimeError(
                "SERVO_TWIST mode is not installed; call start_twist() and wait for "
                "the install ACK before sending or recording a tick"
            )

    def zero(self) -> None:
        self._require_mode_ready()
        self.bus.write(np.zeros(6, dtype=float), hz=self.hz, connected=True)

    def tick(self, vel_m_s: float, phase: str, *, check_abort: bool = True) -> bool:
        from peirastic.core.ipc import Status

        self._require_mode_ready()
        tw = np.zeros(6, dtype=float)
        tw[self.axis] = float(vel_m_s)
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
        y = float(row_m.get("v_tcp_z", float("nan")))
        if self.axis != 2:
            # MotionBus currently publishes tool-Z achieved.  For X/Y the
            # Window A CSV (twist_achieved_*) is the right pair; this column
            # stays Z so a later Window A log can be merged.
            y = float(row_m.get("v_tcp_z", float("nan")))
        age = float(row_m.get("feedback_age_s", float("nan")))
        self.writer.writerow(
            {
                "t_wall_s": _fmt(t_wall),
                "t_mono_s": _fmt(time.monotonic()),
                "dt_actual_s": _fmt(dt_act),
                "phase": phase,
                "command_frame": "tool",
                **mode_provenance(self.mode_install),
                "axis": str(self.axis),
                "vel_ff": _fmt(vel_m_s),
                "vel_ff_vz": _fmt(vel_m_s if self.axis == 2 else float("nan")),
                "v_cmd_z": _fmt(vel_m_s if self.axis == 2 else float("nan")),
                "v_ach": _fmt(y),
                "vz_achieved_tool": _fmt(y),
                "fz": _fmt(fz),
                "feedback_age_s": _fmt(age),
                "sensor_age_s": _fmt(age),
                "a_tcp_z_plus": _fmt(float(row_m.get("a_tcp_z_plus", float("nan")))),
                "motion_seq": str(int(row_m.get("seq", 0))),
                "run_id": self.run_id,
                "dof_requested": self.dof_meta.get("dof_requested"),
                "dof_active": self.dof_meta.get("dof_active"),
                "dof_config_source": self.dof_meta.get("dof_config_source", ""),
                "controller_api_version": self.dof_meta.get("controller_api_version", ""),
            }
        )
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

    def retract_z(
        self,
        vel_m_s: float,
        seconds: float,
        below_n: float,
        phase: str = "retract",
    ) -> bool:
        """Leave contact along tool −Z.  Do not trip abort_n (F is still high)."""
        t0 = time.monotonic()
        speed = -abs(float(vel_m_s))
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(speed, phase, check_abort=False):
                return False
            if math.isfinite(self.last_fz) and self.last_fz < float(below_n):
                print(f"[RETRACT] Fz={self.last_fz:.2f} N < {below_n:.2f} N", flush=True)
                return True
            time.sleep(self.dt)
        print(f"[RETRACT] timed out  Fz={self.last_fz:.2f} N", flush=True)
        return True

    def hold(self, vel_m_s: float, seconds: float, phase: str) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(vel_m_s, phase):
                return False
            time.sleep(self.dt)
        return True

    def chirp(
        self,
        amp_m_s: float,
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
            if not self.tick(float(amp_m_s) * math.sin(ang), phase):
                return False
            time.sleep(self.dt)
        return True

    def play(self, vel: np.ndarray, phase: str) -> bool:
        for vi in vel:
            if not self.tick(float(vi), phase):
                return False
            time.sleep(self.dt)
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
