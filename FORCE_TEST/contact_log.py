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

from id_math import disp_chirp_velocity
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
        secondary: str = "payload_id",
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
        self.last_vz = float("nan")
        self.x_proxy = 0.0
        self.last_twist = np.zeros(6, dtype=float)
        self.aborted = False
        self.unloaded = False
        self._below_min_from: float | None = None
        self.secondary = str(secondary).strip() or "payload_id"

    def reset_proxy(self) -> None:
        """Zero the live tool-Z integral.  Call at each matched-state settle."""

        self.x_proxy = 0.0

    def start_twist(self) -> None:
        from peirastic.core.modes import Mode, ModeRequest

        self.client.set_mode(
            ModeRequest(
                Mode.SERVO_TWIST,
                {"filter": False, "secondary": self.secondary},
            )
        )
        print(
            f"[MODE] SERVO_TWIST  filter OFF  secondary={self.secondary}  "
            f"6-D cmd  force loop OFF  log={self.log_csv}",
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
        min_n: float | None = None,
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
        vz = float(row_m.get("v_tcp_z", float("nan")))
        self.last_vz = vz
        if math.isfinite(vz) and math.isfinite(dt_act):
            self.x_proxy += vz * float(dt_act)
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
        if min_n is not None:
            if math.isfinite(fz) and fz < float(min_n):
                now = time.monotonic()
                if self._below_min_from is None:
                    self._below_min_from = now
                elif now - self._below_min_from >= 0.25:
                    self.unloaded = True
                    print(
                        f"[UNLOAD] Fz={fz:.2f} N < {float(min_n):.2f} N for 0.25s",
                        flush=True,
                    )
                    return False
            else:
                self._below_min_from = None
        return True

    def hold(
        self,
        twist: np.ndarray | float,
        seconds: float,
        phase: str,
        *,
        axis: int | None = None,
        check_abort: bool = True,
        min_n: float | None = None,
    ) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(
                twist, phase, check_abort=check_abort, axis=axis, min_n=min_n
            ):
                return False
            time.sleep(self.dt)
        return True

    def play_until(
        self,
        twist_fn,
        phase: str,
        pred,
        seconds: float,
        *,
        check_abort: bool = True,
        min_n: float | None = None,
    ) -> bool:
        """Send twist_fn(self) until pred(self) or timeout.  True = predicate hit."""

        t0 = time.monotonic()
        while time.monotonic() - t0 < float(seconds):
            if not self.tick(
                twist_fn(self), phase, check_abort=check_abort, min_n=min_n
            ):
                return False
            if pred(self):
                return True
            time.sleep(self.dt)
        return False

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

    def chirp_disp_axis(
        self,
        axis: int,
        ax: float,
        f0: float,
        f1: float,
        seconds: float,
        phase: str,
        *,
        min_n: float | None = None,
        onesided: bool = False,
    ) -> bool:
        """Displacement-limited chirp.  onesided keeps x ≥ the start pose."""

        self._below_min_from = None
        t0 = time.monotonic()
        T = float(seconds)
        while True:
            t = time.monotonic() - t0
            if t >= T:
                break
            vel = float(
                disp_chirp_velocity(
                    np.asarray([t]), ax, f0, f1, T, onesided=onesided
                )[0]
            )
            if not self.tick(axis_twist(axis, vel), phase, min_n=min_n):
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
        seconds: float = 25.0,
        phase: str = "seek",
        max_travel_m: float | None = None,
    ) -> bool:
        """Press +tool-Z until Fz exceeds contact_n, or time/travel runs out."""

        t0 = time.monotonic()
        travel = 0.0
        last_report = t0
        vel = float(vel_m_s)
        cap = None if max_travel_m is None else abs(float(max_travel_m))
        cap_txt = f"{1e3 * cap:.0f} mm cap" if cap is not None else (
            f"{1e3 * abs(vel) * float(seconds):.0f} mm by time"
        )
        print(
            f"[SEEK] +tool-Z  {1e3 * vel:.1f} mm/s  up to {seconds:.1f} s "
            f"({cap_txt})  until Fz>{contact_n:.2f} N",
            flush=True,
        )
        while time.monotonic() - t0 < float(seconds):
            if cap is not None and travel >= cap:
                break
            if not self.tick(axis_twist(2, vel), phase):
                return False
            travel += abs(vel) * self.dt
            now = time.monotonic()
            if now - last_report >= 1.0:
                fz = self.last_fz
                fz_txt = f"{fz:.2f} N" if math.isfinite(fz) else "nan"
                print(
                    f"[SEEK] t={now - t0:.1f}s  travel={1e3 * travel:.1f} mm  "
                    f"Fz={fz_txt}  need>{contact_n:.2f} N",
                    flush=True,
                )
                last_report = now
            if math.isfinite(self.last_fz) and self.last_fz > float(contact_n):
                print(
                    f"[CONTACT] Fz={self.last_fz:.2f} N  after {now - t0:.1f}s  "
                    f"travel={1e3 * travel:.1f} mm",
                    flush=True,
                )
                return True
            time.sleep(self.dt)
        fz = self.last_fz
        fz_txt = f"{fz:.2f} N" if math.isfinite(fz) else "nan (no force snapshot)"
        print(
            f"[ERR] no contact after {seconds:.1f}s  commanded travel={1e3 * travel:.1f} mm  "
            f"last Fz={fz_txt}  threshold={contact_n:.2f} N"
            + (
                f"  — stop, do not skate further. Raise the phantom so mid-stroke "
                f"TCP is 10–25 mm above the pad."
                if cap is not None
                else ""
            ),
            flush=True,
        )
        return False

    def hold_quiet(
        self,
        min_s: float,
        max_s: float,
        phase: str,
        *,
        eps_dfdt: float = 0.80,
        quiet_s: float = 0.40,
    ) -> bool:
        """Hold u=0 until |Ḟ| stays below eps for quiet_s, after at least min_s.

        Preload / settle only.  Identification phases stay open-loop twist.
        """

        hist: list[tuple[float, float]] = []
        t0 = time.monotonic()
        quiet_from: float | None = None
        while time.monotonic() - t0 < float(max_s):
            if not self.tick(0.0, phase, check_abort=False):
                return False
            now = time.monotonic()
            if math.isfinite(self.last_fz):
                hist.append((now, float(self.last_fz)))
                hist = [(ts, ff) for ts, ff in hist if now - ts <= 0.25]
                if len(hist) >= 4:
                    span = hist[-1][0] - hist[0][0]
                    if span > 0.06:
                        dfdt = abs((hist[-1][1] - hist[0][1]) / span)
                        if dfdt < float(eps_dfdt):
                            if quiet_from is None:
                                quiet_from = now
                            if now - quiet_from >= float(quiet_s) and now - t0 >= float(min_s):
                                print(
                                    f"[QUIET] |dF/dt|={dfdt:.2f} N/s for {quiet_s:.2f}s  "
                                    f"F={self.last_fz:.2f} N",
                                    flush=True,
                                )
                                return True
                        else:
                            quiet_from = None
            time.sleep(self.dt)
        print(
            f"[QUIET] timed out after {max_s:.1f}s  Fz={self.last_fz:.2f} N — continuing",
            flush=True,
        )
        return True

    def seek_force(
        self,
        target_n: float,
        *,
        vel_m_s: float = 0.006,
        seconds: float = 8.0,
        band_n: float = 0.15,
        phase: str = "seek_f",
    ) -> bool:
        """Press or retract until |F − target| ≤ band (live Fz only)."""

        t0 = time.monotonic()
        target = float(target_n)
        band = abs(float(band_n))
        while time.monotonic() - t0 < float(seconds):
            fz = self.last_fz
            if math.isfinite(fz) and abs(fz - target) <= band:
                print(f"[FORCE] Fz={fz:.2f} N ~ {target:.2f} N", flush=True)
                return True
            if math.isfinite(fz) and fz > target:
                vel = -abs(float(vel_m_s))
            else:
                vel = abs(float(vel_m_s))
            if not self.tick(axis_twist(2, vel), phase, check_abort=fz <= target if math.isfinite(fz) else True):
                return False
            time.sleep(self.dt)
        print(f"[ERR] seek_force timed out  Fz={self.last_fz:.2f} N  target={target:.2f}", flush=True)
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
