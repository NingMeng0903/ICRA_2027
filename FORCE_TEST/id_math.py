"""Pure ID helpers.  No hardware, no DATA/VISU writes.

Used by 07–15 so excitation, FOPDT prediction, stiffness envelopes,
residual tubes, and prefix energy bounds can be unit-tested offline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Joint 2-DoF lines.  No shared frequency so MIMO H1 is not rank-deficient.
Z_FREQS_HZ = (0.5, 1.3, 2.1, 2.9)
TH_FREQS_HZ = (0.8, 1.6, 2.4, 3.2)
# Independent scan perturbations for 13 (also disjoint from each other).
DZ_FREQS_HZ = (0.7, 1.9)
DTH_FREQS_HZ = (1.1, 2.3)


def exp_chirp_omega_phi(t: np.ndarray, f0: float, f1: float, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous ω(t), φ(t) of a log-frequency chirp."""

    f0 = max(float(f0), 1e-3)
    f1 = max(float(f1), f0 + 1e-3)
    t = np.asarray(t, dtype=float)
    k = math.log(f1 / f0) / max(float(seconds), 1e-6)
    omega = 2.0 * math.pi * f0 * np.exp(k * t)
    phi = 2.0 * math.pi * f0 * (np.exp(k * t) - 1.0) / k
    return omega, phi


def disp_chirp_velocity(
    t: np.ndarray,
    ax_m: float,
    f0: float,
    f1: float,
    seconds: float,
    *,
    onesided: bool = False,
) -> np.ndarray:
    """Displacement-limited chirp velocity.

    Two-sided: x = Ax sin φ ∈ [−Ax, Ax], v = Ax ω cos φ.
    One-sided (contact): x = Ax (1 − cos φ) ∈ [0, 2Ax], v = Ax ω sin φ.
    One-sided never commands a retract past the pose where the chirp started.
    """

    omega, phi = exp_chirp_omega_phi(t, f0, f1, seconds)
    ax = float(ax_m)
    if onesided:
        return ax * omega * np.sin(phi)
    return ax * omega * np.cos(phi)


def disp_multisine(
    dt: float,
    seconds: float,
    ax: float,
    freqs_hz: tuple[float, ...] | list[float],
    *,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean periodic v whose integrated x is bounded by Ax and returns near 0."""

    dt = float(dt)
    n = max(int(round(float(seconds) / dt)), 2)
    t = np.arange(n, dtype=float) * dt
    rng = np.random.default_rng(int(seed))
    x = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    for f in freqs_hz:
        f = float(f)
        if f <= 0.0:
            continue
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        x += np.sin(2.0 * math.pi * f * t + phi)
        v += 2.0 * math.pi * f * np.cos(2.0 * math.pi * f * t + phi)
    peak = float(np.max(np.abs(x))) or 1.0
    scale = float(ax) / peak
    return v * scale, x * scale


def freq_split_disp_multisine(
    dt: float,
    seconds: float,
    ax_z: float,
    ax_th: float,
    *,
    theta_axis: int = 4,
    seed: int = 1,
    z_freqs: tuple[float, ...] = Z_FREQS_HZ,
    th_freqs: tuple[float, ...] = TH_FREQS_HZ,
) -> np.ndarray:
    """N×6 twist.  uz and ωθ occupy disjoint lines."""

    z_set = {round(float(f), 6) for f in z_freqs}
    th_set = {round(float(f), 6) for f in th_freqs}
    if z_set & th_set:
        raise ValueError(f"z and θ frequencies overlap: {sorted(z_set & th_set)}")
    vz, _ = disp_multisine(dt, seconds, ax_z, z_freqs, seed=seed)
    wth, _ = disp_multisine(dt, seconds, ax_th, th_freqs, seed=seed + 17)
    seq = np.zeros((vz.size, 6), dtype=float)
    seq[:, 2] = vz
    seq[:, int(theta_axis)] = wth
    return seq


def scan_perturb_twists(
    dt: float,
    seconds: float,
    rho: float,
    amp_z: float,
    amp_th: float,
    *,
    scan_axis: int = 0,
    theta_axis: int = 4,
    seed: int = 3,
    z_freqs: tuple[float, ...] = DZ_FREQS_HZ,
    th_freqs: tuple[float, ...] = DTH_FREQS_HZ,
) -> np.ndarray:
    """Constant ρ plus small disjoint δvz, δωθ — rank-3 regressor for Ḣ."""

    z_set = {round(float(f), 6) for f in z_freqs}
    th_set = {round(float(f), 6) for f in th_freqs}
    if z_set & th_set:
        raise ValueError(f"δz and δθ frequencies overlap: {sorted(z_set & th_set)}")
    n = max(int(round(float(seconds) / float(dt))), 2)
    t = np.arange(n, dtype=float) * float(dt)
    rng = np.random.default_rng(int(seed))
    seq = np.zeros((n, 6), dtype=float)
    seq[:, int(scan_axis)] = float(rho)
    for f in z_freqs:
        seq[:, 2] += float(amp_z) * np.sin(
            2.0 * math.pi * float(f) * t + float(rng.uniform(0.0, 2.0 * math.pi))
        )
    for f in th_freqs:
        seq[:, int(theta_axis)] += float(amp_th) * np.sin(
            2.0 * math.pi * float(f) * t + float(rng.uniform(0.0, 2.0 * math.pi))
        )
    z_peak = float(np.max(np.abs(seq[:, 2]))) or 1.0
    th_peak = float(np.max(np.abs(seq[:, int(theta_axis)]))) or 1.0
    seq[:, 2] *= float(amp_z) / z_peak
    seq[:, int(theta_axis)] *= float(amp_th) / th_peak
    return seq


def fopdt_predict(u: np.ndarray, dt: float, t0: float, tp: float, gain: float) -> np.ndarray:
    """Discrete K e^{-T0 s}/(Tp s+1) on a uniform grid.  Missing params → zeros."""

    u = np.asarray(u, dtype=float).reshape(-1)
    y = np.zeros_like(u)
    if not (math.isfinite(float(dt)) and dt > 1e-6):
        return y
    if not all(math.isfinite(float(v)) for v in (t0, tp, gain)):
        return y
    delay = max(int(round(float(t0) / float(dt))), 0)
    a = math.exp(-float(dt) / max(float(tp), 1e-6))
    state = 0.0
    k = float(gain)
    for i, ui in enumerate(u):
        ud = float(u[i - delay]) if i >= delay and math.isfinite(u[i - delay]) else 0.0
        if not math.isfinite(ud):
            ud = 0.0
        state = a * state + (1.0 - a) * k * ud
        y[i] = state
        del ui
    return y


def predict_2x2(
    uz: np.ndarray,
    wth: np.ndarray,
    dt: float,
    g: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """ŷ = G u with four FOPDT channels.  Missing channel → 0."""

    def _one(u, name: str) -> np.ndarray:
        ch = g.get(name) or {}
        return fopdt_predict(
            u,
            dt,
            float(ch.get("T0_s", float("nan"))),
            float(ch.get("Tp_s", float("nan"))),
            float(ch.get("K", float("nan"))),
        )

    vz = _one(uz, "zz") + _one(wth, "zth")
    th = _one(uz, "thz") + _one(wth, "thth")
    return vz, th


def residual_tube(err: np.ndarray, *, slack: float = 0.0) -> dict:
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "n": 0,
            "p95": float("nan"),
            "max_abs": float("nan"),
            "bar": float("nan"),
            "slack": float(slack),
        }
    mx = float(np.max(np.abs(e)))
    p95 = float(np.percentile(np.abs(e), 95))
    return {
        "n": int(e.size),
        "p95": p95,
        "max_abs": mx,
        "bar": mx + abs(float(slack)),
        "slack": float(slack),
    }


def cover_fraction(err: np.ndarray, bar: float) -> float:
    if not math.isfinite(float(bar)) or bar < 0.0:
        return float("nan")
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    return float(np.mean(np.abs(e) <= float(bar) + 1e-12))


def local_stiffness(
    x: np.ndarray,
    f: np.ndarray,
    *,
    win: int = 41,
    min_dx: float = 2e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window dF/dx.  Returns (F_center, Ke) aligned with the inputs."""

    x = np.asarray(x, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)
    n = int(min(x.size, f.size))
    ke = np.full(n, np.nan, dtype=float)
    fm = np.full(n, np.nan, dtype=float)
    win = max(int(win), 7)
    if win % 2 == 0:
        win += 1
    h = win // 2
    if n < win:
        return fm, ke
    for i in range(h, n - h):
        sl = slice(i - h, i + h + 1)
        xx = x[sl]
        ff = f[sl]
        m = np.isfinite(xx) & np.isfinite(ff)
        if int(np.count_nonzero(m)) < max(h, 6):
            continue
        xx = xx[m]
        ff = ff[m]
        if float(np.std(xx)) < float(min_dx):
            continue
        xc = xx - float(np.mean(xx))
        a = np.vstack([xc, np.ones(xx.size)]).T
        coef, *_ = np.linalg.lstsq(a, ff, rcond=None)
        ke[i] = float(coef[0])
        fm[i] = float(np.mean(ff))
    return fm, ke


def stiffness_envelope(
    f_center: np.ndarray,
    ke: np.ndarray,
    f_lo: float,
    f_hi: float,
    *,
    eps_f: float = 0.15,
) -> dict:
    """Local Ke on [f_lo, f_hi].  work_band_reached only if the samples span the band."""

    m = (
        np.isfinite(f_center)
        & np.isfinite(ke)
        & (f_center >= float(f_lo))
        & (f_center <= float(f_hi))
    )
    empty = {
        "n": 0,
        "ke_min": float("nan"),
        "ke_max": float("nan"),
        "ke_bar": float("nan"),
        "ke_median": float("nan"),
        "f_valid_min": float("nan"),
        "f_valid_max": float("nan"),
        "work_band_reached": False,
    }
    if not np.any(m):
        return empty
    vals = ke[m]
    fvals = f_center[m]
    fmin = float(np.min(fvals))
    fmax = float(np.max(fvals))
    reached = fmin <= float(f_lo) + float(eps_f) and fmax >= float(f_hi) - float(eps_f)
    return {
        "n": int(np.count_nonzero(m)),
        "ke_min": float(np.min(vals)),
        "ke_max": float(np.max(vals)),
        "ke_bar": float(np.max(np.abs(vals))),
        "ke_median": float(np.median(vals)),
        "f_valid_min": fmin,
        "f_valid_max": fmax,
        "work_band_reached": bool(reached),
        "local_span_reached": bool(reached),
    }


def work_band_spanned(
    f: np.ndarray,
    f_lo: float,
    f_hi: float,
    *,
    eps_f: float = 0.15,
) -> dict:
    """True only if the raw force trajectory covers [f_lo, f_hi] to within ε.

    A single in-band sample is not coverage.  Window-mean F_center can sit
    ~0.2–0.5 N inside the peak on a stiff pad; use the raw limb Fz here.
    """

    ff = np.asarray(f, dtype=float)
    ff = ff[np.isfinite(ff)]
    empty = {
        "n": 0,
        "f_min": float("nan"),
        "f_max": float("nan"),
        "work_band_reached": False,
    }
    if ff.size == 0:
        return empty
    fmin = float(np.min(ff))
    fmax = float(np.max(ff))
    return {
        "n": int(ff.size),
        "f_min": fmin,
        "f_max": fmax,
        "work_band_reached": bool(
            fmin <= float(f_lo) + float(eps_f) and fmax >= float(f_hi) - float(eps_f)
        ),
    }


def so3_log_vee(R: np.ndarray) -> np.ndarray:
    """(Log R)^∨ for R ∈ SO(3)."""

    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    if th < 1e-9:
        return 0.5 * axis
    if abs(th - math.pi) < 1e-6:
        diag = np.clip(0.5 * (np.diag(R) + 1.0), 0.0, 1.0)
        w = np.sqrt(diag)
        if R[2, 1] < 0:
            w[0] = -w[0]
        if R[0, 2] < 0:
            w[1] = -w[1]
        if R[1, 0] < 0:
            w[2] = -w[2]
        nrm = float(np.linalg.norm(w)) or 1.0
        return (th / nrm) * w
    return (th / (2.0 * math.sin(th))) * axis


def twist_at_offset(twist: np.ndarray, r_tool: np.ndarray) -> np.ndarray:
    """V_c = (v + ω × r, ω) at a tool-frame offset.  Same rotation frame."""

    tw = np.asarray(twist, dtype=float)
    r = np.asarray(r_tool, dtype=float).reshape(3)
    out = np.array(tw, copy=True, dtype=float)
    if tw.ndim == 1:
        out[:3] = tw[:3] + np.cross(tw[3:6], r)
        return out
    out[:, :3] = tw[:, :3] + np.cross(tw[:, 3:6], r)
    return out


def wrench_at_offset(wrench: np.ndarray, r_tool: np.ndarray) -> np.ndarray:
    """W_c = (F, τ − r × F) so W^⊤ V is invariant."""

    w = np.asarray(wrench, dtype=float)
    r = np.asarray(r_tool, dtype=float).reshape(3)
    out = np.array(w, copy=True, dtype=float)
    if w.ndim == 1:
        out[3:6] = w[3:6] - np.cross(r, w[:3])
        return out
    out[:, 3:6] = w[:, 3:6] - np.cross(r, w[:, :3])
    return out


def port_power(wrench: np.ndarray, twist: np.ndarray, *, trans_only: bool = False) -> np.ndarray:
    w = np.asarray(wrench, dtype=float)
    v = np.asarray(twist, dtype=float)
    if w.ndim == 1:
        p = float(np.dot(w[:3], v[:3]))
        if not trans_only:
            p += float(np.dot(w[3:6], v[3:6]))
        return np.asarray(p)
    p = np.sum(w[:, :3] * v[:, :3], axis=1)
    if not trans_only:
        p = p + np.sum(w[:, 3:6] * v[:, 3:6], axis=1)
    return p


def power_invariance_rel_err(
    wrench: np.ndarray,
    twist: np.ndarray,
    r_tool: np.ndarray,
    *,
    trans_only: bool = False,
) -> float:
    """|P_tcp − P_contact| / (1 + |P_tcp|) should be ~ machine epsilon."""

    p0 = port_power(wrench, twist, trans_only=trans_only)
    p1 = port_power(
        wrench_at_offset(wrench, r_tool),
        twist_at_offset(twist, r_tool),
        trans_only=trans_only,
    )
    den = 1.0 + np.abs(p0)
    rel = np.abs(p0 - p1) / den
    rel = rel[np.isfinite(rel)]
    return float(np.max(rel)) if rel.size else float("nan")


def prefix_work(power: np.ndarray, dt: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(power), power, 0.0)
    d = np.where(np.isfinite(dt) & (dt > 1e-6), dt, 0.0)
    return np.cumsum(p * d)


def prefix_debt(work: np.ndarray) -> np.ndarray:
    """Current deficit (−W)_+, not the running max.  Needed for ∀k coverage."""

    return np.maximum(-np.asarray(work, dtype=float), 0.0)


def prefix_covers(d_true: np.ndarray, d_bound: np.ndarray, *, slack: float = 0.0) -> dict:
    a = np.asarray(d_true, dtype=float)
    b = np.asarray(d_bound, dtype=float)
    n = int(min(a.size, b.size))
    if n == 0:
        return {"n": 0, "all_ok": False, "frac": float("nan"), "max_violation": float("nan")}
    ok = np.isfinite(a[:n]) & np.isfinite(b[:n])
    if not np.any(ok):
        return {"n": 0, "all_ok": False, "frac": float("nan"), "max_violation": float("nan")}
    viol = a[:n][ok] - (b[:n][ok] + float(slack))
    covered = viol <= 1e-12
    mx = float(np.max(viol)) if viol.size else float("nan")
    return {
        "n": int(np.count_nonzero(ok)),
        "all_ok": bool(np.all(covered)),
        "frac": float(np.mean(covered)),
        "max_violation": mx,
    }


def predicted_twist(cmd: np.ndarray, dt: float, g: dict, *, theta_axis: int = 4) -> np.ndarray:
    """V̂ = Ĝ u on (vz, ωθ); other axes stay as commanded."""

    cmd = np.asarray(cmd, dtype=float)
    hat = np.array(cmd, copy=True)
    if cmd.ndim == 1:
        vz, wth = predict_2x2(
            np.asarray([cmd[2]]), np.asarray([cmd[int(theta_axis)]]), dt, g
        )
        hat[2] = float(vz[0])
        hat[int(theta_axis)] = float(wth[0])
        return hat
    vz, wth = predict_2x2(cmd[:, 2], cmd[:, int(theta_axis)], dt, g)
    hat[:, 2] = vz
    hat[:, int(theta_axis)] = wth
    return hat


def finite_channel(ch: dict) -> bool:
    return all(math.isfinite(float(ch.get(k, float("nan")))) for k in ("T0_s", "Tp_s", "K"))


def dump_jsonable(obj):
    """Drop ndarray fields so a payload can go to JSON."""

    if isinstance(obj, dict):
        return {k: dump_jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [dump_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
