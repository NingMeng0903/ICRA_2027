#!/usr/bin/env python3
"""Offline checks for 07–15 ID math.  No hardware, no DATA wipe."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from id_math import (
    TH_FREQS_HZ,
    Z_FREQS_HZ,
    disp_chirp_velocity,
    fopdt_predict,
    freq_split_disp_multisine,
    local_stiffness,
    port_power,
    power_invariance_rel_err,
    predicted_twist,
    prefix_covers,
    prefix_debt,
    prefix_work,
    residual_tube,
    so3_log_vee,
    stiffness_envelope,
    twist_at_offset,
    work_band_spanned,
    wrench_at_offset,
)
from window_a import pose_body_twist, relative_axis_angle, rot_xyz


def test_disp_chirp_stays_inside_ax() -> None:
    dt = 0.005
    T = 8.0
    ax = 0.0004
    t = np.arange(0.0, T, dt)
    v = disp_chirp_velocity(t, ax, 0.3, 8.0, T)
    x = np.cumsum(v) * dt
    assert float(np.max(np.abs(x))) < 1.15 * ax + 1e-6
    # Old 5 mm/s @ 0.2 Hz would be ~4 mm.  We must stay sub-millimetre.
    assert float(np.max(np.abs(x))) < 0.001
    print("[OK] disp chirp |x| < Ax", flush=True)


def test_freq_split_disjoint() -> None:
    assert not (set(Z_FREQS_HZ) & set(TH_FREQS_HZ))
    seq = freq_split_disp_multisine(0.005, 4.0, 0.0004, 0.01, seed=1)
    assert seq.shape[1] == 6
    assert float(np.max(np.abs(np.cumsum(seq[:, 2]) * 0.005))) < 0.0006
    try:
        freq_split_disp_multisine(0.005, 1.0, 0.001, 0.01, z_freqs=(0.8,), th_freqs=(0.8,))
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping lines must fail")
    print("[OK] frequency-split 2×2", flush=True)


def test_local_ke_envelope() -> None:
    x = np.linspace(0.0, 0.008, 400)
    f = 1.0 + 800.0 * x
    fm, ke = local_stiffness(x, f, win=31, min_dx=1e-6)
    env = stiffness_envelope(fm, ke, 2.0, 5.0)
    assert env["work_band_reached"]
    assert abs(env["ke_bar"] - 800.0) < 20.0
    assert abs(env["ke_min"] - 800.0) < 20.0
    mid = (fm >= 2.0) & (fm <= 3.2) & np.isfinite(ke)
    partial = stiffness_envelope(fm[mid], ke[mid], 2.0, 5.0)
    assert not partial["work_band_reached"]
    # Old default: band [2, 6] N with a 5 N target never spans 6 N.
    to5 = fm <= 5.0
    assert not stiffness_envelope(fm[to5], ke[to5], 2.0, 6.0)["work_band_reached"]
    assert work_band_spanned(np.array([0.5, 2.0, 3.5, 5.0]), 2.0, 5.0)["work_band_reached"]
    assert not work_band_spanned(np.array([0.5, 2.0, 3.5]), 2.0, 5.0)["work_band_reached"]
    assert not work_band_spanned(np.array([2.6, 3.5, 5.0]), 2.0, 5.0)["work_band_reached"]
    print("[OK] local Ke envelope + full-band coverage", flush=True)


def test_so3_omega_and_pose_twist() -> None:
    th = 0.10
    dt = 0.01
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    w = so3_log_vee(R) / dt
    assert abs(w[2] - th / dt) < 1e-8
    assert abs(w[0]) < 1e-8 and abs(w[1]) < 1e-8
    n = 20
    pose = np.zeros((n, 6))
    pose[:, 2] = 0.002 * np.arange(n)
    pose[:, 5] = th * np.arange(n)
    dts = np.full(n, dt)
    tw = pose_body_twist(pose, dts)
    assert abs(float(np.nanmean(tw[2:, 2])) - 0.002 / dt) < 5e-3
    print("[OK] SO(3) log omega", flush=True)


def test_power_invariance() -> None:
    w = np.array([0.1, -0.2, 4.0, 0.05, 0.02, -0.01])
    v = np.array([0.002, 0.0, 0.01, 0.3, -0.1, 0.05])
    r = np.array([0.0, 0.01, 0.03])
    p0 = float(port_power(w, v))
    p1 = float(port_power(wrench_at_offset(w, r), twist_at_offset(v, r)))
    assert abs(p0 - p1) < 1e-12
    rel = power_invariance_rel_err(np.vstack([w, w]), np.vstack([v, v]), r)
    assert rel < 1e-12
    print("[OK] TCP/contact power invariance", flush=True)


def test_prefix_forall_k() -> None:
    dt = np.full(8, 0.1)
    p_true = np.array([-2.0, -1.0, 0.5, 0.5, 0.2, 0.1, 0.0, 0.0])
    p_lo = p_true - 0.2
    d_t = prefix_debt(prefix_work(p_true, dt))
    d_b = prefix_debt(prefix_work(p_lo, dt))
    cov = prefix_covers(d_t, d_b)
    assert cov["all_ok"]
    p_bad = p_true + 1.5
    cov_bad = prefix_covers(d_t, prefix_debt(prefix_work(p_bad, dt)))
    assert not cov_bad["all_ok"]
    print("[OK] prefix ∀k coverage", flush=True)


def test_power_bound_from_gu_not_ach() -> None:
    dt = 0.005
    n = 200
    cmd = np.zeros((n, 6))
    cmd[:, 2] = 0.01
    g = {
        "zz": {"T0_s": 0.0, "Tp_s": 0.01, "K": 1.0},
        "zth": {"T0_s": 0.0, "Tp_s": 0.01, "K": 0.0},
        "thz": {"T0_s": 0.0, "Tp_s": 0.01, "K": 0.0},
        "thth": {"T0_s": 0.0, "Tp_s": 0.01, "K": 1.0},
    }
    hat = predicted_twist(cmd, dt, g, theta_axis=4)
    wrench = np.zeros((n, 6))
    wrench[:, 2] = 4.0
    p_hat = port_power(wrench, hat)
    ach = cmd.copy()
    ach[:, 2] = 0.02
    p_ach = port_power(wrench, ach)
    assert float(np.nanmean(p_hat)) < float(np.nanmean(p_ach)) - 0.01
    print("[OK] energy bound uses Ĝu, not P_ach", flush=True)


def test_fopdt_and_tube() -> None:
    dt = 0.005
    t = np.arange(0.0, 2.0, dt)
    u = np.sin(2.0 * math.pi * 1.3 * t)
    y = fopdt_predict(u, dt, 0.028, 0.014, 0.96)
    assert y.size == u.size
    assert float(np.max(np.abs(y))) < 1.0
    tube = residual_tube(y - 0.96 * u, slack=0.001)
    assert tube["n"] > 0 and math.isfinite(tube["bar"])
    print("[OK] FOPDT + residual tube", flush=True)


def test_rot_xyz_roundtrip() -> None:
    rx = np.array([0.0, 0.2])
    ry = np.array([0.0, -0.1])
    rz = np.array([0.0, 0.3])
    R = rot_xyz(rx, ry, rz)
    assert R.shape == (2, 3, 3)
    assert abs(float(np.linalg.det(R[1])) - 1.0) < 1e-9
    print("[OK] rot_xyz", flush=True)
    pose = np.zeros((12, 6))
    pose[:, 4] = np.linspace(0.0, 0.2, 12)
    th = relative_axis_angle(pose, 1)
    assert abs(float(th[-1]) - 0.2) < 1e-8
    print("[OK] relative SO(3) tilt angle", flush=True)


def test_matched_state_gate() -> None:
    from importlib import import_module

    st = import_module("12_stop_tail")
    a = {"f0_n": 2.50, "x0_m": 0.0030, "v0_m_s": 0.008, "u0_m_s": 0.008}
    b = {"f0_n": 2.62, "x0_m": 0.0032, "v0_m_s": 0.009, "u0_m_s": 0.008}
    ok = st._match_ok(a, b, eps_f=0.25, eps_x=0.00035, eps_v=0.004, eps_u=0.0015)
    assert ok["ok"]
    far = st._match_ok(a, {**b, "f0_n": 3.4}, eps_f=0.25, eps_x=0.00035, eps_v=0.004, eps_u=0.0015)
    assert not far["ok"]
    print("[OK] matched-state gate", flush=True)


if __name__ == "__main__":
    test_disp_chirp_stays_inside_ax()
    test_freq_split_disjoint()
    test_local_ke_envelope()
    test_so3_omega_and_pose_twist()
    test_power_invariance()
    test_prefix_forall_k()
    test_power_bound_from_gu_not_ach()
    test_fopdt_and_tube()
    test_rot_xyz_roundtrip()
    test_matched_state_gate()
    print("[OK] id design suite", flush=True)
