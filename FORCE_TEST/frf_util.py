"""Black-box helpers for vel_ff → v_ach.  No force-law / no Dimeas filter."""

from __future__ import annotations

import math

import numpy as np

from io_csv import col, load_rows, moments


def load_pair(path, u_keys=("vel_ff", "vel_ff_vz"), y_keys=("v_ach", "vz_achieved_tool")):
    rows = load_rows(path)
    t = col(rows, "t_wall_s", "t_mono_s")
    u = col(rows, *u_keys)
    y = col(rows, *y_keys)
    fb = col(rows, "feedback_age_s")
    dt = col(rows, "dt_actual_s")
    mask = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 8:
        raise ValueError(f"{path} has no finite command/achieved velocity pair")
    order = np.argsort(t[mask], kind="stable")
    return (
        t[mask][order],
        u[mask][order],
        y[mask][order],
        fb[mask][order],
        dt[mask][order],
        rows,
    )


def xcorr_delay_s(cmd: np.ndarray, ach: np.ndarray, dt: float) -> float:
    if cmd.size < 8 or float(np.std(cmd)) < 1e-8:
        return float("nan")
    c0 = cmd - np.mean(cmd)
    a0 = ach - np.mean(ach)
    corr = np.correlate(a0, c0, mode="full")
    lags = np.arange(-c0.size + 1, a0.size)
    lag = int(lags[int(np.argmax(corr))])
    return max(lag, 0) * float(dt)


def rise_s(cmd: np.ndarray, ach: np.ndarray, t: np.ndarray) -> float:
    if cmd.size < 4:
        return float("nan")
    target = float(cmd[-1])
    start = float(ach[0])
    span = target - start
    if abs(span) < 1e-5:
        return float("nan")
    lo = start + 0.10 * span
    hi = start + 0.90 * span
    t10 = t90 = float("nan")
    pred = (lambda yi, th: yi >= th) if span > 0 else (lambda yi, th: yi <= th)
    for ti, yi in zip(t, ach):
        if not math.isfinite(t10) and pred(yi, lo):
            t10 = float(ti)
        if math.isfinite(t10) and pred(yi, hi):
            t90 = float(ti)
            break
    if not (math.isfinite(t10) and math.isfinite(t90)):
        return float("nan")
    return max(t90 - t10, 0.0)


def peak_accel(ach: np.ndarray, t: np.ndarray) -> float:
    if ach.size < 3:
        return float("nan")
    dti = np.diff(t)
    dv = np.diff(ach)
    ok = dti > 1e-4
    if not np.any(ok):
        return float("nan")
    return float(np.max(np.abs(dv[ok] / dti[ok])))


def step_edges(t, cmd, ach, *, min_hold_s=0.25, jump_m_s=0.002) -> list[dict]:
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    if not math.isfinite(dt) or dt <= 1e-4:
        dt = 0.005
    edges = []
    i = 1
    while i < cmd.size:
        if abs(float(cmd[i] - cmd[i - 1])) < jump_m_s:
            i += 1
            continue
        cmd_now = float(cmd[i])
        t0 = float(t[i])
        j = i
        while j < cmd.size and abs(float(cmd[j]) - cmd_now) < jump_m_s:
            j += 1
        hold = float(t[min(j, cmd.size) - 1] - t0)
        if hold < min_hold_s:
            i = j
            continue
        sl = slice(i, j)
        pre = max(int(round(0.080 / dt)), 4)
        sl_x = slice(max(0, i - pre), j)
        tail = ach[sl]
        n_tail = max(int(0.2 / dt), 4)
        settled = float(np.mean(tail[-n_tail:])) if tail.size >= n_tail else float(np.mean(tail))
        edges.append(
            {
                "cmd_mm_s": abs(cmd_now) * 1000.0,
                "sign": 1.0 if cmd_now >= 0.0 else -1.0,
                "t_edge_s": t0,
                "delay_s": xcorr_delay_s(cmd[sl_x], ach[sl_x], dt),
                "rise_s": rise_s(cmd[sl], ach[sl], t[sl]),
                "peak_accel_m_s2": peak_accel(ach[sl], t[sl]),
                "kp": settled / cmd_now if abs(cmd_now) > 1e-5 else float("nan"),
            }
        )
        i = j
    return edges


def welch_frf(cmd, ach, dt, *, nperseg=None):
    n = int(cmd.size)
    if n < 64:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    seg = int(nperseg or min(2048, max(256, n // 8)))
    seg = min(seg, n)
    hop = max(seg // 2, 1)
    window = np.hanning(seg)
    u_acc, y_acc, uy_acc = [], [], []
    for start in range(0, n - seg + 1, hop):
        u = np.fft.rfft((cmd[start : start + seg] - np.mean(cmd[start : start + seg])) * window)
        y = np.fft.rfft((ach[start : start + seg] - np.mean(ach[start : start + seg])) * window)
        u_acc.append(u * np.conj(u))
        y_acc.append(y * np.conj(y))
        uy_acc.append(y * np.conj(u))
    puu = np.mean(np.stack(u_acc, axis=0), axis=0)
    pyy = np.mean(np.stack(y_acc, axis=0), axis=0)
    puy = np.mean(np.stack(uy_acc, axis=0), axis=0)
    freq = np.fft.rfftfreq(seg, dt)
    eps = 1e-18
    h = puy / (puu + eps)
    g2 = np.abs(puy) ** 2 / (np.abs(puu) * np.abs(pyy) + eps)
    return freq, np.abs(h), np.angle(h), np.real(g2)


def fopdt_fit(cmd, ach, dt) -> tuple[int, float, float]:
    from paths import add_playground

    add_playground()
    from peirastic.apps.identify_plant import _fit_fopdt, _resample_uniform

    _tu, uu, yy, dtr = _resample_uniform(
        np.arange(cmd.size) * dt, cmd, ach
    )
    delay, tp, gain, _rmse = _fit_fopdt(uu, yy, dtr)
    return int(delay), float(tp), float(gain)


def fopdt_from_frf(freq, mag, phase, coh, *, f_lo=0.3, f_hi=8.0):
    """Fit K exp(-T0 s)/(Tp s+1) to the nonparametric FRF, not time-domain xcorr.

    Whole-record chirp xcorr locks T0 high and then cannot trade it back
    against Tp.  Magnitude gives Tp; leftover phase gives T0.
    """
    m = (
        (coh >= 0.6)
        & (freq >= f_lo)
        & (freq <= f_hi)
        & np.isfinite(mag)
        & np.isfinite(phase)
        & (mag > 1e-4)
    )
    if int(np.count_nonzero(m)) < 6:
        return float("nan"), float("nan"), float("nan")
    f = freq[m]
    g = mag[m]
    ph = np.unwrap(phase[m])
    w = 2.0 * math.pi * f
    low = f <= 1.0
    k0 = float(np.median(g[low])) if np.any(low) else float(np.median(g))
    best = (float("inf"), float("nan"), float("nan"), k0)
    for t0 in np.linspace(0.010, 0.070, 31):
        for tp in np.linspace(0.008, 0.080, 25):
            den = np.sqrt(1.0 + (w * tp) ** 2)
            pred_g = k0 / den
            pred_ph = -w * t0 - np.arctan(w * tp)
            err = float(
                np.mean((np.log(g) - np.log(np.maximum(pred_g, 1e-6))) ** 2)
                + np.mean(((ph - pred_ph) / math.pi) ** 2)
            )
            if err < best[0]:
                best = (err, float(t0), float(tp), k0)
    return best[1], best[2], best[3]


def median_dt(t, dt_col) -> float:
    if dt_col.size and np.isfinite(dt_col).any():
        med = float(np.nanmedian(dt_col))
        if med > 1e-4:
            return med
    if t.size > 2:
        med = float(np.median(np.diff(t)))
        if med > 1e-4:
            return med
    return 0.005


def save_fig(fig, out_dir, name: str) -> None:
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=140)
    fig.savefig((out_dir / name).with_suffix(".svg"))


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
