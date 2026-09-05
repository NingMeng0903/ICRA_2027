# Figures

Latest take only, same rule as `DATA/`: each script overwrites its own folder. Captions live here. The graphic itself does not carry a title, a date, a run id, or a method dump.

## What a figure is for

A figure shows **one phenomenon**. It is not a screenshot of the test, not a lab notebook stamp, and not a place to park script names.

Do:

- Let axes and the curve carry the claim.
- English axis labels. Unit in parentheses on the axis, not in a title.
- One idea per panel. Panel tags are a single bold letter, nothing after it.
- Keep the latest PNG at 300 dpi and the matching SVG.

Do not:

- Put a title on the axes. No “Delay vs speed”, no “System identification”, no parenthetical leftovers.
- Print wall-clock time, `collected_at`, folder stamps, or “XX ms on 2026-…”.
- Annotate the method on the graphic: no SERVO_TWIST, force OFF, xcorr, Window A, yaml keys.
- Mix two claims on one axis unless they are the same phenomenon.
- Invent colors or fonts per script. Use `FORCE_TEST/paper_fig.py`.

If a number matters, it belongs in `DATA/*/…json` or in the caption below. The plot shows the shape.

## 01 — inner-loop delay

Source: `DATA/01_delay/delay.csv`. T0 is the median cross-correlation delay on **onset** steps at 15–40 mm/s. Return-to-zero edges and the 8 mm/s pair are not the median. Whole-record FOPDT on a step train is not T0.

### `01_delay/velocity_steps`

Commanded tool-Z velocity and the velocity the servo actually produced. The achieved trace lags the command by a nearly constant dead time; that lag is the inner-loop delay.

- **a** — full step train, ±8, 15, 25, 40 mm/s.
- **b** — one onset in the linear band. The shade is the dead time of that step, not the campaign median.

### `01_delay/delay_vs_speed`

T0 against command speed. In the linear band the delay does not rise with speed. The 8 mm/s pair is omitted when cross-correlation collapses to zero (step is small relative to velocity noise); the eye still sees ~40 ms of dead time.

## 02 — how the velocity servo climbs

Source: `DATA/02_step/step.csv`. Onset steps only. Return-to-zero edges are not in the medians. Rise time is t10–t90 of the velocity step, not T0.

### `02_step/rise_accel_kp`

Three faces of the same climb.

- **a** — rise time against command speed. Short and flat from 15–40 mm/s; long at a few mm/s (friction) and again at 60–80 mm/s (accel saturates).
- **b** — peak \(|dv/dt|\). The dashed line is 2 m/s². The plant reaches that cap near 40–60 mm/s.
- **c** — settled speed gain. Below ~12 mm/s the gain drops; above 15 mm/s it sits near one.

## 03 — inner-loop velocity FRF

Source: `DATA/03_chirp/chirp.csv`, chirp segment only. T0/Tp/K are fit to the FRF where \(\gamma^2\ge 0.6\). Whole-record chirp cross-correlation is not T0.

### `03_chirp/gv_bode`

How the velocity servo follows a command across frequency.

- **a** — \(|G_v|\). Solid: measured. Dashed: \(K e^{-T_0 s}/(T_p s+1)\).
- **b** — phase. Same two traces. The fit uses ≤8 Hz; the curve may continue a little past that.
- **c** — coherence. Frequency is logarithmic, ticks 0.2, 0.5, 1, 2, 5, 8, 10 Hz. “8 Hz cut-off” is written at the top; no threshold lines on the axes.

Campaign notes live in each folder’s `README.md`. After a collect, that file must hold the numbers and the verdict, not only what the script is for. Uncollected scripts may keep a purpose stub.

## 09 — paper-method multisine Gv

Source: `DATA/09_multisine/multisine.csv`, `20260903_195045`. Same pair as 03. Peak 15 mm/s, 0.2–10 Hz, three 20 s periods.

### `09_multisine/gv_bode`

Same three faces as 03. Delay matches the chirp (28 ms). Magnitude is lower because the multisine spreads the same peak across many lines; do not replace the chirp K or Tp.

## 10 — Tn residual

Source: `DATA/03_chirp/chirp.csv` observed with T0=28 ms, Tp=14 ms, K=0.96. No new drive.

### `10_tn/tn_residual`

Command, achieved velocity, and the FOPDT prediction. The lower trace is what is left after the delay model; it stays small across the chirp.

## 08 — environment stiffness

Source: `DATA/08_ke/press.csv` merged with `window_a.csv`. Δx is TCP pose along the first-press tool-Z axis. Command integration is not Ke.

### `08_ke/press`

- **a** — Fz.
- **b** — tool-Z travel from pose. The shade is the press interval used for ΔF/Δx.

## 11 — two feedback DoFs

Source: `DATA/11_exec_2dof/exec.csv` + `window_a.csv`.

### `11_exec_2dof/exec_bode`

|G| for the channels that were actually collected (air / preload, uz and ωθ). A small ωθ→vz magnitude in 0.3–3 Hz means the diagonal model is enough.

## 12 — stop tail

Source: `DATA/12_stop_tail/tail.csv` + `window_a.csv`.

### `12_stop_tail/stop_tail`

- **a** — Fz. Vertical lines are stop edges.
- **b** — tool-Z travel after those edges. A late rise after a zero command is committed motion.

## 13 — scan-induced force

Source: `DATA/13_contact_hs/hs.csv` + `window_a.csv`.

### `13_contact_hs/hs_scan`

Fz against path travel. One trace is the flat scan, one is the known slope. If they coincide, Hs is not identified.

## 14 — port power

Source: `DATA/14_port_energy/energy.csv` + `window_a.csv`.

### `14_port_energy/port_power`

Same-tick command power, achieved-twist power, pose-increment power, and the conservative bound. A press-then-retract pass goes negative then recovers; the prefix debt is the running max of the negative integral.

## Later scripts

Same rules. Caption goes in this file when the figure is added. Folder name matches the script kind.

Uncollected 07/08/11–14 may keep a purpose stub in `VISU/<kind>/README.md`. After a collect that file must hold numbers and a verdict. If 08 hard-pad Ke and 12 tails stay far below 1 N, C1 is academically true and experimentally slack. If 12 does not split two queues at the same (F, v), drop the delay-augmented state. If 13 cannot separate Hs, drop ρ as a theory core. If 14 fails coverage, the paper has no passivity claim.
