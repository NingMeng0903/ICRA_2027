# LOG

Servo logs for each collected test already live next to the JSON in `DATA/`
(`delay.csv`, `step.csv`, `chirp.csv`, …). This folder holds the extra Window A
controller CSVs that 06 X/Y cannot take from `axis.csv` (MotionBus only
publishes achieved Z).

| file | used by | pair |
|---|---|---|
| `window_a/run_20260903_171646.csv` | `DATA/06_axis_x` | `twist_requested_vx → twist_achieved_vx` |
| `window_a/run_20260903_173037.csv` | `DATA/06_axis_y` | `twist_requested_vy → twist_achieved_vy` |

Z reuses `DATA/03_chirp/chirp.csv`. Scripts 07 and 08 have not been collected.
Do not copy the rest of `rm75_control/apps/logs/peirastic/` (gigabytes, other
campaigns).
