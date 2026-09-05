# FORCE_TEST — inner velocity servo, then contact ID

Every hardware script MOVEJs to mid-stroke first (same as `peirastic.DEMO.movej`: rail 400 mm + taught arm), **waits 5 s for nullspace**, then SERVO_TWIST. Force loop stays off. `--skip-movej` / `--skip-settle` if already settled. `10_tn_observe.py` is analyze-only.

Window A first: `python -m peirastic.apps.run_controller`. Air scripts 01–05/09 do not need `--log-csv`. **06 X/Y and all of 08, 11–14 require Window A `--log-csv` and `--window-a-csv`.** MotionBus only publishes tool-Z.

```bash
cd /media/camp/EXT_DRIVE/ICRA_2027
source /media/camp/EXT_DRIVE/RealUS_playground/rm75_control/env.sh
python FORCE_TEST/01_system_delay.py --dry-run
```

| script | measures | not |
|---|---|---|
| `01_system_delay.py` | T0 / Γd (xcorr on velocity steps) | group delay, yaml 55 ms, force delay |
| `02_step_response.py` | rise, peak accel, Kp vs speed | T0 |
| `03_chirp_gv.py` | Gv Bode, FOPDT K/T0/Tp, one amplitude | Ya / force FRF |
| `04_timing_jitter.py` | feedback_age, dt_actual | plant T0 |
| `05_tracking_error.py` | \|vel_ff − v_ach\| (Lee) | BEFM apply (no τ) |
| `06_axis_gv.py` | Gv on X/Y/Z (Ma macro/mini) | force law |
| `07_contact_gv.py` | same Gv in light contact | force tracking |
| `08_env_ke.py` | pad/tissue Ke from **pose** tool-Z; `DATA/ke_sites.csv` | command-integral dx, inner loop |
| `09_multisine_gv.py` | paper §II.A multisine FRF | position-loop experiment |
| `10_tn_observe.py` | eq.(10) first term, observe only | Q/N1/N2 ID, CDYOB apply |
| `11_exec_2dof.py` | uz / ωθ execution + cross terms | force certificate |
| `12_stop_tail.py` | queue ⊕ stop tail envelopes | the safety filter itself |
| `13_contact_hs.py` | Hsρ, relaxation, tilt sign | 90° bounce law |
| `14_port_energy.py` | physical port work and prefix debt | QP tank / passivity theorem |

`--csv` analyzes an existing file. Each script keeps **only the latest** run: `DATA/01_delay/`, `VISU/01_delay/`, … A new collect (or `--csv` analyze) deletes that script’s previous folder. `06` keeps latest per axis in `DATA/06_axis_x|y|z/` and `VISU/06_axis/{x,y,z}/`. `--dry-run` prints the plan only: no drive, no DATA, no VISU, even with `--csv`. Timestamp of the take is in the JSON `collected_at` field.

Contact ID (08, 11–14) writes a 6-D command log plus a copy of Window A as `DATA/<kind>/window_a.csv`. Analyze refuses to invent pose/wrench from the command file. Window A must log `tx,ty,tz` (playground patch); without torque, 13/14 degrade and say so.

Collect order: **07, 08 (soft and hard pads), 11, 12, 13, 14**. 10 is already done. 15 and QP force rows are not this round.

Paper contract these scripts feed:

- **C1** — force corridor is a delay resource. 08 gives Ke; 12 gives stop tails and a \(\bar w\) proxy. If hard-pad 12 tails stay ~0.1 N, do not call this a viability-governor paper.
- **C2** — ρ is the projection of the same admissible set. 13 must show Hs on the wedge. If Hs ≈ 0, drop ρ from the theory core.
- **Energy** — 14 is an optional port certificate. Coverage failure cancels any passivity claim.

Figure style and captions: [`VISU/README.md`](../VISU/README.md). Graphics have no titles, dates, or method parentheses. Each `VISU/<kind>/README.md` is the analysis of the latest take once data exists; do not leave a purpose-only stub after a collect.
