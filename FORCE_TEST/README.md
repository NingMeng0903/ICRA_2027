# FORCE_TEST — inner velocity servo, then contact ID

Every hardware script MOVEJs to mid-stroke first (same as `peirastic.DEMO.movej`: rail 400 mm + taught arm), **waits 5 s for nullspace**, then SERVO_TWIST. Force loop stays off. `--skip-movej` / `--skip-settle` if already settled. `10_tn_observe.py` is analyze-only.

Window A first: `python -m peirastic.apps.run_controller`. Air scripts 01–05/09 do not need `--log-csv`. **07, 08, 11–15 require Window A `--log-csv` and `--window-a-csv`.** MotionBus only publishes tool-Z.

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
| `07_contact_gv.py` | air vs contact Gv under **bounded Ax ≈ 0.4 mm**; residual set Ev | force tracking, 5 mm/s @ 0.2 Hz chirp |
| `08_env_ke.py` | Ke envelope in F∈[2,6] N, load/unload; `ke_envelope.csv` | single-secant 527 N/m as the paper number |
| `09_multisine_gv.py` | paper §II.A multisine FRF | position-loop experiment |
| `10_tn_observe.py` | eq.(10) first term, observe only | Q/N1/N2 ID, CDYOB apply |
| `11_exec_2dof.py` | full 2×2 G + residual tube W | “cross small yes/no”, force certificate |
| `12_stop_tail.py` | **matched-state** same (F,x,v,u), different 50–100 ms queue | sequential slow/fast that happens to share F |
| `13_contact_hs.py` | Hz, Hθ, Hs from independent δvz, δωθ, ρ; α table | rank-1 scan, 90° bounce law |
| `14_port_energy.py` | TCP↔contact invariance, SO(3) ω, 11-bound, prefix ∀k | same-take ‖v_ach−v_cmd‖ as passivity |
| `15_holdout.py` | independent coverage of 08/11–14 bounds | fitting a new G/Ke/H/w̄ |

`--csv` analyzes an existing file. Each script keeps **only the latest** run: `DATA/01_delay/`, `VISU/01_delay/`, … A new collect (or `--csv` analyze) deletes that script’s previous folder. `06` keeps latest per axis in `DATA/06_axis_x|y|z/` and `VISU/06_axis/{x,y,z}/`. `--dry-run` prints the plan only: no drive, no DATA, no VISU, even with `--csv`. Timestamp of the take is in the JSON `collected_at` field.

Contact ID writes a 6-D command log plus a copy of Window A as `DATA/<kind>/window_a.csv`. Analyze refuses to invent pose/wrench from the command file. Window A must log `tx,ty,tz` (playground patch); without torque, 13/14 degrade and say so.

Collect order: **07 → 08 → 11 → 12 → 端口坐标/符号预校验 → 13 → 14 → 15**. 10 is already done. 08 can be collected now (pose Δx is correct); expand the matrix across pads / sites / speeds. Do **not** collect 07/11–14 on the old defaults.

Importance for the paper: **12 > 11 > 15 > 08 > 13 > 14 > 07**. 14 enters the main text only if the 11-bound covers every prefix on real data.

Paper contract these scripts feed:

- **C1** — force corridor is a delay resource. 08 gives \(\overline K_e\); 12 gives matched-state tails and \(\bar w = \max|\Delta F|+\delta\). If hard-pad 12 tails stay ~0.1 N, do not call this a viability-governor paper.
- **C2** — ρ is the projection of the same admissible set. 13 must produce rank-3 H on the wedge. If Hs ≈ 0, drop ρ from the theory core.
- **Energy** — 14 is an optional port certificate. `passivity_claim_allowed` stays false without 11’s \((\bar e_v,\bar e_\omega)\) and ∀k prefix coverage.
- **15** — independent hold-out. Coverage here is what lets you write “certified operating envelope”.

Figure style and captions: [`VISU/README.md`](../VISU/README.md). Graphics have no titles, dates, or method parentheses. Each `VISU/<kind>/README.md` is the analysis of the latest take once data exists; do not leave a purpose-only stub after a collect.
