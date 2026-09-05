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
| `08_env_ke.py` | Ke envelope in F∈[2,5] N (aligned with `--target-n=5`); `ke_envelope.csv` | single-secant 527 N/m; claiming [2,6] N at a 5 N press |
| `09_multisine_gv.py` | paper §II.A multisine FRF | position-loop experiment |
| `10_tn_observe.py` | eq.(10) first term, observe only | Q/N1/N2 ID, CDYOB apply |
| `11_exec_2dof.py` | **separate** \(G_{\rm air}\) and \(G_{\rm contact}\) 2×2 + \(W_{\rm contact}\) | mixed air+contact plant; auto-diagonal on \|K\|=0.08 |
| `12_stop_tail.py` | **matched-state** same (F,x,v,u), different 50–100 ms queue | sequential slow/fast that happens to share F |
| `13_contact_hs.py` | one \(\alpha\) per take; \(H(\alpha)=[H_z,H_\theta,H_s]\) | flat+wedge mixed into one H |
| `14_port_energy.py` | \(P_{\rm lower}=W^\top\hat G_{\rm contact}u-\|F\|\bar e\); prefix ∀k | \(P_{\rm ach}-\|F\|\bar e_G\); lever “calibration” via invariance |
| `15_holdout.py` | independent coverage of 08/11–14 bounds | fitting a new G/Ke/H/w̄ |

`--csv` analyzes an existing file. Each script keeps **only the latest** run: `DATA/01_delay/`, `VISU/01_delay/`, … A new collect (or `--csv` analyze) deletes that script’s previous folder. `06` keeps latest per axis in `DATA/06_axis_x|y|z/` and `VISU/06_axis/{x,y,z}/`. `--dry-run` prints the plan only: no drive, no DATA, no VISU, even with `--csv`. Timestamp of the take is in the JSON `collected_at` field.

Contact ID writes a 6-D command log plus a copy of Window A as `DATA/<kind>/window_a.csv`. Analyze refuses to invent pose/wrench from the command file. Window A must log `tx,ty,tz` (playground patch); without torque, 13/14 degrade and say so.

Collect order: **07 → 08 → 11 → 12 → 13 → 14**. Do **not** collect 15 now. Do **not** collect 07–14 as one batch.

| exp | collect now? |
|---|---|
| 07 | **GO** — disp-limited chirp + \(\mathcal E_v\) vs air \(G_v\) |
| 08 | **GO** after this band fix — default [2, 5] N = `--target-n`; each hard/soft site at \(v_z=1.5,3,6\) mm/s |
| 11 | **wait** — \(G_{\rm air}\) and \(G_{\rm contact}\) are now separate; collect after 07/08 |
| 12 | **GO** on hard pad first — matched-state is the core motivation experiment |
| 13 | **wait** — one known \(\alpha\) per take (`--alpha-deg`, 0 = flat) |
| 14 | **wait** — needs 11 \(G_{\rm contact}\) first; bound is \(W^\top\hat G u\), not \(P_{\rm ach}\) |

Importance for the paper: **12 > 11 > 15 > 08 > 13 > 14 > 07**. 14 enters the main text only if the 11-bound covers every prefix on real data.

Paper contract these scripts feed:

- **C1** — force corridor is a delay resource. 08 gives \(\overline K_e\); 12 gives matched-state tails and \(\bar w = \max|\Delta F|+\delta\). If hard-pad 12 tails stay ~0.1 N, do not call this a viability-governor paper.
- **C2** — ρ is the projection of the same admissible set. 13 must produce rank-3 H on the wedge. If Hs ≈ 0, drop ρ from the theory core.
- **Energy** — 14 is an optional port certificate. `passivity_claim_allowed` stays false without 11’s \((\bar e_v,\bar e_\omega)\) and ∀k prefix coverage.
- **15** — independent hold-out. Coverage here is what lets you write “certified operating envelope”.

Figure style and captions: [`VISU/README.md`](../VISU/README.md). Graphics have no titles, dates, or method parentheses. Each `VISU/<kind>/README.md` is the analysis of the latest take once data exists; do not leave a purpose-only stub after a collect.
