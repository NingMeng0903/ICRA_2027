# 07_contact_gv — displacement-limited contact Gv

Take `20260905_145323` · filter OFF · force loop OFF · Window A median gap 2.3 ms · clock resets 2 (used latest SERVO slice)

## Verdict

- Air **T0 = 70.0 ms**, Tp = 14.0 ms, K = 0.584.
- Contact **T0 = 10.0 ms**, Tp = 8.0 ms, K = 2.086.
- Residual vs the air model \(\mathcal{E}_v\): p95 = 2.12 mm/s, max = 2.96 mm/s. This set matters more than a single T0.
- Contact pose offset from the start of that chirp: 1.17 mm (commanded Ax = 0.20 mm; this is not peak-to-peak). If it is near 4 mm the excitation is still too large.
- 03 air T0 = 28.0 ms. Do not read this as a force Bode.

## Figure

`gv_compare.png`: air / contact \(|G_v|\).
