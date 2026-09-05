# 13_contact_hs — 力–力矩–扫描耦合

本轮**还没采**。扫描段叠加互不重频的 \(\delta v_z,\delta\omega_\theta\)，同时改变 \(\rho\)。否则 \(X=[v_z,\omega_\theta,\rho]\) 秩到不了 3。

`--alpha-deg` 是这一拍的已知楔角，写入 `DATA/hs_alphas.csv`。±tilt 只做符号。粗角度估计要 −10/−5/0/+5/+10° 各一拍。90° 不是预设定律。Window A 需要 tx, ty, tz。

```bash
python FORCE_TEST/13_contact_hs.py --window-a-csv /path/to/window_a.csv --alpha-deg 5 --dry-run
```
