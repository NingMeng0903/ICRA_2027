# 13_contact_hs — 力–力矩–扫描耦合

本轮**还没采**。**一角度一 take**。不要在同一拍中间把探针从平面换到楔块再混进一个 \(H\)。

`--alpha-deg` 是这一拍的已知攻角（0 = 平面），硬件采集必填。建议 \(\alpha=-10,-5,0,+5,+10\) 各一拍，追加 `DATA/hs_alphas.csv`。

每一拍：reset → seek → settle → 两个 ρ + 独立 \(\delta v_z,\delta\omega_\theta\) → reverse → ±θ → retract。力矩用 \(\tau_C=\tau_{TCP}-r\times F\)，倾角用 \(\mathrm{Log}(R_{\rm ref}^\top R)^\vee\cdot e_\theta\)，不是 global \(r_y\)。90° 不是预设定律。Window A 需要 tx, ty, tz。

```bash
python FORCE_TEST/13_contact_hs.py --window-a-csv /path/to/window_a.csv --alpha-deg 0 --dry-run
python FORCE_TEST/13_contact_hs.py --window-a-csv /path/to/window_a.csv --alpha-deg 5 --dry-run
```
