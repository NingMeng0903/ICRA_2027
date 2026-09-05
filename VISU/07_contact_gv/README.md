# 07_contact_gv — 限位移轻接触 Gv

本轮**还没采**。不要用旧默认 5 mm/s / 0.2 Hz（低频位移约 4 mm）。

预载 F0≈1.2 N。空气和接触都用 \(x=A_x\sin\phi\)，\(A_x\approx0.4\) mm，0.3–8 Hz。输出 T0/Tp/K 对照，以及 \(\mathcal E_v=v_{\rm ach}-\hat G_v^{\rm air}u\)。力环关。需要 `--window-a-csv`。

```bash
python FORCE_TEST/07_contact_gv.py --window-a-csv /path/to/window_a.csv --dry-run
```
