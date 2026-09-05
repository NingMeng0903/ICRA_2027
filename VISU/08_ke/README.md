# 08_ke — 开环环境刚度包络

本轮**还没采**。Δx 必须来自 Window A 实际 TCP 位姿沿 tool-Z 的投影，**禁止命令积分**。

不要只留一个 secant（例如 527 N/m）。分析给出 F∈[2,6] N 内 loading/unloading 的 local / \(\overline K_e\)。矩阵：软/硬垫 × 3 site × 2–3 速度。压完同速卸载，再抬离、MOVEJ 回中位。行追加在 `DATA/ke_envelope.csv`（wipe 不删）。

```bash
python FORCE_TEST/08_env_ke.py --site 软垫 --window-a-csv /path/to/window_a.csv --dry-run
python FORCE_TEST/08_env_ke.py --site 硬垫 --press-mm-s 5 --window-a-csv /path/to/window_a.csv --dry-run
```

没带 `--site` 也可以采；告诉我部位后补进那一行。不要开混合。
