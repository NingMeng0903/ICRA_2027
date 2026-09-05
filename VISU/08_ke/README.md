# 08_ke — 开环环境刚度包络

本轮**还没采**。Δx 必须来自 Window A 实际 TCP 位姿沿 tool-Z 的投影，**禁止命令积分**。

不要只留一个 secant（例如 527 N/m）。默认工作带 **F∈[2,5] N**，与 `--target-n=5` 对齐。`work_band_reached` 要求加载段真实力程跨满该带，不是“带内有一个 local sample”。若论文最终要用 2–6 N，把 `--target-n` 至少加到 6，并提高 abort。

矩阵：每个 hard/soft site 至少 3 个速度，例如 \(v_z=1.5,3,6\) mm/s。压完同速卸载，再抬离、MOVEJ 回中位。行追加在 `DATA/ke_envelope.csv`（wipe 不删）。

```bash
python FORCE_TEST/08_env_ke.py --site 软垫 --press-mm-s 1.5 --window-a-csv /path/to/window_a.csv --dry-run
python FORCE_TEST/08_env_ke.py --site 硬垫 --press-mm-s 3 --window-a-csv /path/to/window_a.csv --dry-run
python FORCE_TEST/08_env_ke.py --site 硬垫 --press-mm-s 6 --window-a-csv /path/to/window_a.csv --dry-run
```

没带 `--site` 也可以采；告诉我部位后补进那一行。不要开混合。
