# 08_ke — 开环环境刚度 Ke

本轮（2026-09-03，filter OFF）**还没采**。这是本目录里唯一不是速度伺服辨识的脚本。

Δx 必须来自 Window A 实际 TCP 位姿沿 tool-Z 的投影，**禁止命令积分**。软垫和硬垫都要采；硬垫才可能让 C1 下界变紧。压完沿工具 −Z 抬离，再 MOVEJ 回中位。`DATA/08_ke/` 只留最新一拍。Ke 和部位追加在 `DATA/ke_sites.csv`（wipe 不删）。

```bash
python FORCE_TEST/08_env_ke.py --site 软垫 --window-a-csv /path/to/window_a.csv --dry-run
python FORCE_TEST/08_env_ke.py --site 硬垫 --window-a-csv /path/to/window_a.csv --dry-run
```

没带 `--site` 也可以采；告诉我部位后补进那一行。不要开混合。
