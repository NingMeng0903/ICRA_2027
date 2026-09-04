# 08_ke — 开环环境刚度 Ke

本轮（2026-09-03，filter OFF）**还没采**。这是本目录里唯一不是速度伺服辨识的脚本。

压完沿工具 −Z 抬离，再 MOVEJ 回中位。`DATA/08_ke/` 只留最新一拍。Ke 和部位追加在 `DATA/ke_sites.csv`（wipe 不删）。

```bash
python FORCE_TEST/08_env_ke.py --site 腹部
```

没带 `--site` 也可以采；告诉我部位后补进那一行。不要开混合。
