# 01_delay — 内环死区 T0

采集：`20260903_161356` · filter OFF · 力环关 · MOVEJ 中位 · `vel_ff → v_ach`

## 结论

- **T0 = 40.0 ms（Γd ≈ 8 tick）**。中位只收 15–40 mm/s 的 onset，不含回零、不含 8 mm/s。
- 8 mm/s 互相关会塌成 0 ms；曲线仍是先静约 40 ms 再爬。图上已去掉这些点。
- 步进整段 FOPDT（T0=45 ms，Tp=14 ms）**不是 T0**。写模型用 03 的 Bode。
- feedback_age p95 = 4.2 ms，是新鲜度，不是植物延迟。
- 关滤波后上升变快（见 02）。本数比开着 50 ms slew 的旧 01（45 ms）干净，仍比 FRF 的 28 ms 多 1–2 tick（互相关会吃一点上升）。

## 图

- `velocity_steps.png` **a** 全列车，**b** 线性档一个 onset，阴影是该步死区。
- `delay_vs_speed.png` 15/25/40 mm/s 围着中位。

不要写 yaml。数据：`DATA/01_delay/`
