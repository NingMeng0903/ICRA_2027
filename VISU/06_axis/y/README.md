# 06_axis / y — 分轴 Gv

采集：`20260903_173037` · filter OFF · 力环关 · 15 mm/s chirp · `twist_requested_vy → twist_achieved_vy`

Window A：`rm75_control/apps/logs/peirastic/run_20260903_173037.csv`（`servo_twist`）。不要用上一份 `171646`（那是 X），也不要用 `axis.csv` 的 v_ach。

## 结论

- **T0 = 28.0 ms，Tp = 14.0 ms，K = 1.066**。0.3–8 Hz 相干约 0.98。采集有效。
- 死区和 Z 同一档，比 X 的 14 ms 长。低频 |G| 略大于 1（0.5–3 Hz 约 1.04–1.22），不是跟丢。
- 伺服段里 vx/vz 指令几乎为 0；实现 vz 有耦合（max 约 12 mm/s），Y 不是完全解耦。
- 不要写 yaml。数据：`DATA/06_axis_y/`
