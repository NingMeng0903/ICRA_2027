# 06_axis / x — 分轴 Gv

采集：`20260903_171646` · filter OFF · 力环关 · 15 mm/s chirp · `twist_requested_vx → twist_achieved_vx`

Window A：`rm75_control/apps/logs/peirastic/run_20260903_171646.csv`（`servo_twist`）。不要用 `axis.csv` 的 v_ach（那是 Z）。

## 结论

- **T0 = 14.0 ms，Tp = 20.0 ms，K = 0.965**。0.3–8 Hz 相干约 0.98。
- 对照 Z：T0 28→14 ms，Tp 14→20 ms。X 多半是臂。
- `vel_ff_vx` 和实现反号，分析用 `twist_requested_vx`。
- 不要写 yaml。数据：`DATA/06_axis_x/`
