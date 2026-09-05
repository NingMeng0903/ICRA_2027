# 11_exec_2dof — 两个反馈自由度的执行合同

本轮**还没采**。空气再单独做 uz 与 ωθ（默认 wy），再做频率分离的联合 multisine；预载垫上复做小幅。

对照 03 的 Tn，不要对力 Bode。交叉项小就保留对角模型。硬 force certificate 本轮不启用。

```bash
python FORCE_TEST/11_exec_2dof.py --window-a-csv /path/to/window_a.csv --dry-run
```
