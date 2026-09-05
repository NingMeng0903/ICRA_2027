# 11_exec_2dof — 2×2 执行合同

本轮**还没采**。辨识完整

\[
\begin{bmatrix}v_z\\\omega_\theta^{\rm ach}\end{bmatrix}
=G(s)
\begin{bmatrix}u_z\\\omega_\theta\end{bmatrix}+w,
\quad
G=\begin{bmatrix}G_{zz}&G_{z\theta}\\G_{\theta z}&G_{\theta\theta}\end{bmatrix}.
\]

联合 multisine 频率分离：z 0.5/1.3/2.1/2.9 Hz，θ 0.8/1.6/2.4/3.2 Hz。接触段 z 用限位移 chirp。残差管是 max\(|e|\)+slack，不是 “cross small yes/no”。硬 force certificate 本轮不启用。

```bash
python FORCE_TEST/11_exec_2dof.py --window-a-csv /path/to/window_a.csv --dry-run
```
