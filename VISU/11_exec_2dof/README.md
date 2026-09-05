# 11_exec_2dof — 2×2 执行合同

本轮**还没采**。空气和接触**分开**拟合

\[
G_{\rm air}(s),\qquad
\boxed{G_{\rm contact}(s)},
\qquad
\begin{bmatrix}v_z\\\omega_\theta^{\rm ach}\end{bmatrix}
=G(s)
\begin{bmatrix}u_z\\\omega_\theta\end{bmatrix}+w.
\]

论文 / QP 合同只用 \(G_{\rm contact}\) 和 \(\mathcal W_{\rm contact}\)。空气是 baseline，两段不得拼成一个 FOPDT。

联合 multisine 频率分离：z 0.5/1.3/2.1/2.9 Hz，θ 0.8/1.6/2.4/3.2 Hz。永远保存 full 2×2。交叉通道量纲不同，**不要**用 \|K\|=0.08 自动对角化。归一化 γ 只留给之后的消融。

```bash
python FORCE_TEST/11_exec_2dof.py --window-a-csv /path/to/window_a.csv --dry-run
```
