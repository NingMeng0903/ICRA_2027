# 14_port_energy — 从命令预测端口功率

本轮**还没采**，而且要等 11 的 \(G_{\rm contact}\) 先出来。

路线 A（服务 QP）：

\[
\hat V=\hat G_{\rm contact}u,\qquad
\hat P=W^\top\hat V,\qquad
P_{\rm lower}=\hat P-\|F\|\bar e_v-\|\tau\|\bar e_\omega.
\]

用 \(P_{\rm pose}=W^\top V_{\rm pose}\) 验证 \(P_{\rm pose}\ge P_{\rm lower}\)，覆盖必须对每一个 prefix。**不要**从 \(P_{\rm ach}\) 上减 11 的 plant residual。

TCP↔接触点 \(P_T=P_C\) 是伴随自检，证明空间变换没写错，**不能**标定 30 mm 杠杆或传感器 frame。JSON 记录 \({}^Tr_C=[r_x,r_y,r_z]\)。缺 11 `G_contact` 则 `passivity_claim_allowed=False`。

```bash
python FORCE_TEST/14_port_energy.py --window-a-csv /path/to/window_a.csv --dry-run
```
