# 14_port_energy — 真实端口与逐段前缀债务

本轮**还没采**。先做 TCP↔接触点功率不变性；角速度用 \(\mathrm{Log}(R_k^\top R_{k+1})^\vee/\Delta t\)。保守界来自 11 的 \((\bar e_v,\bar e_\omega)\)，不是 \(\|v_{\rm ach}-v_{\rm cmd}\|\)。覆盖必须 \(D_{\rm true}(k)\le D_{\rm bound}(k)\) 对每一个 prefix。缺 11 JSON 则 `passivity_claim_allowed=False`。

```bash
python FORCE_TEST/14_port_energy.py --window-a-csv /path/to/window_a.csv --dry-run
```
