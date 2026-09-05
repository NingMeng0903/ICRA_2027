# 12_stop_tail — 同状态不同队列

本轮**还没采**。硬垫可以按当前协议直接采。不要按 slow 0.8 s → fast 0.8 s 直接采完就算 “same state different queue”。

在线构造：A 以恒定 \(u_{\rm match}\) 压到 \(F^\star\)；B 先 burst 再切到同一 \(u_{\rm match}\)，等到 live \((F,x,v)\) 进入 A 的门再施加**同一** backup。分析用 Window A 位姿复核。\(\bar w=\max|\Delta F|+\delta\)，不是 p90。

力反馈只用于 preload 初始化；辨识段（hist / stop）全程开环 twist。软垫 settle 还要求 \(|\dot F|\) 持续 0.3–0.5 s 低于阈值，避免把粘弹性松弛写成队列。

```bash
python FORCE_TEST/12_stop_tail.py --window-a-csv /path/to/window_a.csv --dry-run
```
