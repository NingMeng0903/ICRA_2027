# 04_jitter — 时序，不是植物 T0

采集：`20260903_163657` · filter OFF · 力环关 · MOVEJ 中位 · 静止持 10 s

## 结论

- **age p95 = 4.30 ms**，max 4.92 ms，全程没有 ≥1 tick（5 ms）。
- **td_is_band = False**。新鲜度落在一个 200 Hz 拍里，Td 不用扩成带。
- dt p50 = 5.000 ms。偶发 ~10 ms 是 Window B 记日志漏一拍，那些行的 age 仍 <5 ms。
- 这不是 T0。植物用 01 / 03。yaml 的 55 ms 也不是这个数。

不要写 yaml。数据：`DATA/04_jitter/`
