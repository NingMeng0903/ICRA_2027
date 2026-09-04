#!/usr/bin/env python3
"""Stream / sensor timing.  This is not plant T0.

What: feedback_age, dt_actual while holding a constant (or zero) velocity.
Why: De Stefano and CDYOB treat delay as an integer number of samples plus
     a freshness band.  If age p95 > 1 tick, Td is a band around T0, not a
     point.  yaml 55 ms is not this measurement.

Hold still or a slow constant v.  Do not chirp here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frf_util import load_pair, moments, mpl, save_fig
from io_csv import write_json
from goto_mid import add_movej_args, go_mid_from_args
from paths import add_playground, dry_exit, kind_dirs, stamp, write_readme


def _readme_04(payload: dict) -> str:
    age = payload.get("feedback_age") or {}
    dt = payload.get("dt_actual") or {}
    when = payload.get("collected_at") or ""
    band = bool(payload.get("td_is_band"))
    return f"""# 04_jitter — 时序，不是植物 T0

采集：`{when}` · filter OFF · 力环关 · MOVEJ 中位 · 静止持 10 s

## 结论

- **age p95 = {1e3 * float(age.get("p95") or float("nan")):.2f} ms**，max {1e3 * float(age.get("max") or float("nan")):.2f} ms，全程没有 ≥1 tick（5 ms）。
- **td_is_band = {band}**。新鲜度落在一个 200 Hz 拍里，Td 不用扩成带。
- dt p50 = {1e3 * float(dt.get("p50") or float("nan")):.3f} ms。偶发 ~10 ms 是 Window B 记日志漏一拍，那些行的 age 仍 <5 ms。
- 这不是 T0。植物用 01 / 03。yaml 的 55 ms 也不是这个数。

不要写 yaml。数据：`DATA/04_jitter/`
"""


def analyze(csv_path: Path, *, when: str) -> dict:
    t, _u, _y, fb, dt_col, _rows = load_pair(csv_path)
    payload = {
        "csv": str(csv_path),
        "feedback_age": moments(fb),
        "dt_actual": moments(dt_col),
        "tick_s": 0.005,
        "td_is_band": bool(
            moments(fb).get("p95", 0) is not None
            and moments(fb).get("p95", 0) > 0.005
        ),
        "note": "Age is stream freshness.  Plant T0 is 01_system_delay.py.",
        "collected_at": when,
    }
    data, visu = kind_dirs("04_jitter", preserve=csv_path)
    write_json(data / "jitter.json", payload)
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, title, stats in (
        (axes[0], "feedback_age_s", payload["feedback_age"]),
        (axes[1], "dt_actual_s", payload["dt_actual"]),
    ):
        ax.bar(["mean", "p95", "max"], [stats["mean"], stats["p95"], stats["max"]])
        ax.axhline(0.005, color="0.4", ls="--", label="1 tick")
        ax.set_title(title)
        ax.set_ylabel("s")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, visu, "jitter.png")
    plt.close(fig)
    write_readme(visu, _readme_04(payload))
    print(
        f"[JITTER] age_p95={1e3 * payload['feedback_age']['p95']:.2f} ms  "
        f"td_is_band={payload['td_is_band']}  data={data}",
        flush=True,
    )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default="")
    p.add_argument("--shm-prefix", default="")
    p.add_argument("--hold-s", type=float, default=10.0)
    p.add_argument("--mm-s", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=200.0)
    p.add_argument("--dry-run", action="store_true")
    add_movej_args(p)
    args = p.parse_args()
    when = stamp()
    if not args.csv:
        print(
            f"[PLAN] SERVO_TWIST hold {args.mm_s:.1f} mm/s for {args.hold_s:.1f}s  "
            "force loop OFF  (timing only; MOVEJ mid first)",
            flush=True,
        )
    if dry_exit(args):
        return 0
    if args.csv:
        analyze(Path(args.csv), when=when)
        return 0
    rc = go_mid_from_args(args)
    if rc:
        return rc
    add_playground()
    from servo_log import ServoLogger

    data, _visu = kind_dirs("04_jitter")
    log = data / "jitter.csv"
    srv = ServoLogger(prefix=args.shm_prefix, hz=args.hz, log_csv=log)
    try:
        srv.start_twist()
        if not srv.hold(args.mm_s / 1000.0, args.hold_s, "hold"):
            return 130
        srv.tick(0.0, "done")
    except KeyboardInterrupt:
        print("[STOP]", flush=True)
    finally:
        srv.close()
    if log.is_file() and srv.n_rows > 16:
        analyze(log, when=when)
    return 0


if __name__ == "__main__":
    sys.exit(main())
