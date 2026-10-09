"""Summarize the interleaved latency re-measurement (pre-registration §5).

Usage: python3 analyze_latency_remeasure.py <timings.jsonl> <output.json>
Pure standard library. Ratios are computed per scene (same scene, same repeat)
so host-load drift cancels; the 20% speed gate is "판정 보류" when the gap to
0.80 lies inside the across-repeat range of the median ratio.
"""
import json
import statistics
import sys
from collections import defaultdict

def pct(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

rows = [json.loads(line) for line in open(sys.argv[1])]
by = defaultdict(list)
cell = {}
for r in rows:
    by[r["model"]].append(r)
    cell[(r["model"], r["scene_token"], r["repeat"])] = r["policy_ms"]
models = list(dict.fromkeys(r["model"] for r in rows))
repeats = sorted({r["repeat"] for r in rows})
summary = {"samples": len(rows), "host_load1": {"median": statistics.median(r["host_load1"] for r in rows),
           "min": min(r["host_load1"] for r in rows), "max": max(r["host_load1"] for r in rows)},
           "sm_clock_mhz": {"min": min(r["sm_clock_mhz"] for r in rows), "max": max(r["sm_clock_mhz"] for r in rows)},
           "models": {}, "ratios": {}}
for m in models:
    ms = [r["policy_ms"] for r in by[m]]
    per = {rep: statistics.median(r["policy_ms"] for r in by[m] if r["repeat"] == rep) for rep in repeats}
    low = [r["policy_ms"] for r in by[m] if r["host_load1"] <= summary["host_load1"]["median"]]
    high = [r["policy_ms"] for r in by[m] if r["host_load1"] > summary["host_load1"]["median"]]
    summary["models"][m] = {"n": len(ms), "p50": pct(ms, .5), "p95": pct(ms, .95), "mean": statistics.mean(ms),
        "repeat_p50": per, "repeat_p50_range_pct": (max(per.values()) - min(per.values())) / min(per.values()) * 100,
        "p50_low_load": pct(low, .5) if low else None, "p50_high_load": pct(high, .5) if high else None}
for reference in ["B10", "B02-opt", "B03-opt", "B05-opt"]:
    if reference not in by:
        continue
    out = {}
    for m in models:
        if m == reference:
            continue
        ratios = [cell[(m, t, rep)] / cell[(reference, t, rep)] for (mm, t, rep) in cell if mm == m and (reference, t, rep) in cell]
        per = {rep: statistics.median(cell[(m, t, r)] / cell[(reference, t, r)] for (mm, t, r) in cell
                                      if mm == m and r == rep and (reference, t, r) in cell) for rep in repeats}
        median = statistics.median(ratios)
        lo, hi = min(per.values()), max(per.values())
        verdict = "pass" if hi <= .80 else "fail" if lo > .80 else "판정 보류"
        out[m] = {"median_paired_ratio": median, "repeat_median_ratios": per, "speed_gate_0.80": verdict,
                  "p95_ratio": summary["models"][m]["p95"] / summary["models"][reference]["p95"]}
    summary["ratios"][reference] = out
json.dump(summary, open(sys.argv[2], "w"), indent=2, ensure_ascii=False)
for m, v in summary["models"].items():
    print(f"{m:14s} p50 {v['p50']:7.1f} p95 {v['p95']:7.1f}  repeat range {v['repeat_p50_range_pct']:4.1f}%  low/high load p50 {v['p50_low_load']:.1f}/{v['p50_high_load']:.1f}")
for ref, out in summary["ratios"].items():
    print("== vs", ref)
    for m, v in out.items():
        print(f"  {m:14s} ratio {v['median_paired_ratio']:.3f} repeats {[round(x, 3) for x in v['repeat_median_ratios'].values()]} gate {v['speed_gate_0.80']} p95 ratio {v['p95_ratio']:.3f}")
print("host load1", summary["host_load1"], "sm clock", summary["sm_clock_mhz"])
