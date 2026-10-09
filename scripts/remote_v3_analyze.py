"""Round-3 analysis on navtest-dev (pre-registration §6). Scorer environment."""
import json
from pathlib import Path

import pandas as pd

from remote_baseline_analyze import QUALITY, cluster_bootstrap
from remote_navtest_analyze import load, METRICS

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
V3 = ROOT / "runs/heads_v3_2026-10-03"
RATIO = {"first": 0.467, "terminal": 0.794}   # 10/2 interleaved p50 ratio vs B02-opt

def paired(model, reference):
    d = model[METRICS].subtract(reference[METRICS]); d["log_name"] = reference["log_name"]
    return cluster_bootstrap(d.reset_index(drop=True), METRICS)

if __name__ == "__main__":
    tokens = (ROOT / "splits/navtest-dev-tokens.txt").read_text().splitlines()
    base = {n: load(ROOT / f"runs/navtest_dev_quality_2026-10-02/scores/{n}.csv").loc[tokens] for n in ["B10", "B02-opt"]}
    v2 = {f"{ph}-seed{s}": load(ROOT / f"runs/heads_navtest_dev_2026-10-02/scores/v2/SF3-{ph}@seed{s}.csv").loc[tokens]
          for ph in ["first", "terminal"] for s in [17, 29]}
    results = {}
    for path in sorted((V3 / "scores").glob("*.csv")):
        name = path.stem                                   # e.g. 4k-A-first-seed17
        frame = load(path).loc[tokens]
        data, kind, phase, seed = name.split("-")
        r = {"pdms": float(frame["score"].mean()), "ade_4s_m": float(frame["ade_4s_m"].mean())}
        for ref, rf in [("B10", base["B10"]), ("B02-opt", base["B02-opt"]), ("v2-SF3-same-phase", v2[f"{phase}-{seed}"])]:
            d = paired(frame, rf)
            r[ref] = {"score": d["score"], "nc_mean": d["no_at_fault_collisions"]["mean"], "dac_mean": d["drivable_area_compliance"]["mean"]}
        r["passes"] = all(r[ref]["score"]["ci95_low"] >= -2.0 and r[ref]["nc_mean"] >= -0.5 and r[ref]["dac_mean"] >= -0.5
                          for ref in ["B10", "B02-opt"]) and RATIO[phase] <= .80
        results[name] = r
    configs = {}
    for name, r in results.items():
        key = name.rsplit("-seed", 1)[0]
        configs.setdefault(key, []).append((name, r))
    verdict = {}
    for key, items in configs.items():
        verdict[key] = {"seeds": len(items), "pass_all_seeds": len(items) == 2 and all(r["passes"] for _, r in items),
                        "mean_pdms": sum(r["pdms"] for _, r in items) / len(items),
                        "mean_gain_vs_v2_same_phase": sum(r["v2-SF3-same-phase"]["score"]["mean"] for _, r in items) / len(items)}
    out = {"models": results, "configs": verdict, "rule": "pre-registration §6", "navtest_final_used": False}
    (V3 / "analysis-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for key, v in sorted(verdict.items(), key=lambda kv: -kv[1]["mean_pdms"]):
        print(f"{key:18s} mean PDMS {v['mean_pdms']:6.2f}  gain vs v2 {v['mean_gain_vs_v2_same_phase']:+.2f}  pass {v['pass_all_seeds']}")
    for name, r in sorted(results.items()):
        s = r["B10"]["score"]; b = r["B02-opt"]["score"]
        print(f"  {name:22s} {r['pdms']:6.2f} vsB10 {s['mean']:+.2f} [{s['ci95_low']:+.2f},{s['ci95_high']:+.2f}] vsB* {b['mean']:+.2f} [{b['ci95_low']:+.2f}] DAC {r['B10']['dac_mean']:+.2f} ADE {r['ade_4s_m']:.3f}")
