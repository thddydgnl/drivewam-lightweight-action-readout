"""navtest-final analysis and verdict (pre-registration §8). Scorer environment."""
import json
from pathlib import Path

from remote_baseline_analyze import QUALITY, cluster_bootstrap
from remote_navtest_analyze import METRICS
import pandas as pd

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
RUN = ROOT / "runs/navtest_final_2026-10-03"

def load(path):
    f = pd.read_csv(path, keep_default_na=False)
    assert f["valid"].astype(str).isin(["True", "true", "1"]).all() and len(f) == 8896
    f[QUALITY] = f[QUALITY].astype(float) * 100
    return f.set_index("scene_token")

if __name__ == "__main__":
    tokens = (ROOT / "splits/navtest-final-tokens.txt").read_text().splitlines()
    s = {n: load(RUN / f"scores/{n}.csv").loc[tokens] for n in ["B10", "B02-opt", "student-seed17", "student-seed29"]}
    lat = json.loads((ROOT / "runs/navtest_final_latency_2026-10-03/latency-summary.json").read_text())
    ratio = lat["ratios"]["B02-opt"]["student"]
    out = {"means": {n: {m: float(f[m].mean()) for m in METRICS} for n, f in s.items()}, "paired": {}, "dac_fail_scenes": {}}
    for n in ["student-seed17", "student-seed29", "B02-opt"]:
        for ref in ["B10", "B02-opt"]:
            if n == ref:
                continue
            d = s[n][METRICS].subtract(s[ref][METRICS]); d["log_name"] = s[ref]["log_name"]
            out["paired"][f"{n} - {ref}"] = cluster_bootstrap(d.reset_index(drop=True), METRICS)
        out["dac_fail_scenes"][n] = int((s[n]["drivable_area_compliance"] == 0).sum())
    out["dac_fail_scenes"]["B10"] = int((s["B10"]["drivable_area_compliance"] == 0).sum())
    verdict = {}
    for seed in [17, 29]:
        checks = {}
        for ref in ["B10", "B02-opt"]:
            p = out["paired"][f"student-seed{seed} - {ref}"]
            checks[f"pdms_ci_low_vs_{ref}>=-2"] = p["score"]["ci95_low"] >= -2.0
            checks[f"nc_drop_vs_{ref}<=0.5"] = -p["no_at_fault_collisions"]["mean"] <= .5
            checks[f"dac_drop_vs_{ref}<=0.5"] = -p["drivable_area_compliance"]["mean"] <= .5
        checks["p50_ratio_vs_B*<=0.80"] = ratio["speed_gate_0.80"] == "pass"
        verdict[f"seed{seed}"] = {"checks": checks, "pass": all(checks.values())}
    out.update(verdict=verdict, final_pass=all(v["pass"] for v in verdict.values()), latency_ratio_vs_Bstar=ratio,
               latency_p50=lat["models"], preregistration="§8", scenes=len(tokens), logs=int(s["B10"]["log_name"].nunique()))
    (RUN / "final-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for n, m in out["means"].items():
        print(f"{n:16s} PDMS {m['score']:.2f} NC {m['no_at_fault_collisions']:.2f} DAC {m['drivable_area_compliance']:.2f} EP {m['ego_progress']:.2f} ADE {m['ade_4s_m']:.3f}")
    for k, p in out["paired"].items():
        print(f"  {k:28s} PDMS {p['score']['mean']:+.2f} [{p['score']['ci95_low']:+.2f},{p['score']['ci95_high']:+.2f}] NC {p['no_at_fault_collisions']['mean']:+.2f} DAC {p['drivable_area_compliance']['mean']:+.2f}")
    print(json.dumps({"dac_fail_scenes": out["dac_fail_scenes"], "verdict": verdict, "final_pass": out["final_pass"],
                      "p50_ratio": ratio["median_paired_ratio"], "repeats": ratio["repeat_median_ratios"]}, indent=1))
