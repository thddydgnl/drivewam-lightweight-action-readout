#!/usr/bin/env bash
# navtest-final (pre-registration §8): inference -> scoring -> latency -> verdict.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
run=$root/runs/navtest_final_2026-10-03
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }
stage(){ python3 -c "import json;print(json.load(open('$1')).get('stage',''))" 2>/dev/null || echo none; }

log wait_inputs_and_cache
until [ -f $root/reports/navtest-final-processing.json ] && [ "$(stage $root/reports/navtest-final-cache-status.json)" = verified ]; do
  [ "$(stage $root/reports/navtest-final-cache-status.json)" = failed ] && { log cache_failed; exit 1; }; sleep 30; done

log inference
for s in 0 1; do
  ( export LITE_GPU_IDS=$s; source $root/scripts/remote_env.sh model; cd $root/scripts; python -u remote_navtest_final_infer.py --shard $s ) \
    > $root/logs/navtest-final-infer-shard$s-2026-10-03.log 2>&1 &
done
wait
for s in 0 1; do [ "$(stage $run/inference-shard$s-status.json)" = verified ] || { log inference_failed; exit 1; }; done

log assemble_and_score
( source $root/scripts/remote_env.sh scorer; cd $root/scripts; python - <<'PY'
import json, numpy as np
from pathlib import Path
R = Path.home() / "DriveWAM_Lightweight_Action_Readout"; run = R / "runs/navtest_final_2026-10-03"
tokens = (R / "splits/navtest-final-tokens.txt").read_text().splitlines()
got = {}
for f in sorted(run.glob("shard*/*.npz")):
    with np.load(f) as d:
        for key in ["B10", "B02_opt", "student_seed17", "student_seed29"]:
            for t, x in zip(d["scene_tokens"].astype(str), d[key]):
                got.setdefault(key, {})[t] = x
(run / "predictions").mkdir(exist_ok=True)
for key, v in got.items():
    assert set(v) == set(tokens)
    np.savez(run / "predictions" / (key.replace("_opt", "-opt").replace("_seed", "-seed") + ".npz"), scene_tokens=np.asarray(tokens), trajectories=np.stack([v[t] for t in tokens]))
print("assembled", sorted(got))
PY
  mkdir -p $run/scores
  for f in $run/predictions/*.npz; do n=$(basename $f .npz)
    python -u remote_navtest_score.py score --split final --predictions $f --output $run/scores/$n.csv; done )

log latency
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model; cd $root/scripts; python -u remote_final_latency.py )
( source $root/scripts/remote_env.sh scorer; cd $root/scripts
  python -c "import sys" && python $root/scripts/analyze_latency_remeasure.py $root/runs/navtest_final_latency_2026-10-03/timings.jsonl $root/runs/navtest_final_latency_2026-10-03/latency-summary.json
  python -u remote_final_analyze.py )
log done
