#!/usr/bin/env bash
# Exclusive latency re-measurement (pre-registration 2026-11-01 §1.3). Every other r5 job is either
# finished or waits for this one, so the host runs nothing else. NAVSIM alone on GPU 0, then the
# PhysicalAI shards on GPUs 0 and 1 together (10/4 protocol). Models: B10, every candidate that
# passed the dev quality screening (§1.2, old v=3 included) and the student.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }
log wait_idle
while (( $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) > 0 )); do sleep 20; done
nav=$(python3 -c "import json; e=json.load(open('$root/reports/r5-nav-screen.json'))['eligible']; print(','.join(['B10']+e+['student']))")
pai=$(python3 -c "import json; e=json.load(open('$root/reports/r5-pai-screen.json'))['eligible']; print(','.join(e+['S-first']))")
log "navsim:$nav"
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model >/dev/null; cd $root/scripts
  python -u remote_r5_navsim_latency.py --models "$nav" )
python3 $root/scripts/analyze_latency_remeasure.py $root/runs/r5_navsim_latency_2026-11-01/timings.jsonl \
  $root/runs/r5_navsim_latency_2026-11-01/latency-summary.json
log "physicalai:$pai"
pids=()
for s in 0 1; do
  ( export LITE_GPU_IDS=$s; source $root/scripts/remote_env.sh model >/dev/null; cd $root/scripts
    python -u remote_pai_latency.py --models "$pai" --head $root/runs/pai_heads_2026-10-04/train20k-normal-first-seed17 \
      --clips 256 --reps 3 --shard $s --run-id r5_pai_latency_2026-11-01 ) > $root/logs/r5-pai-latency-shard$s.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
log done
