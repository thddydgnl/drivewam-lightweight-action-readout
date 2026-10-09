#!/usr/bin/env bash
# navtest-dev pipeline after quality inference: assemble -> head predictions ->
# official scoring -> isolated latency re-measurement -> analysis.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
quality=$root/runs/navtest_dev_quality_2026-10-02
heads=$root/runs/heads_navtest_dev_2026-10-02
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }

log assemble
( source $root/scripts/remote_env.sh scorer; cd $root/scripts
  python -u remote_navtest_score.py assemble --run-id navtest_dev_quality_2026-10-02
  python -u remote_navtest_score.py cv --run-id navtest_dev_quality_2026-10-02 )

log head_predictions
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_navtest_heads.py v1 --output-run heads_navtest_dev_2026-10-02/v1 ) &
v1=$!
( export LITE_GPU_IDS=1; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_navtest_heads.py predict-v2 ) &
v2=$!
wait $v1; wait $v2

log scoring
( source $root/scripts/remote_env.sh scorer; cd $root/scripts
  mkdir -p $quality/scores $heads/scores/v1 $heads/scores/v2
  for f in $quality/predictions/*.npz; do
    name=$(basename $f .npz); [ "$name" = B10-seed-sensitivity ] && continue
    [ -f $quality/scores/$name.csv ] || python -u remote_navtest_score.py score --predictions $f --output $quality/scores/$name.csv
  done
  for f in $heads/v1/*.npz; do
    name=$(basename $f .npz)
    [ -f $heads/scores/v1/$name.csv ] || python -u remote_navtest_score.py score --predictions $f --output $heads/scores/v1/$name.csv
  done
  for f in $heads/seed*/*/navtest-dev-predictions.npz; do
    model=$(basename $(dirname $f)); seed=$(basename $(dirname $(dirname $f)))
    out=$heads/scores/v2/$model@$seed.csv
    [ -f $out ] || python -u remote_navtest_score.py score --predictions $f --output $out
  done )

log analysis
( source $root/scripts/remote_env.sh scorer; cd $root/scripts; python -u remote_navtest_analyze.py )

log latency
while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100{n++}END{print n+0}') > 0 )); do sleep 10; done
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_latency_remeasure.py )
log done
