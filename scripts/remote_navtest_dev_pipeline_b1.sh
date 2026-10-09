#!/usr/bin/env bash
# Redo head predictions at batch one (deployed policy), keeping batch-64 outputs
# for the record, then rescore heads, re-analyse and run latency re-measurement.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
heads=$root/runs/heads_navtest_dev_2026-10-02
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }

log archive_batch64
[ -d $heads/v1_batch64 ] || mv $heads/v1 $heads/v1_batch64
mkdir -p $heads/scores_batch64
[ -d $heads/scores_batch64/v1 ] || mv $heads/scores/v1 $heads/scores_batch64/v1
[ -d $heads/scores_batch64/v2 ] || mv $heads/scores/v2 $heads/scores_batch64/v2
[ -f $heads/v2-predictions.batch64.json ] || mv $heads/v2-predictions.json $heads/v2-predictions.batch64.json
for f in $heads/seed*/*/navtest-dev-predictions.npz; do mv $f ${f%.npz}.batch64.npz; done

log head_predictions_batch1
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_navtest_heads.py v1 --output-run heads_navtest_dev_2026-10-02/v1 ) &
a=$!
( export LITE_GPU_IDS=1; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_navtest_heads.py predict-v2 ) &
b=$!
wait $a; wait $b

log rescoring
( source $root/scripts/remote_env.sh scorer; cd $root/scripts
  mkdir -p $heads/scores/v1 $heads/scores/v2
  for f in $heads/v1/*.npz; do n=$(basename $f .npz)
    python -u remote_navtest_score.py score --predictions $f --output $heads/scores/v1/$n.csv; done
  for f in $heads/seed*/*/navtest-dev-predictions.npz; do
    m=$(basename $(dirname $f)); s=$(basename $(dirname $(dirname $f)))
    python -u remote_navtest_score.py score --predictions $f --output $heads/scores/v2/$m@$s.csv; done
  python -u remote_navtest_analyze.py )

log latency
while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100{n++}END{print n+0}') > 0 )); do sleep 10; done
mv $root/runs/latency_remeasure_2026-10-02 $root/runs/latency_remeasure_2026-10-02.failed-batch64-reference 2>/dev/null || true
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_latency_remeasure.py )
log done
