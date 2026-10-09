#!/usr/bin/env bash
# Round 3: X1 features -> 20k training; score all v3 heads; analysis.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
v3=$root/runs/heads_v3_2026-10-03
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }
stage(){ python3 -c "import json;print(json.load(open('$1')).get('stage',''))" 2>/dev/null || echo none; }

log wait_x1_processed
until [ "$(stage $root/reports/x1-prepare-status.json)" = processed_and_verified ]; do
  [ "$(stage $root/reports/x1-prepare-status.json)" = failed ] && { log x1_prepare_failed; exit 1; }; sleep 30; done

log x1_features
for s in 0 1; do
  ( export LITE_GPU_IDS=$s; source $root/scripts/remote_env.sh model; cd $root/scripts; python -u remote_x1_features.py --shard $s ) \
    > $root/logs/x1-features-shard$s-2026-10-03.log 2>&1 &
done
wait
( source $root/scripts/remote_env.sh model; cd $root/scripts; python -u remote_x1_features.py --finalize )

log wait_4k_training
while pgrep -f "remote_heads_v3.py --phase .* --data 4k" >/dev/null; do sleep 30; done
[ $(ls $v3/4k-*/summary.json | wc -l) -eq 8 ] || { log 4k_training_incomplete; exit 1; }

log train_20k
for pair in "0 first" "1 terminal"; do set -- $pair
  ( export LITE_GPU_IDS=$1; source $root/scripts/remote_env.sh model; cd $root/scripts; python -u remote_heads_v3.py --phase $2 --data 20k ) \
    > $root/logs/heads-v3-20k-$2-2026-10-03.log 2>&1 &
done
( source $root/scripts/remote_env.sh scorer; cd $root/scripts; mkdir -p $v3/scores
  for f in $v3/4k-*/navtest-dev-predictions.npz; do n=$(basename $(dirname $f))
    [ -f $v3/scores/$n.csv ] || python -u remote_navtest_score.py score --predictions $f --output $v3/scores/$n.csv; done )
wait
[ $(ls $v3/20k-*/summary.json | wc -l) -eq 8 ] || { log 20k_training_incomplete; exit 1; }

log score_20k
( source $root/scripts/remote_env.sh scorer; cd $root/scripts
  for f in $v3/20k-*/navtest-dev-predictions.npz; do n=$(basename $(dirname $f))
    [ -f $v3/scores/$n.csv ] || python -u remote_navtest_score.py score --predictions $f --output $v3/scores/$n.csv; done
  python -u remote_v3_analyze.py )
log done
