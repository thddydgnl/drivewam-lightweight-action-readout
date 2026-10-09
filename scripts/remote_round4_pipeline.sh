#!/usr/bin/env bash
# Round 4: wait for streaming + features -> finalize -> cleanup -> full training -> score -> analysis.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
v3=$root/runs/heads_v3_2026-10-03
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }
stage(){ python3 -c "import json;print(json.load(open('$1')).get('stage',''))" 2>/dev/null || echo none; }

log wait_stream
until [ "$(stage $root/reports/full-stream-status.json)" = streamed ]; do
  [ "$(stage $root/reports/full-stream-status.json)" = failed ] && { log stream_failed; exit 1; }; sleep 60; done
( source $root/scripts/remote_env.sh scorer; cd $root/scripts; python -u remote_full_stream.py finalize )
[ "$(stage $root/reports/full-stream-status.json)" = verified ] || { log exclusion_over_limit; exit 1; }

log wait_features
until [ "$(stage $root/caches/features/full_first_2026-10-03/shard0-status.json)" = verified ] && \
      [ "$(stage $root/caches/features/full_first_2026-10-03/shard1-status.json)" = verified ]; do
  pgrep -f remote_full_features.py >/dev/null || { log features_died; exit 1; }; sleep 60; done
( source $root/scripts/remote_env.sh model; cd $root/scripts; python -u remote_full_features.py --finalize )

log cleanup_archives
( source $root/scripts/remote_env.sh scorer; cd $root/scripts; python -u remote_full_stream.py cleanup )

log train_full
( export LITE_GPU_IDS=0; source $root/scripts/remote_env.sh model; cd $root/scripts
  python -u remote_heads_v3.py --phase first --data full --kinds R --seeds 17,29 ) > $root/logs/heads-v3-full-first-2026-10-03.log 2>&1

log score_full
( source $root/scripts/remote_env.sh scorer; cd $root/scripts
  for f in $v3/full-*/navtest-dev-predictions.npz; do n=$(basename $(dirname $f))
    [ -f $v3/scores/$n.csv ] || python -u remote_navtest_score.py score --predictions $f --output $v3/scores/$n.csv; done
  python -u remote_v3_analyze.py )
log done
