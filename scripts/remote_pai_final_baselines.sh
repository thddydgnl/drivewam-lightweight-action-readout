#!/usr/bin/env bash
# Final-set baselines (fused opt rollouts), shards given as arguments, on GPU ${GPU:-0}.
# Resumable; dev selection may preempt it. The parity gate ran first (reports/pai-parity.json).
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
cd $root/scripts
export LITE_GPU_IDS=${GPU:-0}
source remote_env.sh model >/dev/null
for s in "$@"; do
  python -u remote_pai_baselines.py --set final --models B10-opt,B05-opt,B03-opt,B02-opt,B01-opt \
    --shard $s --num-shards 2 --run-id pai_final_baselines_2026-10-04
done
