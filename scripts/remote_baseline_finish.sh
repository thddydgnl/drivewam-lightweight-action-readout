#!/usr/bin/env bash
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
run_id=baselines_devval256_2026-10-01
deadline=$((SECONDS + 3600))
while true; do
  stage=$(python3 - "$root/runs/$run_id/inference-status.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);print(json.loads(p.read_text()).get('stage','pending') if p.exists() else 'pending')
PY
)
  if [[ "$stage" == verified ]]; then break; fi
  if [[ "$stage" == failed ]]; then echo 'Inference failed; refusing partial scoring.'; exit 1; fi
  if (( SECONDS >= deadline )); then echo 'Timed out waiting for inference.'; exit 1; fi
  sleep 10
done
source "$root/scripts/remote_env.sh" scorer
python -u "$root/scripts/remote_baseline_score.py" --run-id "$run_id" --backend official
python -u "$root/scripts/remote_baseline_score.py" --run-id "$run_id" --backend vendored
python -u "$root/scripts/remote_baseline_analyze.py" --run-id "$run_id"
