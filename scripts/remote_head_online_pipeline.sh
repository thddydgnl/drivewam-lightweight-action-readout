#!/usr/bin/env bash
# Finish the currently requested pilot and preserve sequential measurement gates.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
run_id=${1:-heads_online_devval256_2026-10-01}
deadline=$((SECONDS + 28800))
while true; do
  stage=$(python3 - "$root" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);paths=[r/'runs/head_pilot_2026-10-01'/f'{g}-status.json' for g in ['probes','fusion']]
paths+=[r/'runs/opt_devval256_2026-10-01/analysis-status.json']
values=[json.loads(p.read_text()).get('stage','pending') if p.exists() else 'pending' for p in paths]
print('failed' if 'failed' in values else 'verified' if all(v=='verified' for v in values) else 'pending')
PY
)
  if [[ "$stage" == verified ]]; then break; fi
  if [[ "$stage" == failed ]]; then echo 'A prerequisite failed; native timing held.'; exit 1; fi
  if (( SECONDS >= deadline )); then echo 'Pilot training wait expired.'; exit 1; fi
  sleep 15
done
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100 {n++} END {print n+0}')
  if [[ "$busy" == 0 ]]; then break; fi
  if (( SECONDS >= deadline )); then echo 'GPUs remain busy; native timing held.'; exit 1; fi
  sleep 10
done
echo 'All training gates passed; starting isolated native online head timing.'
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_head_online_infer.py" --run-id "$run_id"
) > "$root/logs/${run_id}-infer.log" 2>&1
source "$root/scripts/remote_env.sh" scorer
python -u "$root/scripts/remote_opt_score.py" --run-id "$run_id" --backend official
python -u "$root/scripts/remote_opt_score.py" --run-id "$run_id" --backend vendored
python -u "$root/scripts/remote_head_online_analyze.py" --run-id "$run_id"
echo 'Pilot native online evidence verified. Feature control and target diagnostics remain.'
