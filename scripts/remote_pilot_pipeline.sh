#!/usr/bin/env bash
# Run only after isolated GPU baseline timing; each dependent stage is gated.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
deadline=$((SECONDS + 14400))
while true; do
  stage=$(python3 - "$root/runs/opt_devval256_2026-10-01" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
paths=[r/(name+'-status.json') for name in ['inference','score-official','score-vendored','analysis']]
values=[json.loads(p.read_text()).get('stage','pending') if p.exists() else 'pending' for p in paths]
print('failed' if 'failed' in values else 'verified' if all(v=='verified' for v in values) else 'pending')
PY
)
  if [[ "$stage" == verified ]]; then break; fi
  if [[ "$stage" == failed ]]; then echo 'Opt verification failed; feature stage held.'; exit 1; fi
  if (( SECONDS >= deadline )); then echo 'Opt timing wait expired.'; exit 1; fi
  sleep 10
done
# The status is written immediately before process exit; await actual release.
while (( $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits) > 100 )); do
  if (( SECONDS >= deadline )); then echo 'GPU0 remains busy.'; exit 1; fi
  sleep 5
done
echo 'Starting validated feature extraction and pilot1000 CPU metric cache.'
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_feature_extract.py"
) > "$root/logs/features-pilot-2026-10-01.log" 2>&1 &
feature_pid=$!
(
  source "$root/scripts/remote_env.sh" scorer
  python -u "$root/scripts/remote_pilot_val_metric_cache.py"
) > "$root/logs/metric-cache-pilot1000-2026-10-01.log" 2>&1 &
metric_pid=$!
wait "$feature_pid"
wait "$metric_pid"
python3 - "$root" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
assert json.loads((root/'caches/features/world_features_pilot_2026-10-01/status.json').read_text())['stage']=='verified'
assert json.loads((root/'reports/navsim-pilot-val1000-cache-status.json').read_text())['stage']=='verified'
PY
echo 'Feature and scorer caches verified; starting probes and overfit gate.'
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_train_heads.py" --group probes
) > "$root/logs/head-probes-pilot-2026-10-01.log" 2>&1 &
probes_pid=$!
while true; do
  stage=$(python3 - "$root" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1])/'runs/head_pilot_2026-10-01'
p=r/'overfit.json';s=r/'probes-status.json'
if p.exists() and json.loads(p.read_text()).get('passed'): print('passed')
elif s.exists() and json.loads(s.read_text()).get('stage')=='failed': print('failed')
else: print('pending')
PY
)
  if [[ "$stage" == passed ]]; then break; fi
  if [[ "$stage" == failed ]]; then echo '32-scene overfit failed; fusion training held.'; wait "$probes_pid"; exit 1; fi
  sleep 10
done
while (( $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits) > 100 )); do sleep 10; done
echo '32-scene gate passed; starting parameter-matched fusion comparisons.'
(
  export LITE_GPU_IDS=1
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_train_heads.py" --group fusion
) > "$root/logs/head-fusion-pilot-2026-10-01.log" 2>&1 &
fusion_pid=$!
wait "$probes_pid"
wait "$fusion_pid"
echo 'Pilot head training completed. Online latency, controls, and analysis remain.'
