#!/usr/bin/env bash
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python "$root/scripts/remote_pilot_audit.py"
) > "$root/logs/pilot-artifact-audit-2026-10-01.log" 2>&1
python3 - "$root" <<'PY'
import json,sys,subprocess
from pathlib import Path
r=Path(sys.argv[1]);assert json.loads((r/'reports/pilot-artifact-audit-2026-10-01.json').read_text())['status']=='complete_pilot_artifacts_rechecked'
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(line.strip())<100 for line in memory.splitlines()),'Await GPU release; never overlap prior timing'
PY
echo 'Complete artifact audit passed. Starting matched-feature control and paired B10 target extraction.'
(
  export LITE_GPU_IDS=1
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_feature_control.py" --stage predict
  source "$root/scripts/remote_env.sh" scorer
  python -u "$root/scripts/remote_feature_control.py" --stage score
) > "$root/logs/feature-control-2026-10-01.log" 2>&1 &
control_pid=$!
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_target_diagnostic_extract.py"
) > "$root/logs/target-diagnostic-extract-2026-10-01.log" 2>&1
echo 'Paired features and teacher targets verified; preparing official B10 reference scores.'
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python "$root/scripts/remote_target_reference_prepare.py"
  source "$root/scripts/remote_env.sh" scorer
  python "$root/scripts/remote_target_diagnostic_score.py" --predictions "$root/runs/target_diagnostic_2026-10-01/B10-reference/reference-predictions.npz" --output "$root/runs/target_diagnostic_2026-10-01/B10-reference/reference-scores.csv"
) > "$root/logs/target-diagnostic-reference-2026-10-01.log" 2>&1
deadline=$((SECONDS+14400))
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100 {n++} END {print n+0}')
  if [[ "$busy" == 0 ]]; then break; fi
  if (( SECONDS>=deadline )); then echo 'GPUs remain busy; target training held.'; exit 1; fi
  sleep 10
done
echo 'Training identical SF3 heads against recorded GT and same-world B10 trajectories.'
(
  export LITE_GPU_IDS=0
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_target_diagnostic_train.py" --target GT
) > "$root/logs/target-diagnostic-GT-2026-10-01.log" 2>&1 &
gt_pid=$!
(
  export LITE_GPU_IDS=1
  source "$root/scripts/remote_env.sh" model
  python -u "$root/scripts/remote_target_diagnostic_train.py" --target Teacher
) > "$root/logs/target-diagnostic-Teacher-2026-10-01.log" 2>&1 &
teacher_pid=$!
wait "$gt_pid"
wait "$teacher_pid"
wait "$control_pid"
source "$root/scripts/remote_env.sh" scorer
python -u "$root/scripts/remote_target_diagnostic_analyze.py"
echo 'Feature and target diagnostics verified. Full-scale expansion remains a scientific decision.'
