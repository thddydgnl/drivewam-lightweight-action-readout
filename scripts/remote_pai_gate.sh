#!/usr/bin/env bash
# Parity gate (GPU0) as soon as four dev-pool clips are processed with their 10 Hz mp4 kept.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
ready(){ python3 - <<PY
from pathlib import Path
r = Path("$root"); pool = (r / "splits/pai-devpool-3k.txt").read_text().split()
n = sum((r / "data/physicalai/front" / c / "camera_front_wide_120fov.mp4").exists()
        and (r / "processed/physicalai-v1" / c / "receipt.json").exists() for c in pool[:200])
print(n)
PY
}
until [ "$(ready)" -ge 4 ]; do sleep 30; done
cd $root/scripts
export LITE_GPU_IDS=0
source remote_env.sh model >/dev/null
python -u remote_pai_parity.py --set final --n 3
