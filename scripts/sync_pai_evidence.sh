#!/usr/bin/env bash
# Copy PhysicalAI evidence (reports, job markers, small run summaries) from the server.
set -euo pipefail
dst=$(cd "$(dirname "$0")" && pwd)/evidence/physicalai_2026-10-04
mkdir -p $dst/reports $dst/jobs $dst/heads
R=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
rsync -a kiis-mvf-gpu:$R/reports/pai-*.json $dst/reports/ 2>/dev/null || true
rsync -a kiis-mvf-gpu:$R/reports/navsim-cache-cleanup.jsonl $dst/reports/ 2>/dev/null || true
rsync -a kiis-mvf-gpu:$R/reports/pai-jobs/ $dst/jobs/
rsync -a --include='*/' --include='summary.json' --include='history.json' --exclude='*' kiis-mvf-gpu:$R/runs/pai_heads_2026-10-04/ $dst/heads/ 2>/dev/null || true
rsync -a kiis-mvf-gpu:$R/splits/pai-*.txt $dst/splits/ 2>/dev/null || { mkdir -p $dst/splits; rsync -a kiis-mvf-gpu:$R/splits/pai-*.txt $dst/splits/; }
ls -R $dst | head -50
