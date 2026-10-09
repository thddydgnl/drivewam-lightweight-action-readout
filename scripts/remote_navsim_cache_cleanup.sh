#!/usr/bin/env bash
# Free disk for the PhysicalAI features (user permission 2026-10-04: NAVSIM intermediate caches may be
# deleted only when disk is short; final evidence, predictions, scores and pre-registrations are kept).
# Deletes, in order, the NAVSIM pilot and X1 feature caches until >= $1 GB are free. Lock files with
# the shard SHA-256 values are archived first.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
need=${1:-260}
arch=$root/reports/navsim-cache-locks; mkdir -p $arch
free(){ df -BG --output=avail / | tail -1 | tr -dc 0-9; }
log=$root/reports/navsim-cache-cleanup.jsonl
for c in world_features_pilot_2026-10-01 x1_features_2026-10-03; do
  d=$root/caches/features/$c
  [ -d $d ] || continue
  if [ "$(free)" -ge "$need" ]; then break; fi
  size=$(du -s -BG $d | cut -f1 | tr -dc 0-9)
  cp $d/FEATURE_LOCK.json $arch/$c.FEATURE_LOCK.json 2>/dev/null || true
  ls -la $d > $arch/$c.listing.txt
  rm -rf $d
  echo "{\"deleted\": \"$c\", \"size_gb\": $size, \"free_after_gb\": $(free), \"utc\": \"$(date -u +%FT%TZ)\"}" | tee -a $log
done
echo "{\"free_gb\": $(free), \"need_gb\": $need}"
