#!/usr/bin/env bash
# Final-set runs for a re-selected B* (pre-registration 2026-11-01 §1.4). Each step is a no-op when
# that benchmark's B* did not change (remote_r5_analyze.py changed exits 3).
# Usage: remote_r5_final.sh nav-regen | nav-infer SHARD | nav-score | pai-infer SHARD
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
cd $root/scripts
bench=$([[ $1 == nav-* ]] && echo navsim || echo physicalai)
set +e; B=$(python remote_r5_analyze.py changed --bench $bench); rc=$?; set -e
if [ $rc -eq 3 ]; then echo "$bench B* unchanged: $1 skipped"; exit 0; fi
[ $rc -eq 0 ] || exit $rc
echo "$bench re-selected B*: $B"
run=r5_navsim_final_bstar_2026-11-01
case $1 in
  nav-regen)
    python -u remote_navtest_regen_inputs.py --split navtest_final \
      --keep-from splits/navtest-final-manifest.regen-2026-11-01-latency256.jsonl ;;
  nav-infer)
    python -u remote_r5_navsim_grid.py --split final --models $B --run-id $run --shard $2 ;;
  nav-score)
    python -u remote_r5_analyze.py assemble --run $run --split final
    mkdir -p $root/runs/$run/scores
    for f in $root/runs/$run/predictions/*.npz; do
      n=$(basename $f .npz)
      python -u remote_navtest_score.py score --split final --predictions $f --output $root/runs/$run/scores/$n.csv
    done ;;
  pai-infer)
    python -u remote_pai_baselines.py --set final --models $B --shard $2 --num-shards 2 --run-id pai_final_bstar_2026-11-01 --raw-check 0 ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
