#!/usr/bin/env bash
# PhysicalAI dev guidance (pre-registration §2 step 3, dev = D2 list): route + BEV on CPU, then
# scene-evolving guidance with Qwen3-VL-8B (HF engine) on the GPUs given as arguments, then merge.
# Usage: remote_pai_dev_guidance.sh <stage: routebev|guidance|merge> [gpu shard pairs "0:0 1:1"] [batch]
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
src=$root/work/DriveWAM/src
front=$root/data/physicalai/front
guid=$root/data/physicalai/dev_guidance
model=$root/checkpoints/qwen3-vl-8b-instruct
source $root/scripts/remote_env.sh model >/dev/null
export PYTHONPATH=$root/.venvs/vlm-overlay${PYTHONPATH:+:$PYTHONPATH}
export MPLBACKEND=Agg
cd $src
common=(--dataset_root $front --clip_index $root/data/physicalai/clip_index.parquet --output_dir $guid
        --num_clips -1 --num_chunks 5 --chunk_seconds 4.0 --seed 0)
case "$1" in
  routebev)
    python -m physicalai.vlm_preprocess.generate_route "${common[@]}" --clip_ids_file $guid/dev_clip_ids.json \
      --num_workers 16 --route_store_name route_store.json
    python -m physicalai.vlm_preprocess.generate_bev "${common[@]}" --clip_ids_file $guid/dev_clip_ids.json \
      --num_workers 16 --bev_store_name bev_store.npz
    ;;
  guidance)
    for pair in $2; do
      g=${pair%%:*}; k=${pair##*:}
      ( export CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python $root/scripts/pai_vlm_patch.py "${common[@]}" --clip_ids_file $guid/dev_clip_ids_shard$k.json \
          --model_path $model --tensor_parallel_size 1 --batch_size ${3:-4} --max_model_len 8192 \
          --gpu_memory_utilization 0.9 --route_store $guid/route_store.json --bev_store $guid/bev_store.npz \
          --output_name guidance_shard$k.json ) > $root/logs/pai-guidance-shard$k.log 2>&1 &
    done
    wait
    ;;
  merge)
    python3 - <<PY
import json
from pathlib import Path
g = Path("$guid"); store = {}
for k in range(2):
    store.update(json.loads((g / f"guidance_shard{k}.json").read_text()))
ids = json.loads((g / "dev_clip_ids.json").read_text())
routes = json.loads((g / "route_store.json").read_text())
expect = {c: sum(1 for i in range(5) if f"{c}__c{i}" in routes) for c in ids}
missing = [c for c in ids if len(store.get(c, {})) < expect[c]]
(g / "scene_evolving_guidance_store.json").write_text(json.dumps(store, indent=2, ensure_ascii=False))
print("dev guidance clips", len(store), "chunks", sum(len(v) for v in store.values()), "expected", sum(expect.values()),
      "clips missing chunks", len(missing))
assert len(missing) <= 10, missing[:5]
PY
    ;;
esac
