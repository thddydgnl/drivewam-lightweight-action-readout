#!/usr/bin/env bash
# PhysicalAI dev construction (pre-registration §2, D1):
#   1) tag the 3,000-clip val pool with DriveWAM qwvl_scene_select (Qwen3-VL-8B, 2 GPUs)
#   2) DriveWAM stratified test sampling -> 1,000 dev clips
#   3) route + BEV (CPU) and scene-evolving guidance (Qwen3-VL-8B, HF engine, 2 GPU shards)
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
src=$root/work/DriveWAM/src
front=$root/data/physicalai/front
tags=$root/data/physicalai/dev_tags
guid=$root/data/physicalai/dev_guidance
model=$root/checkpoints/qwen3-vl-8b-instruct
log(){ echo "{\"stage\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}"; }
source $root/scripts/remote_env.sh model >/dev/null
export PYTHONPATH=$root/.venvs/vlm-overlay${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $tags $guid
cd $src

log wait_pool
pool_state(){ python3 - <<PY
from pathlib import Path
r = Path("$root"); pool = (r / "splits/pai-devpool-3k.txt").read_text().split()
ok = [c for c in pool if (r / "processed/physicalai-v1" / c / "receipt.json").exists()
      and (r / "data/physicalai/front" / c / "camera_front_wide_120fov.mp4").exists()]
failed = [c for c in pool if (r / "logs/pai-process-errors" / (c + ".txt")).exists()]
print(len(ok) + len(failed), len(ok), len(failed))
PY
}
until [ "$(pool_state | cut -d' ' -f1)" -ge 3000 ]; do sleep 60; done
python3 - <<PY
import json
from pathlib import Path
r = Path("$root"); pool = set((r / "splits/pai-devpool-3k.txt").read_text().split())
have = {p.name for p in Path("$front").iterdir() if (p / "camera_front_wide_120fov.mp4").exists()}
failed = sorted(c for c in pool if (r / "logs/pai-process-errors" / (c + ".txt")).exists())
assert have <= pool and pool - have == set(failed), (len(pool - have), len(have - pool), len(failed))
(r / "reports/pai-devpool-failed.json").write_text(json.dumps({"processed": len(have), "failed": failed}, indent=2) + "\n")
print("pool processed", len(have), "failed", len(failed))
PY

log preempt
pkill -f remote_pai_final_baselines.sh || true
pkill -f "remote_pai_baselines.py --set final" || true
pkill -f "remote_pai_prompts.py --set train20k" || true
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')" -lt 2000 ]; do sleep 10; done

log tag_pool
export CUDA_VISIBLE_DEVICES=0,1
python -m physicalai.vlm_data_sample.qwvl_scene_select --model_path $model --dataset_root $front \
  --clip_index $root/data/physicalai/clip_index.parquet --split val --output_dir $tags --num_gpus 2 \
  --batch_size 4 --num_sample_frames 20 --max_new_tokens 300 --loader_workers 12 --load_stagger_s 10

log sample_dev
python -m physicalai.vlm_data_sample.qwvl_scene_stats --input_dir $tags --dataset_root $front \
  --save_csv $tags/ranked.csv --sample_out $tags/dev_sample.csv --sample_total 1000 --sample_quantile 0.75 \
  --sample_rare_per_tag 30 --sample_baseline 200 --no_print_events
python3 - <<PY
import json, hashlib, pandas as pd
from pathlib import Path
d = pd.read_csv("$tags/dev_sample.csv"); ids = d.clip_id.astype(str).tolist()
pool = set(Path("$root/splits/pai-devpool-3k.txt").read_text().split())
assert len(ids) == len(set(ids)) and set(ids) <= pool, (len(ids), len(set(ids) - pool))
Path("$root/splits/pai-dev-1k.txt").write_text("".join(c + "\n" for c in ids))
Path("$guid/dev_clip_ids.json").write_text(json.dumps(ids))
for k in range(2):
    Path(f"$guid/dev_clip_ids_shard{k}.json").write_text(json.dumps(ids[k::2]))
print("dev clips", len(ids), hashlib.sha256(Path("$root/splits/pai-dev-1k.txt").read_bytes()).hexdigest())
PY

log route_bev
common=(--dataset_root $front --clip_index $root/data/physicalai/clip_index.parquet --output_dir $guid
        --num_clips -1 --num_chunks 5 --chunk_seconds 4.0 --seed 0 --clip_ids_file $guid/dev_clip_ids.json)
python -m physicalai.vlm_preprocess.generate_route "${common[@]}" --num_workers 16 --route_store_name route_store.json
python -m physicalai.vlm_preprocess.generate_bev "${common[@]}" --num_workers 16 --bev_store_name bev_store.npz

log guidance
for k in 0 1; do
  ( export CUDA_VISIBLE_DEVICES=$k
    python $root/scripts/pai_vlm_patch.py --dataset_root $front --clip_index $root/data/physicalai/clip_index.parquet \
      --output_dir $guid --num_clips -1 --num_chunks 5 --chunk_seconds 4.0 --seed 0 \
      --clip_ids_file $guid/dev_clip_ids_shard$k.json --model_path $model --tensor_parallel_size 1 --batch_size 8 \
      --max_model_len 8192 --gpu_memory_utilization 0.9 --route_store $guid/route_store.json \
      --bev_store $guid/bev_store.npz --output_name guidance_shard$k.json ) > $root/logs/pai-guidance-shard$k.log 2>&1 &
done
wait
python3 - <<PY
import json
from pathlib import Path
g = Path("$guid"); store = {}
for k in range(2):
    store.update(json.loads((g / f"guidance_shard{k}.json").read_text()))
ids = json.loads((g / "dev_clip_ids.json").read_text())
missing = [c for c in ids if c not in store or len(store[c]) < 5]
(g / "scene_evolving_guidance_store.json").write_text(json.dumps(store, indent=2, ensure_ascii=False))
print("dev guidance clips", len(store), "missing_or_short", len(missing))
assert len(missing) <= 10, missing[:5]
PY
log launch_workers
for g in 0 1; do
  nohup python3 $root/scripts/remote_pai_worker.py --gpu $g > $root/logs/pai-worker-gpu$g.log 2>&1 &
  sleep 20
done
log done
