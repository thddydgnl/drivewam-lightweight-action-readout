#!/usr/bin/env bash
# VLM pipeline smoke test on 4 processed dev-pool clips (GPU0): tagging, route/BEV, guidance.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
src=$root/work/DriveWAM/src
sm=$root/data/physicalai/smoke_vlm
model=$root/checkpoints/qwen3-vl-8b-instruct
rm -rf $sm; mkdir -p $sm/front $sm/tags $sm/guid
python3 - <<PY
import json
from pathlib import Path
r = Path("$root"); pool = (r / "splits/pai-devpool-3k.txt").read_text().split()
have = [c for c in pool if (r / "data/physicalai/front" / c / "camera_front_wide_120fov.mp4").exists()][:4]
for c in have:
    (Path("$sm/front") / c).symlink_to(r / "data/physicalai/front" / c)
Path("$sm/guid/ids.json").write_text(json.dumps(have))
print(have)
PY
source $root/scripts/remote_env.sh model >/dev/null
export PYTHONPATH=$root/.venvs/vlm-overlay${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
cd $src
t0=$(date +%s)
python -m physicalai.vlm_data_sample.qwvl_scene_select --model_path $model --dataset_root $sm/front \
  --clip_index $root/data/physicalai/clip_index.parquet --split val --output_dir $sm/tags --num_gpus 1 \
  --batch_size 4 --num_sample_frames 20 --max_new_tokens 300 --loader_workers 4
t1=$(date +%s); echo "{\"tagging_s\": $((t1-t0))}"
common=(--dataset_root $sm/front --clip_index $root/data/physicalai/clip_index.parquet --output_dir $sm/guid
        --num_clips -1 --num_chunks 5 --chunk_seconds 4.0 --seed 0 --clip_ids_file $sm/guid/ids.json)
python -m physicalai.vlm_preprocess.generate_route "${common[@]}" --num_workers 4 --route_store_name route_store.json
python -m physicalai.vlm_preprocess.generate_bev "${common[@]}" --num_workers 4 --bev_store_name bev_store.npz
t2=$(date +%s)
python $root/scripts/pai_vlm_patch.py "${common[@]}" --model_path $model --tensor_parallel_size 1 --batch_size 4 \
  --max_model_len 8192 --gpu_memory_utilization 0.9 --route_store $sm/guid/route_store.json \
  --bev_store $sm/guid/bev_store.npz --output_name guidance.json
t3=$(date +%s); echo "{\"guidance_s\": $((t3-t2))}"
ls -la $sm/tags $sm/guid
python3 -c "import json; d=json.load(open('$sm/guid/guidance.json')); print(json.dumps(d, indent=1)[:3000])"
