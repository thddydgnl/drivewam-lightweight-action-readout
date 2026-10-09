#!/usr/bin/env bash
# Guidance smoke: 8 dev clips (40 chunks) with the HF captioner on GPU $1, batch $2.
set -euo pipefail
root=/home/yonghwi/DriveWAM_Lightweight_Action_Readout
src=$root/work/DriveWAM/src
guid=$root/data/physicalai/dev_guidance
sm=$root/data/physicalai/smoke_vlm/guid_dev_b$2
mkdir -p $sm
python3 -c "import json; ids=json.load(open('$guid/dev_clip_ids.json'))[:8]; json.dump(ids, open('$sm/ids.json','w'))"
source $root/scripts/remote_env.sh model >/dev/null
export PYTHONPATH=$root/.venvs/vlm-overlay${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $src
t0=$(date +%s)
python $root/scripts/pai_vlm_patch.py --dataset_root $root/data/physicalai/front --clip_index $root/data/physicalai/clip_index.parquet \
  --output_dir $sm --num_clips -1 --num_chunks 5 --chunk_seconds 4.0 --seed 0 --clip_ids_file $sm/ids.json \
  --model_path $root/checkpoints/qwen3-vl-8b-instruct --tensor_parallel_size 1 --batch_size $2 --max_model_len 8192 \
  --gpu_memory_utilization 0.9 --route_store $guid/route_store.json --bev_store $guid/bev_store.npz --output_name guidance.json
t1=$(date +%s); echo "{\"guidance_s\": $((t1-t0)), \"batch\": $2}"
python3 -c "import json; d=json.load(open('$sm/guidance.json')); print(len(d), sum(len(v) for v in d.values())); print(json.dumps(d[list(d)[0]], indent=1, ensure_ascii=False))"
