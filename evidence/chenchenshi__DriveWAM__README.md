<div align="center">
<h1>DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving</h1>

<a href="https://arxiv.org/abs/2605.28544"><img src="https://img.shields.io/badge/Paper-b31b1b" alt="Paper"></a>
<a href="https://chenshi3.github.io/drivewam.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

<br>

Chen Shi\*, Jinrui Xu\*, Shaoshuai Shi, Kehua Sheng, Bo Zhang, Li Jiang†

*The Chinese University of Hong Kong, Shenzhen &amp; Voyager Research, Didi Chuxing*

\*Equal Contribution, †Corresponding Author

</div>

**DriveWAM** is a joint **video generation and action prediction** model for autonomous driving. It adapts a pretrained video diffusion transformer into an **autoregressive video-action policy**, organizing video and action streams into a unified temporal token sequence trained under a joint flow-matching objective — preserving video generation priors while extending the model to ego-motion action prediction.

<!-- Demo video: add a github user-attachments link here -->

## Highlights

### NavSim

Comparison on NAVSIM v1. \*: results with imitation learning. †: trained with multiple trajectory anchors. MV: multi-view cameras; SV: single-view camera; L: LiDAR.

<div align="center">

| Method | Ref | Sensors | NC ↑ | DAC ↑ | TTC ↑ | C. ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---|---|---|---|---|---|
| Human | – | – | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| UniAD | CVPR'23 | MV | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| TransFuser | TPAMI'23 | MV & L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| PARA-Drive | CVPR'24 | MV | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW | ICLR'25 | SV | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| DiffusionDrive | CVPR'25 | MV & L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE | ICCV'25 | MV & L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| *VLA-based Methods* | | | | | | | | |
| ReCogDrive\* | ICLR'26 | MV | 98.1 | 94.7 | 94.2 | 100.0 | 80.9 | 86.5 |
| DriveVLA-W0 | ICLR'26 | SV | 98.7 | 96.2 | 95.5 | 100.0 | 82.2 | 88.4 |
| AutoVLA | NeurIPS'25 | MV | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveDreamer-Policy | arXiv'26 | MV | 98.4 | 97.1 | 95.1 | 100.0 | 83.5 | 89.2 |
| DriveVLA-W0† | ICLR'26 | SV | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| *WA-based Methods* | | | | | | | | |
| Epona | ICCV'25 | SV | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WorldDrive | arXiv'26 | SV | 98.4 | 95.8 | 95.2 | 99.8 | 83.3 | 89.0 |
| **DriveWAM (Ours)** | – | SV | 98.3 | **98.1** | 95.2 | 100.0 | **84.3** | **90.1** |

</div>

### PhysicalAI-AV

Comparison on PhysicalAI-Autonomous-Vehicles. 

<div align="center">

| Method | Source | ADE@3s ↓ | FDE@3s ↓ | ADE@4s ↓ | FDE@4s ↓ |
|---|---|---|---|---|---|
| VaVAM | Valeo | 2.31 | 4.32 | - | - |
| Alpamayo-1.5 | NVIDIA | 0.80 | 2.31 | 1.44 | 4.18 |
| **DriveWAM (Ours)** | — | **0.47** | **1.35** | **0.83** | **2.47** |

</div>

### Qualitative Results

<div align="center">

![Qualitative Results](assets/result_vis.jpg)

</div>

### Data Scaling

DriveWAM's action prediction error improves consistently as training data scales from 4k to 100k clips. Scene-evolving (SE) guidance provides complementary benefit at every scale.

<div align="center">

<img src="assets/data_scaling.jpg" width="300">

| # Clips | # Iters | SE Guidance | ADE@4s ↓ | FDE@4s ↓ |
|---|---|---|---|---|
| 4k | 50k | ✗ | 1.21 | 3.65 |
| 4k | 50k | ✓ | 1.01 | 2.95 |
| 20k | 50k | ✗ | 0.95 | 2.94 |
| 20k | 50k | ✓ | 0.94 | 2.65 |
| 100k | 50k | ✗ | 0.92 | 2.75 |
| 100k | 50k | ✓ | **0.83** | **2.47** |

</div>

## News
- [Jun 7, 2026] We open-source all code and model weights.
- [May 27, 2026] We release the paper and project page.

## Getting Started

### Installation

First, clone this repository and set up the environment.

```bash
git clone <repo-url>
cd DriveWAM

# 1. Create conda environment
conda env create -f environment.yml
conda activate drivewam

# 2. Install PyTorch (CUDA 12.6)
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Install Flash Attention
pip install flash-attn==2.8.3 --no-build-isolation
```

Two optional extras, installed when you need the corresponding feature:

```bash
# NavSim evaluation extras (for the NavSim benchmark)
pip install -r requirements-navsim.txt

# VLM preprocessing extras (to generate navigation guidance)
pip install vllm qwen-vl-utils
```

## Data Preparation

DriveWAM trains and evaluates on two benchmarks. Prepare whichever you need.

### NavSim

Follow the [NavSim installation guide](https://github.com/autonomousvision/navsim) to download the nuPlan-based dataset and cache metric files, and export the environment variables from that guide (`OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_MAP_VERSION`). Then extract per-scene samples:

```bash
# navtrain split (training)
python -m src.navsim.process_data --output-path ./data/navsim/trainval

# navtest split (evaluation)
python -m src.navsim.process_data \
    --navsim-log-path $OPENSCENE_DATA_ROOT/navsim_logs/test \
    --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/test \
    --scene-filter-config navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml \
    --output-path ./data/navsim/test
```

Each scene becomes one pkl file, which is what the training and evaluation scripts read by default:

```
./data/navsim/trainval/
    sample_000000.pkl
    sample_000001.pkl
    ...
```

### PhysicalAI-Autonomous-Vehicles

The raw dataset is hosted on [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) and accessed through the [physical_ai_av](https://github.com/NVlabs/physical_ai_av) devkit. The devkit requires Python ≥ 3.11, so install it in a separate environment from `drivewam`:

```bash
pip install physical_ai_av
```

Accept the dataset license on the Hugging Face page, then download the dataset (or a subset of chunks) to `./data/physicalai`. DriveWAM only needs the **`camera_front_wide_120fov`** camera plus the egomotion and calibration features. Extract 10 Hz clips from the download:

```bash
python -m src.physicalai.process_data \
    --dataset_root ./data/physicalai \
    --output_dir ./data/physicalai/front \
    --num_workers 16
```

This writes one directory per clip:

```
./data/physicalai/
├── clip_index.parquet          # official train/test split; keep it even if you prune the raw chunks
└── front/
    └── <clip_id>/
        ├── camera_front_wide_120fov.mp4
        └── camera_front_wide_120fov_ego.pkl
```

VLM navigation prompts are used as conditioning during training and inference. We provide pregenerated prompts: training-split prompts are available on [Hugging Face](https://huggingface.co/chenchenshi/DriveWAM); the 1k-sample test-split prompts used for evaluation are included in the repo at `src/physicalai/eval_data/prompts_test_sample_1k.json`. To regenerate them yourself:

```bash
# Step 1 – generate route / BEV / scene-evolving guidance
bash scripts/drivewam_physicalai_vlm_preprocess.sh

# Step 2 – VLM-based clip quality filtering and sub-sampling
SPLIT=train \
bash scripts/drivewam_physicalai_vlm_data_sample.sh
```

## Training

DriveWAM model checkpoints are available on [Hugging Face](https://huggingface.co/chenchenshi/DriveWAM). DriveWAM is trained on top of [LingBot-VA Base](https://huggingface.co/robbyant/lingbot-va-base), a pretrained autoregressive diffusion transformer. Download the base model weights before training.

Key training hyperparameters (see configs for full details):

| Hyperparameter | NavSim / PhysicalAI |
|---|---|
| Training steps | 50 000 |
| Learning rate | 1e-5 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| Warmup steps | 10 |
| Batch size (per GPU) | 1 |
| Precision | bfloat16 |
| Input resolution | 256×448 |
| SNR shift (video / action) | 5.0 / 1.0 |

All experiments are conducted on 48 × NVIDIA H20 GPUs.

Edit the config (`src/configs/navsim_cfg.py` or `src/configs/physicalai_cfg.py`) to set your paths and hyperparameters, then launch with the matching script:

| Benchmark | Config | Launch script |
|---|---|---|
| NavSim | `src/configs/navsim_cfg.py` | `scripts/drivewam_navsim_train.sh` |
| PhysicalAI | `src/configs/physicalai_cfg.py` | `scripts/drivewam_physicalai_train.sh` |

```bash
# NavSim
bash scripts/drivewam_navsim_train.sh

# PhysicalAI
bash scripts/drivewam_physicalai_train.sh
```

For PhysicalAI, we provide three CSV files listing clip IDs at different training data scales (4k / 20k / 100k clips), available on [Hugging Face](https://huggingface.co/chenchenshi/DriveWAM). Set the `clip_csv` field in `src/configs/physicalai_cfg.py` to the desired scale before training.

## Evaluation

### NavSim (PDM Score)

PDM score evaluation requires a metric cache — a set of per-scenario `.pkl` files that store precomputed map and route information to score each predicted trajectory. The precomputed metric cache for the navtest split is available for download on [Hugging Face](https://huggingface.co/chenchenshi/DriveWAM). To generate it yourself, run:

```bash
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=./data/navsim/metric_cache
```

This writes one `metric_cache.pkl` per scenario token under `./data/navsim/metric_cache/`. Pass the resulting directory to the evaluation script via `--metric-cache-path`.

```bash
python -m src.navsim.eval \
    --checkpoint-path /path/to/checkpoint \
    --config-name navsim_cfg \
    --dataset-path ./data/navsim/test \
    --metric-cache-path ./data/navsim/metric_cache
```

### PhysicalAI

```bash
python -m src.physicalai.eval \
    --checkpoint-path /path/to/checkpoint \
    --config-name physicalai_cfg
```

## Citation

If you find DriveWAM useful, please cite:

```bibtex
@article{shi2026drivewam,
  title={DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving},
  author={Shi, Chen and Xu, Jinrui and Shi, Shaoshuai and Sheng, Kehua and Zhang, Bo and Jiang, Li},
  journal={arXiv preprint arXiv:2605.28544},
  year={2026}
}
```

## Acknowledgements

We gratefully acknowledge the following open-source projects that DriveWAM builds upon: [Wan2.2](https://github.com/Wan-Video/Wan2.2), [LingBot-VA](https://github.com/robbyant/lingbot-va), [NavSim](https://github.com/autonomousvision/navsim), [NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
