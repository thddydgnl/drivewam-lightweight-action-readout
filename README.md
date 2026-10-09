# DriveWAM Lightweight Action Readout

**Reading driving actions from a frozen World-Action Model's first imagination pass**

[한국어 README](README.ko.md) · [Technical report (KO, PDF)](docs/report/DriveWAM_Lightweight_Action_Readout_연구설명.pdf) · [Integrated research log (KO)](docs/통합_연구기록.md) · [Evidence](evidence/)

Author: **YongHwi Song (송용휘)**, B.S. candidate, Chungbuk National University · [homepage](https://thddydgnl.github.io/) · Research period: 2026-09-15 to 2026-10-08

---

## TL;DR

[DriveWAM](https://github.com/chenshi3/DriveWAM) is a 6.24B-parameter driving World-Action Model built on a video diffusion transformer (Wan2.2-TI2V-5B). It first *imagines* future video latents and then runs the same 5B backbone **10 more times** to denoise a 4-second action chunk. Action denoising is the single largest latency term in the pipeline.

This project asks: **without retraining the backbone, can a small head read the trajectory directly from the representation DriveWAM forms the moment it imagines the future once?**

- We keep the official checkpoints frozen, hook the hidden states of transformer blocks 10/20/30 **during the first future-denoising pass**, and regress the 4-second trajectory with a ~10M-parameter head (**SF3**, State-Fusion over 3 levels). Inference stops right after that pass. The remaining future denoising and all 10 action-denoising passes are skipped.
- Evaluation followed **pre-registered protocols** on two benchmarks: NAVSIM v1.1 (single-scene planning, closed-form PDMS) and NVIDIA PhysicalAI-AV (continuous driving with memory and language guidance, ADE/FDE).
- The proposed readout is compared not only against official DriveWAM (**B10**) but against **B\***, the fastest reduced-step action-denoising variant that still meets a pre-registered quality bar on the dev set. This is the fair baseline: "just use fewer steps".

### Headline results (locked final sets, evaluated once)

| | NAVSIM `navtest-final` (8,896 scenes) | PhysicalAI-AV official test 1k |
|---|---|---|
| Official DriveWAM (B10) | PDMS 89.67, policy p50 797.9 ms | ADE 0.802 m / FDE 2.384 m, decision p50 837.8 ms |
| B\* (fastest quality-preserving low-step baseline) | B02-opt (2 action steps), 420.7 ms | B03-opt (3 action steps), 434.0 ms |
| **Proposed (SF3-first)** quality | PDMS 89.25 / 89.49 (2 seeds) | ADE 0.750 / 0.748 / 0.762 m, FDE 2.244 / 2.244 / 2.287 m (3 seeds) |
| **Proposed (SF3-first)** decision latency | **189.0 ms** (2.2× faster than B\*, 4.2× faster than B10) | **70.9 ms** (6.1× faster than B\*, 11.8× faster than B10) |
| Pre-registered verdict | **Near miss**: one seed violated the DAC rule by a single scene (0.51 vs 0.50 allowed) | **Pass**: 3 seeds × 7 criteria |

![Architecture comparison](docs/report/figures/fig2_compare.png)

*Top: original DriveWAM. Bottom: the proposed path stops after future pass 1 and reads blocks 10/20/30 with the SF3 head. The backbone is never updated.*

![Quality vs latency on the final sets](docs/report/figures/fig9_tradeoff.png)

*Lower-left is better. On both benchmarks the proposed model sits left of B\* at near-B\* quality.*

---

## Key findings

1. **One imagination pass is what matters.** Features from the *first* future-denoising pass (noise tokens that have attended once to the observation cache) beat features from the observation-only pass (*current*) by a wide margin: +3.3 PDMS on NAVSIM, −0.11 m ADE on PhysicalAI. Running the imagination to completion (*terminal*) adds nothing measurable for readout (ΔADE +0.008 to +0.014 m, CI includes 0). This is what licenses early exit.
2. **Removing the action generator beats shrinking it.** B\* already halves DriveWAM's latency by cutting action steps to 2–3. The readout is a further 2.2× (NAVSIM) and 6.1× (PhysicalAI) faster than B\* because it drops the remaining future passes and the whole action-denoising loop.
3. **Head training data scale decides the outcome.** Same architecture, more labels: NAVSIM gap to B10 went −4.2 → −1.3 → −0.4 PDMS with 4k → 20k → 92.6k scenes. On PhysicalAI the 4k head failed the tail-error criterion and the 20k head passed. Concluding "world features contain no action information" from a small head-train set would have been wrong.
4. **Memory and language guidance help, but less than imagination.** In the continuous setting, removing past-chunk memory costs +0.02–0.04 m ADE and removing the VLM guidance sentence costs +0.03–0.045 m. Both are smaller than the imagination effect (+0.11 m).
5. **Methodological traps we hit and documented.** (a) The public NAVSIM backbone has memorised navtrain scenes: B10's 4-s ADE is 0.076 m on navtrain-internal validation vs 0.598 m on unseen navtest, so all conclusions were moved to unseen navtest and a locked final split. (b) The official single-seed PhysicalAI protocol is seed-sensitive: B10 dev ADE ranges 0.821–0.930 over seeds 0–3, and the official seed 0 is the luckiest. (c) A deterministic regressor vs a stochastic generator is not a fair accuracy comparison on mean-error metrics: against a 4-sample B10 average the proposed model is on par in ADE and ~0.1 m worse in FDE. **We therefore do not claim the readout is more accurate than DriveWAM.** The claim is: mean-prediction-level quality in a single pass, at a fraction of the latency.

---

## Method in brief

**Frozen backbone and taps.** Official DriveWAM checkpoints for NAVSIM and PhysicalAI are used unmodified. The official inference loop is re-implemented step by step (verified bitwise against upstream) and forward hooks read block outputs without altering them.

**Three readout timings** (same head, trained separately for each):

| Timing | What the backbone has done so far | Cost |
|---|---|---|
| `current` | registered the observation frames; no imagination | cheapest |
| **`first`** | one future pass over pure-noise future tokens (first imagination) | **what the proposed model uses** |
| `terminal` | all three future-latent updates | requires full future generation |

**SF3 head** (`scripts/lightweight_heads.py`, `heads_v3.py`, `pai_heads.py`), ~9.98M trainable parameters (0.16% of the backbone):

- Per level (blocks 10/20/30; 112 tokens × 3072): LayerNorm → Linear 3072→512 → 16 learned queries × 8-head attention → learned weighted sum → LayerNorm.
- Ego MLP (NAVSIM: 4 past poses + velocity/acceleration + command, 20-d; PhysicalAI: velocity/acceleration/curvature, 5-d) → 128-d.
- Fusion: concat (1,664) → Linear 512 + GELU → residual MLP 512→1024→512.
- Output: driving state + per-step time embedding → LayerNorm → MLP → normalised (Δx, Δy, Δθ) per step, FP32 output layer; K = 8 poses @ 2 Hz (NAVSIM) or 40 steps @ 10 Hz (PhysicalAI). Quantile (1%/99%) normalisation, cumulative sum to poses.
- Loss: Huber on positions + 0.2 × (1 − cos Δθ) + 0.1 × Huber on normalised deltas.
- Training protocol (frozen from v2 on): 40 epochs, AdamW 3e-4, wd 0.01, batch 32, 5% warm-up + cosine, **last checkpoint, no selection on the evaluation set**, batch-1 prediction as in deployment.

**Baselines.** B10 (official), and *opt* variants B10/B05/B03/B02/B01-opt that skip the output-irrelevant final action cache pass and reduce action steps (verified bitwise identical to raw). **B\*** = fastest opt variant that meets the dev-set quality rule. Reference: ego-only head, constant-velocity.

**Statistics.** Cluster-paired bootstrap, 2,000 resamples (NAVSIM: per log; PhysicalAI: per NVIDIA chunk). Latency: three interleaved passes in the same session, models alternated per scene/clip; verdicts use ratios because absolute times drift on the shared host.

**Bitwise verification before any result was read.** Rollout re-implementation = upstream (B10, B05); opt = raw; shared multi-baseline rollout = individual runs; hooks do not perturb outputs; early-exit features = features captured during a full run; online (backbone + head) = offline (cached features) predictions; forward counts per clip (B10 87, proposed 17).

---

## Experimental setup

| | NAVSIM v1.1 | PhysicalAI-AV |
|---|---|---|
| Data | OpenScene revision `a76f840b` | NVIDIA revision `33f9bf44`, front camera, 23,994 / 24,000 clips processed |
| Unit | one scene → one 4-s trajectory | one clip (21 frames @ 1 fps) → 5 consecutive 4-s chunks, teacher forcing |
| Head-train | navtrain head-train 4k → 20k → 92,599 scenes (1,072 logs) | DriveWAM public train 4k ⊂ 20k clips |
| Dev / final | navtest-dev 34 logs / 3,250 scenes; **locked** navtest-final 102 logs / 8,896 scenes | val 1,000 clips (hash-ranked); official test 1k |
| Metrics | official PDMS (+ NC, DAC, EP, ADE4) | official ADE / FDE (+ chunk FDE > 2 m rate) |
| Quality rule for B\* and the proposed model | PDMS CI lower bound ≥ −2.0 vs B10 and B\*, NC/DAC mean degradation ≤ 0.5, p50 ≤ 0.8 × B\*, both seeds | ΔADE CI upper ≤ +0.05 m, ΔFDE CI upper ≤ +0.15 m, FDE>2m rate increase ≤ 1 pp, p50 ≤ 0.8 × B\*, no p95 regression, all 3 seeds |
| Hardware | 2 × RTX A5000 24 GB, shared host (load avg 12–60 from other tenants) | same |

Guidance sentences for the PhysicalAI dev set were regenerated with the DriveWAM pipeline (Qwen3-VL-8B; mean 27.8 words vs 28.0 in the official test prompts).

---

## Research timeline

| Date (2026) | Step | Outcome |
|---|---|---|
| 09-15 → 09-30 | Plan, source audit (20 items), environment & data locks | Pinned commits/revisions with SHA-256; no input leakage vs upstream |
| 10-01 | Baselines on 256 dev scenes; 12-head pilot; feature-swap control | B05 −26% latency at equal PDMS; swapping in another scene's features collapses PDMS to 35–41 (features carry the signal) |
| 10-02 | **Found backbone memorisation of navtrain**; fixed head protocol (v2); switched to interleaved latency; wrote NAVSIM pre-registration | navtest-dev: B\* = B02-opt; all 4k heads −3.6 to −7.7 PDMS; first ≈ terminal ≫ current |
| 10-03 | Round 3 (FP32 output, anchor head, 4k → 20k); Round 4 (92.6k) | 20k closes gap to −1.3; 92.6k to −0.4 PDMS; anchor head memorises training scenes |
| 10-03 | NAVSIM final (one run, recorded as a user-approved exception) | Near miss (§ above) |
| 10-04 | PhysicalAI extension: parity gate → dev baselines → heads (4k, 20k) → latency → dev verdict → final (one run) → diagnostics → determinism control | Pass; first ≫ current, first ≈ terminal; memory/language effects quantified |
| 10-05 → 10-08 | Integrated log, 15-page technical report, manuscript | — |

---

## Repository layout

```
.
├── README.md / README.ko.md
├── docs/
│   ├── README.md                       NAVSIM progress record (KO; was research_plan/README.md)
│   ├── 통합_연구기록.md                 integrated log of both benchmarks: process, results, limits (KO)
│   ├── 실행계획서.md                    detailed execution plan (KO)
│   ├── 사전등록_navtest-dev_2026-10-02.md     NAVSIM pre-registration + change log (KO)
│   ├── 사전등록_PhysicalAI_2026-10-04.md      PhysicalAI pre-registration + change log (KO)
│   ├── PhysicalAI_확장_실행계획서.md      PhysicalAI extension plan (KO)
│   ├── PhysicalAI_결과보고서.md           PhysicalAI results report (KO)
│   ├── 외부코드_검수보고서.md             upstream code audit, 20 items (KO)
│   ├── DriveWAM_Lightweight_Action_Readout_Research_Idea.md   original research proposal (KO)
│   ├── WAM-AD_투고계획.md                 venue planning notes (KO)
│   ├── report/                          15-page technical report (PDF/DOCX), figures, figure/doc build scripts
│   └── paper/                           manuscript PDF (to be added)
├── scripts/                             all experiment code (see below)
├── configs/
│   ├── SOURCE_LOCK.json                 pinned upstream commits, model revisions, file hashes
│   ├── requirements-model.candidate.txt direct dependencies validated on the GPU host
│   └── pai_jobs.json, pai_jobs_diag.json   PhysicalAI job-queue definitions
├── evidence/                            raw results: JSON/CSV/JSONL reports, splits, logs, per-scene predictions
├── CITATION.cff · LICENSE
└── vendor/                              (local only, git-ignored) audited copies of upstream repos
```

### Code map (`scripts/`)

| Module | Role |
|---|---|
| `world_feature_tap.py` | Forward hooks on video blocks; `current` / `first` / `terminal` phases; early exit with KV-cache cleanup |
| `lightweight_heads.py` | `QueryPooler`, `EgoEncoder`, trajectory decoder, loss; all head variants used in the pilot |
| `heads_v3.py` | SF3 trunk with FP32 output; regression (R) and anchor (A) heads from round 3 |
| `pai_heads.py` | PhysicalAI SF3 head: 40-step output, quantile normalisation, chunk-frame ↔ clip-frame rotation |
| `pai_common.py` | PhysicalAI runtime: step-by-step rollout (baselines, opt, early exit), prompt cache, feature taps |
| `pai_vlm_patch.py` | Qwen3-VL patch-embed workaround for slow bf16 Conv3d on torch 2.9 |
| `navsim_action_opt.py`, `baseline_common.py`, `navsim_current_dataset.py` | NAVSIM opt baselines, frozen-model init, rollout batch construction |
| `head_policy_runtime.py` | Batch-1 online policy (backbone + head) used for the final evaluations |
| `remote_*_pipeline.sh`, `remote_pai_gate.sh`, `remote_pai_worker.py` | Pipeline drivers: pilot, navtest-dev, rounds 3/4, NAVSIM final, PhysicalAI job queue |
| `remote_*_infer.py` / `_score.py` / `_analyze.py` | Inference, official scoring, paired-bootstrap analysis per stage |
| `remote_*latency*.py` | Interleaved same-session latency measurement |
| `make_pai_report.py` | Builds `evidence/physicalai_2026-10-04/tables.md` from the analysis JSONs |
| `remote_env.sh`, `remote_bootstrap.py`, `remote_prepare.py` | Server environment (two isolated venvs: model and official scorer) |

### Where each number comes from

| Result | Evidence |
|---|---|
| NAVSIM dev baselines (256 scenes) | `evidence/baselines_devval256_2026-10-01/`, `opt_devval256_2026-10-01/` |
| NAVSIM pilot heads, feature-swap control | `evidence/head_pilot_2026-10-01/`, `heads_online_devval256_2026-10-01*/`, `world_features_pilot_2026-10-01/` |
| NAVSIM navtest-dev (B\*, timing, depth) | `evidence/navtest_dev_2026-10-02/` |
| NAVSIM scale curve (20k, 92.6k) | `evidence/round3_2026-10-03/`, `evidence/round4_2026-10-03/` |
| NAVSIM final | `evidence/navtest_final_2026-10-03/` |
| PhysicalAI (parity, dev/final analysis, diagnostics, determinism control) | `evidence/physicalai_2026-10-04/reports/*.json`, `tables.md` |
| Environment and data locks | `evidence/remote_environment_2026-09-30/`, `evidence/navsim_data_2026-09-30/` |
| Disk clean-up record (what was deleted on the server) | `evidence/disk_cleanup_2026-10-05/` |

---

## Reproducibility notes

This repository is a **complete record** of the experiments, not a turnkey package. Please read this before trying to run anything.

- **Not included (size or licence):** model weights (`*.safetensors`, including trained heads), the 221 MB pilot evidence tarball, NAVSIM/PhysicalAI data, and the audited copies of upstream repositories. Everything needed to re-obtain them is pinned in [`configs/SOURCE_LOCK.json`](configs/SOURCE_LOCK.json): DriveWAM `abbefb40`, Light-WAM `db7d05e7`, navsim `3e8291bf`, lingbot-va `7c6ffa9b`; HF `chenchenshi/DriveWAM` revision `c4fb6d2c`; OpenScene `a76f840b`; PhysicalAI-AV `33f9bf44`.
- **Server layout assumed by the scripts:** `$LITE_ROOT = ~/DriveWAM_Lightweight_Action_Readout` with `work/DriveWAM/src` (pinned commit), `upstream/navsim-v1.1`, `data/`, `runs/`, `caches/`, `reports/`, `scripts/` (this folder), and two virtual environments activated by `scripts/remote_env.sh model|scorer`. The official NAVSIM scorer must live in a separate CPU environment (numpy 1.23 + OpenBLAS pinned to the Haswell kernel; see the comment in `remote_env.sh`).
- **Environment:** torch 2.9.0 + cu126, flash-attn 2.8.3, diffusers 0.36.0, transformers 4.55.2; full resolved lists in `evidence/remote_environment_2026-09-30/locks/`.
- **Order of operations:** `remote_pilot_pipeline.sh` → `remote_navtest_dev_pipeline.sh` → `remote_round3_pipeline.sh` → `remote_round4_pipeline.sh` → `remote_final_pipeline.sh` for NAVSIM; `remote_pai_gate.sh` (parity) then `remote_pai_worker.py` consuming `configs/pai_jobs.json` / `pai_jobs_diag.json` for PhysicalAI; `make_pai_report.py` for tables.
- **Known gotchas recorded in the docs:** bf16 Conv3d patch-embed is ~10 s/call on torch 2.9 + cuDNN 9.10 (use `pai_vlm_patch.py`); the upstream PhysicalAI evaluator caches prompt embeddings on GPU indefinitely (cleared per clip here, values identical); head predictions must use batch 1 and an FP32 output layer (bf16 output changed results at the cm level with batch size).
- **Server-side state:** NAVSIM intermediate features and preprocessed inputs were deleted on 2026-10-05 to free disk; NAVSIM retraining requires re-extraction. The log is in `evidence/disk_cleanup_2026-10-05/log.jsonl`.

---

## Limitations

- One world model (DriveWAM); open-loop, non-reactive scoring. Closed-loop safety is untested.
- NAVSIM final is a near miss, and that run was made after the dev set had missed by one scene (recorded as an exception).
- PhysicalAI dev is a random val subset while the official test is VLM-curated; B10 ADE was similar on both (0.821 vs 0.802).
- The chunk-frame output rotation on PhysicalAI was decided before results but strongly affects them (without it, ADE +0.26 m).
- Deterministic head vs stochastic generator is favourable to the head on mean-error metrics (see finding 5c). Multimodal or safety-oriented metrics are needed.
- No comparison against one-step distillation of the action generator (scoped out).
- Latency is a ratio measured on one A5000 on a shared host; VAE and text-encoder costs are common to all methods and excluded.
- Hypothesis H2 (multi-level > single-level) is directionally consistent but significant in only one seed; left open.

---

## Documents

All research documents are in Korean. Start with [`docs/통합_연구기록.md`](docs/통합_연구기록.md) for the complete story, then [`docs/PhysicalAI_결과보고서.md`](docs/PhysicalAI_결과보고서.md) and [`docs/README.md`](docs/README.md) for benchmark-level detail. The pre-registration files record every rule that was fixed before results were read and every deviation with its timestamp.

The 15-page technical report [`docs/report/DriveWAM_Lightweight_Action_Readout_연구설명.pdf`](docs/report/DriveWAM_Lightweight_Action_Readout_연구설명.pdf) (figures 1–9 in `docs/report/figures/`) is the basis of the manuscript. The manuscript PDF will be added under `docs/paper/`.

## Citation

```bibtex
@misc{song2026drivewamreadout,
  author = {Song, YongHwi},
  title  = {DriveWAM Lightweight Action Readout: reading driving actions from a frozen world-action model's first imagination pass},
  year   = {2026},
  url    = {https://github.com/thddydgnl/drivewam-lightweight-action-readout}
}
```

## Acknowledgements and licences

Built on the public code and checkpoints of [DriveWAM](https://github.com/chenshi3/DriveWAM) (Apache-2.0), [NAVSIM](https://github.com/autonomousvision/navsim) (Apache-2.0), [Light-WAM](https://github.com/L1ziang/Light-WAM) (MIT), [lingbot-va](https://github.com/robbyant/lingbot-va) (Apache-2.0), and the NVIDIA PhysicalAI-AV dataset. None of their code or weights is redistributed here. Code and documents in this repository are released under the [MIT License](LICENSE).
