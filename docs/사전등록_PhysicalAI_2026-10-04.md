# 사전등록: PhysicalAI-AV 확장 · 2026-10-04

**작성 시점:** 모델을 하나도 실행하기 전. 데이터는 메타데이터(clip_index)와 clip 목록만 확인했다.
**상위 문서:** [PhysicalAI 확장 실행계획서](PhysicalAI_확장_실행계획서.md)
**고정 자산:**
- DriveWAM checkpoint `chenchenshi/DriveWAM@c4fb6d2c` `physicalai/` (SHA-256 `ad90feae…0d98`)
- NVIDIA PhysicalAI-AV revision `33f9bf447ed3bcb7d545ce13f4226f824214fafb`
- DriveWAM 코드 commit `abbefb40`

## 1. 데이터 분할 (고정, 서버 `reports/pai-splits.json`)

| 세트 | 구성 | 목록 SHA-256 |
|---|---|---|
| final (잠금) | 공식 `test_sample_1k.csv` 1,000 clip(test split) + 공개 chunk별 안내 문장 | `3760a84e…e94d` |
| dev 후보 | **val split**(clip_is_valid) 중 `SHA256("17:pai-devpool:"+clip_id)` 순위 상위 3,000 clip | `bd5dd9d5…acd5` |
| dev | dev 후보에 DriveWAM의 test 선별 절차를 그대로 적용해 고른 1,000 clip(§2) | 생성 후 해시 기록 |
| head-train | DriveWAM 공개 `train_4k.csv` ⊂ `train_20k.csv`(train split, 백본 학습 clip) | `b9bdf75f…eb95` / `8d4e247f…cff9` |

- **실행 전 계획 변경:** 실행계획서의 "test split에서 1k를 뺀 dev"를 **공식 val split**으로 바꾼다. val은 백본 학습에 쓰이지 않았고, 이렇게 하면 test split은 final에만 쓰인다.
- 세션 정보가 없으므로 통계 단위(cluster)는 **NVIDIA chunk 번호**로 한다. 같은 chunk의 clip이 같은 수집 구간일 수 있다는 보수적 가정이다.

## 2. dev 선별 (D1, 사용자 승인 2026-10-04)

1. dev 후보 3,000 clip을 DriveWAM `qwvl_scene_select.py`(Qwen3-VL-8B-Instruct, 20프레임, 기본 설정)로 태깅한다.
2. `qwvl_scene_stats.py`의 층화 표집(sample_total 1000, quantile 0.75, rare_per_tag 30, baseline 200, sample_seed 0)으로 1,000 clip을 고른다.
3. dev clip의 chunk별 안내 문장은 DriveWAM `vlm_preprocess` 3단계(route → BEV → scene-evolving guidance, Qwen3-VL-8B-Instruct, temperature 0)로 생성한다. 실행 환경 차이로 문구가 원 저자와 다를 수 있음을 기록한다.

## 3. 평가 protocol

- 공식 `predict_autoregressive`와 같다.
  - chunk마다 미래 영상 3단계(denoise fraction 0.6)를 거친 뒤 예측 기억을 지우고, 실제 영상·action을 등록한다.
  - clip마다 seed 0. 문장 임베딩은 원본 `_encode_single_prompt`와 비트 단위로 같게 미리 계산한다.
- 지표: 공식 ADE(160스텝, chunk별 실제 위치 재설정), FDE(chunk 끝 오차 평균), chunk 단위 FDE > 2m 비율, heading 오차(보조).
- 학생 경로는 B10과 **같은 순서로 난수를 소비**한다(action noise를 뽑고 버림). "B10 실행에 hook으로 얻은 first feature = 조기 종료 학생 경로 feature"를 32 clip에서 비트 단위로 검증한다.

## 4. 비교군과 B\*

- 기준: B10, B05, B10-opt, B05-opt, B03-opt, B02-opt, B01-opt(raw/opt 출력 비트 동일을 dev 32 clip에서 확인), CV.
- **B\* 규칙:** opt 후보 중 dev에서 다음을 만족하는 것 가운데 chunk당 결정 지연 p50이 가장 작은 모델이다.
  - B10 대비 ADE 차이 CI 상한 ≤ +0.05m, FDE 차이 CI 상한 ≤ +0.15m
  - 동률이면 action 단계가 많은 쪽을 고른다.
- 지연은 dev 고정 256 clip에서 장면 단위로 번갈아 3회 측정한다. 결정 지연 = 실제 기록 등록 이후의 미래 영상 단계 + action 단계(학생은 head)다.

## 5. 학생 모델과 학습

- SF3 회귀 head, **first 시점**, layer [10,20,30], query 16, 폭 512, 출력층 FP32, 출력 40스텝 × (dx, dy, dheading)(PhysicalAI quantile 정규화).
- ego 입력은 백본 조건과 같은 정보(속도·가속도 4 + 곡률 1)다.
- 학습 표본은 clip마다 공식 protocol과 같은 실제 이력 상태의 chunk 1–4다. (변경 기록 1: 실제 chunk 수는 5개이므로 chunk 1–5)
- 40 epoch 고정, AdamW 3e-4, wd 0.01, batch 32, warmup 5% + cosine, 마지막 checkpoint, dev 예측은 batch 1.
- 데이터 규모는 4k → 20k다. 최종 seed는 **17, 29, 43**이다.
- 시점 비교(current/first/terminal, 4k)와 진단 ablation(이력 절단, 고정 문장; 20k first → 변경 기록 2: 4k first)은 dev에서만 하며 판정에 넣지 않는다.

## 6. 판정 규칙

- **통과:** 학습 seed 3개 모두에서 다음을 만족해야 한다.
  - 학생 − B10, 학생 − B\*의 ADE 차이 95% CI 상한 ≤ **+0.05m**, FDE 차이 95% CI 상한 ≤ **+0.15m**(chunk-cluster paired bootstrap 2,000회)
  - chunk 단위 FDE > 2m 비율 증가(평균) ≤ **1%p**(B10 대비)
  - 결정 지연 p50 ≤ **0.80 × B\***, p95 악화 없음
- dev에서 설정(데이터 규모 4k/20k)을 고정한 뒤 final에서 1회 평가한다. **final 실행은 사용자가 2026-10-04에 사전 승인했으므로 추가 확인 없이 진행한다.**
- 결과와 관계없이 모두 보고하고, final 결과로 설정을 다시 고르지 않는다.
- 100k 확대는 20k에서 미통과이고 4k → 20k의 ADE 개선이 0.03m 이상일 때만 검토한다. 저장 공간 때문에 별도 압축 검증이 필요해, 이번 실행 범위에서는 보고만 한다.

## 변경 기록

- 2026-10-04: 최초 작성(모델 실행 전).
- 2026-10-04, 결과 산출 전(parity gate 통과 직후, 어떤 ADE/FDE도 보기 전) 수정:
  1. **chunk 수 정정.** 공식 test clip은 1fps 21프레임(잠재 프레임 6개)이라 미래 chunk가 **5개**다. parity gate의 forward 횟수로 확인했다(B10은 clip당 87회 = 등록 12 + 영상 20 + action 55, first 학생은 17회). §5의 "chunk 1–4"는 4개로 잘못 안 데서 나온 표현이고, 의도대로 **공식 protocol의 모든 chunk(1–5)** 로 학습한다.
  2. **진단 ablation 규모.** 이력 절단·고정 문장 진단은 20k가 아닌 **4k first**로 한다(시점 비교와 같은 규모). 특징 저장 공간과 GPU 시간 때문이다. 판정에는 쓰지 않는다.
  3. **데이터 규모 고정 규칙(§6 보완).** dev에서 seed 3개 평균 ADE가 낮은 규모를 고른다. 두 규모 차이가 0.01m 미만이면 20k를 고른다.
  4. **처리 실패 clip.** devkit egomotion 보간 범위 오류로 전처리가 실패한 clip은 제외한다. dev 후보에서 빠진 clip은 `reports/pai-devpool-failed.json`, 학습 세트에서 빠진 clip은 특징 status 파일에 기록한다. final 1,000 clip은 모두 처리됐다.
  5. **기준선 실행 방식.** opt 기준선 5개는 한 rollout에서 영상 단계·noise·실제 기록 등록을 공유하고 action 단계만 따로 돌린다(`rollout_multi`). 따로 돌린 결과와 비트 단위로 같음을 parity gate에서 확인했다. 각 shard 앞 16 clip에서는 raw B10을 함께 돌려 B10-opt와 비트 단위로 같은지 본다(dev 32 clip 확인).
  6. **지연 측정용 head.** 학생 지연은 처음 학습된 head(4k, seed 17)로 잰다. head 연산량은 가중치와 무관하다. 같은 실행에서 온라인 학생 출력과 캐시 특징으로 낸 출력이 같은지도 비교한다.
  7. **dev 선별을 D1에서 D2로 변경(2026-10-04 05:30 KST, dev 결과 산출 전).** DriveWAM 태깅 스크립트(`qwvl_scene_select`, 기본 batch 4, 20프레임)를 dev 후보 4 clip에서 시험했다. A5000 24GB에서 11분 동안 생성한 뒤 메모리 부족(OOM)으로 4 clip 모두 실패했다. batch 1로 줄여도 3,000 clip이면 GPU 50시간 이상으로 추정돼 실행할 수 없다. vLLM은 서버 드라이버(CUDA 12.6) 때문에 쓸 수 없다. 그래서 계획서의 대안 **D2**를 쓴다. dev = 처리에 성공한 dev 후보 clip을 사전등록 해시 순위(`SHA256("17:pai-devpool:"+clip_id)`)대로 앞에서 1,000개(목록 SHA-256 `1016200c…1f69`, `reports/pai-dev-split.json`)로 한다. 전처리 실패 6 clip은 순위에서 건너뛴다. chunk별 안내 문장 생성(route → BEV → guidance)은 §2-3 그대로 한다. 영향: dev는 무작위 val 장면이라 VLM이 고른 '흥미 장면' 위주의 final보다 쉬울 수 있다. B\* 선정과 규모 선택이 이 차이의 영향을 받을 수 있음을 보고서에 적는다.
  8. **지연 측정 clip 수(§4).** 256 clip이 아닌 dev 앞 **128 clip × 3회**로 잰다. 모든 clip의 입력 크기가 같아 chunk별 지연은 clip에 거의 무관하다. 모델당 결정 1,920개(128 × 3 × 5 chunk)면 p50·p95 추정에 충분하다. 이렇게 하면 GPU 시간이 약 2.3시간 줄어든다.
  9. **학생 출력 좌표계(§5 보완, 2026-10-04 06:40 KST, head 학습·dev 결과 전).** 데이터의 action은 **clip 시작 시점 좌표계**의 변위다(`physicalai_dataset.py`: `T_curr_inv = inv(pose[0])`). 학습 표본 1,600 chunk에서 chunk 시작 heading이 clip 시작과 15° 넘게 다른 경우가 18%, 45° 넘는 경우가 10%였다. 사전등록 입력(속도·가속도·곡률, 모두 현재 차량 좌표계)만으로는 출력을 얼마나 회전할지 알 수 없다. 그래서 head는 **chunk 시작 차량 좌표계**의 변위를 예측하고, 이미 등록된 실제 action에서 누적한 heading으로 clip 좌표계로 회전한다(`to_clip_frame`, `heading_before`). 이 heading은 공식 protocol에서 모든 방법이 기억으로 받는 정보다(실제 action 등록). 원래 사양(회전 없이 clip 좌표계 직접 예측)은 4k·seed 3개 ablation(`-clipframe`)으로 dev에서 함께 보고한다. 판정에는 회전 버전을 쓴다.
  10. **구현 메모(결과 영향 없음).** 원본 평가 코드는 문장 임베딩을 GPU에 무기한 캐시해 24GB GPU에서 메모리 부족이 났다. clip마다 캐시를 비우고 미리 계산한 값을 다시 읽는다(값 동일).

## 결과 기록 (2026-10-04)

- **dev(§4–§6):** B\*=**B03-opt**(B05-opt, B03-opt만 자격, 결정 지연 p50 434.0ms). 규모는 **20k**(seed 평균 dev ADE 0.762m, 4k 0.813m). 20k 학생 seed 3개 모두 7개 기준 통과. 4k 학생은 FDE>2m 비율 기준(+2.3–3.7%p)에서 미달.
- **final(1회, 16:30 KST):** seed 17/29/43 모두 통과.
  - ADE 0.750 / 0.748 / 0.762m, FDE 2.244 / 2.244 / 2.287m.
  - B10 대비 ΔADE −0.051 / −0.054 / −0.040m, 95% CI 상한 모두 0 미만.
  - FDE>2m 비율은 B10 대비 −2.5 / −2.9 / −1.8%p.
  - 결정 지연 p50 70.9ms로 B\*의 0.163배, p95는 84.2ms 대 502.9ms.
  - B10 재현은 ADE 0.802 / FDE 2.384(논문 0.83 / 2.47).
- **배포 경로 확인:** 온라인 학생(백본 + head) 출력이 캐시 feature 예측과 비트 단위로 같았다(dev 128 clip, 640 chunk).

## 사후 분석 (사전등록 외, 결과를 본 뒤 추가)

- **결정성 대조(2026-10-04 16:40 KST 결정):** 학생의 정확도 우위가 결정적 회귀(평균 예측) 때문인지 확인한다.
  - dev에서 B10-opt를 seed 0–3으로 돌리고, K=1–4개 표본 평균의 ADE/FDE를 공식 지표로 잰다.
  - ADE(K)≈a+b/K로 무한 표본 평균을 외삽한다.
  - 20k 학생과 4표본 평균을 같은 bootstrap으로 비교한다.
  - final은 다시 쓰지 않는다. 판정에는 영향이 없다.
- **결정성 대조 결과(2026-10-04 21:21 KST):**
  - B10 seed 0–3의 dev ADE는 0.821 / 0.869 / 0.882 / 0.930이다. 4개 평균은 ADE 0.777, FDE 2.200, 외삽값은 0.750 / 2.080이다.
  - 20k 학생 대 4개 평균: ΔADE −0.011 ~ −0.021, ΔFDE +0.084 ~ +0.117(CI > 0).
  - 공식 B10 대비 우위의 상당 부분은 결정성 효과다. 판정(사전등록 기준)은 바뀌지 않는다.
- **진단 결과(사전등록 진단, dev, 4k, seed 3개; 2026-10-04 23:40 KST, 판정 무관):** 모두 first 대비 ΔADE다.
  - current: +0.106 ~ +0.123
  - terminal: +0.008 ~ +0.014(CI가 0 포함)
  - 기억 제거: +0.020 ~ +0.037
  - 고정 문장: +0.034 ~ +0.045
  - 원래 좌표계 사양(clipframe): +0.260 ~ +0.269
