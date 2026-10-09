# PhysicalAI-AV 확장 실행계획서 — 연속 주행 World-Action Model에서의 경량 action readout

**작성일:** 2026-10-04
**상태:** 2026-10-04 실행 중. 확정 규칙은 [사전등록 문서](사전등록_PhysicalAI_2026-10-04.md)가 우선한다. 실행 중 확인된 정정: ① test clip은 미래 chunk **5개**(1fps 21프레임, 잠재 프레임 6개)라 B10은 clip당 **87회**, first 학생은 **17회** forward다(parity gate 측정). ② dev는 test split이 아니라 **공식 val split**에서 만든다. ③ 진단 ablation은 4k에서 한다. 아래 본문의 '4 chunk·70회·14회'는 작성 당시 추정이다.
**선행 결과:** [README](README.md)의 NAVSIM 결과(navtest-dev·navtest-final), [사전등록 문서](사전등록_navtest-dev_2026-10-02.md)
**근거 코드:** 고정 DriveWAM commit `abbefb40` 사본 [source_audit/DriveWAM](https://github.com/chenshi3/DriveWAM/tree/abbefb407c9c553f221522fc26077d307620adb4)

> 이 문서는 실행 전 설계다. 수치 중 "추정"으로 표시한 것은 측정값이 아니다. 실제 실행 전에 §11의 사용자 결정을 받고, 판정 규칙(§6)을 별도 사전등록 파일로 해시와 함께 고정한다.

---

## 0. 요약

- **무엇을 하는가:** NAVSIM에서 검증한 방법을 DriveWAM의 두 번째 벤치마크인 NVIDIA **PhysicalAI-Autonomous-Vehicles**로 확장한다. 방법은 "고정된 DriveWAM의 첫 미래 패스 표현 + 작은 SF3 head"로 반복 action denoising을 대체하는 것이다. PhysicalAI는 분야 이름이 아니라 **주행 데이터셋 이름**이고, 모델은 같은 DriveWAM 월드모델(PhysicalAI용 별도 checkpoint)이다.
- **왜 하는가:** NAVSIM은 장면마다 한 번 계획하는 스냅샷 평가였다. PhysicalAI는 월드모델이 **과거 영상·action을 기억(KV cache)하며 chunk 단위로 미래를 이어서 상상**하는 연속 주행 설정이다. 또 chunk마다 VLM이 만든 **장면 안내 문장**으로 조건화된다. 논문이 action 병목(H20에서 765ms)을 제기한 것도 이 설정이다.
- **핵심 질문(월드모델 관점):**
  1. 기억을 가진 월드모델에서도 첫 상상 단계 표현만으로 action을 읽을 수 있는가?
  2. 월드모델의 기억(과거 chunk)이 action readout에 실제로 도움이 되는가?
  3. 언어 안내가 readout에 필요한가?
  4. 같은 설정에서 실제 속도 이득은 얼마인가?
- **코드 확인에서 나온 중요한 사실(§2):**
  - 공식 평가는 chunk마다 예측 기억을 지우고 **기록된 실제 영상·action을 기억에 등록**한다(teacher forcing).
  - test clip은 **5개 chunk(20초)**이고 B10은 clip당 87회 forward를 실행한다. 학생 방식은 **17회**가 된다(실행 중 정정, 처음에는 4 chunk·70회·14회로 잘못 계산했다).
  - 12–20초 clip에서는 선택적 기억 삭제(Selective KV)가 작동하지 않는다.
- **예상 기간:** 약 1.5–2.5주(추정). 데이터 접근 승인 시점과 다운로드 방식(§7 X1)에 크게 좌우된다.

---

## 1. NAVSIM 결과에서 이어지는 질문

| NAVSIM에서 확인한 것 | PhysicalAI에서 새로 확인할 것 |
|---|---|
| navtest-final 기준, 고정 백본 + SF3 first head가 B10 대비 PDMS −0.17 ~ −0.42, B\* 대비 0.449배 지연 | 기억과 언어 안내가 있는 연속 설정에서도 성립하는가 |
| first ≈ terminal > current (미래 반복 갱신은 불필요) | 기억이 있을 때도 같은가. current가 사실상 공짜가 되는 구조(§2.4)에서 그 차이는 얼마인가 |
| head 학습 데이터 규모가 결정적(4k → 92.6k) | DriveWAM이 공개한 4k/20k/100k clip 목록으로 같은 규모 곡선을 재현하는가 |
| 공개 checkpoint가 학습 장면을 기억(navtrain 내부 검증 무효) | PhysicalAI checkpoint도 학습 clip을 기억하는가. 그렇다면 검증 세트를 어떻게 만드는가 |
| 남은 약점은 주행 가능 영역 위반(DAC) | PhysicalAI는 PDMS가 없는 open-loop 평가(ADE/FDE)다. 같은 약점이 오차 꼬리로 나타나는가 |

---

## 2. PhysicalAI 설정의 실제 구조 (고정 코드 확인 결과)

### 2.1 데이터셋 (공개 페이지 확인, 2026-10-04)

- 1,700시간, **306,152개 clip × 20초**, 25개국, 전체 133TB(카메라 7대, LiDAR, radar 포함).
- **접근:** Hugging Face gated dataset이고 *NVIDIA Autonomous Vehicle Dataset License* 동의가 필요하다. 자율주행 관련 용도만 허용되며 감시·생체정보·번호판 추적 등은 금지다.
- **DriveWAM이 쓰는 것:** `camera_front_wide_120fov`(1080p 30fps mp4), egomotion label, calibration뿐이다.
- **파일 구조:** 약 100 clip씩 묶은 chunk zip. 전방 광각 카메라 zip은 공개 목록 첫 1,000개 기준 중앙값 1.29GB이고, clip당 약 13MB다. 전방 카메라 전체는 약 4TB로 추정한다.
- **분할:** `clip_index.parquet`(11MB)의 `split`, `clip_is_valid`, `chunk` 열로 clip과 zip의 대응을 얻는다.
- **devkit:** `physical_ai_av`(pip, Python 3.11 이상)로, 현재 모델 환경(3.10)과 분리해야 한다.

### 2.2 DriveWAM PhysicalAI 입력·출력 ([physicalai_cfg.py](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/configs/physicalai_cfg.py), [physicalai_dataset.py](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/dataset/physicalai_dataset.py))

| 항목 | 값 | NAVSIM과 비교 |
|---|---|---|
| 영상 | 전방 광각, 10Hz에서 **1fps로 추출**, 256×448 | NAVSIM: 현재 영상 1장 |
| test clip 구간 | 시작부터 **21프레임(20초)**. VAE 시간 압축 후 latent 6개 = 등록 1 + **미래 chunk 5개**(정정: 처음 17프레임·4 chunk로 잘못 적었다) | NAVSIM: 미래 latent 1개 |
| action | chunk당 **40스텝(10Hz, 4초)**, (dx, dy, dheading) quantile 정규화 | NAVSIM: 8스텝(2Hz) |
| ego 조건 | chunk 시작 시점의 속도·가속도(4) + 곡률(1) | NAVSIM: 과거 4개 pose + 속도·가속도 + command |
| 문장 조건 | **chunk마다 다른 장면 안내 문장**(test 1k는 공개 JSON, 키 0–3). 첫 등록 chunk는 빈 문장 | NAVSIM: command별 고정 문장 4개 |
| 기억 | attention window 24 = 영상 12 chunk + action 12 chunk 슬롯 | NAVSIM: 사실상 1 chunk |

### 2.3 공식 평가 프로토콜 ([eval.py](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/physicalai/eval.py))

clip마다 `predict_autoregressive`가 다음 순서로 실행된다(L269–461).

1. 0번 실제 프레임과 0-action을 기억에 등록한다(`update_cache=2`, 2회).
2. 미래 chunk마다(4회):
   - (a) 미래 영상 latent를 노이즈에서 3단계 갱신하고, 마지막에 예측 기억 쓰기 1회를 한다(`update_cache=1`, `is_pred=True`).
   - (b) action을 10단계 생성하고, 마지막에 예측 기억 쓰기 1회를 한다.
   - (c) `clear_pred_cache`로 **예측으로 쓴 기억을 모두 지운다**(L429–430, [model.py L335–340](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/modules/model.py#L335)).
   - (d) **기록된 실제 영상 latent와 실제 action chunk를 기억에 등록**한다(L432–450, 2회).
3. 지표: 160스텝 전체 ADE와 chunk 끝 오차 평균 FDE. chunk마다 시작 위치를 실제 위치로 맞춘다(`actions_to_trajectory_with_reset`, L218–252). 논문의 ADE@4s 0.83 / FDE@4s 2.47은 이 정의다.
4. 공식 test는 VLM 점수로 고른 **흥미·희귀 장면 1,000개**다(`test_sample_1k.csv`의 `interest_score`, `tags`, `source`). seed는 clip마다 0으로 고정한다.

**함의:**
- **기억은 항상 기록된 실제 이력으로 채워진다.** 예측 action은 다음 chunk에 들어가지 않는다. 앞선 대화에서 걱정한 "학생 action이 기억에 쌓여 오차가 누적되는 문제"는 공식 평가에는 없다. 실제 배치에서도 "실행한 action 등록"은 모든 방법이 똑같이 1회씩 하는 공통 비용이다.
- action 생성의 마지막 예측 기억 쓰기는 곧바로 지워지므로, **NAVSIM의 opt 최적화(마지막 action pass 생략)가 여기서도 출력 변화 없이 성립할 것으로 예상**한다(X3에서 비트 단위로 검증).
- 미래 영상의 마지막 예측 기억 쓰기는 같은 chunk의 action 생성에만 쓰이고 지워진다. 학생 경로에서는 필요 없다.
- window(12 chunk)가 test clip의 chunk 수(최대 5)보다 커서 **Selective KV 삭제가 한 번도 일어나지 않는다.** 이번 확장으로 "긴 시간 기억"에 대한 주장은 할 수 없다.
- 원본 평가는 UMT5 문장 encoder(약 10.6GiB)를 transformer(11.6GiB), VAE와 **같은 GPU에 올린다.** A5000 24GB에서는 문장 임베딩을 미리 계산하거나 GPU를 나눠야 한다.

### 2.4 clip당 백본 forward 횟수 (test clip, 미래 chunk 5개 — 정정된 값)

| 단계 | B10 | B05 | B10-opt | B03-opt | B02-opt | **학생 (first)** | 학생 (current) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 시작 등록 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| chunk당 미래 영상 | 4 | 4 | 4 | 4 | 4 | **1** | 0 |
| chunk당 action | 11 | 6 | 10 | 3 | 2 | **0 (+ 작은 head)** | 0 (+ head) |
| chunk당 실제 기록 등록 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| **chunk당 합계** | 17 | 12 | 16 | 9 | 8 | **3** | **2** |
| **clip당 합계** | 87 | 62 | 82 | 47 | 42 | **17** | **12** |

- current 표현은 직전 실제 프레임을 기억에 등록하는 forward의 중간 출력이다. 이 forward는 어차피 실행하므로, current 학생은 공통 비용 외 추가 forward가 없다. NAVSIM과 달리 current가 사실상 공짜다.
- 회수 비율은 지연 비율이 아니다(영상과 action의 token 수가 다르다). 실제 지연은 X7에서 잰다.

---

## 3. 연구 질문과 가설

| ID | 질문 | 판단 근거 |
|---|---|---|
| **P1 (충분성)** | 실제 이력 기억을 가진 고정 DriveWAM의 first 표현에 head만 학습해 B10·B\*와 비슷한 ADE/FDE를 얻는가 | §6 품질 기준(사전등록) |
| **P2 (효율)** | chunk당 결정 지연이 B\* 대비 20% 이상 줄어드는가. VLM·UMT5까지 포함한 전체 주기에서는 얼마인가 | §6 속도 기준 + 전체 주기 보고 |
| **P3 (상상 단계)** | first ≈ terminal, current < first가 기억이 있는 설정에서도 유지되는가 | 같은 head·예산으로 세 시점 비교 |
| **P4 (기억의 가치)** | 월드모델의 과거 chunk 기억이 readout에 도움이 되는가 | 이력 절단 ablation: 매 chunk 기억을 비우고 현재 프레임만 등록 |
| **P5 (언어 안내의 가치)** | chunk별 VLM 장면 안내가 readout에 필요한가. 고정 문장으로 바꾸면 얼마나 잃는가 | 안내 ablation: config의 기본 문장 사용 |
| P6 (보조) | head 학습 데이터 4k → 20k(→ 100k)의 규모 곡선 | DriveWAM 공개 목록 사용 |
| P7 (보조) | PhysicalAI checkpoint도 학습 clip을 기억하는가 | B10의 학습 clip / dev / final ADE 비교 |

P4·P5는 월드모델 관점에서 이번 확장의 핵심 분석이다. 두 결과는 결과와 관계없이 모두 보고한다.

---

## 4. 비교군과 모델

**기준 모델(학습 없음):**
- B10(원본, 영상 3단계 + action 10단계, 영상 denoise fraction 0.6)
- B05, B10-opt, B05-opt, B03-opt, B02-opt, B01-opt
- B\*: dev에서 §6 규칙으로 선정한다.
- 등속 기준(CV): ego 속도로 4초 직진. 오류·좌표계 점검용이다.

**학생 모델(백본 고정, head만 학습):**
- 주 모델: **SF3 first 회귀 head.**
  - layer [10,20,30], query 16, 폭 512, 출력층 FP32(NAVSIM v3 구조)
  - 출력 40스텝 × (dx, dy, dheading) 정규화 delta. 시간 embedding은 40개다.
  - ego 입력은 백본과 같은 정보(속도·가속도 4 + 곡률 1)다.
- 시점 비교: 같은 구조로 SF3-current, SF3-terminal을 각각 학습한다.
- 진단 ablation(dev 전용, 주 결과 아님):
  - **이력 절단:** feature 추출 시 chunk마다 기억을 비우고 현재 실제 프레임 + 0-action만 등록한다. 기억의 기여를 측정한다.
  - **고정 문장:** chunk별 안내 대신 config 기본 문장을 사용한다. 언어 안내의 기여를 측정한다.

**noise 정합 원칙:**
- 원본 평가는 clip 시작에 seed를 한 번 고정하고 영상 noise와 action noise를 차례로 뽑는다.
- 학생 경로가 action noise를 뽑지 않으면 **다음 chunk의 영상 noise가 B10과 달라진다.**
- 따라서 학생 runner는 원본과 같은 순서로 난수를 소비하거나(action noise를 뽑고 버림), chunk별 결정적 seed를 쓰는 별도 protocol을 사전등록한다.
- 어느 쪽이든 X4에서 "B10 실행에 hook으로 얻은 first feature = 조기 종료 학생 경로의 first feature"를 비트 단위로 검증한다.

---

## 5. 데이터 분할 (NAVSIM 교훈 반영)

| 세트 | 구성 | 용도 | 백본이 학습했나 |
|---|---|---|---|
| **head-train** | DriveWAM 공개 `train_4k.csv`, `train_20k.csv`(필요 시 `train_100k.csv`)의 학습 split clip | head 학습, 규모 곡선 | **예**(NAVSIM의 navtrain과 같은 상황) |
| **dev** | `clip_index`의 test split 중 공식 1k를 **제외한** clip. 약 1,000개를 고정 seed로 선택 | B\* 선정, 시점·ablation 판정, 설정 고정 | 아니오 |
| **final (잠금)** | 공식 `test_sample_1k.csv` 1,000개 + 공개 문장 | 최종 1회 평가. 논문 수치와 직접 비교 | 아니오 |

**dev 구성 방식(사용자 결정, §11):**
- **D1(권장):** DriveWAM의 VLM 장면 선별 스크립트(`vlm_data_sample/qwvl_scene_*.py`)를 test split 후보에 적용해 final과 비슷한 흥미·희귀 장면 분포로 고른다.
- **D2:** test split에서 무작위로 고른다. 비용이 적은 대신 final보다 쉬운 장면이 많을 수 있다.
- 어느 방식이든 dev clip의 **chunk별 안내 문장은 직접 생성**해야 한다. 공개 문장은 학습 clip과 test 1k에만 있다. 사용 스크립트는 `vlm_preprocess/generate_route.py`, `generate_bev.py`, `generate_scene_evolving_guidance.py`이고 Qwen3-VL-8B를 vLLM으로 돌린다. 우리가 생성한 문장은 원 저자의 문장과 문구가 다를 수 있음을 기록한다.

**기억(암기) 점검:** B10의 ADE를 head-train clip 일부, dev, final에서 비교한다. 학습 clip에서만 크게 낮으면 NAVSIM과 같은 기억 현상으로 보고, head-train 내부 검증은 의사결정에 쓰지 않는다.

**재현 확인:** final에서 B10을 1회 실행해 논문 수치(ADE 0.83 / FDE 2.47)와 비교하는 것은 **파이프라인 검증으로 사전에 허용**한다. B10은 조정할 설정이 없는 기준 모델이다. 학생은 설정 고정 전에 final에서 절대 실행하지 않는다. 재현 차이가 크면 설정·데이터 처리부터 점검하되, final 결과로 학생 설계를 바꾸지 않는다.

**통계 단위:** clip(또는 `clip_index`에 주행 세션 정보가 있으면 세션) 단위 paired bootstrap 2,000회. chunk는 같은 clip 안에서 독립이 아니므로 표본으로 세지 않는다.

---

## 6. 성공 기준 (실행 전 사전등록 예정안)

NAVSIM의 PDMS −2점 기준 대신 open-loop 지표에 맞춘 여유폭(margin)을 둔다.

- **품질:** 학생 − 기준의 paired 차이 95% CI 상한이 **B10 및 B\* 대비 각각** 다음 이내여야 한다.
  - ADE ≤ **+0.05m**
  - FDE ≤ **+0.15m**
  - 근거: 논문 수치의 약 6%다. 논문에서 장면 안내가 준 개선(100k 기준 ADE 0.92 → 0.83)의 절반보다 작고, 논문의 action 10 → 5단계 변화(+0.01m)는 이 안에 든다.
- **꼬리 오차(안전 대용 지표):** FDE > 2m인 chunk 비율의 증가가 B10 대비 1%p 이내여야 한다. NAVSIM의 DAC 약점이 open-loop에서 어떻게 나타나는지 보는 지표다.
- **속도:** chunk당 결정 지연 p50이 **B\* 대비 0.80배 이하**이고 p95도 악화하지 않아야 한다. 같은 세션에서 장면 단위로 번갈아 측정한다.
  - 결정 지연 = 기억 등록 이후, 현재 chunk의 미래 영상 단계 + action 단계(학생은 head). VAE 인코딩과 실제 기록 등록은 공통 비용으로 따로 보고한다.
- **seed:** 학습 seed **3개(17, 29, 43)** 모두에서 통과해야 한다. 원래 계획의 3개를 지킨다(NAVSIM은 2개로 진행했다).
- **보고만 하는 항목(판정 미포함):** VLM·UMT5를 포함한 chunk 전체 주기 지연, P3–P7 분석.

통과하면 설정을 고정하고 final 실행을 사용자에게 확인받는다. 미통과라도 모든 결과를 보고하고, final 결과로 설정을 다시 고르지 않는다.

---

## 7. 단계별 실행 계획

| 단계 | 내용 | 산출물·검증 | 추정 기간 |
|---|---|---|---|
| **X0 접근·환경** | ① 사용자가 HF에서 데이터 라이선스에 동의하고 서버에서 직접 로그인한다(token은 문서·로그에 남기지 않음) ② `physical_ai_av` 전용 Python 3.11 환경 ③ DriveWAM PhysicalAI checkpoint(12.5GB)·학습 clip 목록·학습용 안내 문장(28.7MB) 다운로드와 SHA-256 검증 ④ vLLM + Qwen3-VL-8B 환경(dev 문장 생성·VLM 지연 측정용) ⑤ 사전등록 파일 고정 | 환경 lock, 자산 해시, 사전등록 해시 | 0.5–1일 (+승인 대기) |
| **X1 데이터** | ① `clip_index` 확보 → split·chunk 대응, 세션 정보 확인 ② dev 선정(D1/D2) ③ **zip 부분 다운로드 검증**: zip은 tar.gz와 달리 임의 접근이 되므로, HTTP range로 목록을 읽고 필요한 clip의 mp4만 받는다. 실패하면 chunk zip 전체를 받아 추출한 뒤 삭제한다 ④ 필요한 egomotion·calibration ⑤ 1fps 프레임 + ego 처리. DriveWAM loader와 프레임·action·ego 비트 단위 일치 확인 | clip manifest, 처리 보고서, 원본 loader parity | 1–2일 |
| **X2 문장** | ① dev clip의 chunk별 안내 문장 생성(D1이면 장면 선별 포함) ② 모든 문장의 UMT5 임베딩을 미리 계산(유효 token만 저장)한 뒤 feature 추출이 끝나면 삭제 ③ 원본 `_encode_single_prompt`와 임베딩 비트 일치 확인 | 문장 JSON, 임베딩 manifest | 1–2일 |
| **X3 기준 모델** | ① final에서 B10 재현 확인(파이프라인 검증) ② dev에서 B10/B05/opt 1·2·3·5·10단계 실행, 32 clip에서 raw/opt 출력 비트 일치 ③ 학습 clip 일부로 기억 점검(P7) ④ B\* 선정 | 재현 보고서, B\* 선정 기록 | 1일 |
| **X4 feature** | ① head-train(4k, 20k)과 dev에서 chunk마다 실제 이력이 등록된 상태의 first feature(layer [10,20,30]) 추출 ② 4k와 dev는 current·terminal도 추출 ③ dev는 이력 절단·고정 문장 ablation feature도 추출 ④ hook 비간섭, 조기 종료 동일성, 미래 영상·action 정답 치환 불변 검증 | feature lock, parity 보고서 | 0.5–1일 |
| **X5 head 학습** | NAVSIM v3 protocol(40 epoch 고정, 마지막 checkpoint, batch-1 예측, FP32 출력)을 따른다. 출력 40스텝. 4k → 20k 규모 곡선. 최종 설정 seed 17/29/43 | checkpoint 해시, 학습 기록 | 0.5–1일 |
| **X6 dev 판정·진단** | §6 규칙 판정. P3(시점), P4(기억), P5(안내), 꼬리 오차, chunk 순번(1–4)별 오차 분석. 100k 확대 여부를 사전등록 규칙으로 결정 | dev 분석 보고서 | 1일 |
| **X7 final + 지연** | 고정 설정을 final 1,000 clip에서 1회 평가. 지연은 B10·B\*·학생을 장면 단위로 번갈아 측정한다. VLM·UMT5 지연은 다른 GPU에서 따로 잰다 | final 판정, 지연 보고서 | 0.5일 |
| **X8 정리** | NAVSIM 결과와 통합한 결과 보고서, 재현 bundle | 문서 갱신 | 1일 |

**100k 확대 규칙(사전등록 예정):** 20k에서 미통과이고 4k → 20k 개선이 ADE 0.03m 이상이면 100k를 검토한다. 다만 저장 공간 문제(§8)로 token pooling 같은 압축 방식을 4k에서 먼저 검증한 경우에만 진행한다.

---

## 8. 자원·시간 추정

| 항목 | 추정치 | 근거·비고 |
|---|---|---|
| 전방 카메라 다운로드 (20k + dev 1k + final 1k) | clip 단위 부분 다운로드 시 약 0.3TB | clip당 약 13MB |
| 〃 chunk zip 전체를 받아야 할 때 | 최대 약 4TB(학습 clip이 대부분의 chunk에 흩어져 있으면) | 스트리밍 후 삭제로 디스크는 감당 가능. 다운로드만 수일 걸릴 수 있다 |
| 처리 후 영상 저장 | clip당 약 1MB(1fps 17프레임) → 22k clip 약 22GB | |
| first feature (20k clip × 4 chunk) | 약 165GB (chunk당 2.06MB) | 4k에서 current·terminal 추가 약 66GB |
| UMT5 임베딩 (임시) | 수십 GB | feature 추출 후 삭제 |
| GPU: feature 추출 | 20k clip 약 3–4시간(GPU 2장) | clip당 forward 14회 + VAE 17프레임 |
| GPU: 기준 모델 (dev 1k × 7종 + final B10) | 약 5–8시간 | B10 clip당 forward 70회 |
| GPU: dev 문장 생성(VLM) | 약 2–6시간(GPU 1장) | 1k clip × 4 chunk + route·BEV 단계 |
| head 학습 | 모델당 수십 분 | 20k × 4 = 80k 표본(NAVSIM 92.6k와 비슷) |
| 현재 서버 여유 | 디스크 약 403GB, RAM 503GB, A5000 × 2 | NAVSIM 중간 feature cache(약 300GB)를 정리해야 할 수 있다 → §11 |

**전체 기간:** 작업일 기준 7–11일, 대기 포함 약 1.5–2.5주(추정).

---

## 9. 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| 데이터 승인 지연, 라이선스 조건 | 시작 지연 | 승인 전에 코드·runner를 공개 test 1k 문장과 소수 clip 기준으로 먼저 준비한다 |
| clip 부분 다운로드 불가 | 다운로드가 수 TB·수일로 늘어남 | chunk 단위로 받아 추출 후 삭제. 필요하면 head-train을 적은 chunk에 몰린 clip으로 제한하되 편향을 기록한다 |
| checkpoint 출처 불명확, 재현 실패 | 기준선 신뢰 하락 | X3에서 재현 확인. 실패하면 설정·전처리·attention backend를 점검하고 "공개 checkpoint 기준"으로 보고한다 |
| 24GB 메모리(UMT5 동시 적재) | OOM | 임베딩 사전 계산. 실제 배치 지연은 GPU 분리 구성으로 따로 측정한다 |
| dev 문장을 우리가 생성 | 원 저자 문장과 분포 차이 | 생성 설정을 기록하고, final은 공개 문장을 그대로 쓴다 |
| dev와 final 장면 분포 차이 | B\*·설정 선택의 편향 | D1 선별 방식 사용. tag별 subgroup 보고 |
| teacher forcing open-loop 평가 | 실제 주행(자기 action이 기억에 들어감)과 다름 | 주장 범위를 open-loop로 제한한다. 자기 action 등록 진단은 선택 실험으로 둔다 |
| Selective KV 미작동 | 장기 기억에 대한 결론 불가 | 주장 범위를 "4 chunk 이내 이력"으로 명시한다 |
| 난수 소비 순서 차이 | 학생과 B10의 미래 noise 불일치 | §4 noise 정합 원칙 + 비트 단위 검증 |
| 공유 서버 부하 | 지연의 절대값 변동 | 장면 단위 교차 측정, 비율로 판정 |
| 디스크 부족 | feature 저장 불가 | §11 결정 후 NAVSIM 중간 cache 정리. 최종 증거·예측·점수는 보존 |

---

## 10. 주장할 수 있는 것과 없는 것

- **할 수 있는 주장(결과가 나오면):** "기억과 언어 안내를 갖춘 연속 주행 WAM에서도, 고정 백본의 첫 상상 단계 표현만으로 반복 action 생성을 대체할 수 있다(open-loop, 4 chunk)." "그 설정에서 기억과 안내가 readout에 기여하는 정도는 이만큼이다."
- **할 수 없는 주장:** 폐루프(closed-loop) 주행 안전성, 4 chunk를 넘는 장기 기억, 다른 WAM 전반으로의 일반화.

---

## 11. 실행 전에 필요한 사용자 결정

1. **데이터 접근(필수, 사용자 직접):** Hugging Face에서 NVIDIA PhysicalAI-AV 라이선스에 동의하고, 서버에서 `huggingface-cli login`을 직접 실행해 주세요. 저는 계정 동의나 token 입력을 대신할 수 없습니다.
2. **dev 구성 방식:** D1(VLM 선별, 권장) / D2(무작위).
3. **데이터 규모:** 4k + 20k로 시작(권장)하고, 100k는 §7 규칙으로 결정.
4. **디스크 정리:** 공간이 부족하면 NAVSIM 중간 feature cache(pilot 63GB, X1 약 100GB 등)를 삭제해도 되는지 확인. 최종 결과·예측·점수·사전등록 기록은 보존한다.
5. **VLM 지연 측정 포함 여부:** 포함하면 vLLM 환경과 GPU 1장이 수 시간 추가로 필요하다.

---

## 부록 A. 코드 근거

| 사실 | 위치 |
|---|---|
| 실제 기록 등록과 예측 기억 삭제 | [eval.py L429–450](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/physicalai/eval.py#L429) |
| 예측 기억 표시(`is_pred`)와 삭제 | [model.py L335–340, L466](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/modules/model.py#L335) |
| ADE/FDE 정의(chunk별 실제 위치 재설정) | [eval.py L218–252, L467–479](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/physicalai/eval.py#L218) |
| test clip 16초·17프레임·4 chunk 구성 | [physicalai_dataset.py L548–600 부근](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/dataset/physicalai_dataset.py) |
| UMT5를 같은 GPU에 적재 | [eval.py L66–75](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/physicalai/eval.py#L66) |
| 기억 크기(window 24) | [physicalai_cfg.py](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/configs/physicalai_cfg.py), [physicalai_model.py L617–618](https://github.com/chenshi3/DriveWAM/blob/abbefb407c9c553f221522fc26077d307620adb4/src/modules/physicalai_model.py#L617) |
| 공식 test 1k(VLM 점수 선별)와 chunk별 문장 | `src/physicalai/eval_data/test_sample_1k.csv`, `prompts_test_sample_1k.json` |
