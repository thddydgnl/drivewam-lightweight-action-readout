# 사전등록: navtest-dev 도입과 pilot protocol 수정 · 2026-10-02

**작성 시점:** navtest 장면에 대한 어떤 예측·점수도 만들기 전. 아래 규칙은 navtest-dev 결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 날짜·이유를 이 문서 끝에 추가하고 해당 결과를 "사후 변경"으로 표시한다.

## 0. 왜 바꾸는가

공개 DriveWAM NAVSIM checkpoint는 navtrain으로 학습되었다(공식 README의 학습 데이터 경로). 본 연구의 head-train·head-val은 모두 navtrain 내부다. 2026-10-01 pilot에서 다음을 관측했다.

- B10의 head-val dev256 ADE4 0.076m, PDMS 93.1. 공개 navtest PDMS 90.1, 사람 운전 94.8.
- generation seed 0/1/2 사이 B10 궤적 RMS 차이 3.5cm, 장면 내 PDMS 표준편차 0.05점.
- 비슷한 조건의 다른 장면 feature로 바꾸면 모든 head의 PDMS가 76–87에서 35–41로 떨어짐. ego-only 63.5보다 낮다.

이는 백본이 head-val 장면을 기억하고 있을 가능성을 강하게 시사한다. 이 데이터로는 (a) B10·저-step 기준의 품질, (b) head와 B10의 격차, (c) current/first/terminal 비교, (d) GT/teacher target 진단을 처음 보는 데이터에 대한 결론으로 쓸 수 없다.

## 1. navtest의 로그 단위 사전 분할 (고정됨)

- 파일: 서버 `splits/navtest-log-split-seed17.json`, SHA-256 `68178d2266df41c3f5d618f81ba75f00a9b99589bab026f271a8b82b4ef031a4`.
- 방법: 고정 navtest.yaml(SHA-256 `61284edf…c9cd`)의 136개 로그를 정렬 → `random.Random(17).shuffle` → 앞 ceil(25%)=34개 로그를 **navtest-dev**, 나머지 102개 로그를 **navtest-final**로 둔다. 장면은 로그를 따른다. 다시 뽑지 않는다.
- 생성 스크립트: `remote_navtest_split.py`. 센서·라벨·예측을 읽지 않는다.

**navtest-dev의 용도:** 백본 기억 효과 측정, B* 선정(§1.3 규칙을 full head-val 대신 navtest-dev에 적용), pilot 가설 판정, head 구조·학습 설정 결정.

**navtest-final 잠금 규칙:**
- 모든 설정(B*, head 구조, 학습 길이, generation protocol, checkpoint 규칙)을 고정하기 전에는 어떤 모델도 navtest-final에서 실행하지 않는다.
- 실행 후에는 결과로 모델·설정을 다시 고르지 않는다.
- 최종 표는 navtest-final 기준이다. 공식 navtest 전체(12,146장면)와 장면 집합이 다르므로 공개 navtest 수치와 직접 비교하지 않는다. 필요하면 전체 navtest 결과를 보조로 함께 적되, 그 25%가 dev로 사용되었음을 표기한다.

head 학습 데이터는 계속 navtrain(head-train 로그)이다. navtrain head-val은 학습 곡선 진단에만 쓰고 의사결정에 쓰지 않는다.

## 2. GT/teacher target 진단(D-GT/D-Teacher) 취소

- 이유: navtrain 장면에서는 B10이 GT와 평균 약 8cm, seed 간 3.5cm만 다르다. 두 target이 사실상 같아 진단이 결론을 낼 수 없다.
- 조치: 2026-10-01에 추출한 8개 shard(head_train seed 0·1, 1,024행)는 해시와 함께 보존하고 학습은 하지 않는다.
- 재설계 여부는 navtest-dev에서 B10–GT 차이와 seed 간 변동을 측정한 뒤 별도로 정한다.

## 3. head 학습 protocol v2

pilot v1의 문제: ADE 기준 early stopping과 cosine 학습률을 함께 써서 모델별 학습 길이가 달랐다(SF1-wide 15 epoch, ego-only 18 epoch, SF3-current 28 epoch, 나머지 30 epoch). 또한 checkpoint 선택(PDMS)과 보고 세트(dev256)가 겹쳤다.

v2 규칙:

- 데이터: 기존 4k head-train feature cache 그대로(백본·feature 변경 없음).
- 모든 모델 **40 epoch 고정**(v1에서 12개 중 7개가 30번째 마지막 epoch에서 최고였기 때문), warmup 5% + cosine, early stopping 없음.
- **마지막 epoch checkpoint만 사용**. 평가 세트로 checkpoint를 고르지 않는다.
- 학습 seed 17, 29 두 개. 결과는 seed별로 보고하고 평균만으로 판정하지 않는다.
- 나머지(optimizer, lr 3e-4, wd 0.01, batch 32, loss, 구조, layer [10,20,30])는 v1과 같다.
- 평가: navtest-dev 전체에서 마지막 checkpoint 1회. navtrain head-val ADE는 epoch마다 기록하되 선택에 쓰지 않는다.

## 4. navtest-dev에서 계산하고 해석하는 방법

- 기준 모델: B10, B05-opt, B03-opt, B02-opt, B01-opt(raw와 opt 출력 동일은 기존 검증으로 확인됨), CV. generation seed 0.
- head: v1 선택 checkpoint 12개(기존 그대로)와 v2 12개 × 2 seed.
- 통계: 로그 단위 paired bootstrap 2,000회(34개 로그), sample-weighted 평균.
- **기억 효과:** B10의 head-val dev256 대비 navtest-dev의 PDMS·ADE 차이를 보고한다(서로 다른 장면 집합이므로 기술 통계로만 제시).
- **B\*:** 실행계획서 §1.3 규칙(B10 대비 PDMS 차이 CI 하한 ≥ −2.0, NC/DAC 평균 악화 ≤ 0.5점, 그중 policy p50 최소)을 navtest-dev 품질과 §5의 재측정 지연으로 적용한다.
- **pilot 판정:** 실행계획서 §14.4의 표를 그대로 적용한다. "비슷함"은 동등성 입증이 아니다. seed 두 개의 방향이 다르면 "판정 보류"로 적는다.
- **generation seed 민감도:** navtest-dev의 고정 32장면에서 B10 seed 0/1/2를 실행해 seed 간 궤적 차이를 기록한다(target 진단 재설계 판단용).

## 5. 지연 재측정 protocol

- 서버 호스트 load average가 다른 사용자 컨테이너 때문에 50 이상인 상태가 확인되었다. 이를 통제할 수 없으므로 측정마다 load average와 GPU clock을 기록한다.
- 모델 순서를 반복마다 섞어(interleave) 3회 반복, 반복별 p50/p95와 반복 간 범위를 함께 보고한다. CUDA event 기준 GPU 시간도 보조로 보고한다.
- 20% 속도 기준과의 차이가 반복 간 변동 범위 안에 있으면 "판정 보류"로 적는다.

## 6. 추가 사전등록: 3차 보완 라운드 (2026-10-03, 실행 전 작성)

**근거(2차 결과의 진단):** v2 SF3-terminal(seed 17)의 PDMS 손실 전체가 navtest-dev의 125개 장면(3.8%)에서 나온다. 이 장면들에서 B10은 주행 가능 영역 안에 있고 head만 벗어난다. 이 장면들의 ADE는 0.71m vs 0.58m로 크게 다르지 않다. 손실은 고속(≥8m/s)과 회전 구간에서 크다. 회귀 head의 mode averaging과 출력 정밀도 부족을 원인 후보로 둔다.

**실험(모두 실행하고 모두 보고):**
- 출력: (R) SF3 회귀 head, 출력층 FP32 / (A) SF3 trunk + anchor head. A는 head-train GT 궤적의 k-means(K=256, x/y 8점, seed 17) anchor 중 하나를 고르는 분류 + anchor별 보정값이고, 학습은 GT에 가장 가까운 anchor의 분류 CE + 그 anchor 궤적의 위치/yaw 손실, 추론은 argmax anchor + 보정값이다. 출력층은 FP32다.
- 시점: first, terminal. layer [10,20,30], query 16, 폭 512는 v2와 같다.
- 데이터: (4k) 기존 pilot head-train / (20k) 기존 4k + 새 navtrain head-train 16,000장면. 새 장면은 기존 pilot 로그를 제외한 head-train 로그만 쓴다. archive를 `random.Random(17)` 순서로 추가하다가 가용 장면이 16,000 이상이 되면 멈추고, `SHA256("17:x1:"+token)` 순위로 16,000개를 고른다.
- 학습: v2와 같다(40 epoch 고정, 마지막 checkpoint, AdamW 3e-4, batch 32, cosine). seed 17, 29. 예측은 batch 1.
- 총 2(R/A) × 2(first/terminal) × 2(4k/20k) × 2 seed = 16개 모델.

**판정 규칙:**
- **통과:** 어떤 설정이 두 seed 모두에서 다음을 만족하면 통과다.
  - PDMS 차이 95% CI 하한이 B10 및 B\*(B02-opt) 대비 각각 −2.0 이상
  - NC·DAC 평균 악화가 각각 0.5점 이내
  - B\* 대비 policy p50 비율 0.80 이하(10/2 교차 측정값 사용: first 0.467, terminal 0.794. head 교체에 따른 차이는 작지만 통과 후보는 재측정한다)
- 통과 설정이 있으면 그 설정을 고정하고, navtest-final 실행을 **사용자에게 확인받는다**.
- 통과 설정이 없으면 두 경우로 나눈다.
  - 최고 설정이 같은 시점의 v2 SF3보다 seed 평균 1.5점 이상 좋으면 부분 개선으로 보고하고 다음 방향을 사용자에게 묻는다.
  - 그 미만이면 direct readout 보완은 종료하고 (a) 부정 결과 정리로 간다.
- 8개 설정을 dev에서 비교하므로, 최고 설정의 dev 추정은 낙관적으로 편향된다. 최종 주장은 navtest-final로만 한다.

## 7. 추가 사전등록: 4차 라운드, head-train 전체 확대 (2026-10-03, 실행 전 작성)

**근거:** 3차에서 학습 데이터를 4k에서 20k로 늘리자 SF3 회귀 head가 v2 대비 +2.9점(first) 좋아졌다. first와 terminal은 비슷하고 first가 2배 빠르다.

**실험:**
- 모델: SF3 회귀(출력층 FP32), **first 시점만**, layer [10,20,30], query 16, 폭 512. 학습 설정은 §6과 같다(40 epoch 고정, 마지막 checkpoint, AdamW 3e-4, batch 32, cosine).
- seed: 17, 29. dev 예측은 batch 1.
- 데이터: frozen log split의 navtrain head-train 전체 장면(92,599). 기존 4k와 X1 16k에 나머지 장면을 더한다. 영상 누락이나 처리 실패로 빠지는 장면은 수를 보고하며, 0.5%를 넘으면 실행을 멈추고 원인을 확인한다.
- 디스크 관리: 사용자 허락(2026-10-03)에 따라 카메라 archive 원본은 필요한 전방 영상의 추출·검증 후 삭제한다. 다운로드 receipt는 보존한다.

**판정 규칙(§6과 동일):**
- 두 seed 모두에서 다음을 만족하면 통과다.
  - PDMS 차이 95% CI 하한이 B10과 B\* 대비 각각 −2.0 이상
  - NC·DAC 평균 악화가 각각 0.5점 이내
  - B\* 대비 p50 비율 0.80 이하(first 0.467. 통과 시 재측정)
- 통과하면 설정을 고정하고 navtest-final 실행을 사용자에게 확인받는다.
- 통과하지 못하면 navtrain에서 더 늘릴 데이터가 없으므로 데이터 확대 축은 종료한다. 4k/20k/전체의 데이터 규모 곡선을 결과로 보고하고 정리 방향을 사용자와 정한다.
- 3차까지 dev에서 여러 설정을 비교했으므로 dev 추정은 낙관적일 수 있다. 최종 주장은 navtest-final로만 한다.

## 8. navtest-final 1회 평가 (2026-10-03, 실행 전 작성)

**일탈 기록:** §7 규칙상 4차 결과는 "근소 미통과"다. seed 17의 DAC 악화가 0.52점으로, 위반 장면이 허용치보다 1개 많았다. 원래 규칙은 통과 시에만 final 실행을 확인받게 되어 있다. 이번에는 사용자 결정(2026-10-03)으로, dev 기준을 통과하지 못한 상태에서 final을 1회 실행한다. 이 실행은 사전등록 규칙의 예외로 기록한다.

**고정 설정(변경 불가):**
- 학생 모델: SF3 first 회귀(출력층 FP32), head-train 92,599장면, 40 epoch, 마지막 checkpoint.
  - seed 17 checkpoint SHA-256 `7af22cee…2c90`, seed 29 `0c5ad62f…d6b4`
  - ego 정규화 통계 `stats-full-first.safetensors` `a4c4b471…67d8`
  - 예측은 batch 1
- 기준 모델: B10, B\*=B02-opt(navtest-dev에서 선정). generation seed 0.
- 학생의 first feature는 B10 원본 rollout의 hook에서 얻는다(조기 종료 경로와 비트 단위 동일성 검증됨).
- 평가: navtest-final 102개 로그 8,896장면, 공식 NAVSIM v1.1 scorer, 로그 단위 paired bootstrap 2,000회.

**판정 규칙(§6·§7과 동일, 결과를 보기 전에 고정):**
- 두 seed 모두에서 다음을 만족하면 통과다.
  - PDMS 차이 95% CI 하한이 B10 및 B\* 대비 각각 −2.0 이상
  - NC·DAC 평균 악화가 각각 0.5점 이내
  - B\* 대비 policy p50 비율 0.80 이하. final 장면 256개에서 B10·B\*·학생을 장면 단위로 번갈아 3회 측정한다.
- 결과와 관계없이 모두 보고하고, final 결과로 설정·모델을 다시 고르거나 고치지 않는다.
- 다른 opt 기준(B05/B03/B01-opt)은 시간 절약을 위해 final에서 실행하지 않는다. dev에서 B10과 품질이 같거나 탈락했다.

## 변경 기록

- 2026-10-02: 최초 작성.
- 2026-10-02 (사후 절차 변경, 판정 규칙은 그대로): navtest-dev의 head 예측을 처음에는 batch 64로 계산했다. head 출력층의 BF16 autocast 때문에 batch-1 온라인 정책과 궤적이 cm 단위로 달랐다. 그래서 모든 head를 batch 1로 다시 예측·채점했다. batch-64 결과는 `scores_batch64`에 보존하며, 차이는 PDMS ±0.2점 이내다. 결과를 보기 전에 정한 비교·판정 규칙은 바꾸지 않았다.
- 2026-10-02 (결과): B\*=B02-opt. 모든 head가 품질 기준 미달. §14.4에 따라 full 확대 보류.
- 2026-10-03 (3차 결과): 통과 설정 없음. 최고 설정 20k 회귀 first(89.36 / 90.00)가 v2 대비 +2.9점이라 §6의 "부분 개선"에 해당한다. 통과하지 못한 이유는 seed 17의 CI 하한 −2.62와 두 seed의 DAC 악화 1.05–1.51점이다.
- 2026-10-03 (4차 결과): 92,599장면 학습 SF3 first 회귀 head는 90.43 / 90.58점이다(B10 대비 −0.51 [−1.12, +0.14] / −0.36 [−0.95, +0.28]). seed 29는 통과, seed 17은 DAC 평균 악화 0.52점(위반 장면 +17, 허용 16.25)으로 미통과다. 규칙상 "두 seed 모두 통과"가 아니므로 미통과로 기록한다. 규칙은 바꾸지 않았다.
- 2026-10-03 (navtest-final 결과, §8): B10 89.67, B\* 89.65, 학생 89.25 / 89.49점.
  - 학생 − B10: −0.42 [−0.76, −0.08] / −0.17 [−0.56, +0.27]
  - 속도: B\* 대비 0.449배
  - seed 29 통과. seed 17은 B10 대비 DAC 악화 0.51점(위반 +45장면, 허용 44.48)으로 미통과.
  - 최종 판정은 "미통과(근소)"이며, 결과로 설정을 바꾸지 않았다.
- 2026-10-03 (기록 누락 보완): 3차 4k 학습 스크립트가 ego 정규화 통계를 저장하지 않았다. 학습 데이터에서 결정적으로 다시 계산할 수 있으며, 20k 학습부터는 `stats-<data>-<phase>.safetensors`로 저장한다.
