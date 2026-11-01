# TMLR 원고 내부 검토 (2026-10-31)

대상: `tmlr/main.tex` v1.1 (16쪽). 두 가지를 합쳤다.

- **A. 독립 리뷰:** 실험 기록을 보지 않고 원고·그림·bib만 읽은 리뷰어 관점 검토(Claude 서브에이전트). 줄 번호는 v1.1 `main.tex` 기준이다.
- **B. 사실 확인:** A의 지적을 코드와 `evidence/`로 확인한 결과와 우선순위 계획.

판정 요약: 독립 리뷰는 **major revision, 확신도 4/5**다. TMLR 결정은 accept as is / accept with minor revision / reject 세 가지뿐이라([editorial policies](https://jmlr.org/tmlr/editorial-policies.html)), 큰 수정이 필요하다는 평가는 사실상 reject 후 재제출에 가깝다.

---

## A. 독립 리뷰 전문

### 1. 요약

frozen 상태의 DriveWAM(6.24B)에서 첫 future-denoising pass(`first`)가 진행될 때 block 10/20/30의 hidden state를 hook으로 읽는다. 이를 10M 파라미터 SF3 head로 궤적에 회귀시키고, 그 시점에서 추론을 끝낸다. 주장은 네 가지다.

- **(a) 속도와 품질**
  - NAVSIM: B10 대비 −0.17/−0.42 PDMS, latency는 B\*(B02-opt)의 0.449배. 결과는 near miss다.
  - PhysicalAI-AV: 3개 seed 모두 사전 등록 기준을 통과했고, latency는 B\*(B03-opt)의 0.163배다.
- **(b) phase 효과:** `first`가 `current`보다 크게 낫고, `terminal`과는 비슷하다.
- **(c) 데이터 규모:** head 학습 데이터 규모가 결과를 좌우한다.
- **(d) 평가 함정:** NAVSIM backbone의 학습 장면 암기와 단일 seed 민감성을 보고한다. 또 readout의 정확도 우위는 대부분 결정론적 예측이라서 생긴다고 본다.

### 2. 주장별 근거

#### (a) 속도와 품질: Table 4 (l.266–287), Table 6 (l.324–348), Fig. 6

잠근 final set을 한 번만 평가했고, paired cluster bootstrap을 썼으며, B\* 선정 규칙을 미리 고정했다. 근거는 강하다. 부족한 점은 다음과 같다.

- **헤드라인 배율의 측정 구간.** abstract의 PhysicalAI 6.1×/11.8×(l.47)는 registration을 뺀 구간이다. registration을 포함하면 chunk당 189.9 대 552.5 ms(l.322), 즉 2.9×로 NAVSIM의 2.2×와 비슷하다. 두 벤치마크 간 배율 차이는 대부분 측정 구간 차이에서 나온다.
- **final 표의 latency 출처.** PhysicalAI final 표의 latency는 dev 128 clip에서 잰 값이다(l.298, l.490). 그런데 Fig. 6 캡션(l.416)은 "locked final sets"라고 적고 있다.
- **측정 환경.** 부하 12–60의 공유 호스트에서 쟀고(l.153), GPU를 전용으로 썼는지 알 수 없다. 하드웨어와 무관한 비용(forward 수, FLOPs)은 PhysicalAI의 B10/readout에만 있다(l.93).
- **NAVSIM B10의 cache pass.** NAVSIM "B10 (official)"이 불필요한 action cache pass를 포함하는지 불분명하다(l.144, l.276). 포함한다면 4.2×가 부풀려진다.
- **재현값 비교 누락.** NAVSIM B10 재현값을 DriveWAM 논문의 발표 수치와 비교하지 않았다(PhysicalAI는 l.322에서 비교).

#### (b) phase 효과(제목의 핵심 주장): Table 3 (l.226), Table 8 (l.389)

- **4k 규모에서만 확인.**
  - 두 표 모두 4k head를 썼는데, 이 규모에서는 모든 head가 품질 기준에 실패했다.
  - 저자들은 4k에서 멈췄다면 잘못된 결론을 냈을 것이라고 직접 쓴다(l.420). 같은 논리가 phase 결론에도 적용된다.
  - 92.6k/20k 규모의 `current` 결과는 없다. 20k에서 `first`≈`terminal`이라는 결과는 appendix Table 10(l.507–508)에만 있다.
  - `current`는 `first`보다 약 2배 빠르다(0.248 대 0.467). 큰 데이터에서 격차가 줄면 제안 방법의 선택 자체가 바뀐다.
- **"imagination" 해석의 대안 설명이 통제되지 않았다.**
  - pass 1의 future token이 관측 cache에 대한 query로 작동해 정보를 모으는 것일 수 있다.
  - t=1 timestep modulation 효과일 수도 있다.
  - 다음 chunk guidance 문장의 효과는 고정 prompt 행으로 일부만 통제됐다(l.383). 0.11 m 격차 중 약 0.04 m가 guidance 몫이다.
- **pass 1 noise 미보고.** 학습·평가 때 넣은 noise와 noise seed 민감도가 보고되지 않았다. 공식 프로토콜처럼 모든 clip에 seed 0(l.353)을 고정해서 썼다면 "deterministic regressor"라는 주장은 그 고정 noise 위에서만 성립한다.
- **동등성 margin 부재.** "indistinguishable"(l.64)이라 했지만 NAVSIM `first`−`terminal`의 CI [−2.11, +0.41](l.224)는 넓다. 동등성 margin을 정해야 한다.
- **depth 질문 미답변.** intro의 "from which depth"(l.55)에는 답이 없다.

#### (c) 데이터 규모: Fig. 5, Table 10

- **데이터 점이 적음.** 방향은 분명하지만 NAVSIM은 3개 점에 seed 2개, PhysicalAI는 2개 점이다.
- **step 수와의 혼재.** epoch을 40으로 고정해서 데이터 양과 optimizer step 수의 효과가 섞여 있다.
- **"tail-error" 표현.** PhysicalAI 4k의 실패는 중앙값 근처인 2 m 임계값에서 난 것이다(l.295). 따라서 l.65의 "tail-error criterion"은 부정확하다.

#### (d) 결정론 대조와 평가 함정: Table 7 (사후 분석, dev만)

저자 스스로 헤드라인을 깎는 정직한 분석으로, 높이 평가한다. 다만 다음이 걸린다.

- **FDE에서는 평균 수준에 못 미침.**
  - 4-sample 평균 대비 FDE가 +0.08~0.12 m 나쁘다(CI>0). 3-sample 평균 대비 +0.11~0.14 m, 외삽 극한 대비 약 +0.2 m 나쁘다.
  - CI 상한(l.372)이 보고되지 않았는데, 사전 허용치 +0.15 m에 가깝다.
  - 따라서 "approximates the mean"(l.47, l.353, l.436)은 FDE에서 성립하지 않는다. readout과 sample-mean 궤적 사이의 거리도 직접 재지 않았다.
- **외삽 근거가 약함.** k에 따른 값이 단조롭지 않아(0.765→0.777) a+b/k 외삽의 근거가 약하다.
- **anchor head 예시.** anchor head의 100%/18.8%(l.199)는 head 자체의 과적합이지 backbone 암기의 증거가 아니다. 검증 집합도 밝히지 않았다.
- **PhysicalAI 쪽 암기 미확인.** PhysicalAI backbone이 head 학습 clip을 암기했는지 확인하지 않았다.

### 3. 구조

주장은 claim별(l.63–66)인데 결과는 벤치마크별(§6, §7)로 배치되어 있다. 그래서 독자가 §8(l.420)에 가서야 근거를 조립하게 된다.

1. **Fig. 6의 위치.** 가장 설득력 있는 Fig. 6(l.412)이 Discussion에 있어 PDF 12쪽에 나온다. Fig. 1은 3쪽이다.
2. **phase 증거의 분산.** phase 증거가 §6.2와 §7.4(마지막 소절)로 나뉘어 있다. figures/fig_phase_results.png가 있는데도 main.tex에서 쓰지 않았다.
3. **결과 장의 시작.** 결과 장이 과정 서술로 시작한다("Our first development set was…", l.199).
4. **실험 노트 용어.** 본문에 실험 노트 용어가 남아 있다.
   - "protocol v2"(l.227), "(v1)"(l.213), "round 3/4"
   - "change 9"(l.387), "user decision"(l.451)
5. **appendix에 묻힌 핵심 증거.**
   - feature-swap 대조(l.515)
   - 20k phase 비교(Table 10)
   - NAVSIM seed 간 궤적 차이 6 cm(l.515): §8 "Why the verdicts differ"(l.422)를 바로 설명하는 근거인데 인용되지 않는다.
6. **§3·§4.4의 위치와 정의 불일치.** §4.4 Evaluation protocol은 Setup에 속한다. §3에는 불일치가 있다.
   - l.93과 Fig. 1은 future pass를 3회로 쓴다.
   - Fig. 2에는 4번째 "Terminal pass"가 있고, clip당 video forward 20회는 chunk당 4회를 뜻한다.
   - 따라서 `terminal`의 정의(l.107)를 다시 써야 한다.
7. **abstract.** abstract(l.47)는 약 300단어에 숫자가 15개를 넘고, 핵심 단서는 맨 끝에 있다.

**제안 outline**

1. **Intro:** Fig. 6과 Fig. 1을 teaser로 쓰고, 범위를 좁힌 기여를 제시한다.
2. **Related work**
3. **Method:** 방법별 forward/FLOPs 표, phase 정의, head 요약을 둔다.
4. **Experimental design:** 벤치마크, split 근거(암기 결과는 한 문단), B\*, 통계, latency를 다룬다.
5. **주결과:** 두 final set의 품질–latency를 보인다(Table 4+6 통합, Fig. 6).
6. **Where/when to read:** 모든 규모의 phase 결과(fig_phase_results 확장), noise 대조, feature-swap을 넣는다.
7. **What makes it work:** 데이터 규모, memory/guidance/출력 좌표계를 다룬다.
8. **What the accuracy means:** 두 벤치마크의 결정론 효과로 verdict 차이를 설명한다.
9. **Limitations·Conclusion:** 사전 등록 이력, 검증, seed별 표는 appendix로 보낸다.

### 4. 누락된 baseline·대조 (결정에 미칠 영향이 큰 순서)

1. **전체 규모의 `current`·ego-only.** `current`와 ego-only head를 92.6k/20k로, `terminal`을 92.6k로 학습해야 한다. 제목의 주장과 방법 선택이 바뀔 수 있다.
2. **pass 1 noise 대조.**
   - 사용한 noise를 명시하고, test 때 noise seed 민감도를 보고하고, 0이나 상수 입력으로 바꿔 보고, pass 1의 x̂0를 decode해 봐야 한다.
   - flow matching에서 순수 noise에서 한 번에 예측한 값은 조건부 기댓값의 추정치다. 이 성질로 `first`≈`terminal`과 readout의 평균 예측 거동을 설명할 수 있으니 명시하고 검증하기 바란다.
   - 같은 논리로 B01-opt는 DriveWAM 자신의 평균 추정치여야 하는데, ADE가 1.26으로 나쁜 이유를 논의해야 한다.
3. **training-free baseline 확장.**
   - future pass 수 × action step 수의 격자가 필요하다. 예컨대 future pass 1회 + action k step은 논문 자체 결과가 시사하는 baseline이다.
   - batch로 돌린 multi-sample B\*/B10 평균의 실측 latency도 필요하다. "1/47"(l.353)은 batch 병렬화를 무시한 값이다.
4. **latency 정비.** 측정 구간을 일관되게 맞춘 헤드라인, forward/FLOPs 표, 유휴 GPU 재측정이 필요하다.
5. **PhysicalAI ego-only head와 guidance 인과성.** 같은 chunk 좌표계 회전을 쓴 ego-only head(20k)로 open-loop ego-status shortcut의 크기를 재야 한다. guidance 문장이 과거 관측만으로 생성되는지도 밝혀야 한다.
6. **재현·암기 검사.** NAVSIM 전체 navtest에서 B10을 발표 수치와 비교하고, PhysicalAI backbone의 암기 여부를 검사해야 한다.
7. **나머지 대조.**
   - step 수를 맞춘 규모 대조와 seed 추가
   - 두 번째 WAM이나 범용 frozen encoder에 같은 head를 붙인 결과
   - one-step distillation
   - DAC 위반 장면의 정성적 사례

### 5. 표현

**과대 주장**

- "6.1×/11.8×"(l.47): 2.9×(registration 포함)를 함께 적어야 한다.
- "approximates the mean"(l.47), "mean-prediction-level"(l.69, l.436): "4-sample 평균과 ADE는 같고 FDE는 0.08~0.12 m 나쁘다"로 고쳐야 한다.
- "near miss by a single scene"(l.47, l.264): seed 17의 DAC 위반 장면은 181에서 226으로 25% 늘었고, seed 29는 약 +31장면(+17%)이다. 정확한 서술은 "허용치를 1장면 넘었다"이다.
- "decides the outcome"(l.47, l.65), "tail-error criterion"(l.65).
- "rules this out"(l.383): "격차의 약 1/3을 설명한다"가 맞다.
- "provably"(l.422).
- "data is not the explanation"(l.422): 20k clip에서 나온 상관된 chunk는 독립 샘플 100k와 같지 않다.
- "almost twice"(l.224): 실제로는 1.7×다.
- "far better"(abstract): 4k에서만 확인됐다고 밝혀야 한다.
- "Imagine": 근거가 나오기 전까지는 "first future-denoising pass"로 쓰는 것이 낫다.
- "pre-registered": 변경 10건이 있었고 gate 실패 뒤에도 final을 실행했다. 문서를 익명화하고 hash를 붙여 supplementary로 제출하고, 표현은 "protocol-locked with logged amendments"가 적절하다.

**과소 주장**

- 공식 seed 0이 4개 중 가장 좋았는데도(l.426) readout이 앞선다. 그러니 공식 비교는 보수적이며, 단일 sample 기대값(0.876) 대비 결과도 제시할 만하다.
- NAVSIM의 6 cm seed 간 차이는 verdict 차이를 설명하는 강한 근거인데 활용되지 않았다.

### 6. 강점

- **설계:** 공개 checkpoint를 frozen 상태로 쓰고, 사전 규칙으로 고른 training-free B\*를 둔 것은 효율 연구의 모범이다.
- **정직한 보고:** 잠근 final set을 한 번만 평가했고, near miss와 사후 분석을 정직하게 표기했으며, deviation을 기록했다.
- **재현성:** 재구현이 원본과 bitwise로 일치함을 검증했다(Appendix B).
- **평가:** 상호 보완적인 두 벤치마크, paired cluster bootstrap, 복수 seed를 썼다.
- **교훈:** 결정론 대조와, 암기 및 단일 seed 민감성은 커뮤니티에 유용한 평가 교훈이다.

### 7. 권고

**Major revision, 확신도 4/5** (TMLR 체계로는 "재제출 권장").

- 기준 (2), 독자의 관심은 충족한다.
- 기준 (1), 주장과 근거의 일치는 속도–품질 주장에서는 대체로 충족한다. 해석적 주장(imagination, 규모가 결과를 좌우함, 평균의 근사)은 근거보다 강하다.

**평가를 가장 크게 올릴 변경**

1. 전체 규모의 `current`/ego-only 결과와 pass 1 noise 대조를 추가하거나, imagination 주장을 축소할 것.
2. future pass × action step 격자와 batch multi-sample baseline을 추가할 것.
3. 측정 구간이 일관된 latency와 forward/FLOPs 표를 제시할 것.
4. claim별로 재구성하고, Fig. 6과 phase 그림을 앞으로, appendix의 핵심 증거를 본문으로 옮길 것.
5. 과대 표현을, 특히 abstract에서 고칠 것.

**부수 사항**

- **그림:** Fig. 1/2의 글자가 박스를 넘치고, Fig. 3 하단의 "trajecto"가 잘려 있다. Fig. 5는 범례와 선 모양이 다르고 왼쪽 패널에 B\* 선이 없으며, Fig. 6에서는 라벨이 겹친다.
- **본문:** NAVSIM token 수가 빠져 있고(l.101), "chunk"가 두 뜻으로 쓰이며(l.151), "closed-form PDMS"(l.47)는 비표준 용어다.
- **참고문헌:** 18편으로 얇다. 빠진 분야는 diffusion early-exit/feature caching, open-loop ego-status shortcut(AD-MLP, "Is Ego Status All You Need"), one-step diffusion policy다.

---

## B. 사실 확인 (코드·evidence 대조)

| 리뷰 지적 | 확인 결과 | 근거 |
|---|---|---|
| `terminal` 정의 불일치 | **맞음.** `terminal`은 3회 denoising 뒤 denoise된 latent를 cache에 쓰는 4번째 video pass다. §3·Fig. 1은 3회만 그린다. | `scripts/world_feature_tap.py`의 `PHASES = [current_primer, future_first_preupdate, future_cache_terminal]`, `fig_phases.png` |
| pass 1 noise 고정 | **맞음.** PhysicalAI는 clip마다 `torch.manual_seed(0)`, NAVSIM은 장면별 `sample_seed(token, 0)`이다. readout은 이 noise 위에서만 결정적이다. | `scripts/remote_pai_features.py:50`, `scripts/remote_feature_extract.py:78` |
| guidance 인과성 | **부분적으로 미래 정보를 쓴다.** 장면 문장은 과거 관측만 쓰지만, 경로 명령(직진/좌/우)은 미래 GT ego 자세로 만든다(NAVSIM 주행 명령과 같은 관행). 원고에 명시해야 한다. | `vendor/source_audit/DriveWAM/src/physicalai/vlm_preprocess/generate_route.py`, `generate_scene_evolving_guidance.py` |
| 4표본 평균 대비 FDE | **맞음.** ΔFDE +0.096 [0.049, 0.148] / +0.084 [0.037, 0.132] / +0.117 [0.066, 0.167]. FDE>2m 비율은 +1.9~2.3%p다. 일부 seed는 B\* 허용치(+0.15 m, +1%p)를 넘는다. | `evidence/physicalai_2026-10-04/reports/pai-ensemble-dev.json` |
| NAVSIM B10 재현값 | navtest 전체(dev 3,250 + final 8,896)로 합치면 **90.01**이다. DriveWAM 공개 README 값은 90.1이다. 원고에 넣으면 된다. | `evidence/navtest_dev_2026-10-02/analysis-means.csv`, `evidence/navtest_final_2026-10-03/.../final-summary.json`, `evidence/chenchenshi__DriveWAM__README.md` |
| NAVSIM B10의 cache pass | NAVSIM 지연 측정의 B10은 원본 그대로(마지막 action cache pass 포함)다. PhysicalAI는 B10-opt다. 두 벤치마크의 B10 정의를 명시해야 한다. 4.2×는 B10-opt 기준이면 약 4.0×다. | `scripts/remote_final_latency.py`, `scripts/remote_pai_latency.py` docstring |
| 단일 표본 기대값 | dev seed 0~3의 B10 ADE 평균은 0.876이다. 공식 seed 0(0.821)이 가장 좋았으므로 공식 비교는 보수적이다. | 원고 Table 7 |
| "almost twice" | terminal/first 지연비는 0.794/0.467 = **1.70**이다. | 원고 Table 3 |
| head 학습 비용 | PhysicalAI head 학습은 seed당 4k 약 0.3 h, 20k 약 1.5 h다. feature 추출 비용과 함께 보고하면 증류와 비용을 비교할 수 있다. | `evidence/physicalai_2026-10-04/heads/*/summary.json` (`elapsed_s`) |

### 무학습 격자 기준의 지연 추정 (리뷰 지적 4-3)

측정된 중앙값에서 pass당 비용을 더하는 방식으로 역산했다. 품질은 미지수다. B01-opt는 이미 두 벤치마크에서 탈락했으므로, 관건은 V1-A2(NAVSIM)와 V1-A3(PhysicalAI)다.

| 변형 (future pass 1회 + action k step) | NAVSIM 추정 지연 | 제안 모델 비율 | PhysicalAI 추정 지연 | 제안 모델 비율 |
|---|---:|---:|---:|---:|
| V1-A1 | 231 ms | 0.82 (0.8 기준 미달) | 128 ms | 0.55 |
| V1-A2 | 273 ms | 0.69 | 186 ms | 0.38 |
| V1-A3 | — | — | 244 ms | 0.29 |
| 현재 B\* | 420.7 ms | 0.449 | 434.0 ms | 0.163 |

pass당 비용: NAVSIM action 약 42 ms, future 약 74 ms. PhysicalAI action 약 58 ms, future 약 96 ms.

DriveWAM `predict(num_video_steps, num_action_steps)`와 `scripts/pai_common.py`의 `rollout(video_steps=...)`에서 인자 하나만 바꾸면 돌릴 수 있다. 다만 10/5 서버 정리로 다음 데이터가 지워져 있어 다시 준비해야 한다.

- NAVSIM navtest·navtrain 전처리 입력
- NAVSIM 92.6k feature
- PhysicalAI dev 전방 카메라

근거: `evidence/disk_cleanup_2026-10-05/log.jsonl`

### 리뷰와 판단이 다른 점

- **"pre-registered" 표현:** 바꿀 필요까지는 없다고 본다. 결과 전에 기록한 변경 10건은 사전등록 관행과 양립한다. 대신 익명화한 사전등록 문서와 hash를 보충자료로 내고, final 예외 실행을 본문에 분명히 쓰면 된다.
- **PhysicalAI backbone 암기:** 평가는 처음 보는 test에서 했으므로 결과의 타당성을 위협하지는 않는다. 우선순위는 낮다.

## C. 권장 순서

1. **글만으로 고칠 것 (새 실험 없음)**
   - 주장 중심 재구성과 abstract 축소
   - 과대 표현 수정
   - `terminal` 정의 정정과 forward 수 표
   - 지연 병기(등록 포함 2.9×, B10 정의)
   - B10 재현값 90.0 대 90.1 추가
   - guidance 경로 명령의 출처와 pass 1 noise 명시
   - 실험 노트 용어 제거, 부록 핵심 근거를 본문으로 이동, 관련 연구 보강
2. **서버 실험 (결정 영향 순)**
   1. 무학습 격자: future pass 1~3 × action step
   2. PhysicalAI 20k `current`·ego-only head. ego-only는 backbone feature가 필요 없다.
   3. pass 1 noise seed 민감도
   4. 배치로 돌린 다표본 평균 지연
3. **결과에 따라 고칠 것**
   - V1 변형이 품질 기준을 통과하면 B\*를 그 변형으로 다시 정의하고 속도 주장을 고친다.
   - 20k `current`가 `first`를 따라오면 "imagination" 주장을 축소한다.
