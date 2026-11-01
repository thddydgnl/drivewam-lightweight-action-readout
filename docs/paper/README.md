# Paper drafts / 논문 초안

두 편의 초안이 있다. 둘 다 LaTeX 원본이고 `tectonic`으로 바로 빌드된다 (`brew install tectonic`; pdflatex/Overleaf도 가능).

| 폴더 | 대상 | 양식 | 상태 (2026-10-15) |
|---|---|---|---|
| [`jkaia2026/`](jkaia2026/) | 2026 한국인공지능학회 추계학술대회, **General (Poster) 트랙** | 학회 LaTeX 템플릿 `kaia.cls` (2026-10 배포본), 2쪽 | 초안 v1, 2쪽, 영문 제출본 `main.tex` + 한국어 작업본 `main_ko.tex` |
| [`tmlr/`](tmlr/) | **Transactions on Machine Learning Research (TMLR)** | 공식 `tmlr.sty` / `tmlr.bst`, 익명 제출 모드 | 초안 v1 → v1.1 (2026-10-31) → **v2 (2026-11-01)**, 21쪽 (본문 15쪽 + 참고문헌 15–17쪽 + 부록 A–F 17–21쪽). 주장 중심으로 재구성하고 11/1 보완실험을 반영. 이전 원고는 `tmlr/archive/main_v1.1.tex`, PDF는 `TMLR_draft_v1.1.pdf`. 아래 §2 수정 기록 참고 |

공통 파일: [`references.bib`](references.bib) (두 논문이 함께 씀), [`figures/`](figures/) (영문 그림, PDF+PNG), [`build/make_figs_en.py`](build/make_figs_en.py) (그림 재생성), [`build/pdf2png.py`](build/pdf2png.py) (페이지 미리보기).

```bash
cd docs/paper/jkaia2026 && tectonic -X compile main.tex
cd docs/paper/tmlr && tectonic -X compile main.tex
python3 docs/paper/build/make_figs_en.py        # figures/*.pdf 재생성
```

모든 수치는 [`../통합_연구기록.md`](../통합_연구기록.md), [`../PhysicalAI_결과보고서.md`](../PhysicalAI_결과보고서.md), [`../README.md`](../README.md)에서 가져왔고 근거 파일은 `evidence/`에 있다. 주장 범위는 통합 연구기록의 "쓰면 안 되는 주장"을 따른다("B10보다 정확하다"는 주장 없음, NAVSIM은 근소 미통과로 기술).

---

## 1. 한국인공지능학회 2026 추계학술대회 (JKAIA 2026) — 포스터 트랙

학회 홈페이지 [논문제출 안내](https://aiassociation.kr/Conference/ConferenceView.asp?AC=0&CODE=CC20260901&CpPage=377)와 [CFP 공지](https://aiassociation.kr/board/board.asp?b_code=290&Action=content&GotoPage=1&B_CATE=BBS1)에서 2026-10-15에 확인한 규정:

- 일시/장소: 2026-11-19(목)–20(금), 서울대학교 호암교수회관 베리타스홀
- **논문 제출 마감 2026-10-25(일)**, 심사결과 통보 11-09(월)
- 제출: EasyChair <https://easychair.org/conferences/?conf=jkaia2026> (계정 필요). 트랙 선택: **General (Poster)**
- 논문은 **영문**으로 작성. EasyChair 입력 시 제목은 영문, **초록은 국문**으로 입력
- 심사는 **single-blind** → 저자 명기
- 포스터 트랙: 학회 LaTeX 템플릿을 그대로 쓰고 **1쪽 또는 2쪽**으로 작성. 제목·저자·소속·초록(선택)을 포함한 2장 분량 요약본 PDF 제출
- 저작권 양도는 선택 사항. 시상과 JARA 출판은 일반(Full Paper) 트랙만 해당. 포스터 트랙 논문은 **논문집에 포함되지 않으며 국제학회 제출·게재 가능**하다고 안내함 (CFP 이미지 4번 항목)

템플릿 원본: `https://aiassociation.kr/UploadData/Editor/Conference/202610/4C732AD210FF48878CC64322E9E5470A.zip` (`main.tex`, `kaia.cls`, `ama.bst`, `references.bib`). 2026 배포본의 `kaia.cls`는 `onecolumn`이고 `main.tex`는 초록·키워드·A. Problem Setup / B. Novelty / C. Algorithms / D. Experiments 네 항목을 틀(`\fbox`) 안에 넣는 구조다. 초안은 이 네 항목 구조를 그대로 따랐다.

언어와 작업 방식: 학회 규정대로 **영문 `main.tex`를 제출**한다. `main_ko.tex`는 같은 문단 구조를 가진 한국어 작업본으로, 저자가 한국어판을 읽고 고치면 같은 자리의 영문을 `main.tex`에 반영한다(Claude Code 세션에서 "한국어판 n번째 문단을 이렇게 고쳤다"고 알려 주면 영문을 맞춘다). 두 파일의 절·문단 순서는 1:1로 같다. 한국어판은 한글 보고서 그림(`../report/figures/fig2_compare.png`)을 쓰고 XeTeX에서 kotex + Apple SD Gothic Neo로 조판한다.

템플릿 대비 바꾼 점 (파일 머리의 주석에도 적어 둠):
- 템플릿의 `\fbox{\parbox}`는 쪽을 넘기지 못하므로 두 쪽짜리 요약을 위해 `mdframed`로 같은 모양의 테두리를 그렸다.
- `kaia.cls`의 `\maketitle`이 `\twocolumn[...]`을 호출하므로 `\twocolumn`을 1단으로 재정의해 템플릿 PDF와 같은 1단 모양을 유지했다.
- tectonic(XeTeX)에서는 Times(`mathptmx`)가 매핑되지 않아 `\ifXeTeX`일 때만 시스템 Times New Roman을 쓴다. pdflatex/Overleaf에서는 클래스 기본값이 그대로 적용된다.

제출 전 할 일:
- [ ] `main_ko.tex`를 읽고 고칠 부분을 정한 뒤 영문 `main.tex`에 반영
- [x] 저자: 송용휘, 강현수(지도교수, hskang@chungbuk.ac.kr), 충북대학교 정보통신공학부 (2026-10-15 반영). 본인 이메일을 학교 계정으로 바꿀지 결정
- [ ] EasyChair 국문 초록 작성 (아래 초안 참고)
- [ ] 본인 문장으로 다시 읽고 수정. 2쪽 초과 여부 확인 (`pypdf`로 쪽수 확인됨)
- [ ] TMLR에 낼 경우를 대비해 저작권 양도는 선택하지 않음

국문 초록 초안 (EasyChair용; `main_ko.tex`의 초록과 동일):

> DriveWAM과 같은 주행 World-Action Model(WAM)은 5B 비디오 확산 트랜스포머로 미래 영상 잠재(latent)를 상상한 뒤, 같은 백본을 10회 더 실행해 4초 행동 chunk를 denoising하며, 이것이 추론 지연의 가장 큰 항이다. 본 연구는 고정된 백본의 첫 미래 denoising 패스(미래 패스 1) 직후 표현에서 작은 head가 궤적을 읽어낼 수 있는지 묻는다. 이 패스에서 세 트랜스포머 블록을 hook으로 읽고, 1천만 파라미터의 state-fusion head(SF3)로 궤적을 회귀한 뒤 추론을 종료한다. 공개 checkpoint는 갱신하지 않는다. NAVSIM에서는 공식 DriveWAM보다 4.2배, 품질을 유지하는 가장 빠른 저단계 기준보다 2.2배 빠르면서 PDMS 손실은 0.2–0.4점이다. NVIDIA PhysicalAI-AV 공식 test에서는 11.8배 빠르면서 궤적 오차가 DriveWAM과 비슷하다. 공식 단일 표본보다는 낮고, dev 세트에서 표본 4개 평균과 비교하면 ADE는 대등, FDE는 약 0.1 m 높다. 진단 결과 미래 패스 1에 행동 정보 대부분이 이미 들어 있고, 미래 denoising을 더 해도 유의한 이득이 없으며, head 학습 데이터가 늘수록 격차가 꾸준히 줄었다.

---

## 2. TMLR (Transactions on Machine Learning Research)

공식 안내 ([author guide](https://jmlr.org/tmlr/author-guide.html), [editorial policies](https://jmlr.org/tmlr/editorial-policies.html), [acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html), 2026-10-15 확인):

- 제출: OpenReview <https://openreview.net/group?id=TMLR>, 상시 접수. 모든 저자가 활성 OpenReview 프로필(소속·이해충돌·출판 이력) 필요
- 양식: `tmlr.sty` + `tmlr.bst` 필수 (GitHub `JmlrOrg/tmlr-style-file`). 10pt, 1단, US Letter. 제출본은 `\usepackage{tmlr}`(익명). 채택 후 `[accepted]`, arXiv용 `[preprint]`
- **쪽수 제한 없음.** 다만 길이는 내용으로 정당화해야 하며 본문이 지나치게 길면 심사가 늦어질 수 있음. 부록은 참고문헌 뒤
- **이중맹검.** 저자명·소속·개인 GitHub·서버명 등 식별 정보 제거. arXiv 게시는 허용되지만 제출본에서 실명 버전을 링크하면 안 됨
- 보충자료 100 MB 이하 (PDF/ZIP), 코드 제출 권장(익명화 필요)
- 심사 기준 두 가지: (1) 주장이 정확하고 설득력 있는 근거로 뒷받침되는가 (2) TMLR 독자 일부가 관심을 가질 만하고 명확히 전달되는가. 새로운 SOTA·방법의 novelty는 필수 조건이 아님. 주장과 근거의 간극은 실험 추가 또는 주장 축소로 메워야 함
- 중복 게재: 다른 **archival** 심사 venue에 게재·채택·동시 제출된 내용 재사용 불가. 명시적 비archival venue(워크숍 등)·arXiv 공개는 허용. 학회 논문의 확장판은 받지 않음
- Broader Impact Statement는 위험이 있을 때 필수(초안에 포함함). 저자 기여·사사는 채택 후에만 추가
- LLM은 보조 도구로 허용, 저자가 내용 전체에 책임. LLM은 저자가 될 수 없음
- 저작권은 저자 보유, CC BY 4.0

v2 구성 (주장 중심): Abstract / 1 Introduction (Fig. 1 세 경로 개요, Fig. 2 품질–지연) / 2 Related work / 3 DriveWAM inference and the readout phases (forward 수 표) / 4 The readout head / 5 Experimental design (벤치마크·분할, 기준 모델과 무학습 격자, 사전 고정 margin 한 절, 통계, 지연, 검증) / 6 Main results on the locked final sets / 7 Where the action information is (phase, ego-only, state-swap, noise seed, guidance 교란) / 8 How much of the future the action generator needs (격자) / 9 What the readout needs (규모, 출력 좌표계, memory·guidance, head 구조) / 10 What the accuracy numbers mean (결정성 대조, 벤치마크 차이, 평가 관행) / 11 Limitations / 12 Conclusion / Broader Impact / References / 부록 A 프로토콜 문서와 변경 기록, B 비트 동일성 검증, C 구현 세부(head 그림, 지연 세션), D NAVSIM 추가, E PhysicalAI 추가, F 격자 전체 표.

제출 전 할 일:
- [ ] 익명성 점검: 코드 링크를 익명 저장소(예: anonymous.4open.science)로 바꾸고, 공개 GitHub(실명)·연구설명 PDF·서버명은 본문에 넣지 않는다. 현재 초안에는 외부 링크 없음
- [ ] 한국인공지능학회 포스터가 "비archival"임을 제출 시 밝힐 수 있도록 CFP 안내문(논문집 미수록) 보관. 저작권 양도는 하지 않음
- [ ] 관련 연구 인용 재확인 (DriveWAM/Light-WAM/Faster-WAM은 2026-10-15, Qwen3-VL(arXiv:2511.21631)·Wan(arXiv:2503.20314)은 2026-10-31, v2에서 추가한 4편은 2026-11-01에 확인함. 나머지 학회 논문 항목(VPP, NAVSIM, DiffusionDrive, Vista 등)은 아직 확인 안 함)
- [ ] `main.tex`의 본문을 본인 문장으로 읽고 수정 (저자 책임)
- [ ] OpenReview 프로필(기관 이메일), 공저자 확정, action editor 추천, 이해충돌 입력
- [ ] 보충자료: 익명화한 코드 zip (≤100 MB), 사전등록 문서 영문 요약

수정 기록:

**v2 (2026-11-01).** 내부 검토(`REVIEW_internal_2026-10-31.md`)와 11/1 보완실험(`evidence/r5_2026-11-01/`, 사전등록 `docs/사전등록_보완실험_2026-11-01.md`)을 반영해 전체를 다시 썼다. v1.1 원고는 `tmlr/archive/main_v1.1.tex`에 보관.
- 결과 반영: 무학습 격자 V{v}-A{k}에서 **B\*가 두 벤치마크 모두 V0-A2**(future denoising 0단계 + action 2단계)로 바뀜. NAVSIM final V0-A2 89.72 PDMS(B10 89.67), PhysicalAI test V0-A2 ADE 0.778(B10-opt 0.802). 제안 모델 지연비는 V0-A2 대비 0.691(NAVSIM)·0.384(PhysicalAI), B10 대비 0.234·0.084. 사전 고정 margin 기준: NAVSIM seed 29 통과·seed 17 DAC 초과(1장면), PhysicalAI seed 17·29 통과·seed 43은 V0-A2 대비 FDE margin 초과. 20k `current`·ego-only head(PhysicalAI): `first` 우위 유지(current−first ADE +0.09~+0.11, ego-only 0.873 ≈ current 0.860~0.868). noise seed: NAVSIM 영향 작음(PDMS ±0.2, RMS 0.14 m ≈ B10 0.13 m), PhysicalAI는 ADE +0.045~0.067이지만 B10-opt도 같은 seed에서 +0.048~0.061이고 seed를 맞추면 제안 모델이 0.06 m 앞섬. 지연은 유휴 호스트 세션(11/1) 값으로 전부 교체(제안 모델 186.3 ms / 67.4 ms).
- 구조: 벤치마크별 장(§6 NAVSIM, §7 PhysicalAI)을 없애고 주장별로 재배치(위 구성 참고). 품질–지연 그림을 2쪽으로, phase 그림(`fig_phase_results`, 20k 패널 추가)과 격자 heatmap(`fig_grid`, 신규)을 본문에 넣음. SF3 head 그림은 부록 C로.
- 표현: "pre-registered" 표현 제거(사전 고정 margin은 §5.3 한 절과 부록 A로 축소). "approximates the mean" → "4표본 평균과 ADE는 같고 FDE는 0.08~0.12 m 나쁨", "almost twice"·"provably"·"rules this out"·"tail-error"·"data is not the explanation" 삭제/수정. terminal pass 정의 정정(denoising 3회 + σ=0.4 cache pass, V0는 σ=1 pass 1회가 cache를 씀). guidance 경로 명령이 미래 GT에서 나옴을 명시, pass-1 noise 고정 명시, B10 재현값(navtest 전체 90.01 vs 90.1) 추가, 등록 포함 chunk당 지연 병기, 실험 노트 용어(protocol v2, round 3/4, change 9) 제거.
- 용어 통일: 제안 모델은 "the readout"(표·그림은 SF3-first), DriveWAM 자체 행동 생성은 "action generator", hidden state는 "states"(feature는 관련 연구 용어에만), noise seed/training seed 구분, NVIDIA chunk는 "recording", B\* 선별 규칙은 "B\* rule", 제안 모델 기준은 "margins".
- 참고문헌 추가(2026-11-01 arXiv·CVF·RSS 페이지에서 저자·학회·연도 확인함): AD-MLP(Zhai et al. 2023, arXiv 2305.10430), Is Ego Status All You Need(Li et al. CVPR 2024), DeepCache(Ma et al. CVPR 2024), Consistency Policy(Prasad et al. RSS 2024).
- 빌드: overfull 0개, 21쪽, `\raggedbottom` 추가.

**v1.1 (2026-10-31).** 실험 기록·evidence와 원고를 대조한 뒤 고쳤다. v1 PDF는 `TMLR_draft_v1.pdf`로 남겨 두었고, 최신 빌드는 `main.pdf`(= `TMLR_draft_v1.1.pdf`)다.
- 지연 정의를 벤치마크별로 나눴다(§4.4). NAVSIM은 policy latency(메모리에 있는 카메라 프레임에서 궤적까지, VAE 인코딩·관측 패스 포함, 경로 명령 4개의 문장 임베딩은 미리 계산). PhysicalAI는 decision latency(기록 등록이 끝난 뒤부터 action chunk까지, VAE latent·안내 문장 임베딩은 미리 계산해 제외). v1의 "VAE 제외" 정의는 PhysicalAI에만 맞았다. 근거: `scripts/remote_final_latency.py`, `scripts/remote_pai_latency.py`. 기여 1, 표 1·4, 그림 6 캡션, 한계, 결론, 부록 C도 맞췄다.
- 인용 추가: Qwen3-VL(표 1, 부록 C), bootstrap(Efron & Tibshirani), 사전등록(Nosek et al.). Qwen3-VL·Wan 서지를 arXiv 기준으로 바꾸고, Efron은 `@book`, GAIA-1은 `@article`로 형식을 고쳤다.
- 포스터에서 고친 표현을 옮겼다: 미국식 철자(memorized, normalized 등), SF3 head가 Light-WAM action expert의 state-fusion 설계를 따른다는 출처와 차이점(§1, §2, §4.2).
- seed 수를 적었다: NAVSIM 2개(17, 29), PhysicalAI 3개(17, 29, 43), 원래 계획은 3개(부록 A, 한계). NAVSIM에서 2개로 정한 이유는 기록에 없어 쓰지 않았다.
- 부록 D의 seed 간 궤적 차이를 근거 파일의 지표대로 "mean RMS spread"로 고쳤다(`evidence/navtest_dev_2026-10-02/analysis-summary.json`).
- 넘치던 표 2개(결정성 대조, 부록 3·4차)의 열 간격을 줄였다. overfull 경고 0개.

---

## 3. 그림

`build/make_figs_en.py`가 `figures/`에 영문 그림을 만든다 (한글 보고서 그림 `../report/build/make_figs.py`를 영문으로 옮긴 것, 수치는 동일):

| 파일 | 내용 | 사용 |
|---|---|---|
| `fig_overview` | B10 / 무학습 V0-A2 / 제안 경로(조기 종료 + SF3) 세 줄 (v2) | TMLR |
| `fig_overview_poster` | B10 vs 제안 경로 두 줄 (포스터용, 변경 없음) | JKAIA |
| `fig_phases` | current / first / terminal 읽기 시점 | TMLR |
| `fig_sf3` | SF3 head 구조 | TMLR |
| `fig_pai` | PhysicalAI clip/chunk, teacher forcing, 백본 실행 횟수 | TMLR |
| `fig_phase_results` | 시점별 결과 막대 (NAVSIM 4k, PhysicalAI 4k, PhysicalAI 20k + ego-only) | TMLR |
| `fig_grid` | 무학습 격자 heatmap (v × k, dev, 지연 포함) | TMLR |
| `fig_scale` | head 학습 데이터 규모 곡선 | TMLR |
| `fig_tradeoff` | 품질 대 지연 (final, 11/1 유휴 호스트 세션 지연) | TMLR |
