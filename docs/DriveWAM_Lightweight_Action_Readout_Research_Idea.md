# DriveWAM 기반 Lightweight Action Readout 연구 아이디어

**2026-10-04 최종:** 잠금 navtest-final(8,896장면)에서 평가했다.
- 고정 백본의 첫 future pass 표현에 head-train 전체(92.6k)로 학습한 작은 SF3 head는 PDMS 89.25 / 89.49점이다(B10 89.67, B\* 89.65). 차이는 −0.42 / −0.17점이다.
- 지연은 B\*의 0.449배, B10의 0.237배다.
- 사전 기준은 한 seed의 주행 가능 영역(DAC) 조건을 한 장면 차이로 놓쳐 근소 미통과다.
- 학습 데이터 규모(4k → 20k → 92.6k)가 성능을 결정했다. 미래 latent 반복 갱신은 추가 가치가 없었다.

[README](README.md#navtest-final-최종-결과--2026-10-03-사전등록-8-1회-실행)를 참고한다.

**2026-10-02 개정:** navtrain 내부 검증 세트는 공개 백본이 이미 학습한 장면이었다. B10의 ADE가 처음 보는 navtest-dev보다 약 8배 작았다. 그래서 판정 기준을 사전등록한 navtest-dev로 옮겼다. 결과:
- B10 90.93점. 가장 빠른 품질 통과 기준 B\*는 B02-opt.
- 고정 백본의 direct head 최고값은 SF3-terminal 86–87점으로, B10 대비 −3.6 / −4.7점이라 사전 품질 목표(−2점)에 미달했다.
- 첫 future pass는 terminal과 비슷하고 current보다 나았다.

direct readout만으로는 목표에 미치지 못한다는 예비 결론이며, 다음 방향은 [README](README.md#navtest-dev-결과--2026-10-02-사전등록-protocol)에서 결정한다.

**2026-09-30 개정:** frozen 주행 checkpoint의 사후 action 경로 대체로 연구 질문을 좁혔다. 저-step/opt baseline, current·첫 future pass·terminal 비교, GT/teacher target 진단을 필수 pilot에 추가했다. 정확한 구현·평가·성공 기준은 [상세 실행계획서](실행계획서.md)를 따른다. 중계 SSH를 통한 GPU 환경, NAVSIM 4k/1k 파일럿·devval256 metric cache 준비와 합성·실제 관측 B10 실행 확인을 완료했다. 2026-10-01 dev-val 256장면에서 B10/B05 기준 평가를 완료했다. 공식 PDMS는 93.115 / 93.111, policy p50은 792.70 / 583.79ms다. B05의 지연은 26.35% 감소했고 paired PDMS 차이 95% CI는 [−0.0163, +0.0029] point다. 이는 navtrain 내부 12개 로그의 파일럿 결과이며 navtest 일반화 성능이나 B* 선정을 뜻하지 않는다. head 학습과 opt·확대 평가는 남아 있다. [결과·protocol·lock](README.md#b10b05-기준-평가-결과--2026-10-01)을 보관했다.

## 0. 문서 목적

이 문서는 다음 네 논문을 직접적인 출발점과 핵심 관련 연구로 하여, **자율주행 World-Action Model(WAM)의 Action-side 계산을 경량화하는 연구 아이디어**를 구체화한 연구 기획서이다.

- **DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving**
- **Light-WAM: Efficient World Action Models with State-Fusion Action Decoding**
- **Faster-WAM: Do World Action Models Need Deep Action Modules?** — Ma et al., arXiv:2608.02365
- **Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models** — Zhao et al., arXiv:2608.04404

핵심 아이디어는 다음 한 문장으로 요약할 수 있다.

> **이미 학습된 DriveWAM의 backbone을 재학습하지 않고, 관측 및 미래 생성 과정의 어느 시점·깊이에서 얻은 표현이 반복 action denoising을 대체할 만큼 충분한지 검증한다.**

주 방법 SF3-terminal은 기존 미래 latent 생성 경로를 유지한다. SF3-first/current를 함께 비교하여 latent 갱신을 유지할 필요성도 검증한다. 같은 future latent를 보존한다는 사실만으로 학생 action이 그 미래에 의존한다고 단정하지 않는다.

본 문서는 **원 논문에서 확인된 사실**과 **본 연구에서 새롭게 제안하는 설계**를 구분하여 작성한다.

---

# 1. 연구 배경

## 1.1 World-Action Model의 기본 아이디어

World-Action Model(WAM)은 단순히 현재 observation에서 바로 action을 출력하는 정책이 아니라, **미래 world evolution에 대한 표현과 action prediction을 함께 학습하거나 결합하는 모델**이다.

WAM의 기본 관점은 다음과 같다.

```text
Observation
    ↓
World Representation / Future Prediction
    ↓
Action Prediction
```

로봇 조작 분야에서는 대규모 video generative model이 물체 움직임, 상호작용, 장면 변화 등 물리적 동역학을 이미 학습하고 있다는 점을 이용하여, 이를 robot action prediction에 활용하는 연구가 빠르게 진행되고 있다.

자율주행에서도 동일한 관점이 중요하다. 차량의 action은 주변 차량의 이동, 보행자의 움직임, 차선 및 도로 geometry, traffic-light 상태 변화, ego vehicle의 미래 위치 및 방향, route command에 따른 장기 행동과 직접 연결된다.

따라서 단순한 현재 장면 기반 policy보다 **future dynamics를 명시적으로 고려하는 WAM 구조**가 자연스럽다.

---

# 2. 연구의 출발점: DriveWAM

이 장은 DriveWAM 논문 전반의 배경이다. 본 연구의 공개 NAVSIM 경로는 현재 전방 영상 한 장, ego history와 고정 route 문장을 사용한다. Qwen guidance 및 장기 action KV memory 검증은 PhysicalAI 확장으로 분리한다. 논문의 일반 구조와 NAVSIM 실행 범위를 구분한다.

## 2.1 DriveWAM의 핵심 구조

DriveWAM은 pretrained video diffusion transformer를 자율주행 policy의 중심 backbone으로 사용한다.

핵심 구성 요소는 다음과 같다.

1. **Video VAE**
2. **Action Encoder**
3. **Ego-state Encoder**
4. **Wan2.2-TI2V-5B Video Diffusion Transformer**
5. **Scene-Evolving VLM Guidance**
6. **Action Decoder**
7. **Selective KV Memory**

전체 구조를 단순화하면 다음과 같다.

```text
                    Current Observation
                            │
                         Video VAE
                            │
                         Video Token
                            │
Past Video ────────────────┐
Past Action ───────────────┤
Ego State ─────────────────┤
VLM Guidance ──────────────┤
                           ↓
                 Wan2.2-TI2V-5B
              Video Diffusion Transformer
                           │
                           ↓
                    Future Video Latent
                           │
                           ↓
                Same 5B Video Transformer
                   for Action Generation
                           │
                           ↓
                      MLP Decoder
                           │
                           ↓
                     Ego Trajectory
```

DriveWAM의 중요한 특징은 **World Model과 Action Model이 완전히 별도의 대형 네트워크가 아니라, 동일한 Video Diffusion Transformer를 둘 다에 사용한다는 점**이다.

## 2.2 입력 표현

### Video

현재 및 과거 frame은 pretrained VAE를 통해 latent로 변환된다.

\[
z_k = \mathrm{VAE}(x_k)
\]

여기서 \(x_k\)는 video chunk, \(z_k\)는 encoded video latent이다.

### Action

차량의 ego action은 ego-frame translation과 yaw increment를 이용하여 표현된다.

\[
u_k = E_a(a_k)
\]

여기서 \(E_a\)는 MLP 기반 action encoder이다.

### History

과거 video와 action은 다음과 같이 history를 구성한다.

\[
H_k = \{(z_i, u_i)\}_{i\leq k}
\]

즉 DriveWAM은 단순한 single-frame policy가 아니라, **과거 video-action history를 조건으로 미래를 예측하는 autoregressive policy**이다.

## 2.3 Future World Modeling

DriveWAM은 먼저 미래 video latent를 생성한다.

\[
\hat v^z_{k+1,\tau}
=
T_\omega
(
z_{k+1,\tau};
H_k,e_k,g_k,\tau
)
\]

여기서:

- \(T_\omega\): Wan2.2-TI2V-5B 기반 Video Diffusion Transformer
- \(H_k\): past video-action history
- \(e_k\): ego state
- \(g_k\): VLM guidance
- \(\tau\): flow-matching timestep

이 과정을 통해 future latent \(\hat z_{k+1}\)를 생성한다.

즉 먼저:

```text
Current Scene
    ↓
World Model
    ↓
Imagined Future World
```

가 수행된다.

## 2.4 Inverse-Dynamics Action Generation

Future world를 만든 후, DriveWAM은 이 future latent를 condition으로 action을 생성한다.

\[
\hat v^a_{k+1,\tau}
=
D_a
\left(
T_\omega
(
u_{k+1,\tau};
\tilde z_{k+1},
H_k,e_k,g_k,\tau
)
\right)
\]

여기서:

- \(\tilde z_{k+1}\): future latent
- \(T_\omega\): 동일한 5B Video Transformer
- \(D_a\): MLP action decoder

중요한 점은 **최종 MLP decoder 자체가 큰 것이 아니라, action generation 과정에서 다시 5B Transformer를 사용한다는 점**이다.

DriveWAM은 이 구조를 **predicted future에 grounded된 inverse-dynamics readout**으로 설명한다.

```text
Predicted Future World
        ↓
What ego action is consistent with this future?
        ↓
Action
```

## 2.5 Scene-Evolving Driving Guidance

DriveWAM은 Qwen3-VL-8B를 semantic guidance 용도로 사용한다.

입력:

- latest observation
- recent ego trajectory
- route command

출력 예시:

```text
"Proceed straight with caution,
slowing near pedestrians and
maintaining safe distance."
```

이를 다음처럼 표현할 수 있다.

\[
g_k=\Phi_{VLM}(x_k,a_k,c_k)
\]

중요한 점은 VLM이 직접 trajectory를 생성하는 것이 아니며, chunk마다 한 번 guidance를 생성한 뒤 같은 guidance를 여러 denoising step에서 재사용한다는 것이다.

## 2.6 Selective KV Memory

DriveWAM은 long-horizon rollout에서 history KV cache가 계속 증가하는 문제를 해결하기 위해 **Selective KV Memory**를 사용한다.

Video와 Action memory를 별도로 관리한다.

\[
H_k^v,\quad H_k^a
\]

각 token에 대해 relevance와 redundancy를 함께 고려한 score를 계산하여, 반복되는 background token은 제거하고 moving vehicle, lane geometry 등 중요한 token을 유지한다.

이 부분은 이미 DriveWAM 내부에서 상당히 직접적인 efficiency optimization이 적용되어 있으므로, 본 연구의 주된 target으로 두지 않는다.

---

# 3. DriveWAM의 병목 분석

아래 H20 수치와 action 10→5 step 결과는 논문의 PhysicalAI 조건을 설명한다. NAVSIM/A5000의 병목 비중이나 baseline 성능으로 복사하지 않으며, 본 실험에서는 실제 호출 수와 전체 policy latency를 다시 측정한다.

## 3.1 논문에서 보고된 실제 latency

DriveWAM은 H20 GPU 1장 기준 한 4-second chunk에 대해 다음 inference cost를 보고한다.

| Stage | Latency |
|---|---:|
| VLM guidance | 125 ms |
| Future video generation | 372 ms |
| Action denoising 10-step | 765 ms |
| Action denoising 5-step | 374 ms |

즉 가장 큰 single bottleneck은 **Action Denoising**이다.

## 3.2 왜 Action Generation이 비싼가?

겉으로 보기에는 Action은 몇 개의 trajectory point만 출력하므로 가벼워 보여야 한다.

하지만 실제로는 다음과 같다.

### Video

```text
5B Video DiT
× 3 ODE steps
```

### Action

```text
5B Video DiT
× 10 ODE steps
```

기본 설정에서:

\[
T_{\text{video}}=3
\]

\[
T_{\text{action}}=10
\]

이다.

즉 action을 생성할 때마다 **5B Video Transformer 전체를 반복 실행한다.**

## 3.3 병목의 본질

따라서 문제를 단순히 "Action denoising step이 많다"라고 정의하기보다 다음처럼 정의하는 것이 더 정확하다.

> **DriveWAM은 이미 Future World Representation을 생성했음에도 불구하고, Action Representation을 만들기 위해 동일한 5B World Backbone을 여러 번 다시 실행한다.**

기존:

```text
World Generation
5B DiT × 3
      ↓
Future

Action Generation
5B DiT × 10
      ↓
MLP
      ↓
Trajectory
```

이 구조에서 action side가 전체 inference cost의 가장 큰 비중을 차지한다.

## 3.4 DriveWAM 논문 자체가 제공하는 중요한 단서

DriveWAM은 action denoising을 10 → 5 step으로 줄였을 때:

\[
ADE@4s: 0.83 \rightarrow 0.84
\]

\[
FDE@4s: 2.47 \rightarrow 2.45
\]

수준으로 성능 변화가 거의 없었다.

반면 action latency는:

\[
765\,ms \rightarrow 374\,ms
\]

로 크게 감소하였다.

이 결과는 다음 가능성을 강하게 시사한다.

> **Action-side generative computation에 상당한 redundancy가 존재할 수 있다.**

본 연구는 여기서 한 단계 더 나아간다.

> 10 step이 아니라 5 step도 충분하다면, **이미 계산된 World Representation으로 Action을 직접 읽어내는 것은 가능한가?**

---

# 4. 핵심 참고 연구 1: Light-WAM

## 4.1 Light-WAM의 문제 정의

Light-WAM은 기존 WAM이 large video backbone과 heavy generative action expert 때문에 비싸다고 본다.

특히 inference에서 future video rollout과 iterative action generation을 모두 수행하면 closed-loop control에 부담이 된다.

Light-WAM의 철학은 다음과 같다.

> **Future-video supervision은 representation learning을 위해 training에 남겨두되, inference에서는 current observation representation에서 action을 직접 decode한다.**

## 4.2 Light-WAM의 전체 구조

```text
Training
────────────────────────────

Future video
   ↓
Downsampled latent
   ↓
Video Backbone
   ↓
Future-video supervision

Current Observation
   ↓
Video Backbone
   ↓
Multi-level features
   ↓
StateFusionActionExpert
   ↓
Action Chunk


Inference
────────────────────────────

Current Observation
   ↓
Video Backbone × 1
   ↓
Multi-level features
   ↓
StateFusionActionExpert
   ↓
Action Chunk
```

Inference에서는 future-video rollout, generative action DiT, iterative denoising을 사용하지 않는다.

Light-WAM의 결과에는 backbone의 LoRA·adapter 적응이 포함된다. 완전히 고정된 DriveWAM에서 head만 학습하는 본 계획과 조건이 다르므로, 이를 frozen readout 성공의 보장으로 사용하지 않는다. [원 논문](https://arxiv.org/html/2606.08242v1)

## 4.3 Multi-Level Backbone State

Light-WAM은 backbone의 마지막 layer만 사용하지 않는다.

선택된 layer:

\[
I=\{8,16,24\}
\]

에서 hidden representation을 가져온다.

\[
H=\{H_\ell\}_{\ell\in I}
\]

즉:

```text
Layer 8  ───┐
Layer 16 ───┼──→ State Fusion
Layer 24 ───┘
```

이다.

## 4.4 Query-Bottlenecked State Fusion

각 layer에는 매우 많은 video token이 존재한다.

\[
H_\ell \in \mathbb{R}^{B\times N\times d}
\]

이를 action decoder가 그대로 처리하면 비용이 커진다.

그래서 Light-WAM은 작은 수의 learnable query를 사용한다.

\[
Q_\ell\in\mathbb{R}^{N_q\times d}
\]

그리고:

\[
P_\ell=MHA(Q_\ell,H_\ell,H_\ell)
\]

를 계산한다.

이를 pooling하여:

\[
s_\ell
=
LN\left(
\frac{1}{N_q}
\sum_{j=1}^{N_q}P_{\ell,j}
\right)
\]

를 만든다.

즉:

```text
Dense Video Tokens
       ↓
Learned Queries
       ↓
Compact State
```

가 된다.

## 4.5 Multi-Level Fusion

각 layer의 compact state를 projection하고 결합한다.

\[
h
=
\phi_{\text{trunk}}
(
\phi_{\text{fuse}}
(
[M_\ell(s_\ell)]_{\ell\in I}
)
)
\]

그리고 각 action timestep의 positional embedding을 추가한다.

\[
r_k=h+\psi(e_k)
\]

최종 action:

\[
\hat a_k
=
\phi_{\text{out}}
(
LN(r_k)
)
\]

따라서 전체 action chunk:

\[
\hat A=[\hat a_1,\cdots,\hat a_K]
\]

를 **single forward pass**로 직접 출력한다.

## 4.6 Light-WAM의 핵심 의미

Light-WAM의 가장 중요한 메시지는 단순한 모델 압축이 아니다.

> **World Model backbone의 representation이 충분히 좋다면, Action을 다시 거대한 generative process로 생성할 필요가 없을 수 있다.**

즉:

```text
World Representation
       ↓
Direct Lightweight Readout
       ↓
Action
```

이라는 관점이다.

## 4.7 Efficiency 결과

Light-WAM 논문에서 inference breakdown은:

| Component | Latency |
|---|---:|
| VAE Encoding | 12.7 ms |
| Visual Branch | 56.5 ms |
| Action Branch | 2.1 ms |
| Total | 72.03 ms |

이다.

이 값은 DriveWAM과 GPU, dataset, input structure가 다르므로 **절대 수치를 직접 비교하면 안 된다.**

하지만 중요한 구조적 결과는 **Action-side computation을 극단적으로 작게 만들 수 있었다는 것**이다.

---

# 5. 핵심 참고 연구 2: Faster-WAM / Dock of Transformers

## 5.1 Faster-WAM이 던지는 질문

Faster-WAM의 제목 자체가 연구 질문이다.

> **Do World Action Models Need Deep Action Modules?**

기존 WAM 중 Mixture-of-Transformers(MoT) 계열은:

```text
Video Layer 1 ↔ Action Layer 1
Video Layer 2 ↔ Action Layer 2
...
Video Layer L ↔ Action Layer L
```

처럼 video/action module의 depth가 거의 같이 증가한다.

Faster-WAM은 이것을 비효율적이라고 본다.

핵심 가정:

> **Video World Model이 이미 physical dynamics를 충분히 학습했다면, Action Head가 다시 깊은 network일 필요가 없다.**

## 5.2 Dock of Transformers(DoT)

Faster-WAM은 Video DiT를 **Representation Hub**로 정의한다.

```text
             Video World Model
              Representation Hub
          ┌────────┬────────┬────────┐
          ↓        ↓        ↓        ↓
       Layer1   Layer2    ...     Layer30
          └────────┴────────┴────────┘
                       ↓
               Docking Interface
                       ↓
               Lightweight Head
                       ↓
                     Action
```

Action module은 Video Backbone의 peer가 아니라 **task-specific output head**가 된다.

DoT Faster-WAM은 video backbone과 action head를 공동 학습한다. 이 절의 구조를 frozen DriveWAM에 붙이는 실험은 원 논문 재현이 아닌 별도 적응 실험이다. [원 논문](https://arxiv.org/html/2608.02365v1)

## 5.3 30-layer Video Backbone + 1-layer Action Head

Faster-WAM은 실제로:

\[
L_v=30
\]

인 Wan2.2-TI2V-5B Video DiT를 사용하고,

Action Head는:

\[
L_a=1
\]

인 single Transformer layer를 사용한다.

즉:

```text
World: 30 layers
Action: 1 layer
```

라는 극단적으로 asymmetric한 구조를 사용한다.

## 5.4 KV-Fusion

문제는 작은 Action Head가 어떻게 30-layer World Backbone의 정보를 충분히 활용하느냐이다.

각 video layer에서 key/value:

\[
K_v^{(\ell)},V_v^{(\ell)}
\]

를 추출한다.

모든 layer에 대해:

\[
\{K_v^{(\ell)},V_v^{(\ell)}\}_{\ell=1}^{L_v}
\]

를 얻는다.

### Channel Mixing

Video Hub와 Action Head가 서로 다른 feature space를 가지므로 먼저 channel projection을 수행한다.

개념적으로:

\[
\bar K^v=W_KK^v
\]

\[
\bar V^v=W_VV^v
\]

이다.

### Layer Mixing

그 다음 여러 layer를 학습 가능한 weight로 결합한다.

\[
\tilde K_h^v
=
\bar K_h^v\times_1A_h
\]

\[
\tilde V_h^v
=
\bar V_h^v\times_1A_h
\]

여기서 \(A_h\)는 attention head별 learned layer mixing coefficient이다.

즉 모델이 **어떤 Action Query를 위해 Video Backbone의 어느 depth를 얼마나 사용할 것인가**를 스스로 학습한다.

## 5.5 Multi-Level Representation의 중요성

Faster-WAM의 ablation에서 final layer만 사용하는 것보다 all-layer KV-Fusion을 사용했을 때 더 좋은 성능을 보였다.

또한 learned layer-mixing signal은 특정 intermediate layer에서 강하지만, 전체 backbone에 걸쳐 의미 있는 signal이 분포한다.

따라서 중요한 결론은:

> **작은 Action Head에서는 World Backbone의 multi-level representation을 잘 전달하는 interface가 중요하다.**

이다.

## 5.6 Video–Action RoPE Alignment

Faster-WAM의 구현에서 중요한 기술적 요소이다.

Video token은 3D RoPE, Action token은 1D RoPE를 사용한다.

따라서 Video Key를 Action Query가 그대로 사용하면 positional basis mismatch가 발생한다.

Faster-WAM은:

```text
Video Key
3D RoPE
    ↓
Undo 3D RoPE
    ↓
Canonical Feature Space
    ↓
KV Fusion
    ↓
Apply 1D RoPE
    ↓
Action Attention
```

을 사용한다.

즉 단순히 여러 layer의 KV를 concatenate하는 것이 아니라, **공통 coordinate basis로 정렬한 뒤 fusion한다.**

## 5.7 Faster-WAM의 핵심 의미

Light-WAM과의 차이는 다음과 같다.

### Light-WAM

```text
Multi-level State
      ↓
Direct Regression
      ↓
Action
```

### Faster-WAM

```text
Multi-level KV
      ↓
1-layer Generative Action Transformer
      ↓
Action
```

즉 Faster-WAM은 Action Generation 자체를 없애지 않고 **Action-specific depth만 극단적으로 줄인다.**

---

# 6. 두 연구를 DriveWAM과 어떻게 연관시킬 것인가

## 6.1 세 논문의 관계

### DriveWAM

> **Future World를 생성하고, 그 Future에 grounded된 Action을 생성한다.**

### Light-WAM

> **Action은 World Representation에서 direct single-pass decoding할 수 있다.**

### Faster-WAM

> **작은 Action Head를 쓰려면 World Backbone의 multi-level information을 효과적으로 전달해야 한다.**

이 세 가지를 결합하면 자연스럽게 다음 질문이 나온다.

> **DriveWAM이 이미 Future World를 계산했다면, 그 과정에서 형성된 multi-level World Representation을 재사용하여 Action을 직접 또는 매우 얕은 Head로 읽어낼 수 있지 않을까?**

## 6.2 제안하는 핵심 구조

### Original DriveWAM

```text
Observation
    ↓
Video VAE
    ↓
5B Video DiT × 3
    ↓
Future World
    ↓
5B Video DiT × 10
    ↓
MLP
    ↓
Trajectory
```

### Proposed Lightweight DriveWAM

```text
Observation
    ↓
Video VAE
    ↓
5B Video DiT × 3
    ↓
Future World Generation

During World Generation:
    ├─ Hidden State @ Layer l1
    ├─ Hidden State @ Layer l2
    ├─ Hidden State @ Layer l3
    └─ Optional KV @ selected layers
            ↓
      Multi-Level Fusion
            ↓
 Lightweight Action Readout
            ↓
      4-second Trajectory
```

즉 \(5B\times10\)에 해당하는 반복 action-side backbone computation을 제거하는 것이 목표이다.

---

# 7. 제안 연구의 핵심 아이디어

## 가제

**LiteDriveWAM: Post-hoc Action Readout from Frozen Driving World Models**

## 7.1 핵심 가설 H1 — frozen 표현의 충분성

> 학습된 주행 backbone을 변경하지 않아도, 관측에서 계산한 표현에 작은 head를 학습하여 기준 모델에 가까운 trajectory 품질을 얻을 수 있는가?

head의 정상 학습, ego-only 대비 가치, held-out log와 navtest의 성능을 확인한다. backbone은 이미 주행 action/video supervision으로 학습된 checkpoint이므로 action supervision 없이 능력이 생겼다고 주장하지 않는다.

## 7.2 핵심 가설 H2 — 여러 깊이의 추가 가치

> 같은 parameter 및 학습 예산에서도 multi-level 표현이 final-only보다 유리한가?

SF1-wide와 SF3-terminal을 비교한다. FMLP와 SF3만 비교하면 pooling 방식과 parameter 수의 차이가 함께 섞인다.

## 7.3 핵심 가설 H3 — 실용적인 효율 개선

> 품질 기준을 만족하는 가장 빠른 저-step baseline보다 작은 head의 전체 policy latency가 낮은가?

원본 B10/B05뿐 아니라 마지막 action cache pass를 생략한 1/2/3/5/10-step 모델을 비교한다. action 연산 감소, 전체 latency, 상주 parameter 수를 분리해 보고한다.

## 7.4 핵심 가설 H4 — 미래 latent 갱신의 추가 가치

> 같은 head에서 current, 첫 future forward, terminal 표현은 어떤 품질·속도 차이를 보이는가?

첫 future pass는 Gaussian slots를 처리한 hidden state이며 아직 solver update를 수행하지 않았다. terminal은 원본의 3회 update 뒤 표현이다. terminal이 유리하더라도 계산량·noise 차이가 남으므로, 미래의 물리적 의미에 대한 강한 해석에는 추가 control이 필요하다. first/current가 더 좋은 결과를 내면 해당 경로로 연구 질문을 수정할 수 있다.

---

# 8. 제안 방법론 1: Final-Layer Direct Action Head

가장 먼저 구현해야 하는 baseline이다.

```text
Generated Future World
       ↓
Final Video-DiT Hidden State
       ↓
Pooling
       ↓
2~3 Layer MLP
       ↓
4-second Trajectory
```

수식:

\[
h_L
=
Pool(H_L)
\]

\[
\hat A
=
D_{\phi}(h_L,e_k,g_k)
\]

여기서:

- \(H_L\): 마지막 Video DiT layer representation
- \(e_k\): ego state
- \(g_k\): optional semantic guidance
- \(D_\phi\): lightweight action decoder

목적은 다음 질문에 답하는 것이다.

> Future World의 final representation만으로도 Action을 직접 읽을 수 있는가?

---

# 9. 제안 방법론 2: Light-WAM식 StateFusion Readout

Final-layer baseline보다 한 단계 발전된 구조이다.

선택 layer:

\[
I=\{\ell_1,\ell_2,\ell_3\}
\]

예:

```text
Layer 10
Layer 20
Layer 30
```

정확한 layer는 ablation으로 정한다.

각 layer \(H_\ell\)에 대해 learned query \(Q_\ell\)를 사용한다.

\[
P_\ell=MHA(Q_\ell,H_\ell,H_\ell)
\]

\[
s_\ell
=
LN\left(
\frac1{N_q}
\sum_jP_{\ell,j}
\right)
\]

fusion:

\[
h
=
\phi_{\text{fuse}}
(
[s_{\ell_1};s_{\ell_2};s_{\ell_3}]
)
\]

trajectory:

\[
\hat a_t
=
\phi_{\text{out}}
(
LN(h+\psi(e_t))
)
\]

전체:

\[
\hat A=[\hat a_1,\cdots,\hat a_K]
\]

---

# 10. 제안 방법론 3: DoT-style Lightweight Transformer Head

StateFusion이 충분하지 않을 경우 확장한다.

World Generation 중 selected/all layer KV를 저장한다.

\[
\mathcal K=\{K_v^{(\ell)}\}_{\ell=1}^{L_v}
\]

\[
\mathcal V=\{V_v^{(\ell)}\}_{\ell=1}^{L_v}
\]

KV Fusion:

\[
\tilde K,\tilde V
=
g(\mathcal K,\mathcal V)
\]

Action Query \(Q_a\)와 mixed-context attention:

\[
O_a
=
Attention
(
Q_a,
[K_a,\tilde K],
[V_a,\tilde V]
)
\]

이 과정을 1-layer Action Transformer로 수행한다.

---

# 11. 단계별 연구 설계

상세 실행계획서의 P0–P7과 같은 순서로 진행한다. DoT·전체 KD·LoRA는 핵심 pilot 이후의 조건부 확장이다.

| 단계 | 실험 목적 | 핵심 산출물·진행 조건 |
|---|---|---|
| P0 환경·단일 sample | A5000 24GB에서 공개 checkpoint 실행 | UMT5 prompt 선계산, 환경 lock, B10 한 sample 성공 |
| P1 데이터·baseline | 공정한 비교 기준 확보 | log split, official scorer coverage, raw/opt parity, 저-step 결과 |
| P2 표현 추출 | 세 시점의 학습 자료 확보 | current/first/terminal cache, GT 누출 방지, 조기 종료 parity |
| P3 최소 head | 학습·출력 계약 및 표현 가치 확인 | 32개 overfit, 4k/1k의 ego-only·linear·FMLP |
| P4 핵심 가설 pilot | 시점·depth·target 효과 분리 | SF3 세 시점, SF1-wide, 작은 GT/teacher 진단 |
| P5 선별 확대 | 필요한 축의 ablation과 최종 학습 | full head-val의 B*, 설정 고정, 핵심 모델 학습 seed 3개 |
| P6 최종 평가 | 잠근 navtest의 품질과 실제 효율 | full coverage PDMS·CI, online p50/p95·VRAM |
| P7 결과 정리 | 증거와 주장 대응 | Pareto·실패 분석·재현 bundle |

SF3-current/first/terminal은 같은 구조의 head를 각 표현으로 각각 학습한다. 하나의 terminal head에 다른 시점의 feature만 넣는 실험으로 시점 우열을 판단하지 않는다. first/terminal 비교가 핵심이므로 query 수 전체 탐색보다 먼저 수행한다.

---

# 12. Training Strategy

## 12.1 가장 낮은 비용의 1차 실험

DriveWAM 전체를 freeze한다.

```text
Trainable:
- Query Pooling
- State Fusion
- Trajectory Head
```

가능하면 feature를 offline cache한다.

```text
DriveWAM Future Generation
          ↓
Extract Hidden States
          ↓
Save to Disk
          ↓
Train Lightweight Head
```

장점:

- 5B backbone backpropagation 불필요
- 반복 실험 빠름
- layer selection 실험 가능
- head architecture ablation 용이

## 12.2 Loss

가장 단순한 형태:

\[
L_{\text{traj}}
=
\sum_{t=1}^{K}
w_t
\|
\hat a_t-a_t
\|_2^2
\]

또는 Cartesian trajectory라면:

\[
L_{\text{pos}}
=
\sum_t
\|
\hat p_t-p_t
\|_1
\]

\[
L_{\text{yaw}}
=
\sum_t
\|
\hat \theta_t-\theta_t
\|_1
\]

\[
L
=
L_{\text{pos}}
+
\lambda_\theta L_{\text{yaw}}
\]

첫 실험에서는 복잡한 diffusion/distillation loss를 추가하지 않는다.

## 12.3 소규모 target 진단과 Optional Teacher Distillation

full KD 학습에 앞서 작은 head-train/head-val subset과 generation seed 3개에서 두 head를 비교한다. D-GT는 기록된 trajectory, D-Teacher는 동일한 generated future에 조건화한 B10 trajectory를 target으로 한다. 나머지 구조·loss·학습 예산은 맞춘다. 두 head 모두 GT 기준 ADE/FDE·PDMS와 teacher imitation error로 평가한다.

생성된 미래와 기록된 action이 서로 다른 가능한 전개를 반영할 위험을 진단하는 것이다. D-Teacher만 잘 되는 결과는 target 불일치나 teacher target의 단순성을 시사할 수 있지만 이를 입증하지는 않는다. teacher는 oracle이 아니다. 이 **작은 진단은 pilot 필수**, 아래 전체 KD는 조건부다.

성능이 부족할 경우 Original DriveWAM Action Output을 teacher로 사용할 수 있다.

\[
L_{\text{KD}}
=
\|
\hat A_{\text{student}}
-
\hat A_{\text{DriveWAM}}
\|_2^2
\]

최종:

\[
L
=
L_{\text{GT}}
+
\lambda_{\text{KD}}L_{\text{KD}}
\]

이 단계도 초기 버전에서는 필수가 아니다.

---

# 13. 핵심 Ablation

## A. Action Architecture — 먼저 수행

| Variant | Action 경로 | 목적 |
|---|---|---|
| B10 / B05 | 원본 10/5 solver step + terminal action cache pass | 공개 실행 재현 |
| B10-opt / B05-opt / B03-opt / B02-opt / B01-opt | 마지막 action cache pass를 생략한 10/5/3/2/1 solver step | 학습 없는 효율 기준 |
| FMLP | final mean + 작은 MLP | 가장 작은 readout |
| SF1 / SF1-wide | final query pooling, 기본/동급 parameter 폭 | pooling·parameter 효과 |
| SF3 | 3-layer query pooling + direct decoder | multi-level readout |
| DoT 변형 | KV-Fusion + 1-layer generative head | direct head의 한계가 확인되면 수행 |

`opt`는 NAVSIM 단일 chunk에서 출력 동일성을 검증한 뒤 적용한다. 다음 chunk에서 action KV를 사용하는 경로에는 그대로 적용하지 않는다.

## B. 표현 시점 — 먼저 수행

| 모델 | Feature | Latent update | Backbone 호출 예상 |
|---|---|---:|---:|
| SF3-current | current primer | 0 | 1 |
| SF3-first | 첫 future forward, sigma=1 | 0 | 3 |
| SF3-terminal | terminal pass, sigma≈0.769231 | 3 | 6 |

모두 작은 head 1회를 추가한다. 호출 수는 latency 비율이 아니며 서버에서 확인한다. feature 교체 control은 장면 의존성을 진단하고, noise·계산량의 혼동을 한계로 기록한다.

## C. Supervision — 먼저 수행

같은 generated future에 대해 GT-target head와 B10-target head를 작은 subset에서 비교한다. 실패를 표현 부족 또는 반복 reasoning의 필요성으로 곧바로 해석하지 않는다.

## D. 세부 탐색 — 핵심 pilot 뒤 수행

- query 수 `{4,8,16,32}`.
- layer 수 1/3/6 및 중간 depth 선택; 동일 parameter 예산 비교 유지.
- current+future 결합, last-update feature, terminal latent-only.
- 미래 solver update 수 1/2 또는 완전 denoising 변형.

first의 0-update 비교는 필수이고, 위 추가 조합은 필요한 것만 실행한다. 모든 조합을 곱해 전체 학습하지 않는다.

---

# 14. 평가 지표

## 14.1 Driving Performance

NAVSIM:

- NC: No at-fault Collisions
- DAC: Drivable Area Compliance
- TTC: Time-To-Collision
- Comfort
- Ego Progress
- PDMS

## 14.2 Trajectory Accuracy

- ADE@3s
- FDE@3s
- ADE@4s
- FDE@4s

## 14.3 Efficiency

반드시 함께 측정:

- Action latency
- Total latency
- Peak VRAM
- FLOPs
- Parameters
- Trainable parameters

가능하면 throughput, real-time Hz도 포함한다.

## 14.4 가장 중요한 그래프

### Performance–Latency Pareto

```text
PDMS
 ↑
 │          ● Original DriveWAM
 │       ● StateFusion
 │    ●
 │ ● Final MLP
 └────────────────────────→ Latency
```

목표는 **Original DriveWAM보다 훨씬 낮은 latency에서 유사한 PDMS를 달성하는 Pareto point 확보**이다.

---

# 15. 예상 주요 표

## Table 1. Main Efficiency Comparison

| Method | Action Method | Action Latency | Total Latency | PDMS | ADE@4s |
|---|---|---:|---:|---:|---:|
| DriveWAM | 10-step shared DiT | - | - | - | - |
| DriveWAM-5 | 5-step shared DiT | - | - | - | - |
| B10/05/03/02/01-opt | 각 solver step, 마지막 action cache pass 생략 | - | - | - | - |
| Ours-Final | final state + MLP | - | - | - | - |
| SF1-wide | 동급 parameter final-only | - | - | - | - |
| SF3-current | current 표현의 multi-level direct | - | - | - | - |
| SF3-first | 첫 future pass의 multi-level direct | - | - | - | - |
| SF3-terminal | terminal 표현의 multi-level direct | - | - | - | - |
| Ours-DoT | 1-layer Transformer | - | - | - | - |

## Table 2. Layer Ablation

| Layers | PDMS | ADE | Latency |
|---|---:|---:|---:|
| Last | | | |
| Mid | | | |
| Early+Mid+Late | | | |
| All selected | | | |

## Table 3. Future Representation Ablation

| Input | PDMS | ADE | FDE |
|---|---:|---:|---:|
| Current | | | |
| First future pass | | | |
| Terminal future pass | | | |
| Current + Future | | | |

---

# 16. 예상 논문 Figure

## Figure 1. Overall Architecture

Original DriveWAM과 Proposed Lightweight DriveWAM의 action path를 비교한다.

## Figure 2. Multi-Level Representation Fusion

Video DiT layer별 state가 Query Pooling/StateFusion으로 들어가는 구조를 그린다.

## Figure 3. Performance–Latency Trade-off

PDMS vs latency를 표시한다.

## Figure 4. Layer Importance

```text
Layer index
    ↓
Action usefulness / learned fusion weight
```

## Figure 5. Qualitative Driving

- intersection
- pedestrian
- vehicle following
- lane change

에서 Original과 Ours trajectory를 비교한다.

---

# 17. 검증할 Contribution 후보

아래는 결과가 확보되었을 때 주장할 후보이며, 이미 입증된 사실이 아니다.

1. **사후 action 경로 대체:** 학습된 DriveWAM을 고정하고 작은 head만 학습하여 반복 action denoising을 대체할 수 있는 조건을 검증한다.
2. **표현 시점과 깊이 분석:** current/first/terminal 및 parameter-matched final/multi-level 비교로 어떤 표현이 주행 readout에 충분한지 분석한다.
3. **강한 효율 기준:** raw/opt·저-step baseline과 같은 품질 기준에서 전체 policy latency를 비교한다.
4. **실패 원인 진단:** generated future와 GT/teacher target의 적합성, head 용량 및 stochastic future 민감도를 구분한다.

미래 표현 재사용이나 작은 action head 자체를 최초의 아이디어로 주장하지 않는다. NAVSIM 단일 checkpoint 결과를 장기 reactive closed-loop, VLM guidance 또는 모든 WAM의 결론으로 확대하지 않는다.

---

# 18. 연구 범위: 추론 경로와 표현의 충분성

이 연구는 학습된 world backbone의 가중치를 유지한 채 추론에 필요한 action 연산을 줄이는 실험이다. 전체 상주 모델 parameter는 대부분 남는다. 따라서 작은 head의 parameter 수를 전체 policy parameter 수처럼 보고하지 않는다.

핵심은 어느 시점·깊이의 frozen 표현이 작은 decoder에 충분한지, 추가 backbone 계산이 어떤 조건에서 실제 품질 이득을 주는지 검증하는 것이다. 기존 WAM이 받은 action/video supervision은 유지되므로 action 학습 전체의 불필요성을 주장하지 않는다.

---

# 19. 기존 연구와의 차별화 포인트

| 연구 | 주요 접근 | 이번 연구에서 구분할 조건 |
|---|---|---|
| DriveWAM | shared backbone으로 future와 action을 반복 생성 | 학습 완료된 주행 checkpoint의 action 추론 경로를 사후 교체 |
| Light-WAM, 2606.08242 | current 표현의 직접 action decoding, LoRA·adapter 적응 | 완전히 고정된 DriveWAM에서 head만 학습하는 충분성 검증 |
| DoT Faster-WAM, 2608.02365 | backbone과 1-layer action head 공동 학습, KV fusion | 공동 재학습 없이 current/first/terminal hidden state를 비교 |
| future-conditioning Faster-WAM, 2608.04404 | 한 번 계산한 noisy future 표현을 action denoising에 재사용 | 주행 도메인의 post-hoc 직접 readout과 latent 갱신의 추가 가치 |

두 Faster-WAM은 다른 논문이며 모두 핵심 관련 연구다. 특히 2608.04404 때문에 “미래 표현을 유지하고 재사용한다”는 설명만으로 차별성을 확보할 수 없다. **주행 checkpoint 고정, head-only 학습, 실제 저-step 대안과 비교, 표현 시점 분석**의 조합을 근거로 연구 가치를 설명한다. 이것이 자동으로 최초성을 보장하는 것은 아니다.

출처: [Light-WAM](https://arxiv.org/html/2606.08242v1), [DoT Faster-WAM](https://arxiv.org/html/2608.02365v1), [future-conditioning Faster-WAM](https://arxiv.org/html/2608.04404v1).

---

# 20. 예상 리스크

## Risk 1. Direct Head의 성능이 크게 떨어짐

가능성:

- DriveWAM action flow가 단순 trajectory decoder가 아니라 중요한 inverse-dynamics reasoning을 수행할 수 있음

대응:

1. Final layer → multi-level
2. MLP → StateFusion
3. StateFusion → 1-layer Transformer
4. Teacher distillation
5. 일부 action denoising step 유지

## Risk 2. Future Generation Feature가 Action에 적합하지 않음

World Generation의 representation은 video reconstruction에 최적화되어 있을 수 있다.

대응:

- layer-wise probe
- adapter
- small LoRA
- future/current feature fusion

## Risk 3. Feature 저장 비용이 커짐

모든 30-layer hidden state를 저장하면 VRAM 증가 가능.

대응:

```text
selected layers only
```

예:

\[
\{10,20,30\}
\]

또는 hidden state 대신 pooled representation만 즉시 생성한다.

## Risk 4. Latency가 생각보다 줄지 않음

Future Video 자체가 이미 372ms이므로 Action을 거의 제거해도 전체 latency의 lower bound가 존재한다.

이 연구의 목표는 "DriveWAM 전체를 10× 빠르게 만든다"가 아니라 **현재 가장 큰 Action-side bottleneck을 제거한다**는 것이다.

후속 연구에서:

- future step reduction
- latent/token reduction
- sparse interaction

등과 결합 가능하다.

---

## Risk 5. 생성된 미래와 기록된 행동의 target 불일치

서로 다른 가능한 전개를 하나의 GT trajectory로 회귀시키면 head가 미래 표현을 무시할 수 있다. 같은 future에 조건화된 B10 행동과 GT를 각각 학습하는 소규모 진단을 먼저 수행한다. teacher의 smoothness·편향도 함께 확인하며, 어느 target도 물리적 일관성의 oracle이라고 간주하지 않는다.

---

# 21. 최소 구현 버전

첫 연구라는 조건에서 가장 현실적인 최소 버전은 다음이다.

## MVP

### 그대로 사용

- pretrained DriveWAM
- Future Video Generation
- NAVSIM의 고정 route 문장 embedding 및 ego 조건
- Existing VAE

### 새로 구현

```text
Selected hidden-state hooks
        ↓
Pooling
        ↓
Small MLP trajectory head
```

### 학습

```text
DriveWAM = frozen
Action Head = train
```

### 비교

```text
Raw B10 / B05
Opt 1 / 2 / 3 / 5 / 10-step
Final-state MLP
Parameter-matched SF1-wide
SF3-current / first / terminal
Small GT / teacher target diagnosis
```

DoT는 처음에는 제외한다.

---

# 22. 구현 진행 순서

- [ ] P0: 환경·prompt cache·체크포인트 경로·B10 단일 sample.
- [ ] P1: log split·official scorer·raw/opt parity·1/2/3/5/10-step 비교.
- [ ] P2: current/first/terminal 수집·GT 누출 검사·cache 및 조기 종료 parity.
- [ ] P3: 32-sample overfit·4k/1k의 최소 head 학습.
- [ ] P4: 같은 SF3의 세 시점 비교·SF1-wide·GT/teacher target 진단·pilot 판정.
- [ ] P5: 필요한 ablation만 수행·full head-val의 B* 고정·최종 학습 seed 3개.
- [ ] P6: 잠근 navtest 전체·PDMS/하위 지표·CI·온라인 latency/VRAM.
- [ ] P7: 결과·미지원 가설·실패 사례·재현 bundle.

각 단계의 입력·산출물·진행 조건은 [상세 실행계획서 5장](실행계획서.md#5-전체-일정과-단계-통과-조건)을 따른다. DoT·전체 KD·LoRA·PhysicalAI는 pilot 결과와 예산을 근거로 별도 결정한다.

---

# 23. 연구 성공 기준

`B*`는 full head-val에서 B10 대비 품질 기준을 만족하는 opt baseline 중 policy p50이 가장 낮은 모델이다. 후보는 1/2/3/5/10-step이며 선택은 navtest 전에 끝낸다. 자세한 CI·NC/DAC 기준은 상세 실행계획서 §1.3을 따른다.

최종 학생은 다음을 동시에 목표로 한다.

- PDMS 차이의 95% CI 하한이 B10 및 B* 각각 대비 −2.0 point 이상.
- policy p50이 B* 대비 20% 이상 감소하고 p95도 악화하지 않음.
- NC/DAC 평균 악화가 두 기준 각각 대비 0.5 point 이내인지 사건 수·CI와 함께 확인.
- 핵심 모델의 학습 seed 3개에서 품질 기준 충족 여부를 각각 확인.

B10 대비 action latency 50% 감소는 보조 지표다. 위 수치는 **사전 연구 목표이며 실측 결과가 아니다.** 표현 분석의 가치와 실용적 효율 성공은 분리한다. B*의 손실에 학생의 허용 손실을 누적해 B10 대비 4 point 저하를 허용하지 않는다.

---

# 24. 연구 결과에 따른 해석

| 관찰 | 지지되는 해석과 한계 |
|---|---|
| parameter-matched final-only보다 multi-level이 우수 | 이 head·데이터 조건에서 여러 깊이의 표현이 유용함을 지지 |
| first와 terminal 품질이 비슷하고 first가 빠름 | 기존 latent 반복 갱신의 추가 가치가 작을 가능성; 모든 미래 conditioning의 무용성을 뜻하지 않음 |
| current가 first/terminal과 비슷하고 더 빠름 | 이 benchmark에서 미래 경로의 추가 가치를 확인하지 못함 |
| GT head는 약하고 teacher head는 강함 | target 적합성·teacher smoothness를 원인 후보로 추가 조사 |
| direct head가 약하고 shallow generative head가 강함 | 시험한 구조에서 generative head가 유리함; 반복 reasoning이 반드시 필요하다는 증명은 아님 |
| B10은 이기지만 B* 대비 이득 없음 | 실용적 효율 목표 미달; 느린 기준만 골라 성공으로 보고하지 않음 |

통계적으로 차이를 발견하지 못한 것과 동등성을 입증한 것은 다르다. 작은 pilot의 결과는 full 평가 또는 추가 seed로 확인하기 전 예비 분석으로 남긴다.

---

# 25. 최종 연구 질문

## Main RQ

> **Can a frozen driving world model support lightweight action readout without retraining its backbone?**

> **이미 학습된 주행 WAM의 backbone을 재학습하지 않고, 어떤 표현으로 반복 action denoising을 대체할 수 있는가?**

## RQ1

> 관측에서 계산한 frozen 표현에 head만 학습해 공개 기준에 가까운 trajectory 품질을 얻을 수 있는가?

## RQ2

> parameter 및 학습 예산을 맞춰도 multi-level 표현이 final-only보다 유리한가?

## RQ3

> 품질 기준을 만족하는 가장 빠른 저-step baseline보다 전체 policy latency가 얼마나 감소하는가?

## RQ4

> current, 첫 future forward, terminal 중 어떤 시점의 표현이 필요한가? latent 반복 갱신의 추가 가치는 무엇인가?

---

# 26. 논문 제목 후보

1. **LiteDriveWAM: Lightweight Action Readout from Multi-Level World Representations for Autonomous Driving**
2. **Do Driving World-Action Models Need Iterative Action Denoising?**
3. **World First, Act Light: Efficient Action Decoding for Driving World-Action Models**
4. **Future-Aware Lightweight Action Decoding for Autonomous Driving World-Action Models**
5. **From Imagination to Action: Lightweight Multi-Level Readout for Driving World Models**

---

# 27. 논문 구성 초안

## 1. Introduction

- WAM의 장점
- Driving WAM의 계산비용 문제
- DriveWAM Action bottleneck
- 핵심 연구 질문
- contribution

## 2. Related Work

### 2.1 Autonomous Driving World Models
### 2.2 World-Action Models
### 2.3 Efficient WAMs
### 2.4 Efficient Action Decoding

## 3. Preliminary: DriveWAM

- architecture
- future generation
- inverse dynamics
- latency bottleneck

## 4. Method

### 4.1 Multi-Level World Representation Extraction
### 4.2 Query-Bottlenecked State Fusion
### 4.3 Lightweight Trajectory Decoder
### 4.4 Optional DoT Action Head
### 4.5 Training Objective

## 5. Experiments

### 5.1 Setup
### 5.2 Main Results
### 5.3 Efficiency
### 5.4 Layer Ablation
### 5.5 Current vs Future
### 5.6 Query Capacity
### 5.7 Qualitative Analysis

## 6. Discussion

- World vs Action capacity
- Future imagination utility
- limitations

## 7. Conclusion

---

# 28. 최종 연구 범위

본 연구는 **학습된 DriveWAM을 고정하고 작은 action head만 학습하는 사후 경량화**를 검증한다. NAVSIM 단일 chunk에서 원본 전처리·ego/route 조건·평가 protocol을 유지한다.

SF3-terminal은 기존 미래 생성 경로를 유지하고, SF3-first/current는 어느 시점의 표현으로 충분한지 비교한다. 작은 head 및 미래 표현 재사용은 선행 연구와 공유하는 아이디어다. 이번 연구의 검증 대상은 **frozen 주행 표현의 충분성, latent 갱신·depth의 추가 가치, 강한 저-step baseline 대비 전체 효율**이다.

query/layer의 넓은 탐색과 DoT·LoRA·PhysicalAI 확장에 앞서 4k/1k pilot으로 이 질문들을 확인한다. 연구 설계와 외부 서버 환경 검증을 완료했으며, 합성 fixture의 메모리 값은 실행계획서에 구분해 기록했다. 실제 주행 데이터의 성능·latency·운영 메모리는 이후 실험에서 측정한다.

---

# 참고 논문

1. Shi et al., **DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving**, arXiv:2605.28544, 2026.
2. Li et al., **Light-WAM: Efficient World Action Models with State-Fusion Action Decoding**, arXiv:2606.08242, 2026.
3. Ma et al., **Faster-WAM: Do World Action Models Need Deep Action Modules?**, arXiv:2608.02365, 2026.
4. Zhao et al., **Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models**, [arXiv:2608.04404](https://arxiv.org/abs/2608.04404), 2026.
