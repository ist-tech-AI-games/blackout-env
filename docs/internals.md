# 내부 구조 및 개발자 가이드

## 목차

- [패키지 구조](#패키지-구조)
- [BlackOutEnv 내부 흐름](#blackoutenv-내부-흐름)
- [Graphic Obs 최적화 구조](#graphic-obs-최적화-구조)
- [Obs 수정 가이드](#obs-수정-가이드)
- [테스트](#테스트)
- [의존성 버전 제약](#의존성-버전-제약)

---

## 패키지 구조

```
blackout_env/
├── __init__.py              — 공개 API 재출
├── env/
│   ├── blackout_env.py      — BlackOutEnv: PettingZoo ParallelEnv
│   ├── obs_preprocessor.py  — ObsPreprocessor: raw obs 변환
│   ├── seed_channel.py      — SeedChannel: Python → Unity 시드 전달
│   ├── semantic_id.py       — SemanticId: graphic 채널 인덱스 상수
│   └── constants.py         — 에이전트 이름/팀 유틸리티
├── competition/
│   ├── match.py             — run_match(), run_series(), MatchResult, SeriesResult
│   └── __init__.py
└── model/
    ├── base.py              — BaseModel ABC
    ├── loader.py            — CheckpointModel, load_checkpoint()
    └── __init__.py
```

---

## BlackOutEnv 내부 흐름

### 초기화

```
__init__()
├── semantic_map_config.json 로드 → n_items, n_classes 파싱
├── ObsPreprocessor 생성
│     vector_obs_size, n_graphic_channels 계산
├── observation_space / action_space 정의
├── SeedChannel, EngineConfigurationChannel 생성
└── UnityEnvironment 연결 (gRPC)
```

### step당 흐름

```
step(actions)
├── _send_actions(actions)
│     MapObsAgent에 빈 액션 전송 (Continuous Actions = 0)
│     BlackOutUnit × 10에 clipped float32[2] 전송
│
├── unity_env.step()  ← gRPC 1 라운드트립
│
└── _collect_obs()
    ├── _collect_map_obs()
    │     MapObsAgent.DecisionSteps 에서 visual_obs 수신 (C, H, W)
    │     transpose → (H, W, C)
    │     preprocess_graphic() → TeamA binary masks
    │     flip_team_perspective() → TeamB binary masks
    │     _team_graphics[0] = TeamA, _team_graphics[1] = TeamB  (캐시)
    │
    └── BlackOutUnit × 10 (DecisionSteps + TerminalSteps)
          _extract_step() → (agent_name, raw_vector[45])
            raw_vector[44] = unitIndex → agent_name 결정
          _preprocess(agent_name, raw_vector)
            preprocess_vector(raw_vector) → float32[N]
            _team_graphics[team_id]       → float32[H, W, C]
          _extract_scalars()
            raw_vector[41~43] → score_0, score_1, time_left
```

### 내부 상태

| 속성 | 타입 | 설명 |
|---|---|---|
| `_preprocessor` | `ObsPreprocessor` | obs 변환기 |
| `_team_graphics` | `dict[int, ndarray]` | 팀별 graphic 캐시. `_collect_map_obs()` 에서 매 스텝 갱신 |
| `_agent_name_cache` | `dict[tuple, str]` | `(behavior, agent_id) → agent_name`. `_send_actions` 시 재사용 |
| `_latest_rewards` | `dict[str, float]` | `_collect_obs()` 완료 시 갱신 |
| `_latest_terminations` | `dict[str, bool]` | 위와 동일 |
| `_latest_winner` | `int \| None` | 첫 TerminalStep 보상에서 추론. `0`=팀A, `1`=팀B, `-1`=무승부 |
| `_latest_scalars` | `dict[str, float]` | `score_0`, `score_1`, `time_left` |
| `_warned_*` | `bool` | 반복 경고 억제 플래그 (one-shot warning) |

---

## Graphic Obs 최적화 구조

BlackOutUnit 10개가 각각 visual obs를 전송하면 매 스텝 gRPC 패킷이 10배 커집니다.
전용 에이전트 `MapObsAgent` (behavior `"BlackOutMap"`)가 팀 A semantic map 1개만 전송하고,
Python이 ally/enemy 채널을 스왑해 팀 B 맵을 생성합니다.

```
Unity                          Python
MapObsAgent → visual_obs[1개] → _collect_map_obs()
                                  preprocess_graphic()      → _team_graphics[0] (TeamA)
                                  flip_team_perspective()   → _team_graphics[1] (TeamB)

BlackOutUnit × 10 → vector_obs만 전송
                                  _preprocess(agent)
                                    graphic = _team_graphics[team_id]
```

**gRPC 절감 효과 (96×96 텍스처 기준):**

| 구조 | 스텝당 graphic 전송 |
|---|---|
| 구조 (에이전트별 전송) | 10개 × 3채널 |
| 현재 (MapObsAgent) | 1개 × 1채널 |
| 절감 | **30배** |

---

## Obs 수정 가이드

### Vector obs에 float 추가하는 경우

1. **Unity** — `BlackOutAgent.CollectObservations()` 에 `sensor.AddObservation(value)` 추가
2. **Unity** — `BehaviorParameters.Vector Observation Size` +1
3. **`obs_preprocessor.py`** — `RAW_VECTOR_SIZE` 및 이후 인덱스 상수 업데이트

   ```python
   # 예: 슬롯 44 앞에 새 float 삽입하는 경우
   RAW_CLASS_SLOT        = N_UNITS * UNIT_BLOCK_SIZE    # 40
   RAW_SCALAR_START      = RAW_CLASS_SLOT + 1           # 41
   RAW_NEW_VALUE_SLOT    = RAW_SCALAR_START + 3         # 44  ← 추가
   RAW_UNIT_INDEX_SLOT   = RAW_NEW_VALUE_SLOT + 1       # 45  ← 밀림
   RAW_VECTOR_SIZE       = RAW_UNIT_INDEX_SLOT + 1      # 46
   ```

4. **`preprocess_vector()`** — `parts`에 새 값 추가 (또는 pass-through)
5. **`vector_obs_size`** — 재계산 확인

> **주의:** `unitIndex`는 반드시 마지막 슬롯 유지 (`RAW_UNIT_INDEX_SLOT`).
> `_extract_step()`의 agent_id 라우팅 로직이 이 위치에 의존합니다.

### Graphic에 새 semantic 채널 추가 (아이템 추가)

→ Unity 저장소의 `adding_item_type.md` 참고

1. `semantic_map_config.json` 의 `n_items` +1
2. `SemanticMapRenderer` 에서 새 ID 픽셀 기록 로직 추가 (Unity)
3. ally/enemy 관계 있는 채널이면 `flip_team_perspective()` 에 스왑 로직 추가
4. `ObsPreprocessor._channel_ids`는 `n_graphic_channels`에서 자동 갱신 → 별도 수정 불필요

### 체크포인트 호환성

`vector_obs_size` 또는 `n_graphic_channels`가 바뀌면 **기존 체크포인트 로드 불가**. 재훈련 필요.

---

## 테스트

**파일:** `test_env.py` (프로젝트 루트)

Unity 프로세스 없이 실행 가능한 단위 테스트입니다.

```bash
python test_env.py
```

테스트 항목:

| 테스트 함수 | 내용 |
|---|---|
| `test_preprocessor` | `ObsPreprocessor` 초기화, vector/graphic 크기, 값 범위 |
| `test_flip_team_perspective` | ally/enemy 채널 스왑 정확성 |
| `test_semantic_id` | `SemanticId` 상수, `item_channel`, `is_item` |
| `test_constants` | `agent_name`, `unit_index`, `team_of`, `team_a/b_agents` |
| `test_spaces` | `observation_space`, `action_space` 크기 |

새 기능을 추가할 때 `test_env.py`에 검증 케이스를 함께 추가하십시오.

---

## 의존성 버전 제약

```toml
# pyproject.toml
requires-python = ">=3.10"
dependencies = [
    "pettingzoo>=1.24.0",
    "gymnasium>=0.29.0",
    "numpy>=1.23.5",
    "mlagents-envs>=1.1.0",
]
```

**주의사항:**

| 항목 | 내용 |
|---|---|
| Python 버전 | 3.10.x 필수. `mlagents-envs 1.1.0`이 3.11+ 미지원 |
| `mlagents-envs` | `pettingzoo==1.15.0` 의존성 선언하지만 실제 임포트 없음 → `--no-deps` 선행 설치 필요 |
| Unity 버전 대응 | `com.unity.ml-agents 4.0.2` ↔ Python `mlagents-envs 1.1.0` |

**설치 순서:**

```bash
pip install "mlagents-envs==1.1.0" --no-deps
pip install blackout-env
```

**editable 설치 (개발):**

```bash
pip install "mlagents-envs==1.1.0" --no-deps
pip install -e .
```
