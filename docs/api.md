# API 레퍼런스

## 목차

- [BlackOutEnv](#blackoutenv)
- [ObsPreprocessor](#obspreprocessor)
- [SemanticId](#semanticid)
- [에이전트 유틸리티](#에이전트-유틸리티)
- [Competition — run_match / run_series](#competition)
- [Competition — BaseModel / load_checkpoint](#basemodel--load_checkpoint)

---

## BlackOutEnv

```python
from blackout_env import BlackOutEnv
```

PettingZoo `ParallelEnv` 구현. `mlagents_envs.UnityEnvironment` 래퍼.

### 생성자

```python
env = BlackOutEnv(
    env_path="path/to/BlackOut.exe",   # None = Unity Editor에 연결
    semantic_config_path="semantic_map_config.json",
    map_w=96,            # 텍스처 가로 (Unity resolutionScale × mapWidth와 일치해야 함)
    map_h=96,            # 텍스처 세로
    worker_id=0,         # base_port에 더해지는 오프셋
    base_port=None,      # None이면 OS가 빈 포트 자동 선택 (병렬 훈련 권장)
    no_graphics=True,    # Unity 빌드에 --no-graphics 전달
    time_scale=1.0,      # Unity Time.timeScale (헤드리스 빌드: 20~100 권장)
)
```

**파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `env_path` | `str \| None` | — | Unity 빌드 경로. `None`이면 에디터에 연결 |
| `semantic_config_path` | `str \| Path` | — | `semantic_map_config.json` 경로 |
| `map_w`, `map_h` | `int` | `96` | visual obs 텍스처 크기 |
| `worker_id` | `int` | `0` | `base_port`가 지정된 경우에만 사용 |
| `base_port` | `int \| None` | `None` | `None`이면 자동 포트 (Ray 등 병렬 환경에 권장) |
| `no_graphics` | `bool` | `True` | Unity 빌드 전용. 에디터 연결 시 무시 |
| `time_scale` | `float` | `1.0` | 에디터 연결 시 `1.0` 유지 |

### 메서드

#### `reset(seed=None, options=None)`

```python
obs, infos = env.reset(seed=42)
```

- `seed`: `int | None`. 지정 시 Unity `Random.InitState(seed)` 호출 → 맵/아이템 배치 재현
- 반환: `(obs_dict, infos_dict)`

#### `step(actions)`

```python
obs, rewards, terminations, truncations, infos = env.step(actions)
```

- `actions`: `dict[agent_name, float32[2]]`
- `truncations`: 항상 `False` (모든 에피소드 종료는 termination)
- `infos`: 매 스텝 `score_0`, `score_1`, `time_left` 포함. 에피소드 종료 스텝에만 `winner` 추가

**infos 구조:**

| 키 | 타입 | 설명 |
|---|---|---|
| `score_0` | `float` | 팀 A 점수 / 목표 점수 |
| `score_1` | `float` | 팀 B 점수 / 목표 점수 |
| `time_left` | `float` | 남은 시간 비율 (0~1) |
| `winner` | `int` | 종료 스텝에만 존재. `0`=팀A, `1`=팀B, `-1`=무승부 |

#### `close()`

Unity 프로세스 종료. 훈련 루프 종료 후 반드시 호출.

#### `observation_space(agent)` / `action_space(agent)`

모든 에이전트가 동일한 공간을 공유합니다.

**Observation space** (`spaces.Dict`):

| 키 | 형태 | 범위 | 설명 |
|---|---|---|---|
| `"vector"` | `float32[N]` | `[-1, 1]` | 전처리된 vector obs |
| `"graphic"` | `float32[H, W, C]` | `[0, 1]` | binary channel masks |

`N = 10×(2+1+(n_items+1)) + n_classes + 3`  
`C = item_id_offset + n_items`

**Action space** (`spaces.Box`):

| 형태 | 범위 | 설명 |
|---|---|---|
| `float32[2]` | `[-1, 1]` | `[dx, dy]` 이동 벡터 |

### Observation 상세 — vector

`N = 10×(2+1+(n_items+1)) + n_classes + 3`

| 세그먼트 | 길이 | 설명 |
|---|---|---|
| pos × 10 유닛 | 20 | `pos_x`, `pos_y` 정규화 좌표 |
| team_sign × 10 | 10 | 아군 `+1.0`, 적군 `-1.0` |
| holding_item one-hot × 10 | `10 × (n_items+1)` | index 0 = 없음, index i = 아이템 i 소지 |
| class one-hot | `n_classes` | 이 에이전트의 유닛 클래스 |
| scalars | 3 | `own_score`, `opp_score`, `time_left` |

유닛 순서: unit_0~9 전체 (agent_id 순). `team_sign`으로 아군/적군 구분.

### Observation 상세 — graphic

`float32[H, W, C]` — 각 채널은 `0.0` / `1.0` binary mask.

| 채널 | 의미 | `SemanticId` 상수 |
|---|---|---|
| 0 | 빈 공간 | `SemanticId.EMPTY` |
| 1 | 벽 | `SemanticId.WALL` |
| 2 | 아군 창고 | `SemanticId.ALLY_STORAGE` |
| 3 | 적군 창고 | `SemanticId.ENEMY_STORAGE` |
| 4 | 아군 유닛 | `SemanticId.ALLY_UNIT` |
| 5 | 적군 유닛 | `SemanticId.ENEMY_UNIT` |
| 6+i | 아이템 타입 i | `SemanticId.item_channel(i)` |

`ally/enemy` 기준은 관찰 주체 팀의 시점. 팀별 flip은 `BlackOutEnv` 내부에서 자동 처리.

---

## ObsPreprocessor

```python
from blackout_env import ObsPreprocessor, load_semantic_config
```

Unity raw obs → 모델 입력 변환. 스테이트리스.

### 생성자

```python
cfg = load_semantic_config("semantic_map_config.json")
preprocessor = ObsPreprocessor(cfg, n_items=1, n_classes=3)
```

### 속성

| 속성 | 타입 | 설명 |
|---|---|---|
| `vector_obs_size` | `int` | 전처리 후 vector 크기 |
| `n_graphic_channels` | `int` | graphic 채널 수 = `item_id_offset + n_items` |
| `RAW_VECTOR_SIZE` | `int` | raw vector 크기 (45) |
| `RAW_CLASS_SLOT` | `int` | raw vector에서 classId 위치 (40) |
| `RAW_SCALAR_START` | `int` | own_score 시작 위치 (41) |
| `RAW_UNIT_INDEX_SLOT` | `int` | unitIndex 위치 (44) — 전처리 시 제거 |

### 메서드

#### `preprocess_vector(raw)`

```python
vector = preprocessor.preprocess_vector(raw_float32_45)
# → float32[vector_obs_size]
```

holdingItemId, classId를 one-hot으로 확장. unitIndex 제거.

#### `preprocess_graphic(raw)`

```python
graphic = preprocessor.preprocess_graphic(raw_float32_HxWx1)
# → float32[H, W, n_graphic_channels]
```

픽셀 ID를 binary channel masks로 변환. 입력은 ML-Agents가 정규화한 `[0, 1]` 값.

#### `flip_team_perspective(graphic)`

```python
team_b_graphic = preprocessor.flip_team_perspective(team_a_graphic)
```

ally/enemy 채널 스왑 (ch2↔ch3, ch4↔ch5). 팀 B 관점 graphic 생성.

---

## SemanticId

```python
from blackout_env import SemanticId
```

전처리 후 graphic obs의 채널 인덱스 상수.

### 클래스 상수

```python
SemanticId.EMPTY          # 0
SemanticId.WALL           # 1
SemanticId.ALLY_STORAGE   # 2
SemanticId.ENEMY_STORAGE  # 3
SemanticId.ALLY_UNIT      # 4
SemanticId.ENEMY_UNIT     # 5
SemanticId.ITEM_ID_OFFSET # 6
SemanticId.BASE_IDS       # (0, 1, 2, 3, 4, 5)
```

### 클래스 메서드

| 메서드 | 설명 |
|---|---|
| `item_channel(item_index)` | KnownItems 인덱스 → 채널 인덱스 (`6 + item_index`) |
| `item_index(channel)` | 채널 인덱스 → KnownItems 인덱스. 아이템 채널이 아니면 `ValueError` |
| `is_item(channel, n_items)` | 채널이 유효한 아이템 채널인지 확인 |
| `all_channels(n_items)` | 전체 채널 인덱스 리스트 반환 |
| `name(channel, n_items=0)` | 채널 인덱스 → 이름 문자열 (예: `"ally_unit"`, `"item_0"`) |

```python
# 사용 예시
ally_mask = graphic[:, :, SemanticId.ALLY_UNIT]
item0     = graphic[:, :, SemanticId.item_channel(0)]
SemanticId.name(4)          # "ally_unit"
SemanticId.is_item(7, n_items=2)  # True
```

---

## 에이전트 유틸리티

```python
from blackout_env import team_of, team_a_agents, team_b_agents, all_agents
# 또는
from blackout_env.env.constants import agent_name, unit_index, team_of, ...
```

| 함수 | 설명 | 예시 |
|---|---|---|
| `agent_name(unit_index)` | unitIndex → agent name | `agent_name(3)` → `"unit_3"` |
| `unit_index(agent)` | agent name → unitIndex | `unit_index("unit_3")` → `3` |
| `team_of(agent)` | agent name → 팀 번호 (0 or 1) | `team_of("unit_6")` → `1` |
| `team_a_agents()` | Team A agent name 리스트 | `["unit_0", ..., "unit_4"]` |
| `team_b_agents()` | Team B agent name 리스트 | `["unit_5", ..., "unit_9"]` |
| `all_agents()` | 전체 agent name 리스트 | `["unit_0", ..., "unit_9"]` |

```python
# 팀별 obs 분리
a_obs = {k: v for k, v in obs.items() if team_of(k) == 0}
b_obs = {k: v for k, v in obs.items() if team_of(k) == 1}
```

상수:

| 상수 | 값 | 설명 |
|---|---|---|
| `BEHAVIOR_NAME` | `"BlackOutUnit"` | Unity behavior 이름 |
| `MAP_BEHAVIOR_NAME` | `"BlackOutMap"` | MapObsAgent behavior 이름 |
| `N_AGENTS` | `10` | 전체 에이전트 수 |
| `N_TEAM_A` | `5` | 팀당 에이전트 수 |

---

## Competition

```python
from blackout_env import run_match, run_series
```

### `run_match(env, model_a, model_b, swap_teams=False)`

단일 에피소드 실행.

```python
result = run_match(env, model_a, model_b)
print(result.winner)              # 0=model_a, 1=model_b, None=무승부
print(result.team_a_total_reward) # float
print(result.episode_steps)       # int
```

- `swap_teams=True`: model_a가 팀 B로 플레이. 페어니스를 위해 `run_series`가 내부에서 자동 스왑.
- `winner`는 항상 model_a/model_b 기준으로 보고 (팀 스왑 여부 반영됨).

### `run_series(env, model_a, model_b, n_matches=10)`

N 경기 시리즈. 짝수 번째 경기마다 팀 스왑.

```python
series = run_series(env, model_a, model_b, n_matches=10)
print(series.model_a_wins)   # int
print(series.model_b_wins)   # int
print(series.draws)          # int
print(series.series_winner)  # 0 or 1 or None
```

---

## BaseModel / load_checkpoint

```python
from blackout_env import BaseModel, load_checkpoint
```

### `BaseModel` (ABC)

```python
class MyModel(BaseModel):
    def act(
        self,
        obs: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """
        obs   : {agent_name: {"vector": float32[N], "graphic": float32[H, W, C]}}
        return: {agent_name: float32[2]}  — (dx, dy) in [-1, 1]
        """
        ...
```

### `load_checkpoint(model_class, checkpoint_path, ...)`

PyTorch `nn.Module` 체크포인트를 로드해 `BaseModel`로 래핑.

```python
model = load_checkpoint(
    MyPolicy,                      # nn.Module 서브클래스
    "checkpoint.pt",
    state_dict_key="policy_state", # None이면 raw state dict
    device="cuda",
    # model_class 생성자 kwargs
    vector_size=56,
    n_channels=7,
)
```

**체크포인트 저장 형식:**

```python
# raw state dict
torch.save(model.state_dict(), "checkpoint.pt")
# → load_checkpoint(..., state_dict_key=None)

# dict 형식 (추가 메타데이터 포함 가능)
torch.save({"policy_state": model.state_dict(), "step": 1000}, "checkpoint.pt")
# → load_checkpoint(..., state_dict_key="policy_state")  ← 기본값
```

**`nn.Module` forward 인터페이스:**

```python
class MyPolicy(nn.Module):
    def forward(
        self,
        vector: torch.Tensor,   # (B, N)      float32
        graphic: torch.Tensor,  # (B, C, H, W) float32  ← CHW 순서
    ) -> torch.Tensor:          # (B, 2)      float32
        ...
```

`load_checkpoint`가 반환하는 `CheckpointModel`은 `graphic`을 `(B, H, W, C) → (B, C, H, W)`로 자동 변환합니다.
