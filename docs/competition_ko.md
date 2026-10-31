# BlackOut — 대회 환경

공식 경기에서는 각 팀이 낸 ONNX 모델을 팀별 Docker 컨테이너에서 돌립니다. 게임은 서버에서 하나만 실행되고, 매 스텝 각 팀 컨테이너에 그 팀 유닛 5명의 관측을 보내 행동을 받아옵니다. 참가자 코드는 실행하지 않고 ONNX 모델만 실행합니다.

## 제출

폴더 하나에 `meta.json`과 모델을 넣어 제출합니다.

```
my_team/
├── meta.json      {"mode": "team"} 또는 {"mode": "unit"}
└── model.onnx
```

`mode`는 모델이 유닛을 어떻게 제어하는지 정합니다.

**`team`** 은 모델 하나가 5명을 한 번에 제어합니다. 매 스텝 한 번 호출되고, 입력은 학습 때 `run_match`가 넘기던 것과 같은 `vector (5, 96)`, `graphic (5, 11, 96, 96)`이며 출력은 `(5, 2)`입니다. 팀원을 한 배치로 받으니 유닛끼리 정보를 섞어 협동하는 모델을 만들 수 있습니다.

**`unit`** 은 유닛마다 따로 호출합니다. 매 스텝 5번, 배치 크기 1로 `vector (1, 96)`, `graphic (1, 11, 96, 96)`을 받아 `(1, 2)`를 냅니다. `model.onnx` 하나를 내면 5명이 같이 쓰고, `unit_0.onnx` ~ `unit_4.onnx`를 내면 i번째 유닛이 `unit_i.onnx`를 씁니다. 클래스별로 다른 모델을 쓰고 싶을 때 쓰면 됩니다.

모델 파일은 2GB 이하의 단일 `.onnx` 파일이어야 하고, 표준 연산만 쓸 수 있습니다.

## 입력과 출력

입력은 이름으로 넣어줍니다. `vector`, `graphic`, `slot` 중 모델이 쓰는 것만 있으면 되고, 다른 이름이 있으면 로드할 때 거부합니다. PyTorch에서 export하면 안 쓰는 입력은 알아서 빠지니 신경 쓰지 않아도 됩니다.

| 이름 | 자료형 | 모양 |
|---|---|---|
| `vector` | float32 | `(B, 96)` |
| `graphic` | float32 | `(B, 11, 96, 96)` |
| `slot` | int64 | `(B,)` |

출력은 첫 번째 출력 하나만 보며, float32 `(B, 2)`로 유닛별 `(dx, dy)`입니다. 값은 `[-1, 1]`로 자르고 NaN은 0으로 바꿉니다. 호출 사이에 상태는 남지 않습니다.

### 행 순서와 slot

행은 항상 유닛 번호순입니다. A팀은 unit_0 ~ unit_4, B팀은 unit_5 ~ unit_9가 0~4행에 들어갑니다. `slot`은 이 팀 안에서의 순번(0~4)입니다.

`vector`에는 10명 전원의 정보가 unit_0부터 들어 있어서, 자기 블록을 찾으려면 `slot`과 unit_0의 `team_sign`을 같이 봐야 합니다. `unit` 방식은 배치에 자기 하나만 들어오므로 `slot` 없이는 자기가 누군지 알 수 없습니다.

```python
is_team_a = vector[:, 2] > 0                    # unit_0이 아군이면 A팀
self_idx = slot + torch.where(is_team_a, 0, 5)  # 10명 중 자기 번호
```

### vector (96)

앞의 90칸은 유닛 10명 × 9칸입니다. 각 유닛은 위치 x, y(0~1), team_sign(아군 +1, 적 −1), 들고 있는 아이템 one-hot 6칸(0번이 없음) 순입니다. 그 뒤로 자기 유닛 클래스 one-hot 3칸(90~92), 우리 점수·상대 점수·남은 시간 3칸(93~95)이 옵니다.

### graphic (11 × 96 × 96)

채널마다 0/1 마스크입니다. 순서는 빈칸, 벽, 우리 창고, 상대 창고, 아군, 적군, 그리고 아이템 5종(배터리, 속도 버프, 속도 디버프, 크기 버프, 크기 디버프)입니다.

env는 `(H, W, C)`로 주지만 모델에는 `(C, H, W)`로 들어갑니다. `load_checkpoint`/`run_match`로 학습했다면 이미 이 순서입니다.

## ONNX로 내보내기

`onnx` 패키지는 mlagents-envs와 protobuf 버전이 충돌합니다. export와 검사는 학습용과 별도의 가상환경에서 하세요.

```bash
pip install torch onnx onnxruntime numpy
```

```python
# team
torch.onnx.export(
    model.eval(),
    (torch.zeros(5, 96), torch.zeros(5, 11, 96, 96)),
    "my_team/model.onnx",
    input_names=["vector", "graphic"], output_names=["action"],
    opset_version=17, dynamo=False,
)

# unit, slot 사용 (forward(vector, graphic, slot))
torch.onnx.export(
    model.eval(),
    (torch.zeros(1, 96), torch.zeros(1, 11, 96, 96), torch.zeros(1, dtype=torch.long)),
    "my_team/model.onnx",
    input_names=["vector", "graphic", "slot"], output_names=["action"],
    opset_version=17, dynamo=False,
)
```

export는 예시 입력으로 모델을 한 번 돌려 그래프를 기록합니다. 그래서 입력값에 따라 갈리는 Python `if`나 `for`는 그때 탄 경로로 굳어버립니다. 이런 부분은 `torch.where` 같은 텐서 연산으로 바꿔두세요.

## 제출 전 검사

```bash
python blackout_env/competition/check_submission.py my_team/
```

공식 경기와 같은 코드로 모델을 불러서 폴더 구성, 입력 이름·모양, 출력 모양을 확인하고 스텝당 실행 시간을 잽니다. `OK`가 나오면 형식은 문제없습니다. 시간은 내 컴퓨터 기준이니 여유를 두세요.

## 제한

팀마다 RTX 3090의 SM 50%와 GPU 메모리 11GB, CPU 4코어, RAM 16GB를 씁니다. GPU 제한은 서버의 CUDA MPS가 걸기 때문에 컨테이너 안에서 바꿀 수 없습니다. 네트워크는 없습니다. 실행 환경은 onnxruntime-gpu 1.22, CUDA 12.4입니다.

한 스텝에 팀당 100ms가 주어집니다(`unit` 방식은 5번 호출을 합친 시간, 전송 포함). 모델 로딩과 워밍업은 경기 전에 끝내고, 첫 스텝만 30초까지 기다립니다.

시간을 넘기거나 실행 중 오류가 나면 그 스텝은 팀 전체가 제자리에 섭니다. 모델을 아예 불러오지 못하면 경기를 시작하지 않습니다. 시간 초과와 오류 횟수는 결과 파일에 남습니다.

## 경기 규칙

**승패.** 경기 승자는 최종 게임 점수로 정합니다. 목표 점수에 먼저 도달하거나, 시간이 끝났을 때 점수가 높은 팀이 이깁니다. env가 주는 reward는 학습용이라 승패와 상관없습니다.

**시드.** 맵을 만드는 시드는 경기를 시작할 때 무작위로 뽑습니다. 미리 알려주지 않고, 경기가 끝난 뒤 결과와 함께 공개합니다. 맵 유불리를 없애기 위해 같은 시드로 진영을 바꿔 두 판씩 합니다.

**동점.** 두 판을 마치고 승수가 같으면 새 시드를 뽑아 다시 두 판을 합니다. 승부가 날 때까지 반복합니다.

**모델 로딩 실패.** 제출한 모델을 불러오지 못하면 경기를 진행하지 않고 다시 제출할 기회를 줍니다. 본 경기 전에 모의 경기를 진행하니, 그때 공식 환경에서 모델이 제대로 도는지 확인하세요. 아래 [직접 돌려보기](#직접-돌려보기)로 미리 테스트해볼 수도 있습니다.

**녹화.** 모든 경기는 녹화해서 공개합니다.

## 직접 돌려보기

Linux, NVIDIA GPU, Docker가 있으면 공식 경기와 똑같이 돌려볼 수 있습니다.

```bash
docker build -f Dockerfile.policy -t blackout-policy:latest .
xvfb-run -a python -m blackout_env.competition.run_competition \
    --build ~/blackout_build/blackout_linux_build.x86_64 \
    --team-a my_team --team-b other_team \
    --seeds 11 22 --out results/my_team_vs_other_team.json
```

시드 하나당 진영을 바꿔 두 판을 하고, 결과 JSON에 판별 점수, 승자, 팀별 시간 초과·오류 횟수가 기록됩니다. GPU 없이 돌리려면 `--no-gpu`를 붙이세요.

### 녹화

`--record videos/`를 붙이면 판마다 리플레이(`.npz`)와 영상(`.mp4`)이 저장됩니다. 경기는 원래 속도로 돌리고 행동만 기록한 뒤, 경기가 끝나면 기록한 행동을 1배속으로 다시 재생하면서 화면을 녹화합니다. 경기 중에 바로 녹화하면 모델이 느린 팀 때문에 영상이 끊기기 때문입니다. 한 판에 7분 정도 걸리고, Xvfb 화면은 1280×720 이상이어야 합니다.

```bash
xvfb-run -a -s "-screen 0 1280x720x24" python -m blackout_env.competition.run_competition \
    ... --record videos/
```

리플레이만 있으면 나중에 영상을 다시 뽑을 수 있습니다.

```bash
xvfb-run -a -s "-screen 0 1280x720x24" python -m blackout_env.competition.replay \
    --build ~/blackout_build/blackout_linux_build.x86_64 --out videos/ videos/*.npz
```

ffmpeg가 없으면 `pip install imageio-ffmpeg`로 대신할 수 있습니다.
