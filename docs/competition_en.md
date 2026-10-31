# BlackOut — Competition Environment

Official matches run each team's ONNX model in its own Docker container. The game runs once on the server; every step, each team's container gets the observations of its 5 units and sends back their actions. No participant code is executed, only the ONNX model.

## Submission

Submit one directory with `meta.json` and your model.

```
my_team/
├── meta.json      {"mode": "team"} or {"mode": "unit"}, optionally "frame_stack": K
└── model.onnx
```

`mode` sets how the model controls your units.

**`team`**: one model controls all 5 units at once. It is called once per step with the same inputs `run_match` passes during training, `vector (5, 96)` and `graphic (5, 11, 96, 96)`, and returns `(5, 2)`. Since teammates come in as one batch, the model can mix information across units to coordinate.

**`unit`**: the model is called separately for each unit, 5 times per step, with batch size 1: `vector (1, 96)`, `graphic (1, 11, 96, 96)` in, `(1, 2)` out. Submit one `model.onnx` and all 5 units share it, or submit `unit_0.onnx` ~ `unit_4.onnx` and unit i uses `unit_i.onnx`. Useful if you want a different model per class.

Model files must be a single `.onnx` file under 2 GB, using standard operators only.

## Inputs and outputs

Inputs are fed by name. A model needs only the ones it uses out of `vector`, `graphic` and `slot`; any other name is rejected at load time. PyTorch export drops unused inputs on its own, so you don't need to worry about that.

| Name | Type | Shape |
|---|---|---|
| `vector` | float32 | `(B, 96)` |
| `graphic` | float32 | `(B, 11, 96, 96)` |
| `slot` | int64 | `(B,)` |

The action is the first output whose name does not start with `state_out`: float32 `(B, 2)`, `(dx, dy)` per unit. Values are clipped to `[-1, 1]` and NaN becomes 0.

### Row order and slot

Rows are always in unit-number order: unit_0 ~ unit_4 for team A and unit_5 ~ unit_9 for team B fill rows 0~4. `slot` is the position within the team (0~4).

`vector` holds all 10 units starting from unit_0, so finding your own block takes `slot` plus unit_0's `team_sign`. In `unit` mode the batch holds only the unit itself, so without `slot` the model cannot tell which unit it is.

```python
is_team_a = vector[:, 2] > 0                    # unit_0 is an ally -> team A
self_idx = slot + torch.where(is_team_a, 0, 5)  # own unit number among the 10
```

### vector (96)

The first 90 values are 10 units × 9: position x, y (0~1), team_sign (ally +1, enemy −1), and a 6-way one-hot of the held item (index 0 = none). Then the unit's class one-hot (90~92) and own score, opponent score, time left (93~95).

### graphic (11 × 96 × 96)

One 0/1 mask per channel: empty, wall, ally storage, enemy storage, ally unit, enemy unit, then the 5 items (battery, speed buff, speed debuff, size buff, size debuff).

The env returns `(H, W, C)`, but the model receives `(C, H, W)`. If you trained with `load_checkpoint`/`run_match`, your model already sees this order.

### Stacking past frames (frame_stack)

Put `"frame_stack": K` (1~16) in `meta.json` and the model gets the last K steps of observations. The server keeps a separate stack for each unit.

| Name | Shape (K > 1) |
|---|---|
| `vector` | `(B, K, 96)` |
| `graphic` | `(B, K, 11, 96, 96)` |

Oldest frame first; `[:, -1]` is the current one. On the first step of an episode there is nothing stacked yet, so the first observation is copied into all K frames. Steps that timed out still add their observation. `slot` is not stacked.

Stack the same way in training. `blackout_env.competition.onnx_policy.FrameStack` is the code the server uses, so you can import it directly.

```python
from blackout_env.competition.onnx_policy import FrameStack

stack = FrameStack(4)
stack.reset()                                   # at the start of every episode
v, g = stack.push(slots, vector, graphic_chw)   # (N, 4, 96), (N, 4, 11, 96, 96)
```

### Recurrent state (RNN)

For models that carry state between steps, like an LSTM or GRU, declare inputs starting with `state_in` paired with outputs that start with `state_out` and have the same suffix (`state_in` ↔ `state_out`, `state_in_c` ↔ `state_out_c`). Shape is float32 `(B, ...)` with fixed sizes after the batch dim.

The server stores each unit's `state_out` and feeds it back as `state_in` on that unit's next step. State starts at zero every episode. In `team` mode, a unit that is dead and missing from the batch keeps its state. A step that wasn't answered does not update the state.

You can use it together with `frame_stack`.

## Exporting to ONNX

The `onnx` package conflicts with mlagents-envs over protobuf, so export and check in a separate virtual environment from training.

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

# unit, with slot (forward(vector, graphic, slot))
torch.onnx.export(
    model.eval(),
    (torch.zeros(1, 96), torch.zeros(1, 11, 96, 96), torch.zeros(1, dtype=torch.long)),
    "my_team/model.onnx",
    input_names=["vector", "graphic", "slot"], output_names=["action"],
    opset_version=17, dynamo=False,
)
```

```python
# team, frame_stack 4 + GRU state (forward(vector, graphic, state_in) -> (action, state_out))
torch.onnx.export(
    model.eval(),
    (torch.zeros(5, 4, 96), torch.zeros(5, 4, 11, 96, 96), torch.zeros(5, 64)),
    "my_team/model.onnx",
    input_names=["vector", "graphic", "state_in"], output_names=["action", "state_out"],
    opset_version=17, dynamo=False,
)
```

Export runs the model once on the example inputs and records the graph, so a Python `if` or `for` that depends on input values gets frozen to whichever path it took. Rewrite those with tensor ops like `torch.where`.

## Checking before you submit

```bash
python blackout_env/competition/check_submission.py my_team/
```

It loads the model with the same code as official matches, checks the layout, input names and shapes, frame_stack, state input/output pairs, and output shape, and times each step. `OK` means the format is fine. Times are from your machine, so leave some margin.

## Limits

Each team gets 50% of the SMs and 11 GB of memory on an RTX 3090, 4 CPU cores and 16 GB RAM. The GPU limits are set by CUDA MPS on the server and can't be changed from inside the container. There is no network. The runtime is onnxruntime-gpu 1.22 on CUDA 12.4.

Each team has 100 ms per step (in `unit` mode, the 5 calls together, including transfer). Loading and warm-up happen before the match; only the first step may take up to 30 s.

If a team runs out of time or hits a runtime error, all its units stand still for that step. If the model can't be loaded at all, the match doesn't start. Timeouts and errors are counted in the result file.

## Match rules

**Winner.** Each match is decided by the final game score: whoever reaches the target score first, or has more points when time runs out. Rewards from the env are for training only and don't count.

**Seeds.** The map seed is drawn at random when the match starts. It isn't announced beforehand; it's published with the results afterwards. Each seed is played twice with sides swapped to cancel out map advantages.

**Ties.** If both teams have the same number of wins after the two matches, a new seed is drawn and two more matches are played, until one team wins.

**Model fails to load.** If your model can't be loaded, the match isn't played and you get a chance to resubmit. Mock matches are held before the main event, so use them to check that your model runs in the official environment. You can also test beforehand with [Running it yourself](#running-it-yourself) below.

**Recording.** Every match is recorded and published.

## Running it yourself

With Linux, an NVIDIA GPU and Docker you can run matches exactly like the official ones.

```bash
docker build -f Dockerfile.policy -t blackout-policy:latest .
xvfb-run -a python -m blackout_env.competition.run_competition \
    --build ~/blackout_build/blackout_linux_build.x86_64 \
    --team-a my_team --team-b other_team \
    --seeds 11 22 --out results/my_team_vs_other_team.json
```

Each seed is played twice with sides swapped. The result JSON has per-match scores and winners, and per-team timeout and error counts. Add `--no-gpu` to run without a GPU.

### Recording

`--record videos/` saves a replay (`.npz`) and a video (`.mp4`) per match. The match runs at normal speed and only the actions are logged; once it's over, the actions are played back at 1x while the screen is recorded. Recording during the match itself would stutter whenever a team's model is slow. Each match takes about 7 minutes to render, and the Xvfb screen must be at least 1280×720.

```bash
xvfb-run -a -s "-screen 0 1280x720x24" python -m blackout_env.competition.run_competition \
    ... --record videos/
```

With the replay files you can render the videos again later:

```bash
xvfb-run -a -s "-screen 0 1280x720x24" python -m blackout_env.competition.replay \
    --build ~/blackout_build/blackout_linux_build.x86_64 --out videos/ videos/*.npz
```

If ffmpeg isn't installed, `pip install imageio-ffmpeg` works instead.
