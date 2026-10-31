"""
Match replays: the seed and every step's actions, enough to play a match again.

Matches run fast (time_scale 20) and policies may take up to the step limit, so recording
the screen during the match would stutter. Instead the match's actions are saved and the
video is made afterwards by replaying them at time_scale 1, with no policy in the loop.

    xvfb-run -a -s "-screen 0 1280x720x24" python -m blackout_env.competition.replay \\
        --build ~/blackout_build/blackout_linux_build.x86_64 --out videos/ replays/*.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..env.blackout_env import BlackOutEnv
from ..env.constants import N_AGENTS, agent_name, unit_index
from .recorder import ScreenRecorder


class ReplayLog:
    """Collects the actions sent each step (pass .add as run_match's on_step)."""

    def __init__(self):
        self.steps: list[dict[str, np.ndarray]] = []

    def add(self, actions: dict[str, np.ndarray]) -> None:
        self.steps.append({a: np.asarray(v, np.float32).copy() for a, v in actions.items()})

    def save(self, path: Path, seed: int | None, meta: dict) -> None:
        actions = np.zeros((len(self.steps), N_AGENTS, 2), np.float32)
        present = np.zeros((len(self.steps), N_AGENTS), bool)
        for t, step in enumerate(self.steps):
            for a, v in step.items():
                actions[t, unit_index(a)] = v
                present[t, unit_index(a)] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, actions=actions, present=present,
                            seed=-1 if seed is None else seed, meta=json.dumps(meta))


def play(env: BlackOutEnv, path: Path, recorder: ScreenRecorder | None = None,
         video: Path | None = None) -> dict:
    """Replay one saved match on env. Returns the final scores, both logged and replayed."""
    data = np.load(path)
    actions, present = data["actions"], data["present"]
    seed = int(data["seed"])
    meta = json.loads(str(data["meta"]))

    env.reset(seed=None if seed < 0 else seed)
    if recorder is not None:
        recorder.start(video)
    info: dict = {}
    t = 0
    try:
        while env.agents and t < len(actions):
            step = {agent_name(i): actions[t, i] for i in np.flatnonzero(present[t])}
            _, _, _, _, infos = env.step(step)
            if infos:
                info = next(iter(infos.values()))
            t += 1
    finally:
        if recorder is not None:
            recorder.stop()
    replayed = {"score_0": float(info.get("score_0", 0.0)), "score_1": float(info.get("score_1", 0.0)),
                "steps": t}
    return {"logged": meta.get("final"), "replayed": replayed}


def render_videos(build: str, replays: list[Path], out_dir: Path, worker_id: int = 0) -> None:
    """Replay each saved match at time_scale 1 and record it to out_dir/<replay name>.mp4."""
    recorder = ScreenRecorder()
    screen = ["-screen-width", str(recorder.width), "-screen-height", str(recorder.height),
              "-screen-fullscreen", "0"]
    env = BlackOutEnv(build, no_graphics=False, time_scale=1.0, worker_id=worker_id, additional_args=screen)
    try:
        for path in replays:
            video = out_dir / f"{path.stem}.mp4"
            r = play(env, path, recorder, video)
            print(f"{video}: logged {r['logged']}, replayed {r['replayed']}", flush=True)
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Render saved BlackOut matches to mp4")
    p.add_argument("replays", type=Path, nargs="+")
    p.add_argument("--build", required=True)
    p.add_argument("--out", type=Path, required=True, help="directory for the mp4 files")
    p.add_argument("--worker-id", type=int, default=0)
    args = p.parse_args()
    render_videos(args.build, args.replays, args.out, args.worker_id)


if __name__ == "__main__":
    main()
