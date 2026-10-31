"""
Run a series between two submissions, each in its own container with half the GPU.

    xvfb-run -a python -m blackout_env.competition.run_competition \\
        --build /path/to/BlackOut.x86_64 \\
        --team-a submissions/team_a --team-b submissions/team_b \\
        --seeds 11 22 --out results/team_a_vs_team_b.json

Add `--record videos/` to also save a replay (.npz) and an mp4 per match. Matches still
run at --time-scale; the videos are rendered afterwards by replaying the saved actions
at time_scale 1 (about 7 minutes per match). The Xvfb screen must be at least
1280x720x24: `xvfb-run -a -s "-screen 0 1280x720x24" python -m ...`.

Each submission directory holds meta.json and ONNX model(s) (see onnx_policy.py).
The Unity build runs on the host (under Xvfb); only the participants' models run in
the containers.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ..env.blackout_env import BlackOutEnv
from .container import POLICY_IMAGE, MpsDaemon, TeamContainer
from .match import SeriesResult, run_match
from .recorder import ScreenRecorder
from .replay import ReplayLog, make_render_env, play
from .remote_model import RemoteModel


def main() -> None:
    p = argparse.ArgumentParser(description="BlackOut competition series runner")
    p.add_argument("--build", required=True, help="Unity Linux build (BlackOut.x86_64)")
    p.add_argument("--team-a", required=True, type=Path)
    p.add_argument("--team-b", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="one seed per pair of matches (each seed is played from both sides)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", default=POLICY_IMAGE)
    p.add_argument("--gpu-percent", type=int, default=50, help="SM share per team (MPS)")
    p.add_argument("--gpu-mem", default="11G", help="GPU memory cap per team (MPS)")
    p.add_argument("--no-gpu", action="store_true", help="run policies on CPU, no MPS")
    p.add_argument("--cpus", type=float, default=4.0)
    p.add_argument("--memory", default="16g")
    p.add_argument("--step-timeout", type=float, default=0.1)
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--record", type=Path, default=None, metavar="DIR",
                   help="save a replay and an mp4 per match to DIR (rendered after the series)")
    args = p.parse_args()

    mps = None if args.no_gpu else MpsDaemon(thread_percentage=args.gpu_percent, mem_limit=args.gpu_mem)
    tag = f"{args.team_a.name}-vs-{args.team_b.name}-{args.worker_id}".lower()
    teams = {
        "a": TeamContainer(f"blackout-{tag}-a", args.team_a, mps, args.image, args.cpus, args.memory),
        "b": TeamContainer(f"blackout-{tag}-b", args.team_b, mps, args.image, args.cpus, args.memory),
    }
    models: dict[str, RemoteModel] = {}
    recorder = ScreenRecorder() if args.record is not None else None  # fail early without DISPLAY/ffmpeg
    replays: list[Path] = []
    env = None
    started = time.time()
    try:
        if mps is not None:
            mps.start()
        for t in teams.values():
            t.start()
        for key, t in teams.items():
            t.wait_ready()
            models[key] = RemoteModel(str(t.socket_path), step_timeout=args.step_timeout)

        env = BlackOutEnv(args.build, no_graphics=False, time_scale=args.time_scale, worker_id=args.worker_id)

        # Same order as run_series: each seed twice, sides swapped on the second match.
        series = SeriesResult()
        for seed in args.seeds:
            for swap in (False, True):
                i = len(series.matches)
                log = ReplayLog() if args.record is not None else None
                match = run_match(env, models["a"], models["b"], swap_teams=swap, seed=seed,
                                  on_step=None if log is None else log.add)
                series.matches.append(match)
                print(f"match {i + 1}: seed={seed} swap={swap} winner={match.winner} "
                      f"score a={match.model_a_score:.2f} b={match.model_b_score:.2f}", flush=True)
                if log is not None:
                    path = args.record / f"{tag}-m{i + 1}-seed{seed}{'-swap' if swap else ''}.npz"
                    score_a, score_b = match.model_a_score, match.model_b_score
                    final = {"score_0": score_b if swap else score_a, "score_1": score_a if swap else score_b}
                    log.save(path, seed, {"team_a": str(args.team_a), "team_b": str(args.team_b),
                                          "swap": swap, "final": final})
                    replays.append(path)

        result = {
            "team_a": str(args.team_a),
            "team_b": str(args.team_b),
            "seeds": args.seeds,
            "gpu": None if mps is None else {"percent": args.gpu_percent, "mem": args.gpu_mem},
            "step_timeout": args.step_timeout,
            "time_scale": args.time_scale,
            "replays": [str(r) for r in replays],
            "wall_time_s": round(time.time() - started, 1),
            "wins": {"team_a": series.model_a_wins, "team_b": series.model_b_wins, "draws": series.draws},
            "series_winner": {0: "team_a", 1: "team_b", None: None}[series.series_winner],
            "matches": [asdict(m) for m in series.matches],
            "policy_stats": {k: asdict(m.stats) for k, m in models.items()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("wins", "series_winner", "policy_stats", "wall_time_s")}, indent=2))
    finally:
        if env is not None:
            env.close()
        for m in models.values():
            m.close()
        for t in teams.values():
            t.cleanup()
        if mps is not None:
            mps.stop()

    if replays:
        env = make_render_env(args.build, recorder, args.worker_id)
        try:
            for path in replays:
                r = play(env, path, recorder, path.with_suffix(".mp4"))
                print(f"{path.with_suffix('.mp4')}: logged {r['logged']}, replayed {r['replayed']}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
