"""
BlackOutEnv integration test.
Connects to a standalone build by default. To use the Unity Editor instead,
set ENV_PATH = None and start Play mode manually before running.

    python test_env.py
    python test_env.py --diagnose   # also run graphic diagnostic (20 steps)
"""

import argparse
import json
import time
import numpy as np
from tqdm import tqdm
from blackout_env import BlackOutEnv
from blackout_env.env.constants import N_AGENTS, all_agents, team_a_agents, team_b_agents

ENV_PATH        = None  # None = Unity Editor
SEMANTIC_CONFIG = "d:/UnityProjects/RLGame2026/Assets/StreamingAssets/semantic_map_config.json"

with open(SEMANTIC_CONFIG) as _f:
    cfg = json.load(_f)
n_items   = cfg["n_items"]
n_classes = cfg["n_classes"]
EXPECTED_VECTOR_SIZE      = 10 * (2 + 1 + (n_items + 1)) + n_classes + 3
EXPECTED_GRAPHIC_CHANNELS = cfg["item_id_offset"] + n_items

class _Results:
    passed: int = 0
    failed: int = 0

_r = _Results()


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [OK  ] {msg}")
        _r.passed += 1
    else:
        print(f"  [FAIL] {msg}")
        _r.failed += 1


def run_episode(env: BlackOutEnv, seed: int | None = None) -> tuple[dict, int, dict, dict, dict, float, float, list]:
    """Reset, run to completion with random actions.
    Returns (first_obs, step_count, last_rewards, last_infos, mid_infos_sample,
             max_score_0, max_score_1, shaping_events).
    shaping_events: list of (step, agent, reward) for non-zero mid-episode rewards.
    """
    obs, _ = env.reset(seed=seed)
    first_obs = {k: dict(v) for k, v in obs.items()}
    step = 0
    last_rewards: dict = {}
    last_infos: dict = {}
    mid_infos_sample: dict = {}
    max_score_0 = 0.0
    max_score_1 = 0.0
    shaping_events: list = []
    t0 = time.perf_counter()
    with tqdm(desc="episode", unit="step") as pbar:
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            last_rewards = rewards
            last_infos = infos
            info = next(iter(infos.values()))
            if step == 0:
                mid_infos_sample = dict(info)
            s0 = info.get("score_0", 0.0)
            s1 = info.get("score_1", 0.0)
            if s0 > max_score_0: max_score_0 = s0
            if s1 > max_score_1: max_score_1 = s1
            for agent, r in rewards.items():
                if not terminations.get(agent, False) and abs(r) > 1e-6:
                    shaping_events.append((step, agent, round(r, 5)))
            step += 1
            elapsed = time.perf_counter() - t0
            pbar.set_postfix_str(
                f"{step / elapsed:.1f} steps/sec  s0={s0:.0f} s1={s1:.0f}"
            )
            pbar.update(1)
    return first_obs, step, last_rewards, last_infos, mid_infos_sample, max_score_0, max_score_1, shaping_events


def test_obs_shapes(obs: dict) -> None:
    print("\n--- obs shapes & dtypes ---")
    check(set(obs.keys()) == set(all_agents()), f"all 10 agents present")
    for agent in all_agents():
        v = obs[agent]["vector"]
        g = obs[agent]["graphic"]
        check(v.shape == (EXPECTED_VECTOR_SIZE,),
              f"{agent} vector {v.shape} == ({EXPECTED_VECTOR_SIZE},)")
        check(v.dtype == np.float32, f"{agent} vector dtype float32")
        check(g.ndim == 3 and g.shape[2] == EXPECTED_GRAPHIC_CHANNELS,
              f"{agent} graphic {g.shape}, ch={EXPECTED_GRAPHIC_CHANNELS}")
        check(g.dtype == np.float32, f"{agent} graphic dtype float32")
        check(g.min() >= 0.0 and g.max() <= 1.0, f"{agent} graphic in [0,1]")


def test_team_split(obs: dict) -> None:
    print("\n--- team split ---")
    a = set(team_a_agents())
    b = set(team_b_agents())
    check(a | b == set(obs.keys()), "team_a + team_b == all agents")
    check(len(a & b) == 0, "team_a ∩ team_b == empty")


def test_episode_completion(step_count: int) -> None:
    print("\n--- episode completion ---")
    check(step_count > 0, f"episode ran {step_count} steps")


def test_rewards_and_infos(last_rewards: dict, last_infos: dict, mid_infos: dict,
                           max_score_0: float, max_score_1: float) -> None:
    print("\n--- rewards & infos ---")

    print(f"  mid-step infos (step 1): {mid_infos}")
    check("score_0"   in mid_infos, "mid-step has score_0")
    check("score_1"   in mid_infos, "mid-step has score_1")
    check("time_left" in mid_infos, "mid-step has time_left")
    check("winner" not in mid_infos, "mid-step has no winner yet")

    print(f"  terminal rewards: { {k: round(v,3) for k,v in last_rewards.items()} }")
    reward_values = set(round(r, 3) for r in last_rewards.values())
    check(reward_values <= {1.0, -1.0, 0.0}, f"terminal rewards ∈ {{-1,0,1}}: {reward_values}")
    check(len(last_rewards) == N_AGENTS, f"all {N_AGENTS} agents have terminal reward")

    sample_info = next(iter(last_infos.values()))
    print(f"  terminal infos: {sample_info}")
    check("winner"    in sample_info, "terminal info has winner")
    check("score_0"   in sample_info, "terminal info has score_0")
    check("score_1"   in sample_info, "terminal info has score_1")
    check("time_left" in sample_info, "terminal info has time_left")
    winner = sample_info.get("winner")
    check(winner in (0, 1, -1), f"winner is 0/1/-1: got {winner}")

    print(f"  max score during episode — team0: {max_score_0:.0f}, team1: {max_score_1:.0f}")
    check(max_score_0 > 0 or max_score_1 > 0, "at least one team scored during episode")


def test_reward_shaping(shaping_events: list) -> None:
    print("\n--- reward shaping (mid-episode) ---")
    check(len(shaping_events) > 0, "non-zero mid-episode rewards received (shaping events fired)")

    if not shaping_events:
        print("  [WARN] no shaping events — rewards may not be wired correctly")
        return

    print(f"  Sample events (step, agent, reward) — showing up to 10 of {len(shaping_events)}:")
    for s, a, r in shaping_events[:10]:
        print(f"    step {s:4d}  {a}  {r:+.5f}")

    rewards_only = [r for _, _, r in shaping_events]
    max_abs = max(abs(r) for r in rewards_only)
    min_abs = min(abs(r) for r in rewards_only)
    check(max_abs <= 1.0, f"shaping rewards ≤ 1.0  (max_abs={max_abs:.5f})")
    check(min_abs >= 0.01, f"shaping rewards ≥ 0.01 (min_abs={min_abs:.5f})")

    has_pos = any(r > 0 for r in rewards_only)
    has_neg = any(r < 0 for r in rewards_only)
    check(has_pos, "at least one positive shaping reward (pickup / kill / deposit / absorb)")
    check(has_neg, "at least one negative shaping reward (death penalty / theft penalty)")

    unique_vals = sorted(set(rewards_only))
    print(f"  Unique shaping values ({len(unique_vals)}): {unique_vals}")
    check(not set(unique_vals) <= {1.0, -1.0}, "shaping values are not just ±1 (terminal leak guard)")

    agents_rewarded = set(a for _, a, _ in shaping_events)
    print(f"  Agents that received shaping rewards: {sorted(agents_rewarded)}")
    print(f"  Total shaping events: {len(shaping_events)}")


def diagnose_graphic(env: BlackOutEnv, seed: int | None = None, n_steps: int = 20) -> None:
    """Run n_steps and print graphic channel maxima each step."""
    print(f"\n--- graphic diagnostic ({n_steps} steps) ---")
    obs, _ = env.reset(seed=seed)
    ch_names = ["empty", "wall", "ally_st", "enemy_st", "ally_u", "enemy_u", "item0"]
    def fmt(g):
        per_ch = " ".join(f"{ch_names[c] if c < len(ch_names) else c}={g[:,:,c].max():.2f}"
                          for c in range(g.shape[2]))
        return f"shape={g.shape}  ch_max[{per_ch}]"

    g = obs.get("unit_0", {}).get("graphic")
    print(f"  reset obs: graphic={'None' if g is None else fmt(g)}")
    for i in range(n_steps):
        if not env.agents:
            print(f"  step {i}: episode ended early")
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, terminations, _, _ = env.step(actions)
        g = obs.get("unit_0", {}).get("graphic")
        terminated = any(terminations.values())
        print(f"  step {i+1:2d}: {fmt(g)}  terminated={terminated}"
              if g is not None else f"  step {i+1:2d}: unit_0 not in obs")
        if terminated:
            break
    # No drain needed — next env.reset() works regardless of episode state.


def _sample_graphic_after_steps(env: BlackOutEnv, seed: int | None, n_steps: int = 5) -> np.ndarray:
    """Reset with seed, advance n_steps to let MapObsAgent populate graphics, return unit_0 graphic."""
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, _, _, _ = env.step(actions)
    # No drain needed — env.reset() in the next call handles mid-episode reset safely.
    return obs.get("unit_0", {}).get("graphic", np.zeros(1, dtype=np.float32))


def test_seed(env: BlackOutEnv) -> None:
    print("\n--- seed reproducibility ---")
    g1 = _sample_graphic_after_steps(env, seed=42)
    g2 = _sample_graphic_after_steps(env, seed=42)
    g3 = _sample_graphic_after_steps(env, seed=99)

    # Compare only static map channels (WALL=1, ALLY_STORAGE=2, ENEMY_STORAGE=3).
    # Dynamic channels (ALLY_UNIT=4, ENEMY_UNIT=5, items=6+) vary between runs because
    # test actions are sampled from unseeded numpy — only map layout is seeded via Unity.
    STATIC = [1, 2, 3]

    check(g1.max() > 0.0, "graphic is non-zero (MapObsAgent data received)")
    check(np.allclose(g1[:, :, STATIC], g2[:, :, STATIC]),
          "seed=42 twice → identical warehouse layout (wall/storage channels)")
    if not np.allclose(g1[:, :, STATIC], g3[:, :, STATIC]):
        check(True, "seed=99 → different warehouse layout")
    else:
        print("  [WARN] seed=99 produced same warehouse layout as seed=42")
        print("         Likely cause: map layout is fixed (not randomised per episode),")
        print("         or Random.InitState() is called after map generation in Unity.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true",
                        help="Run graphic diagnostic before main tests (20 steps)")
    args = parser.parse_args()

    print(f"Config: n_items={n_items}, n_classes={n_classes}")
    print(f"Expected vector_size={EXPECTED_VECTOR_SIZE}, graphic_channels={EXPECTED_GRAPHIC_CHANNELS}")
    print("\nConnecting to Unity Editor...")

    env = BlackOutEnv(
        env_path=ENV_PATH,
        semantic_config_path=SEMANTIC_CONFIG,
        base_port=5004,
        no_graphics=False,
        time_scale=3,
    )

    try:
        if args.diagnose:
            print("\n=== Graphic diagnostic ===")
            diagnose_graphic(env, seed=42, n_steps=20)

        print("\n=== Episode 1 (seed=42) ===")
        obs, steps, last_rewards, last_infos, mid_infos, max_s0, max_s1, shaping = run_episode(env, seed=42)
        test_obs_shapes(obs)
        test_team_split(obs)
        test_episode_completion(steps)
        test_rewards_and_infos(last_rewards, last_infos, mid_infos, max_s0, max_s1)
        test_reward_shaping(shaping)

        print("\n=== Seed reproducibility (episodes 2 & 3) ===")
        test_seed(env)

    finally:
        env.close()

    print(f"\n{'='*40}")
    print(f"Results: {_r.passed} passed, {_r.failed} failed")
    if _r.failed:
        print("SOME TESTS FAILED")
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
