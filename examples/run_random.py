"""
Run random-policy agents in the BlackOut environment.

Usage:
    python run_random.py                        # connect to Unity Editor (Play mode)
    python run_random.py --build path/to/build  # standalone build
    python run_random.py --episodes 3           # run multiple episodes
"""

import argparse

from blackout_env import BlackOutEnv
from random_policy import RandomPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default=None, help="Path to the Unity build executable")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    env = BlackOutEnv(
        env_path=args.build,
    )

    policy = RandomPolicy()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_rewards = {agent: 0.0 for agent in env.possible_agents}
        steps = 0

        while env.agents:
            actions = policy.act(obs)
            obs, rewards, _, _, _ = env.step(actions)
            for agent, reward in rewards.items():
                total_rewards[agent] += reward
            steps += 1

        print(f"Episode {ep + 1} | Steps: {steps}")
        for agent, reward in total_rewards.items():
            print(f"  {agent}: {reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
