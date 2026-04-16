"""
Public constants and agent-naming utilities for BlackOut environment.

Trainer code that needs to split observations or actions by team should import
from here rather than hard-coding magic numbers.
"""

BEHAVIOR_NAME     = "BlackOutUnit"
MAP_BEHAVIOR_NAME = "BlackOutMap"

N_AGENTS = 10
N_TEAM_A = 5   # unitIndex 0~4, TeamId 0
N_TEAM_B = 5   # unitIndex 5~9, TeamId 1

TEAM_A_INDICES = tuple(range(0, 5))
TEAM_B_INDICES = tuple(range(5, 10))


def agent_name(unit_index: int) -> str:
    """Return the agent identifier string for a given unitIndex (0~9)."""
    if not (0 <= unit_index < N_AGENTS):
        raise ValueError(f"unit_index must be 0~{N_AGENTS - 1}, got {unit_index}")
    return f"unit_{unit_index}"


def unit_index(agent: str) -> int:
    """Parse unitIndex from an agent identifier string (e.g. 'unit_3' → 3)."""
    try:
        return int(agent.split("_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid agent name: {agent!r}")


def team_of(agent: str) -> int:
    """Return 0 (Team A) or 1 (Team B) for a given agent identifier."""
    return 0 if unit_index(agent) < N_TEAM_A else 1


def team_a_agents() -> list[str]:
    """Return agent identifiers for Team A (unitIndex 0~4)."""
    return [agent_name(i) for i in TEAM_A_INDICES]


def team_b_agents() -> list[str]:
    """Return agent identifiers for Team B (unitIndex 5~9)."""
    return [agent_name(i) for i in TEAM_B_INDICES]


def all_agents() -> list[str]:
    """Return all agent identifiers in unitIndex order (0~9)."""
    return [agent_name(i) for i in range(N_AGENTS)]
