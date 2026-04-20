"""Visualise DP results on GridWorld.

Run from repo root:
    python -m phase1_tabular.run_dp

Once dp.py is implemented, this script runs both policy iteration and
value iteration, prints the converged value function as a grid, and
draws an arrow map of the optimal policy. Use it to sanity-check your
implementation and to experiment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import GridWorld, ACTION_NAMES
from phase1_tabular.dp import policy_iteration, value_iteration

ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def print_values(env: GridWorld, V: dict, title: str = "Value Function"):
    print(f"\n--- {title} ---")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                row.append("  ### ")
            elif env.is_terminal(s):
                row.append(f" [{V[s]:+.1f}]")
            else:
                row.append(f"  {V[s]:+.2f}")
        print("  ".join(row))


def print_policy(env: GridWorld, policy: dict, title: str = "Policy"):
    print(f"\n--- {title} ---")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                row.append("#")
            elif s == env.goal:
                row.append("G")
            elif s in env.hazards:
                row.append("H")
            else:
                row.append(ARROWS[policy[s]])
        print("  ".join(row))


def run(env: GridWorld, gamma: float):
    print(f"\n{'='*55}")
    print(f"Grid {env.rows}x{env.cols}  |  gamma={gamma}")
    print(f"{'='*55}")
    print(env.render())

    print("\n[Policy Iteration]")
    V_pi, policy_pi = policy_iteration(env, gamma=gamma)
    print_values(env, V_pi)
    print_policy(env, policy_pi)

    print("\n[Value Iteration]")
    V_vi, policy_vi = value_iteration(env, gamma=gamma)
    print_values(env, V_vi)
    print_policy(env, policy_vi)

    # Sanity check: both methods should agree on V*
    max_diff = max(abs(V_pi[s] - V_vi[s]) for s in env.states)
    print(f"\nMax |V_pi - V_vi| across all states: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("Both methods agree.")
    else:
        print("WARNING: large discrepancy — check your implementation.")


if __name__ == "__main__":
    # Basic 4x4, no hazards
    run(GridWorld(rows=4, cols=4), gamma=0.9)

    # With a hazard — watch how V* bends around it
    run(
        GridWorld(rows=4, cols=4, hazards=[(1, 3), (2, 1)]),
        gamma=0.9,
    )

    # With a wall
    run(
        GridWorld(rows=4, cols=4, walls=[(1, 1), (1, 2), (2, 2)]),
        gamma=0.9,
    )
