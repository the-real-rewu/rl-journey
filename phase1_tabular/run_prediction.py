"""Compare MC and TD(0) prediction against the DP ground truth.

Run from repo root:
    python -m phase1_tabular.run_prediction

Once mc.py and td.py are implemented, this script runs both methods on a
4x4 GridWorld with the optimal policy (computed via DP) and tracks the
RMSE against the DP-computed V^π as more episodes are seen.
"""

from __future__ import annotations
import random
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import GridWorld
from phase1_tabular.dp import policy_iteration, policy_evaluation
from phase1_tabular.mc import every_visit_mc_prediction, first_visit_mc_prediction
from phase1_tabular.td import td0_prediction


def rmse(V_est: dict, V_true: dict) -> float:
    diffs = [(V_est[s] - V_true[s]) ** 2 for s in V_true]
    return math.sqrt(sum(diffs) / len(diffs))


def print_value_grid(env: GridWorld, V: dict, label: str):
    print(f"  {label}")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                row.append("  ###  ")
            elif env.is_terminal(s):
                row.append(f" [{V.get(s, 0):+.2f}]")
            else:
                row.append(f"  {V[s]:+.2f}")
        print("    " + "  ".join(row))

def generate_random_policy(env: GridWorld) -> dict:
    policy = {}
    for s in env.states:
        if not env.is_terminal(s) and s not in env.walls:
            policy[s] = random.choice(range(env.num_actions))
    return policy

def main():
    env = GridWorld(rows=4, cols=4)
    gamma = 0.9

    # 1. Compute the optimal policy via DP (this is just the policy we will evaluate).
    _, optimal_policy = policy_iteration(env, gamma=gamma)

    # 2. Get the ground-truth value function for that policy via DP.
    V_true = policy_evaluation(env, optimal_policy, gamma=gamma)

    print("\n" + "=" * 60)
    print("Ground truth V^π (from DP policy evaluation)")
    print("=" * 60)
    print_value_grid(env, V_true, "DP")

    # 3. Run MC and TD with increasing numbers of episodes, track RMSE.
    action_uncertainty = 0.
    print("\n" + "=" * 60)
    print("RMSE vs ground truth")
    print("=" * 60)
    print(f"{'episodes':>10}  {'MC':>10}  {'TD(α=0.1)':>12} (action uncertainty={action_uncertainty})")
    

    for n in [5, 10, 50, 100, 500, 1000, 5000]:
        V_mc, _, _ = first_visit_mc_prediction(env, optimal_policy, gamma=gamma,
                                               num_episodes=n, seed=0, action_uncertainty=action_uncertainty)
        V_td = td0_prediction(env, optimal_policy, gamma=gamma,
                              alpha=0.1, num_episodes=n, seed=0, action_uncertainty=action_uncertainty)
        print(f"{n:>10}  {rmse(V_mc, V_true):>10.4f}  {rmse(V_td, V_true):>12.4f}")

    # # 4. Print final estimates side by side.
    # print("\n" + "=" * 60)
    # print("Final estimates after 5000 episodes")
    # print("=" * 60)
    # V_mc_first_visit = first_visit_mc_prediction(env, optimal_policy, gamma=gamma,
    #                                  num_episodes=5000, seed=0)
    # V_mc_every_visit = every_visit_mc_prediction(env, optimal_policy, gamma=gamma,
    #                                  num_episodes=5000, seed=0)
    # V_td = td0_prediction(env, optimal_policy, gamma=gamma,
    #                       alpha=0.1, num_episodes=5000, seed=0)
    # print()
    # print_value_grid(env, V_true, "DP (truth)")
    # print()
    # print_value_grid(env, V_mc_first_visit, "Monte Carlo (First Visit)")
    # print()
    # print_value_grid(env, V_mc_every_visit, "Monte Carlo (Every Visit)")
    # print()
    # print_value_grid(env, V_td, "TD(0)")


if __name__ == "__main__":
    main()
