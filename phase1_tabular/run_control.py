"""Compare SARSA and Q-Learning on GridWorld.

Run from repo root:
    python -m phase1_tabular.run_control

All experiments are driven by CONFIG below — change one value, re-run.
"""

from __future__ import annotations
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import GridWorld, ACTION_NAMES
from phase1_tabular.dp import policy_iteration, policy_evaluation
from phase1_tabular.sarsa import sarsa
from phase1_tabular.qlearning import qlearning

# ── Experiment config ─────────────────────────────────────────────────────────
# Change these values to run different experiments.
#
# Suggested experiments from the theory doc:
#   1. Default: both algorithms on a plain 4x4 grid.
#   2. Add hazards (e.g. [(2,1),(2,2)]) and compare learned policies.
#   3. Sweep epsilon: try 0.0, 0.05, 0.3 — watch convergence and final policy.
#   4. Swap alpha: 0.5 vs 0.01 — characterize the speed/stability tradeoff.
CONFIG = {
    "algorithm": "both",        # "sarsa", "qlearning", or "both"
    "rows": 4,
    "cols": 4,
    "hazards": [],              # list of (row, col) tuples, e.g. [(2,1),(2,2)]
    "gamma": 0.9,
    "alpha": 0.1,
    "epsilon": 0.1,
    "num_episodes": 3000,
    "seed": 0,
}
# ─────────────────────────────────────────────────────────────────────────────


ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def rmse(Q: dict, V_true: dict, env: GridWorld) -> float:
    """RMSE between max_a Q(s,a) and V_true(s) over non-terminal states."""
    diffs = []
    for s in env.states:
        if env.is_terminal(s):
            continue
        v_est = max(Q[(s, a)] for a in range(env.num_actions))
        diffs.append((v_est - V_true[s]) ** 2)
    return math.sqrt(sum(diffs) / len(diffs))


def print_policy_grid(env: GridWorld, Q: dict, label: str):
    print(f"  {label}")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                row.append(" # ")
            elif env.is_terminal(s):
                row.append(" G " if s == env.goal else " H ")
            else:
                best = max(range(env.num_actions), key=lambda a: Q[(s, a)])
                row.append(f" {ARROWS[best]} ")
        print("    " + " ".join(row))


def print_value_grid(env: GridWorld, Q: dict, label: str):
    print(f"  {label}")
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                row.append("  ### ")
            elif env.is_terminal(s):
                row.append(f"  [G] ")
            else:
                v = max(Q[(s, a)] for a in range(env.num_actions))
                row.append(f" {v:+.2f}")
        print("    " + "  ".join(row))


def run_algorithm(name: str, env: GridWorld, cfg: dict, V_true: dict):
    fn = sarsa if name == "sarsa" else qlearning
    Q = fn(
        env,
        gamma=cfg["gamma"],
        alpha=cfg["alpha"],
        epsilon=cfg["epsilon"],
        num_episodes=cfg["num_episodes"],
        seed=cfg["seed"],
    )
    print(f"\n{'='*60}")
    print(f"{name.upper()}  (ε={cfg['epsilon']}, α={cfg['alpha']}, {cfg['num_episodes']} episodes)")
    print(f"{'='*60}")
    print(f"  Final RMSE vs DP V*: {rmse(Q, V_true, env):.4f}")
    print()
    print_policy_grid(env, Q, "Greedy policy")
    print()
    print_value_grid(env, Q, "max_a Q(s,a)")
    return Q


def main():
    cfg = CONFIG
    env = GridWorld(
        rows=cfg["rows"],
        cols=cfg["cols"],
        hazards=frozenset(cfg["hazards"]),
    )

    _, opt_policy = policy_iteration(env, gamma=cfg["gamma"])
    V_true = policy_evaluation(env, opt_policy, gamma=cfg["gamma"])

    print("\n" + "=" * 60)
    print("DP ground truth V*")
    print("=" * 60)
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = (r, c)
            if env.is_terminal(s):
                row.append(f"  [G] ")
            else:
                row.append(f" {V_true[s]:+.2f}")
        print("    " + "  ".join(row))

    algorithms = (
        ["sarsa", "qlearning"] if cfg["algorithm"] == "both" else [cfg["algorithm"]]
    )
    for name in algorithms:
        run_algorithm(name, env, cfg, V_true)


if __name__ == "__main__":
    main()
