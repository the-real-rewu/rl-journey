"""SARSA: on-policy TD control for GridWorld.

Run from repo root:
    python -m phase1_tabular.sarsa
"""

from __future__ import annotations
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import GridWorld


def epsilon_greedy(Q: dict, state, num_actions: int, epsilon: float) -> int:
    """Select an action using ε-greedy policy over Q."""
    if random.random() < epsilon:
        return random.randrange(num_actions)
    q_vals = [Q[(state, a)] for a in range(num_actions)]
    return q_vals.index(max(q_vals))


def sarsa(
    env: GridWorld,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    num_episodes: int = 2000,
    max_steps: int = 200,
    seed: int | None = None,
) -> dict:
    """On-policy TD control. Returns Q: dict[(state, action) -> float].

    The key distinction from Q-Learning: the bootstrap target uses the action
    you will actually take next under ε-greedy, not the best possible action.
    What does the update equation look like? See section 4 of the theory doc.
    """
    if seed is not None:
        random.seed(seed)

    Q = {(s, a): 0.0 for s in env.states for a in range(env.num_actions)}

    for _ in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.num_actions, epsilon)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)

            # TODO: implement the SARSA update.
            # You need the next action before you can update — choose it now
            # using ε-greedy, then form the TD target from Q(next_state, next_action).
            # At a terminal state, the future value is 0.
            raise NotImplementedError

    return Q


if __name__ == "__main__":
    from envs import RIGHT, DOWN

    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    Q = sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.05, num_episodes=2000, seed=0)

    print("SARSA Q-values (2x2 grid, 2000 episodes):")
    for s in sorted(env.states):
        if env.is_terminal(s):
            continue
        best_a = max(range(env.num_actions), key=lambda a: Q[(s, a)])
        print(f"  Q[{s}, best] = {Q[(s, best_a)]:+.3f}")

    print("\nExpected (close to Q* after convergence):")
    print("  Q[(0,1), DOWN]  ≈ +10.0   (one step from goal)")
    print("  Q[(0,0), RIGHT] ≈  +8.0   (two steps from goal)")
