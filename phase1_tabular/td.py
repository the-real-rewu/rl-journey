"""TD(0) prediction for GridWorld.

Run from repo root:
    python -m phase1_tabular.td

TD(0) updates V[s] each step using a one-step bootstrapped target:
    V[s] ← V[s] + α (r + γ V[s'] - V[s])
"""

from __future__ import annotations
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import GridWorld


def td0_prediction(
    env: GridWorld,
    policy: dict,
    gamma: float = 0.9,
    alpha: float = 0.1,
    num_episodes: int = 1000,
    max_steps: int = 200,
    seed: int | None = None,
) -> dict:
    """Estimate V^π using TD(0).

    For each step in each episode:
        TD target  = r + γ * V[s']      (or just r if s' is terminal)
        TD error   = TD target - V[s]
        V[s]      ← V[s] + α * TD error
    """
    if seed is not None:
        random.seed(seed)

    V = {s: 0.0 for s in env.states}

    # Exploring starts: same reasoning as MC. Without random start states,
    # most states under a deterministic policy are never visited.
    nonterminal_states = [s for s in env.states if not env.is_terminal(s)]

    for ep in range(num_episodes):
        state = env.reset(start=random.choice(nonterminal_states))

        for _ in range(max_steps):
            action = policy[state]
            next_state, reward, done, _ = env.step(action)

            # TODO: Apply the TD(0) update.
            #
            #   td_target = reward + gamma * V[next_state]
            #   (if `done` is True, the episode just ended, so the future
            #    value of next_state is 0 — there is no future. Make sure
            #    the target collapses to just `reward` in that case.)
            #
            #   td_error = td_target - V[state]
            #   V[state] += alpha * td_error
            #
            # Remove the line below and replace with your implementation:
            raise NotImplementedError("implement TD(0) update")

            if done:
                break
            state = next_state

    return V


if __name__ == "__main__":
    # Smoke test: deterministic policy on tiny grid.
    # TD(0) needs more episodes than MC to converge because the +10 reward
    # has to propagate one step backward per episode through bootstrapping.
    from envs import RIGHT, DOWN

    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    policy = {(0, 0): RIGHT, (0, 1): DOWN, (1, 0): RIGHT}
    V = td0_prediction(env, policy, gamma=0.9, alpha=0.5, num_episodes=200, seed=0)
    print("TD(0) prediction (200 episodes, α=0.5, deterministic):")
    for s, v in sorted(V.items()):
        print(f"  V[{s}] = {v:+.3f}")
    print("\nExpected (from DP):")
    print("  V[(0, 0)] = +8.000")
    print("  V[(0, 1)] = +10.000")
    print("  V[(1, 0)] = +0.000  (never visited under this policy)")
