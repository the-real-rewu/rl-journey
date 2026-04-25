"""Monte Carlo prediction for GridWorld.

Run from repo root:
    python -m phase1_tabular.mc

MC prediction estimates V^π by averaging actual returns observed in
complete episodes. No model needed.
"""

from __future__ import annotations
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import GridWorld


def generate_episode(
    env: GridWorld,
    policy: dict,
    max_steps: int = 200,
    start: tuple[int, int] | None = None,
) -> list[tuple]:
    """Run one episode following policy. Returns list of (state, action, reward) tuples.

    The list ends when the episode terminates (or max_steps is reached).
    If `start` is given, the agent begins from that state (exploring starts).
    """
    trajectory = []
    state = env.reset(start=start)
    for _ in range(max_steps):
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward))
        if done:
            break
        state = next_state
    return trajectory


def first_visit_mc_prediction(
    env: GridWorld,
    policy: dict,
    gamma: float = 0.9,
    num_episodes: int = 1000,
    seed: int | None = None,
) -> dict:
    """Estimate V^π using first-visit Monte Carlo.

    For each episode:
      1. Generate a trajectory.
      2. Walk *backward* through it computing G_t = r_{t+1} + γ G_{t+1}.
      3. For each state, the first time it appears in the episode (when
         walking forward, i.e. last time when walking backward), update
         V[s] with the running mean of G_t.
    """
    if seed is not None:
        random.seed(seed)

    V = {s: 0.0 for s in env.states}
    visit_count = {s: 0 for s in env.states}

    # Exploring starts: pick a random non-terminal state to begin each
    # episode. Without this, a deterministic policy might never visit
    # most states and we couldn't estimate their values.
    nonterminal_states = [s for s in env.states if not env.is_terminal(s)]

    for ep in range(num_episodes):
        start = random.choice(nonterminal_states)
        trajectory = generate_episode(env, policy, start=start)

        # TODO: Walk through the trajectory, compute G_t for each step, and
        # update V[s] using a running mean over first visits.
        #
        # Strategy:
        #   1. Compute returns G_0, G_1, ..., G_{T-1} for the trajectory.
        #      G_{T-1} = r_T,  G_{t} = r_{t+1} + γ G_{t+1}
        #      Walking backward is the cheapest way to do this.
        #
        #   2. Find the first occurrence index of each state in the episode
        #      (a state may be revisited; only the first visit counts).
        #
        #   3. For each first-visit state s with return G:
        #         visit_count[s] += 1
        #         V[s] += (G - V[s]) / visit_count[s]   ← incremental mean
        #
        # Remove the line below and replace with your implementation:
        raise NotImplementedError("implement first-visit MC update")

    return V


if __name__ == "__main__":
    # Smoke test: deterministic policy on tiny grid, MC should match the
    # exact return with 1 episode (deterministic env => zero variance).
    from envs import RIGHT, DOWN

    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    policy = {(0, 0): RIGHT, (0, 1): DOWN, (1, 0): RIGHT}
    V = first_visit_mc_prediction(env, policy, gamma=0.9, num_episodes=1, seed=0)
    print("MC prediction (1 episode, deterministic):")
    for s, v in sorted(V.items()):
        print(f"  V[{s}] = {v:+.3f}")
    print("\nExpected (from DP):")
    print("  V[(0, 0)] = +8.000")
    print("  V[(0, 1)] = +10.000")
    print("  V[(1, 0)] = +0.000  (never visited under this policy)")
