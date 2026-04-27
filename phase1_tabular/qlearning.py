"""Q-Learning: off-policy TD control for GridWorld.

Run from repo root:
    python -m phase1_tabular.qlearning
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


def qlearning(
    env: GridWorld,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    num_episodes: int = 2000,
    max_steps: int = 200,
    seed: int | None = None,
) -> dict:
    """Off-policy TD control. Returns Q: dict[(state, action) -> float].

    The key distinction from SARSA: the bootstrap target reflects the optimal
    policy, regardless of which action ε-greedy actually selected at next_state.
    What does the update equation look like? See section 5 of the theory doc.
    """
    if seed is not None:
        random.seed(seed)

    Q = {(s, a): 0.0 for s in env.states for a in range(env.num_actions)}

    for _ in range(num_episodes):
        state = env.reset()

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, env.num_actions, epsilon)
            next_state, reward, done, _ = env.step(action)

            # TODO: implement the Q-Learning update.
            # The target uses the maximum Q value at next_state — you do not
            # need to know which action ε-greedy would actually take there.
            # At a terminal state, the future value is 0.
            best_next_q = max(Q[(next_state, a)] for a in range(env.num_actions))
            Q[(state, action)] += alpha * (reward + (0 if done else gamma * best_next_q) - Q[(state, action)])
            #update the state
            state = next_state
            if done: break

    return Q


if __name__ == "__main__":
    from envs import RIGHT, DOWN

    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    Q = qlearning(env, gamma=0.9, alpha=0.1, epsilon=0.05, num_episodes=2000, seed=0)

    print("Q-Learning Q-values (2x2 grid, 2000 episodes):")
    for s in sorted(env.states):
        if env.is_terminal(s):
            continue
        best_a = max(range(env.num_actions), key=lambda a: Q[(s, a)])
        print(f"  Q[{s}, best] = {Q[(s, best_a)]:+.3f}")

    print("\nExpected (close to Q* after convergence):")
    print("  Q[(0,1), DOWN]  ≈ +10.0   (one step from goal)")
    print("  Q[(0,0), RIGHT] ≈  +8.0   (two steps from goal)")
