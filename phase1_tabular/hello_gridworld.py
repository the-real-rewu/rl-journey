"""Sanity-check script for Section 1.1.

Run this from the repo root:
    python -m phase1_tabular.hello_gridworld

The script walks through a tiny GridWorld, prints the environment after
each step, and shows what reset/step/transitions return. Read the output
alongside 01_foundations.md.

Then try the TODO tasks at the bottom.
"""

import random
from envs import GridWorld, ACTION_NAMES, RIGHT, DOWN


def demo_deterministic_run():
    print("=" * 50)
    print("Demo 1: A deterministic two-step run to the goal")
    print("=" * 50)
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    state = env.reset()
    print(f"Initial state: {state}")
    print(env.render())
    print()

    for action in [RIGHT, DOWN]:
        next_state, reward, done, _ = env.step(action)
        print(f"action={ACTION_NAMES[action]:5}  "
              f"next_state={next_state}  reward={reward:+.1f}  done={done}")
        print(env.render())
        print()


def demo_model_access():
    print("=" * 50)
    print("Demo 2: Inspecting the model (for DP in section 1.2)")
    print("=" * 50)
    env = GridWorld(rows=2, cols=2, start=(0, 0), goal=(1, 1))
    state = (0, 0)
    for action in range(env.num_actions):
        outcomes = env.transitions(state, action)
        for prob, next_state, reward, done in outcomes:
            print(f"P({next_state} | s={state}, a={ACTION_NAMES[action]:5}) "
                  f"= {prob:.2f},  r={reward:+.1f},  done={done}")


def demo_random_rollout():
    print("=" * 50)
    print("Demo 3: Random policy rollout with return calculation")
    print("=" * 50)
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2))
    state = env.reset()
    gamma = 0.9
    rewards = []
    random.seed(0)
    for t in range(20):
        action = random.randrange(env.num_actions)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        print(f"t={t:2}  s={state}  a={ACTION_NAMES[action]:5}  "
              f"r={reward:+.1f}  s'={next_state}  done={done}")
        state = next_state
        if done:
            break

    # G_0 = r_1 + γ r_2 + γ² r_3 + ...
    G0 = sum((gamma ** k) * r for k, r in enumerate(rewards))
    print(f"\nG_0 (γ={gamma}) = {G0:.3f}")

def todo_1():
    print("=" * 50)
    print("TODO 1: Print env.states for a 4x4 grid with a wall at (1, 1)")
    print("=" * 50)
    env = GridWorld(rows=4, cols=4, walls=[(1, 1)])
    print(env.render())


def todo_2():
    print("=" * 50)
    print("TODO 2: Build a GridWorld with a hazard at (0, 2)")
    print("=" * 50)
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), hazards=[(0, 2)])
    print(env.render())
    state = env.reset()
    # Policy: always go right
    while env.is_terminal(state) == False:
        action = RIGHT
        state, reward, done, _ = env.step(action)
        print(f"action={ACTION_NAMES[action]:5}  "
              f"next_state={state}  reward={reward:+.1f}  done={done}")
        print(env.render())
        print()

def todo_3():
    print("=" * 50)
    print("TODO 3: Compute G_0 for different values of gamma")
    print("=" * 50)
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), hazards=[(0, 2)])
    # Policy: always go right
    for gamma in [0.0, 0.5, 0.9, 0.99]:
        g0 = 0.0
        state = env.reset()
        current_step = 0
        while env.is_terminal(state) == False:
            action = RIGHT
            state, reward, done, _ = env.step(action)
            discount_factor = gamma ** current_step
            g0 += discount_factor * reward
            current_step += 1
            print(f"action={ACTION_NAMES[action]:5}  "
                f"next_state={state}  reward={reward:+.1f}  done={done}")
            print()
        print(f"G_0 (γ={gamma}) = {g0:.3f}")
    print("Trend: linearly decreases as gamma increases, since the negative reward from the hazard is discounted less.")


if __name__ == "__main__":
    demo_deterministic_run()
    print()
    demo_model_access()
    print()
    demo_random_rollout()

    # TODO 1: Print env.states for a 4x4 grid with a wall at (1, 1)
    #         and confirm (1, 1) is missing.
    todo_1()

    # TODO 2: Build a GridWorld with a hazard at (0, 2). Roll out a
    #         policy that deliberately walks into it and check the
    #         reward and done flag.
    todo_2()

    # TODO 3: For the 3x3 grid above, vary gamma over [0.0, 0.5, 0.9, 0.99]
    #         and compute G_0 for the same fixed reward sequence.
    #         Explain the trend in one sentence.
    todo_3()
