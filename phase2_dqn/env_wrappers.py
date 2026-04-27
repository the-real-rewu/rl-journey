"""Atari environment preprocessing for DQN.

Run from repo root:
    python -m phase2_dqn.env_wrappers

Gymnasium provides the individual wrappers. Your job is to compose them
into a pipeline that produces (4, 84, 84) float32 observations in [0, 1].
"""

from __future__ import annotations
import numpy as np
import ale_py
import gymnasium
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)

gymnasium.register_envs(ale_py)


def make_env(env_id: str = "ALE/Breakout-v5", render_mode: str | None = None) -> gymnasium.Env:
    """Return a preprocessed Atari environment ready for DQN.

    The output observation must be shape (4, 84, 84) float32 with values in [0, 1].

    Wrappers to compose (in order):
      1. AtariPreprocessing  — grayscale, resize to 84×84, frame skip, done on life loss
      2. FrameStackObservation — stack last 4 frames along a new leading axis

    After stacking, divide by 255.0 so pixel values land in [0, 1].
    The channel-first layout (4, 84, 84) matches PyTorch's Conv2d expectations.

    See the theory doc section 5 for why each step is needed.
    """
    # TODO: implement make_env.
    # Hint: AtariPreprocessing(grayscale_obs=True, scale_obs=False) handles
    # steps 1-3 from the preprocessing table. FrameStackObservation stacks
    # num_stack=4 frames. You'll need a small wrapper to normalise to [0,1]
    # and convert to float32 channel-first — or do it in the agent.
    raise NotImplementedError


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")   # expect (4, 84, 84)
    print(f"Observation dtype: {obs.dtype}")   # expect float32
    print(f"Value range:       [{obs.min():.3f}, {obs.max():.3f}]")  # expect [0, 1]
    print(f"Action space:      {env.action_space.n} discrete actions")
    obs2, reward, term, trunc, info = env.step(env.action_space.sample())
    print(f"Step OK — reward: {reward}")
    env.close()
