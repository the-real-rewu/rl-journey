"""Atari environment preprocessing for DQN.

Run from repo root:
    python3 -m phase2_dqn.env_wrappers

Gymnasium provides the individual wrappers. Your job is to compose them
into a pipeline that produces (4, 84, 84) uint8 observations in [0, 255].
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


class FireResetEnv(gymnasium.Wrapper):
    """Press FIRE on reset so Breakout actually launches the ball.

    ALE/Breakout-v5 does not auto-launch — the ball only moves once action 1 (FIRE)
    is pressed. Without this wrapper, every reset leaves the agent staring at a
    static screen until ε-random play happens to fire.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gymnasium.RewardWrapper):
    """Clip rewards to {-1, 0, +1} (Mnih et al. 2015).

    Breakout gives 1–7 points per brick depending on row. Unclipped rewards plus
    γ=0.99 produce TD targets large enough to make Adam updates unstable.
    """

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


def make_eval_env(env_id: str = "ALE/Breakout-v5", render_mode: str | None = None) -> gymnasium.Env:
    """Return a full-game evaluation environment.

    Differences from make_env (the training environment):
    - terminal_on_life_loss=False: episode runs all lives, not just one.
    - No ClipRewardEnv: returns the true game score so we can compare checkpoints.

    Use this for evaluation and GIF recording. Never use it during training —
    unclipped rewards would destabilize the Q-network.
    """
    env = gymnasium.make(env_id, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    env = FireResetEnv(env)
    env = FrameStackObservation(env, stack_size=4)
    return env


def make_env(env_id: str = "ALE/Breakout-v5", render_mode: str | None = None) -> gymnasium.Env:
    """Return a preprocessed Atari environment ready for DQN.

    The output observation is shape (4, 84, 84) uint8 with values in [0, 255].
    Normalization to float32 [0, 1] is the replay buffer's responsibility — keeping
    the env output as uint8 avoids unnecessary conversion on every step.

    Wrappers composed in order:
      1. AtariPreprocessing    — grayscale, resize to 84×84, frame skip (4),
                                 terminal_on_life_loss for stronger learning signal.
      2. FireResetEnv          — fire on reset so the ball actually launches.
      3. ClipRewardEnv         — clip rewards to {-1, 0, +1} for Q-target stability.
      4. FrameStackObservation — stacks last 4 frames along the leading axis.

    Note: ALE/Breakout-v5 has frameskip=4 built in. Pass frameskip=1 to disable it so
    AtariPreprocessing owns the frame-skip — it max-pools the last 2 frames in each skip
    window to suppress sprite flickering, which the raw ALE skip does not do.

    The channel-first layout (4, 84, 84) matches PyTorch's Conv2d expectations.
    """
    env = gymnasium.make(env_id, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=True,
    )
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = FrameStackObservation(env, stack_size=4)
    return env


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")   # expect (4, 84, 84)
    print(f"Observation dtype: {obs.dtype}")   # expect uint8
    print(f"Value range:       [{obs.min()}, {obs.max()}]")  # expect [0, 255]
    print(f"Action space:      {env.action_space.n} discrete actions")
    obs2, reward, term, trunc, info = env.step(env.action_space.sample())
    print(f"Step OK — reward: {reward}")
    env.close()
