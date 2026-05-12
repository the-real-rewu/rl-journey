"""Evaluate DQN checkpoints and record the best and worst runs as GIFs.

Pass 1: Run greedy episodes for every checkpoint to collect reward statistics.
Pass 2: Re-run the best and worst checkpoints with frame capture to save GIFs.

Run from repo root:
    python3 -m phase2_dqn.eval [--checkpoint_dir checkpoints] [--output_dir videos] [--episodes 5]

GIFs are saved to the output directory and can be opened directly in VSCode.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2_dqn.dqn_agent import DQNAgent
from phase2_dqn.env_wrappers import make_eval_env

# ── Constants ─────────────────────────────────────────────────────────────────
GIF_FPS = 15          # half the env rate — keeps GIF files small
EVAL_EPSILON = 0.05   # tiny exploration so the agent never gets permanently stuck
ASCII_BAR_WIDTH = 40
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CheckpointResult:
    step: int
    mean_reward: float
    max_reward: float
    min_reward: float
    checkpoint_path: str


# ── Agent helpers ──────────────────────────────────────────────────────────────

def load_agent(checkpoint_path: str) -> DQNAgent:
    """Load a checkpoint and configure it for greedy evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_eval_env()
    agent = DQNAgent(num_actions=env.action_space.n, device=device)
    env.close()
    agent.load(checkpoint_path)
    agent.epsilon_start = EVAL_EPSILON
    agent.epsilon_final = EVAL_EPSILON
    return agent


def run_episode(env, agent: DQNAgent, capture_frames: bool) -> tuple[float, list]:
    """Run one episode and return (total_reward, frames).

    frames is a list of (210, 160, 3) uint8 RGB arrays.
    If capture_frames is False, frames is always an empty list.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    frames = []

    while True:
        if capture_frames:
            frames.append(env.render())

        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, frames


# ── Pass 1: reward evaluation ──────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_path: str, num_episodes: int) -> list[float]:
    """Run num_episodes greedy episodes and return the list of episode rewards."""
    agent = load_agent(checkpoint_path)
    env = make_eval_env()
    episode_rewards = []

    for _ in range(num_episodes):
        episode_reward, _ = run_episode(env, agent, capture_frames=False)
        episode_rewards.append(episode_reward)

    env.close()
    return episode_rewards


# ── Pass 2: GIF recording ──────────────────────────────────────────────────────

def record_episode_gif(checkpoint_path: str, gif_output_path: str) -> float:
    """Record one greedy episode and save it as an animated GIF.

    Returns the episode reward so the caller can display it.
    """
    agent = load_agent(checkpoint_path)
    env = make_eval_env(render_mode="rgb_array")

    episode_reward, frames = run_episode(env, agent, capture_frames=True)
    env.close()

    frames_to_gif(frames, gif_output_path, fps=GIF_FPS)
    return episode_reward


def frames_to_gif(frames: list, output_path: str, fps: int) -> None:
    """Convert a list of RGB uint8 arrays to an animated GIF using PIL."""
    frame_duration_ms = 1000 // fps

    pil_frames = [Image.fromarray(frame, "RGB") for frame in frames]

    # GIF requires an 8-bit palette — quantize each frame before saving.
    quantized_frames = [
        frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        for frame in pil_frames
    ]

    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )


# ── Terminal output ────────────────────────────────────────────────────────────

def _ascii_bar(value: float, max_value: float) -> str:
    filled = int(round(ASCII_BAR_WIDTH * value / max_value)) if max_value > 0 else 0
    return "█" * filled + "░" * (ASCII_BAR_WIDTH - filled)


def print_table(results: list[CheckpointResult]) -> None:
    """Print a summary table with an ASCII learning curve."""
    print("\n" + "=" * 74)
    print(f"{'Step':>10}  {'Mean':>6}  {'Max':>5}  {'Min':>5}  Learning curve")
    print("-" * 74)

    max_mean = max(r.mean_reward for r in results) or 1.0
    for result in results:
        bar = _ascii_bar(result.mean_reward, max_mean)
        print(
            f"{result.step:>10,}  "
            f"{result.mean_reward:6.1f}  "
            f"{result.max_reward:5.1f}  "
            f"{result.min_reward:5.1f}  "
            f"{bar}"
        )

    best = max(results, key=lambda r: r.mean_reward)
    print("=" * 74)
    print(f"\nBest checkpoint: step {best.step:,}  mean reward = {best.mean_reward:.1f}")
    print(f"  → {best.checkpoint_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN checkpoints and save GIFs of the best and worst runs."
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="videos")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per checkpoint for reward statistics (default: 5)",
    )
    args = parser.parse_args()

    checkpoint_paths = sorted(
        glob.glob(os.path.join(args.checkpoint_dir, "dqn_step_*.pt")),
        key=lambda path: int(re.search(r"dqn_step_(\d+)", path).group(1)),
    )
    if not checkpoint_paths:
        print(f"No checkpoints found in {args.checkpoint_dir}/")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}")
    print(f"Evaluating {len(checkpoint_paths)} checkpoints × {args.episodes} episodes\n")

    # ── Pass 1: reward statistics ──────────────────────────────────────────────
    results: list[CheckpointResult] = []

    for index, checkpoint_path in enumerate(checkpoint_paths):
        step = int(re.search(r"dqn_step_(\d+)", checkpoint_path).group(1))
        t_start = time.time()

        episode_rewards = evaluate_checkpoint(checkpoint_path, args.episodes)
        elapsed = time.time() - t_start

        result = CheckpointResult(
            step=step,
            mean_reward=float(np.mean(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            checkpoint_path=checkpoint_path,
        )
        results.append(result)

        print(
            f"  [{index + 1:2d}/{len(checkpoint_paths)}] "
            f"step {step:>8,} | "
            f"mean {result.mean_reward:6.1f} | "
            f"max {result.max_reward:5.1f} | "
            f"min {result.min_reward:5.1f} | "
            f"{elapsed:.1f}s"
        )

    print_table(results)

    # ── Pass 2: record best and worst ──────────────────────────────────────────
    best_result = max(results, key=lambda r: r.mean_reward)
    worst_result = min(results, key=lambda r: r.mean_reward)

    best_gif_path = os.path.join(args.output_dir, "best_run.gif")
    worst_gif_path = os.path.join(args.output_dir, "worst_run.gif")

    print(f"\nRecording best checkpoint  (step {best_result.step:,}) …", end=" ", flush=True)
    best_recorded_reward = record_episode_gif(best_result.checkpoint_path, best_gif_path)
    print(f"reward = {best_recorded_reward:.1f}")

    print(f"Recording worst checkpoint (step {worst_result.step:,}) …", end=" ", flush=True)
    worst_recorded_reward = record_episode_gif(worst_result.checkpoint_path, worst_gif_path)
    print(f"reward = {worst_recorded_reward:.1f}")

    print(f"\nGIFs saved to {os.path.abspath(args.output_dir)}/")
    print(f"  best_run.gif   — step {best_result.step:,}  reward {best_recorded_reward:.1f}")
    print(f"  worst_run.gif  — step {worst_result.step:,}  reward {worst_recorded_reward:.1f}")


if __name__ == "__main__":
    main()
