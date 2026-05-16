"""Compare DQN training runs by reading results_*.json files in checkpoints/.

See Task F of phase2_dqn/03_dqn_improvements.md.

Run from repo root:
    python3 -m phase2_dqn.compare_runs [--checkpoint_dir checkpoints] [--output_dir plots]

Produces three PNGs in the output dir:
    episode_rewards.png   — reward vs episode, 50-ep moving average, one line per variant
    mean_max_q.png        — Q-bias tracker (mean max_a Q on a fixed state set) vs env step
    final_summary.png     — bar chart of mean reward over the final 100 episodes per variant

The script must work with however many variants exist in checkpoint_dir — do not
hardcode variant names. Sort the legend in a stable order (alphabetical is fine).
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_results(checkpoint_dir: str) -> dict[str, dict]:
    """Read every checkpoint_dir/results_*.json and return a dict keyed by variant name.

    Each JSON has fields: variant, episode_rewards, q_log, total_steps, use_double_dqn,
    plus whatever fields Task E adds for the new flags.
    """
    # TODO
    raise NotImplementedError


def plot_episode_rewards(results: dict[str, dict], out_path: str, window: int = 50) -> None:
    """Episode reward vs episode number, with a moving average over `window` episodes."""
    # TODO
    raise NotImplementedError


def plot_mean_max_q(results: dict[str, dict], out_path: str) -> None:
    """Mean-max-Q (overestimation tracker) vs env step.

    Each variant's q_log is a list of [env_step, mean_max_q] pairs.
    """
    # TODO
    raise NotImplementedError


def plot_final_summary(results: dict[str, dict], out_path: str, last_n_episodes: int = 100) -> None:
    """Bar chart: mean episode reward over the final `last_n_episodes` per variant."""
    # TODO
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Compare DQN training runs.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.checkpoint_dir)

    if not results:
        print(f"No results_*.json found in {args.checkpoint_dir}/")
        return

    plot_episode_rewards(results, os.path.join(args.output_dir, "episode_rewards.png"))
    plot_mean_max_q(results, os.path.join(args.output_dir, "mean_max_q.png"))
    plot_final_summary(results, os.path.join(args.output_dir, "final_summary.png"))

    print(f"Wrote 3 plots to {args.output_dir}/ comparing {len(results)} variant(s):")
    for variant in sorted(results.keys()):
        print(f"  - {variant}")


if __name__ == "__main__":
    main()
