"""DQN training loop for Atari Breakout.

Run from repo root:
    python3 -m phase2_dqn.train [--use-double-q]

All experiments are driven by CONFIG below.
Command-line arguments override CONFIG values (e.g., --use-double-q to enable Double DQN).
Results are saved to results_*.json based on the experiment variant.
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2_dqn.env_wrappers import make_env
from phase2_dqn.replay_buffer import ReplayBuffer
from phase2_dqn.dqn_agent import DQNAgent

# ── Experiment config ─────────────────────────────────────────────────────────
# Suggested experiments from the theory doc:
#   1. Default: full DQN on Breakout. Watch episode reward increase over time.
#   2. Set target_update_freq=1 (no frozen target). Does training become unstable?
#   3. Set buffer_capacity=500. What happens in the first 500 steps?
#   4. Set epsilon_final=0.0. Does the agent still explore enough to learn?
CONFIG = {
    "env_id":              "ALE/Breakout-v5",
    # For learning experiments: 100k steps is enough to see basic learning.
    # For production: set to 5_000_000 (DQN shows real skill around 3–5M).
    "total_steps":         100_000,
    "buffer_capacity":     100_000,
    "batch_size":          1024,
    "learning_starts":     10_000,   # steps before first gradient update
    "train_freq":          4,        # update every N environment steps
    "target_update_freq":  1_000,
    "gamma":               0.99,
    # Adam lr does NOT scale linearly with batch size — its per-parameter
    # second-moment normalization already adapts step size. 1e-4 is the
    # SB3/Rainbow-range default; 3.2e-3 (=1e-4 × 32) caused divergence.
    "lr":                  1e-4,
    "epsilon_start":       1.0,
    "epsilon_final":       0.1,
    # Epsilon decays over 500k gradient steps ≈ 2M env steps (at train_freq=4).
    # Fully greedy (ε=0.1) for the last 3M steps.
    "epsilon_decay_steps": 500_000,
    "log_freq":            10,       # log every N episodes
    "save_freq":           250_000,  # checkpoint every N steps
    "checkpoint_dir":      "checkpoints",
    "seed":                0,
    "use_double_q":        False,    # enable Double DQN (override with --use-double-q)
}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DQN training on Atari Breakout")
    parser.add_argument("--use-double-q", action="store_true", help="Enable Double DQN (default: vanilla DQN)")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["use_double_q"] = args.use_double_q

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Double DQN: {cfg['use_double_q']}")

    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    env = make_env(cfg["env_id"])
    num_actions = env.action_space.n

    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"], device=device)
    agent = DQNAgent(
        num_actions=num_actions,
        device=device,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_final=cfg["epsilon_final"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        target_update_freq=cfg["target_update_freq"],
        use_double_q=cfg["use_double_q"],
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    obs, _ = env.reset(seed=cfg["seed"])
    episode_reward = 0.0
    episode_count = 0
    episode_rewards = []
    recent_losses = []
    t_start = time.time()

    for step in range(1, cfg["total_steps"] + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)
            if episode_count % cfg["log_freq"] == 0:
                recent = episode_rewards[-cfg["log_freq"]:]
                mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                elapsed = time.time() - t_start
                print(
                    f"step {step:>8,} | ep {episode_count:>5} | "
                    f"reward {sum(recent)/len(recent):>6.1f} | "
                    f"loss {mean_loss:.4f} | "
                    f"ε {agent.epsilon:.3f} | "
                    f"{step/elapsed:.0f} steps/s"
                )
                recent_losses.clear()
            episode_reward = 0.0
            obs, _ = env.reset()

        if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            if len(buffer) >= cfg["batch_size"]:
                batch = buffer.sample(cfg["batch_size"])
                loss = agent.train_step(batch)
                recent_losses.append(loss)

        if step % cfg["save_freq"] == 0:
            path = os.path.join(cfg["checkpoint_dir"], f"dqn_step_{step}.pt")
            agent.save(path)
            print(f"  → checkpoint saved: {path}")

    env.close()
    print("Training complete.")

    # Save episode rewards for analysis
    variant_name = "double_dqn" if cfg["use_double_q"] else "vanilla"
    results_file = os.path.join(cfg["checkpoint_dir"], f"results_{variant_name}.json")
    results = {
        "variant": variant_name,
        "episode_rewards": episode_rewards,
        "total_steps": cfg["total_steps"],
        "use_double_q": cfg["use_double_q"],
    }
    with open(results_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
