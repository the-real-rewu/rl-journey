"""DQN training loop for Atari Breakout.

Run from repo root:
    python -m phase2_dqn.train

All experiments are driven by CONFIG below — change one value, re-run.
"""

from __future__ import annotations
import sys
import os
import time
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
    "total_steps":         2_000_000,
    "buffer_capacity":     100_000,
    "batch_size":          32,
    "learning_starts":     10_000,   # steps before first gradient update
    "train_freq":          4,        # update every N environment steps
    "target_update_freq":  1_000,
    "gamma":               0.99,
    "lr":                  1e-4,
    "epsilon_start":       1.0,
    "epsilon_final":       0.1,
    "epsilon_decay_steps": 500_000,
    "log_freq":            10,       # log every N episodes
    "save_freq":           100_000,  # checkpoint every N steps
    "checkpoint_dir":      "checkpoints",
    "seed":                0,
}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(cfg["seed"])

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
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    obs, _ = env.reset(seed=cfg["seed"])
    episode_reward = 0.0
    episode_count = 0
    episode_rewards = []
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
                elapsed = time.time() - t_start
                print(
                    f"step {step:>8,} | ep {episode_count:>5} | "
                    f"reward {sum(recent)/len(recent):>6.1f} | "
                    f"ε {agent.epsilon:.3f} | "
                    f"{step/elapsed:.0f} steps/s"
                )
            episode_reward = 0.0
            obs, _ = env.reset()

        if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            if len(buffer) >= cfg["batch_size"]:
                batch = buffer.sample(cfg["batch_size"])
                agent.train_step(batch)

        if step % cfg["save_freq"] == 0:
            path = os.path.join(cfg["checkpoint_dir"], f"dqn_step_{step}.pt")
            agent.save(path)
            print(f"  → checkpoint saved: {path}")

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
