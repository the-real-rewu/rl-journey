"""DQN training loop for Atari Breakout.

Run from repo root:
    python3 -m phase2_dqn.train [--use-double-dqn]

All experiments are driven by CONFIG below.
Command-line arguments override CONFIG values (e.g., --use-double-dqn to enable Double DQN).
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
    # 5M env steps — DQN paper scale. Real skill emerges, bias curves clearly
    # diverge between vanilla and Double DQN. Expect 6–14 hours per run on GPU.
    "total_steps":         5_000_000,
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
    # Decay is in GRADIENT steps. Total gradient steps for 5M env steps =
    # (5_000_000 - 10_000) / 4 ≈ 1.25M. Decay over first 250k gradient steps
    # = ~1M env steps (DQN-paper standard). Remaining ~80% of training greedy.
    "epsilon_decay_steps": 250_000,
    "log_freq":            10,       # log every N episodes
    "save_freq":           500_000,  # checkpoint every N env steps
    # Q-value bias tracking: every N env steps, compute mean max_a Q_online(s,a)
    # over a fixed eval state set. Vanilla's curve should drift higher than
    # Double DQN's — direct evidence of overestimation bias, visible long
    # before the reward gap shows up.
    "q_log_freq":          10_000,
    "q_eval_size":         512,      # states held fixed across training for Q tracking
    "checkpoint_dir":      "checkpoints",
    "seed":                0,
    "use_double_dqn":        False,    # enable Double DQN (override with --use-double-dqn)
}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DQN training on Atari Breakout")
    parser.add_argument("--use-double-dqn", action="store_true", help="Enable Double DQN (default: vanilla DQN)")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["use_double_dqn"] = args.use_double_dqn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Double DQN: {cfg['use_double_dqn']}")

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
        use_double_dqn=cfg["use_double_dqn"],
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    obs, _ = env.reset(seed=cfg["seed"])
    episode_reward = 0.0
    episode_count = 0
    episode_rewards = []
    recent_losses = []
    q_log = []                # list of (env_step, mean_max_q) tuples
    eval_states = None        # fixed state set, captured once buffer is warm
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

        # Capture a fixed eval state set once the buffer has enough samples.
        # Holding these constant across training lets us track how Q-estimates
        # evolve on the *same* states — that's where overestimation shows up.
        if eval_states is None and len(buffer) >= cfg["q_eval_size"]:
            eval_states = buffer.sample(cfg["q_eval_size"])[0].detach()

        if eval_states is not None and step % cfg["q_log_freq"] == 0:
            with torch.no_grad():
                mean_max_q = agent.online(eval_states).max(dim=1)[0].mean().item()
            q_log.append((step, mean_max_q))

        if step % cfg["save_freq"] == 0:
            variant_tag = "double_dqn" if cfg["use_double_dqn"] else "vanilla"
            path = os.path.join(cfg["checkpoint_dir"], f"dqn_{variant_tag}_step_{step}.pt")
            agent.save(path)
            print(f"  → checkpoint saved: {path}")

    env.close()
    print("Training complete.")

    # Save episode rewards for analysis
    variant_name = "double_dqn" if cfg["use_double_dqn"] else "vanilla"
    results_file = os.path.join(cfg["checkpoint_dir"], f"results_{variant_name}.json")
    results = {
        "variant": variant_name,
        "episode_rewards": episode_rewards,
        "q_log": q_log,
        "total_steps": cfg["total_steps"],
        "use_double_dqn": cfg["use_double_dqn"],
    }
    with open(results_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
