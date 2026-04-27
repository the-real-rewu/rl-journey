# RL Journey: From Tabular Methods to Advanced Deep RL

A personal learning project building toward mastery of reinforcement learning,
using a Super Mario-like environment as the through-line.

**Philosophy:** Implement everything from scratch. No black boxes.

---

## Project Structure

```
rl-journey/
├── README.md                     ← You are here (roadmap)
├── skills/                       ← Distilled knowledge, your reference library
├── envs/                         ← Custom environments (GridWorld, etc.)
├── utils/                        ← Shared helpers (plotting, replay buffers, etc.)
├── notebooks/                    ← Jupyter notebooks for visualization & experiments
├── phase1_tabular/               ← Q-Learning, SARSA, Dynamic Programming
├── phase2_dqn/                   ← DQN and its improvements on Atari Breakout
├── phase3_policy_gradient/       ← REINFORCE, A2C, PPO
└── phase4_advanced/              ← Rainbow, curiosity, multi-agent, etc.
```

---

## The Big Picture

Reinforcement learning is about an **agent** learning to take **actions** in an
**environment** to maximize cumulative **reward**. The field has evolved from
hand-crafted math over small state spaces to neural networks playing games at
superhuman level. This project walks that history.

```
Environment → State → Agent → Action → Environment → Reward → Agent (loop)
```

---

## Phase 1 — Tabular RL (GridWorld)

**Environment:** A custom GridWorld — a small 2D map with walls, a start, a goal,
and optional hazards. Think of it as a top-down, stripped-down Mario level.

**Why tabular first?** In small environments, we can represent the entire
value function as a table. This makes the math completely transparent before
neural networks obscure it.

### 1.1 — Foundations

- [ ] Markov Decision Process (MDP): states, actions, transitions, rewards, discount γ
- [ ] Bellman equations: the recursive relationship that everything in RL builds on
- [ ] Value functions: V(s) and Q(s,a) — what they mean and why they matter

### 1.2 — Dynamic Programming (requires known model)

- [ ] Policy Evaluation — compute V(s) for a fixed policy
- [ ] Policy Improvement — make a policy greedy w.r.t. its value function
- [ ] Policy Iteration — alternate the two until convergence
- [ ] Value Iteration — one-step shortcut that combines both

**Key insight:** DP requires knowing the full transition model P(s'|s,a).
Real environments don't give you that — which motivates everything that follows.

### 1.3 — Model-Free Prediction

- [ ] Monte Carlo — learn from complete episodes; no model needed
- [ ] TD(0) — learn from incomplete episodes using bootstrapping
- [ ] TD(λ) — unify MC and TD via eligibility traces

### 1.4 — Model-Free Control

- [ ] SARSA (on-policy TD control) — agent learns from what it actually does
- [ ] Q-Learning (off-policy TD control) — agent learns the optimal policy
  regardless of what it does
- [ ] ε-greedy exploration — the simplest exploration strategy

**Milestone:** Agent reliably navigates GridWorld. You can visualize the learned
Q-table as a heatmap.

---

## Phase 2 — Deep Q-Networks (Atari Breakout)

**Environment:** `ALE/Breakout-v5` via `gymnasium` + `ale-py`. Raw pixel
observations (210×160×3), preprocessed to 84×84 grayscale with 4-frame
stacking. This is the original DeepMind DQN benchmark environment and is
identical in structure to Mario — the same preprocessing pipeline applies to
any visual RL task.

**Key idea:** Replace the Q-table with a neural network Q(s,a;θ) that
generalizes across states.

### 2.1 — From Table to Network

- [ ] Function approximation — why and how neural nets replace tables
- [ ] The deadly triad: function approximation + bootstrapping + off-policy = instability
- [ ] Environment preprocessing: grayscale, frame resize, frame stacking

### 2.2 — Vanilla DQN (Mnih et al., 2013/2015)

- [ ] Experience replay — break correlations by sampling from a memory buffer
- [ ] Target network — stabilize training with a slowly-updated copy of Q
- [ ] CNN architecture for Atari-style pixel inputs
- [ ] ε-decay schedule

**Milestone:** Agent learns to hit the ball and score consistently.

### 2.3 — DQN Improvements

- [ ] Double DQN — decouple action selection from evaluation to reduce overestimation
- [ ] Dueling DQN — separate value V(s) and advantage A(s,a) streams
- [ ] Prioritized Experience Replay (PER) — sample important transitions more often
- [ ] Multi-step returns — n-step TD targets

**Milestone:** Agent consistently clears multiple brick rows.

---

## Phase 3 — Policy Gradient Methods

**Key shift:** Instead of learning a value function and acting greedily,
directly learn a policy π(a|s;θ) that maps states to action probabilities.

**Why this matters:** Policy gradients handle continuous action spaces,
are naturally stochastic (good for exploration), and scale to very complex policies.

### 3.1 — REINFORCE (Williams, 1992)

- [ ] Policy gradient theorem — the math behind updating a policy
- [ ] Monte Carlo policy gradient — use episode returns to update θ
- [ ] Baseline subtraction — reduce variance with V(s) as a baseline

### 3.2 — Actor-Critic

- [ ] A2C — synchronous advantage actor-critic; combines policy gradient
  with a learned value baseline
- [ ] Advantage function A(s,a) = Q(s,a) - V(s) — why it reduces variance
- [ ] A3C — asynchronous version with multiple parallel workers

### 3.3 — Proximal Policy Optimization (PPO)

- [ ] The problem with large policy updates — training instability
- [ ] Clipped surrogate objective — constrain how much the policy can change per update
- [ ] PPO in practice: mini-batch updates, entropy bonus, GAE (Generalized Advantage Estimation)

**Milestone:** PPO agent achieves higher average score than DQN on Breakout.
Plot learning curves comparing DQN vs PPO sample efficiency.

---

## Phase 4 — Advanced Topics

By this point you have the fundamentals. Phase 4 is about going deeper on
specific frontiers.

### 4.1 — Rainbow DQN

- [ ] Combine: Double DQN + Dueling + PER + Multi-step + Distributional RL + Noisy Nets
- [ ] Distributional RL: learn the full distribution of returns, not just the mean
- [ ] NoisyNet: learned stochastic exploration replaces ε-greedy

### 4.2 — Curiosity-Driven Exploration

- [ ] The exploration problem in sparse-reward environments
  (Mario's reward is sparse — you only get points by progressing)
- [ ] Intrinsic Curiosity Module (ICM) — reward the agent for encountering
  novel states
- [ ] Random Network Distillation (RND) — simpler curiosity signal

*(Curiosity is especially relevant for sparse-reward environments like hard
Atari games and procedurally-generated levels.)*

### 4.3 — Model-Based RL

- [ ] Learn a world model: predict next state and reward from (s,a)
- [ ] Planning with the model: Dyna-Q
- [ ] Dreamer-style: train entirely inside the learned world model

### 4.4 — Meta-RL & Transfer Learning

- [ ] Train on many Mario levels, test on unseen ones
- [ ] MAML — model-agnostic meta-learning

---

## Skills Reference

See [skills/](skills/) for distilled, permanent notes on every concept encountered.
Each skill is a self-contained reference card written as you learn it.

---

## Key Papers (in order of encounter)

| Phase | Paper | Year |
|-------|-------|------|
| 1 | Sutton & Barto, *Reinforcement Learning: An Introduction* | 1998/2018 |
| 2 | Mnih et al., *Playing Atari with Deep Reinforcement Learning* | 2013 |
| 2 | Mnih et al., *Human-level control through deep RL* (DQN) | 2015 |
| 2 | van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* | 2015 |
| 2 | Wang et al., *Dueling Network Architectures* | 2015 |
| 2 | Schaul et al., *Prioritized Experience Replay* | 2015 |
| 3 | Williams, *Simple Statistical Gradient-Following Algorithms* (REINFORCE) | 1992 |
| 3 | Mnih et al., *Asynchronous Methods for Deep RL* (A3C) | 2016 |
| 3 | Schulman et al., *Proximal Policy Optimization* | 2017 |
| 4 | Hessel et al., *Rainbow: Combining Improvements in Deep RL* | 2017 |
| 4 | Pathak et al., *Curiosity-driven Exploration by Self-Supervised Prediction* | 2017 |
| 4 | Hafner et al., *Dream to Control* (Dreamer) | 2019 |

---

## Progress Tracker

- [ ] Phase 1 — Tabular RL
- [ ] Phase 2 — Deep Q-Networks
- [ ] Phase 3 — Policy Gradient
- [ ] Phase 4 — Advanced Topics
