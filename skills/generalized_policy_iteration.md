---
name: Generalized Policy Iteration (GPI)
description: The unifying pattern behind every RL algorithm — alternating value estimation and policy improvement
type: user
---

# Generalized Policy Iteration (GPI)

## The pattern

```
V (evaluation) → better π (improvement) → better V (evaluation) → ...
```

Any process that alternates between:
1. Making V consistent with the current π (evaluation)
2. Making π greedy with respect to the current V (improvement)

...is doing GPI. The two steps *interact*: each improvement step makes V inaccurate (the policy changed), and each evaluation step makes the current policy suboptimal relative to the new V. They compete and cooperate until neither can improve the other. That fixed point is (π*, V*).

## Why "generalized"

The original policy iteration ran evaluation to full convergence before each improvement. GPI relaxes both steps:
- Evaluation can be partial (1 sweep, 10 sweeps, one sampled episode, one TD step).
- Improvement can be soft (stochastic policies that move *toward* greedy, not fully greedy).

Every RL algorithm in this project is a choice of how incomplete to make each step.

## The same skeleton, all the way to DQN and PPO

| Algorithm | Evaluation step | Improvement step |
|-----------|----------------|-----------------|
| Policy Iteration | Full Bellman sweeps | Greedy argmax |
| Value Iteration | One Bellman sweep | Greedy argmax (implicit) |
| SARSA | One TD update | Greedy (ε-greedy) |
| Q-Learning | One TD update | Hard greedy |
| DQN | Mini-batch gradient step on Q | ε-greedy |
| PPO | Many gradient steps on π | Clipped policy gradient |

The neural networks in DQN and PPO are just flexible function approximators replacing the table. GPI is still the skeleton.

## Key insight

You do not need either step to be exact for GPI to converge — only that each step makes *some* progress and they don't undo each other's work. This is why approximate, sampled, incremental RL algorithms still find good policies.
