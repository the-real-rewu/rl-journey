---
name: On-Policy vs Off-Policy
description: Whether the policy being evaluated matches the policy collecting data — and why it matters near risk
type: concept
---

# On-Policy vs Off-Policy

## The distinction

Every RL algorithm has two implicit policies:

- **Behavior policy:** the policy used to collect experience (take actions, observe transitions).
- **Target policy:** the policy being evaluated and improved.

**On-policy:** behavior policy = target policy. You learn the value of the policy you're following.

**Off-policy:** behavior policy ≠ target policy. You use one policy to explore, but learn the value of a different (usually greedy) policy.

SARSA is on-policy: it evaluates the ε-greedy policy it's following. Q-Learning is off-policy: it uses ε-greedy to collect data but evaluates the greedy policy.

## When it matters

In safe, flat environments the two converge to the same Q* as ε→0. The difference becomes critical when the environment has risk:

**Cliff walking:** two paths from start to goal — a short path along the cliff edge, a long safe path far from it. With ε=0.1:

| | SARSA | Q-Learning |
|---|---|---|
| What the target assumes | Agent follows ε-greedy (may slip toward cliff) | Agent always acts greedily (never slips) |
| Cliff-adjacent states | Penalized — ε can misfire into hazard | Unpenalized — target ignores ε |
| Learned path | Safe (long) | Optimal (cliff-edge, short) |
| During training | Fewer cliff falls | More cliff falls |

## The deployment question

Q-Learning learns the better policy *if* execution is perfect. SARSA learns a policy that is robust to imperfect execution.

Choosing between them depends on the gap between training and deployment:
- **If you deploy with ε=0 and execution is reliable:** Q-Learning's policy is better.
- **If the deployed agent has any action noise or uncertainty:** SARSA's conservatism is worth the longer path.

Most real hardware has some execution noise. SARSA's on-policy accounting for that noise is a feature, not a limitation.

## Why off-policy enables experience replay

Off-policy learning (Q-Learning) has a practical advantage beyond convergence targets: old transitions stored in a replay buffer can still be used for updates, even though they were collected under a different (older) policy. The Bellman optimality target `r + γ max_a Q(s', a)` doesn't depend on which policy generated the data.

This is why DQN uses Q-Learning: experience replay requires off-policy learning.
