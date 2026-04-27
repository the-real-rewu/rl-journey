---
name: SARSA
description: On-policy TD control — learns Q for the ε-greedy policy actually being followed, not Q*
type: concept
---

# SARSA (On-Policy TD Control)

## What it does

SARSA is a model-free algorithm that learns Q values and improves a policy by interacting with the environment. It applies TD(0) to Q(s,a) instead of V(s), updating after each step.

The name comes from the five quantities used in each update: **(S, A, R, S', A')** — current state, action taken, reward received, next state, and next action chosen.

## The update

```
Q(s,a) ← Q(s,a) + α [r + γ Q(s', a') - Q(s,a)]
```

where **a' is the action ε-greedy actually selects at s'** — chosen before the update, carried into the next step.

Compare to Q-Learning: the only difference is `Q(s', a')` vs `max_a Q(s', a)`. SARSA uses the action you will take; Q-Learning uses the best possible action.

## Why "on-policy"

The bootstrap target `Q(s', a')` comes from the same ε-greedy policy being evaluated. SARSA learns Q^{π_ε} — the value of the policy it's following.

This has a precise consequence: SARSA does **not** converge to Q*. It converges to the value of the ε-greedy policy, which is slightly less than Q* because ε-greedy occasionally takes suboptimal actions. As ε→0, Q^{π_ε}→Q*, but for any fixed ε>0, there's a gap. This is visible in practice: SARSA's RMSE vs the DP-computed V* plateaus at a small nonzero value even after convergence.

## Behavior near hazards

Because SARSA's target reflects the actual ε-greedy policy, it "sees" the cost of accidental bad moves. Near a hazard, ε-greedy occasionally steps in, and those transitions appear in the TD updates. SARSA learns to give cliff-adjacent states lower Q values, and the derived policy steers away from risk — even if the greedy path would go through those states.

## Properties

| | SARSA |
|---|---|
| On/off-policy | On-policy |
| Converges to | Q^{π_ε} (ε-greedy policy value) |
| Hazard avoidance | Conservative — accounts for exploration noise |
| Requires model | No |
| Updates | Per step (online) |
