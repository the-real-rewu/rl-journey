---
name: Q-Learning
description: Off-policy TD control — directly approximates Q* by always bootstrapping from the greedy action, regardless of behavior
type: concept
---

# Q-Learning (Off-Policy TD Control)

## What it does

Q-Learning is a model-free algorithm that directly approximates Q* — the optimal action-value function — from sampled experience. It uses ε-greedy to collect data but its update target always reflects the greedy (optimal) policy.

## The update

```
Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s', a') - Q(s,a)]
```

The target `r + γ max_{a'} Q(s', a')` is the Bellman optimality equation sampled at one transition. It doesn't depend on which action ε-greedy actually took at s' — it always uses the best.

Compare to SARSA: swap `Q(s', a')` for `max_{a'} Q(s', a')`. That's the entire difference.

## Why "off-policy"

The data is collected by the ε-greedy **behavior policy**, but the update target represents the **greedy target policy**. These are different policies — hence off-policy.

Because the target policy is greedy, Q-Learning converges to Q* (under standard conditions: all (s,a) visited sufficiently, α decays appropriately). It doesn't matter how random the behavior policy is, as long as it provides coverage. This is what makes off-policy learning powerful: you can learn the optimal policy from any exploration strategy, even a random one.

## Behavior near hazards

Q-Learning's target assumes the agent will act greedily at every future step — it never factors in the chance of ε-greedy accidentally stepping into a hazard. Near a cliff, Q-Learning learns the optimal (cliff-adjacent) path because the target always says "I'll take the best action from here." The hazard risk is invisible to the update.

When deployed with ε > 0, this optimism becomes a problem: the policy is optimal under perfect execution but fragile to exploration noise.

## Properties

| | Q-Learning |
|---|---|
| On/off-policy | Off-policy |
| Converges to | Q* (optimal action-value function) |
| Hazard avoidance | Optimistic — ignores exploration noise |
| Requires model | No |
| Updates | Per step (online) |

## Connection to DQN

Deep Q-Networks (Phase 2) use the same update rule — the only change is replacing the Q table with a neural network. The off-policy property is essential for DQN's experience replay: stored transitions can come from any past policy, and the Q-Learning target still points toward Q*.
