---
name: Bellman Equations
description: The recursive self-consistency equations that define value functions — the foundation of every RL algorithm
type: user
---

# Bellman Equations

The single most important idea in RL. Every algorithm in this project is a way to solve one of these equations.

## Core intuition

**The value of a state = immediate reward + discounted value of the next state.**

This recursive structure means you don't need to unroll full trajectories to estimate value. You only need to look one step ahead and *bootstrap* from the value of the next state. That computational shortcut is what makes RL tractable.

## Bellman Expectation Equation (for a fixed policy π)

```
V^π(s) = Σ_a π(a|s)  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
```

- Outer sum: average over actions according to policy π
- Inner sum: average over next states according to transition model P
- Bracket: immediate reward + discounted value of landing in s'

For deterministic policy and transitions this collapses to:
```
V^π(s) = R(s, π(s), s') + γ V^π(s')
```

## Bellman Optimality Equation (for the optimal policy π*)

```
V*(s) = max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V*(s') ]
Q*(s,a) =      Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ max_{a'} Q*(s',a') ]
```

The only change from expectation to optimality: `Σ_a π(a|s)` becomes `max_a`. Instead of averaging over what the policy does, we take the best possible action.

## Why "bootstrapping" matters

Trajectory unrolling:
```
V^π(s_0) = r_1 + γ r_2 + γ² r_3 + ...    (adds up all future rewards)
```

Bellman bootstrapping:
```
V^π(s_0) = r_1 + γ V^π(s_1)               (one step + already-known next value)
```

Same answer, but bootstrapping lets algorithms update values *incrementally* — no complete episode needed. This is why TD-learning, DQN, and everything else can learn online.

## Two modes of use

| Goal | Equation | Algorithm |
|------|----------|-----------|
| Evaluate a given policy | Bellman Expectation | Policy Evaluation (DP), TD(0), MC |
| Find the optimal policy | Bellman Optimality | Value Iteration (DP), Q-Learning |

## Common pitfall

`V(terminal) = 0` always. The reward belongs to the *transition into* the terminal state, not to being there. Once the episode ends, there is no future — so future expected value is zero.
