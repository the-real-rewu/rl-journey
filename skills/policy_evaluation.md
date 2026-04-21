---
name: Policy Evaluation
description: How to compute V^π for a fixed policy using iterative Bellman sweeps — the inner loop of policy iteration
type: user
---

# Policy Evaluation

**What it solves:** Given a fixed policy π, compute V^π exactly (up to tolerance θ).

## The algorithm

Initialize V[s] = 0 for all s. Repeatedly sweep through all non-terminal states and apply the Bellman expectation equation as an update:

```
V[s] ← Σ_{s',r} P(s',r | s, π(s)) * [r + γ * V[s']]
```

Stop when `max_s |V_new[s] - V_old[s]| < θ`.

## Why it converges

The Bellman expectation operator is a **contraction mapping** with factor γ. Each sweep reduces the error `||V - V^π||` by at least γ. Since γ < 1, the error shrinks geometrically and V converges to V^π.

Terminal states act as anchors: V[terminal] = 0 is always exact. Each sweep propagates that certainty one step outward, which is why the values farthest from the goal converge last.

## In-place vs two-table

- **Two-table:** compute V_new from V_old; swap after each full sweep.
- **In-place:** update V[s] immediately; later states see updated values in the same sweep.

In-place converges faster in practice. Both converge to V^π. Code implementations almost always use in-place.

## What it requires

- The full transition model `P(s'|s,a)` — this is **model-based**. Without it you need Monte Carlo or TD methods (section 1.3).
- A fixed policy π to evaluate.

## Connection to section 1.3

Monte Carlo and TD learning solve the same problem (estimate V^π) without knowing P. Policy evaluation is the model-based baseline to understand what they're approximating.
