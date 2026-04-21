---
name: Value Iteration
description: Find V* directly using the Bellman optimality operator — one sweep per iteration, no explicit policy
type: user
---

# Value Iteration

## The algorithm

```
1. Initialise V[s] = 0 for all s.
2. Loop until max_s |V_new[s] - V[s]| < θ:
   V[s] ← max_a  Σ_{s'} P(s'|s,a) [r + γ V[s']]
3. Extract policy: π*(s) = argmax_a  Σ_{s'} P(s'|s,a) [r + γ V*(s')]
```

## The key difference from policy evaluation

Policy evaluation uses `Σ_a π(a|s)` — averages over the current policy.
Value iteration uses `max_a` — always picks the best action.

This means there is no explicit policy during the sweep. You're solving the Bellman *optimality* equation directly rather than the expectation equation for a fixed π.

## Why it converges

The Bellman optimality operator is also a contraction with factor γ. Each sweep reduces `||V - V*||_∞` by at least γ. Terminal states again anchor the computation — V[terminal] = 0 is exact — and the optimum propagates outward.

## Policy extraction

The policy is not maintained during iteration. After V has converged, extract π* with one pass of `policy_improvement`:

```
π*(s) = argmax_a Q(s,a) = argmax_a Σ_{s'} P(s'|s,a) [r + γ V*(s')]
```

## Comparison to policy iteration

| | Policy Iteration | Value Iteration |
|---|---|---|
| Inner loop | Full eval to convergence | One sweep |
| Outer iterations needed | Few (~6 for 4×4) | More (~7 for 4×4) |
| Cost per outer iteration | High | Low |
| Implementation complexity | Higher | Lower |

On small grids both take ~7 total sweeps. Value iteration is simpler to implement and generally preferred.

## Pitfall: counting iterations

Value iteration reports sweeps over all states. Policy iteration reports outer loop iterations (policy changes) — each of which hides multiple evaluation sweeps internally. Don't compare the two numbers directly.
