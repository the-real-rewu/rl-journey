---
name: Policy Iteration
description: Find π* by alternating policy evaluation and greedy improvement until the policy stabilises
type: user
---

# Policy Iteration

## The algorithm

```
1. Initialise π arbitrarily (e.g. action 0 everywhere).
2. Loop:
   a. Evaluate:  V^π ← policy_evaluation(π)
   b. Improve:   π'(s) = argmax_a Σ_{s'} P(s'|s,a) [r + γ V^π(s')]
   c. If π' == π: STOP — π is optimal.
   d. π ← π'
```

## The Policy Improvement Theorem

The greedy step guarantees: `V^{π'}(s) ≥ V^π(s)` for all s.

Why: if the greedy action differs from π(s), it means there's an action with higher Q(s,a) than what π currently takes — so switching to it strictly improves the value at s, and (by induction through the Bellman equation) everywhere reachable from s.

## Why it terminates in finite steps

There are finitely many deterministic policies (|A|^|S| in total). Each iteration either strictly improves V or we've converged. We cannot cycle, so we must terminate. In practice this happens in very few outer iterations even for large grids.

## Cost

The inner loop (policy evaluation) runs to full convergence before each improvement step. On large state spaces this is expensive. Two alternatives:
- **Truncated policy iteration:** stop evaluation after k sweeps instead of convergence.
- **Value iteration:** take k=1 (one sweep per improvement). This is the extreme case and is usually preferred.

## Common pitfall

Policy iteration counts outer iterations (policy changes), not evaluation sweeps. On a 4×4 grid: ~6 outer iterations, but each one runs 2–7 evaluation sweeps internally.
