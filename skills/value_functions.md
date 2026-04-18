---
name: Value Functions (V and Q)
description: V(s) and Q(s,a) — what they measure, how they relate, and why Q is preferred in practice
type: user
---

# Value Functions

A value function answers the question: **"How good is it to be here?"**

## State-value function V^π(s)

Expected total discounted return starting from state `s` and following policy `π` forever:
```
V^π(s) = E_π [ G_t | s_t = s ]
       = E_π [ r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ... | s_t = s ]
```

One number per state. Compact, but not directly actionable without a model.

## Action-value function Q^π(s, a)

Expected total discounted return from state `s`, taking action `a` first, then following π:
```
Q^π(s,a) = E_π [ G_t | s_t = s, a_t = a ]
```

One number per (state, action) pair. Larger table, but directly actionable.

## The identity linking them

```
V^π(s) = Σ_a π(a|s) · Q^π(s,a)
```

V is just Q averaged over actions according to the policy. Under the *optimal* policy:
```
V*(s) = max_a Q*(s,a)
```

## Why Q is preferred in model-free control

To act greedily using V, you need the transition model P(s'|s,a):
```
π(s) = argmax_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]   ← needs P
```

To act greedily using Q, you don't:
```
π(s) = argmax_a Q(s,a)                                         ← no P needed
```

This is the decisive reason: in model-free settings (where P is unknown), Q(s,a) is the quantity worth estimating. SARSA and Q-Learning both learn Q directly for exactly this reason.

## Optimal vs policy-specific

Both V and Q come in two flavors:
- `V^π`, `Q^π` — the value under a specific policy π (used in prediction / policy evaluation)
- `V*`, `Q*` — the value under the optimal policy (used in control)

The Bellman expectation equation solves for V^π / Q^π.
The Bellman optimality equation solves for V* / Q*.
