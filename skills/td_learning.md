---
name: TD Learning (TD(0))
description: Estimate V^π online by bootstrapping off the next state's value — DP's update rule applied to sampled transitions
type: concept
---

# TD Learning (TD(0))

**What it solves:** The same problem as Monte Carlo — estimate V^π without knowing the transition model — but without waiting for the episode to end.

## Core idea

TD combines two ideas:
- **From Monte Carlo:** learn from experience (sampled transitions, no model needed).
- **From Dynamic Programming:** bootstrap — use your current estimate of V[s'] to update V[s] now, rather than waiting for the full return.

The update after one observed transition (s, r, s'):

```
TD target = r + γ * V[s']
TD error  = TD target - V[s]        (also called δ, the "TD error")
V[s]     ← V[s] + α * δ
```

If s' is terminal, the bootstrap term drops out: target = r (no future).

## Why it works — sinks as anchors

Terminal states have V = 0 by definition (no future reward possible). These are the only states whose values are *certain*. The TD update propagates that certainty outward one step at a time: states adjacent to the terminal get updated toward `r + γ * 0`, then their neighbors get updated toward `r + γ * V[neighbor]`, and so on.

This is identical to how DP policy evaluation works. The difference is that DP averages over all possible (s', r) outcomes using P; TD samples one (s', r) and nudges V[s] toward that single observation. Same mechanism, sampled rather than exact.

## Properties

| Property | TD(0) |
|---|---|
| Bias | Yes — uses estimated V[s'], which is wrong early on |
| Variance | Low — one-step transition is much less noisy than a full return |
| Requires complete episodes | No — updates happen at each step |
| Needs transition model | No |

## The learning rate α

α controls how much weight you put on each new observation.
- **Large α (e.g. 0.5):** fast but noisy — a single bad transition can swing V[s] wildly.
- **Small α (e.g. 0.01):** stable but slow — requires many transitions to move V[s].

For stationary problems, either converges. For non-stationary problems (environment changes over time), a constant α > 0 lets the estimates keep tracking the moving target.

## TD as sampled DP

The structural parallel:

```
DP:    V[s] ← Σ_{s',r} P(s',r|s,π(s)) [r + γ V[s']]     (full expectation)
TD(0): V[s] ← V[s] + α [(r + γ V[s']) - V[s]]            (one sample, incremental)
```

DP is exact but requires the model. TD is approximate but works on anything you can simulate.

## Connection to MC

TD(0) bootstraps after 1 step. MC waits for the full return (infinite steps). The n-step return is the spectrum in between — see `bias_variance_in_rl.md`.
