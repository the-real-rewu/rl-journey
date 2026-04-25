---
name: Monte Carlo Prediction
description: How to estimate V^π from experience alone — average actual returns across trajectories, no transition model needed
type: concept
---

# Monte Carlo Prediction

**What it solves:** Estimate V^π for a fixed policy when you don't know the transition model P(s'|s,a). Instead of computing expectations analytically, you sample them.

## Core idea

The Bellman equation says:

```
V^π(s) = E_π[G_t | s_t = s]
```

You can't compute this expectation without P, but the law of large numbers says: average enough samples of G_t and the sample mean converges to the true expectation. Monte Carlo takes this literally — run trajectories, observe actual returns, average them.

```
V^π(s) ≈ (1/N) Σ G_t   over all visits to s across N episodes
```

## The update rule

Storing all returns and averaging at the end is wasteful. The equivalent incremental update:

```
V[s] ← V[s] + (1/N(s)) * (G_t - V[s])
```

where N(s) is the visit count. This is a running mean — each new G_t nudges V[s] toward the observed return, with diminishing step size as confidence grows.

## First-visit vs every-visit

A state may appear multiple times in one episode.

- **First-visit MC:** only count the first occurrence per episode. Gives unbiased estimates with cleaner theoretical analysis.
- **Every-visit MC:** count every occurrence. Slightly faster convergence in practice.

Both converge to V^π. Either is fine for stationary problems.

## Exploring starts

Under a deterministic policy in a deterministic environment, a fixed start state produces the same trajectory every episode — most states are never visited. Fix: begin each episode from a randomly chosen non-terminal state. This guarantees every state is visited, at the cost of not reflecting the natural start-state distribution.

In Phase 2 we'll replace exploring starts with ε-greedy exploration, which is more practical.

## Properties

| Property | MC |
|---|---|
| Bias | None — G_t is an unbiased sample of V^π(s) |
| Variance | High — G_t accumulates every random choice in the remaining episode |
| Requires complete episodes | Yes — G_t can't be computed until the episode ends |
| Needs transition model | No |

## What it doesn't do

MC can't update V[s] until the episode ends. It can't handle continuing tasks (no episode boundary). And in stochastic environments with long episodes, returns have high variance — you need many episodes to average it down. TD learning fixes these at the cost of introducing bias.
