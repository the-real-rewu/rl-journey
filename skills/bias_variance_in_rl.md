---
name: Bias-Variance Tradeoff in RL
description: Why MC is unbiased but noisy and TD is biased but stable — and the n-step spectrum connecting them
type: concept
---

# Bias-Variance Tradeoff in RL

## The source of the tradeoff

You want to estimate V^π(s) = E[G_t | s_t = s], but you only have finite trajectories. Every estimate you form is based on sampled experience. That immediately introduces a tradeoff: how much do you trust what you just saw?

## The two extremes

**Monte Carlo — full trajectory, no bootstrapping:**

G_t is an unbiased sample of V^π(s). But G_t depends on every random transition and reward in the rest of the episode. In a stochastic environment, each trajectory can look very different from the last. More steps to the terminal → more accumulated randomness → higher variance in G_t → you need many episodes before the average stabilizes.

**TD(0) — one-step bootstrap:**

Instead of waiting for the full return, TD uses `r + γ V[s']` as the target. This target depends on only one transition, so its variance is low. But V[s'] is wrong early in training — you're bootstrapping off a biased estimate. That bias is the price of low variance.

## The key distinction: where bias and variance come from

| | Bias | Variance |
|---|---|---|
| MC | None — G_t is the true return | High — accumulates all future randomness |
| TD | Yes — V[s'] is an estimate | Low — one step of noise |

**Bias in TD comes from bootstrapping, not from α.** Smaller α makes learning slower and more stable (less overreaction to noisy samples), but it doesn't change the fundamental bias introduced by using V[s'] as a proxy for the true future. Bias shrinks as V[s'] improves over training.

## The n-step spectrum

There's no sharp line between MC and TD — there's a continuous spectrum:

```
n-step return: G_t^(n) = r_{t+1} + γ r_{t+2} + ... + γ^(n-1) r_{t+n} + γ^n V[s_{t+n}]
```

- n = 1: TD(0). Minimum variance, maximum early bias.
- n = ∞: Monte Carlo. Zero bias, maximum variance.
- n in between: tunable knob. Larger n → less bias, more variance.

TD(λ) weights all n-step returns geometrically with parameter λ ∈ [0,1]. λ=0 is TD(0); λ=1 is MC. GAE (Generalized Advantage Estimation) in PPO is exactly this idea applied to advantage functions.

## Both converge with enough data

With infinite trajectories:
- MC's variance averages out — the sample mean converges to the true expectation.
- TD's bias disappears — V[s'] converges to V^π(s'), making the bootstrap target exact.

The tradeoff is about *sample efficiency*: how quickly do you get a good estimate with limited data? TD usually wins in practice because low variance lets each α-update carry reliable signal. MC's high variance means many episodes cancel out before the mean settles.

## Practical consequence

Almost every modern RL algorithm (Q-Learning, DQN, A2C, PPO, SAC) uses TD-style bootstrapping. The bias is tolerable because it shrinks as training progresses; the variance reduction is essential because you can't run infinite episodes. When you see n-step returns or GAE in deep RL papers, they are tuning this exact tradeoff.
