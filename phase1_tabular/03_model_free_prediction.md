# 1.3 — Model-Free Prediction

In Phase 1.2 you computed V^π exactly using the model `P(s'|s,a)`.
The world doesn't usually give you that. In every interesting RL problem
— Atari, robotics, Mario — you only get **experience**: trajectories of
`(s, a, r, s')` tuples produced by interacting with the environment.

This section answers: **how do you estimate V^π from experience alone?**

Two foundational answers: **Monte Carlo** and **Temporal-Difference learning**.
The contrast between them — and the bias-variance tradeoff they expose — is
one of the central ideas in RL.

---

## 1. The Setup Has Changed

| | Phase 1.2 (DP) | Phase 1.3 (model-free) |
|--|---------------|-----------------------|
| You have | `P(s'|s,a)`, `R(s,a,s')` | Only what `env.step()` returns |
| Updates use | Expectations over P | Sampled rewards/transitions |
| Cost per state | Sum over all (s', a) | Process whatever you've seen |
| Scales to | Small finite MDPs | Anything you can simulate |

The Bellman equations still hold — but you can no longer evaluate the
expectations directly. You have to **estimate them from samples**.

---

## 2. Monte Carlo Prediction

**Idea:** Run the policy. Wait for the episode to terminate. Use the
*actual* observed return G_t as a sample of V^π(s_t). Average across many
episodes.

```
V^π(s) = E_π[G_t | s_t = s]
       ≈ (1/N) Σ G_t  over visits to s in N episodes
```

This is just the law of large numbers applied to the definition of V.

### First-visit vs Every-visit MC

Within one episode, the same state may appear multiple times.
- **First-visit MC:** count only the first occurrence of s in each episode.
- **Every-visit MC:** count every occurrence.

Both converge to V^π. First-visit gives unbiased estimates with cleaner
analysis; every-visit converges slightly faster in practice. Either is fine.

### The incremental update form

Storing all returns and averaging is wasteful. Equivalent online update:

```
V[s] ← V[s] + (1/N(s)) (G_t - V[s])
```

where N(s) is the number of times we've visited s. This is a running mean.

You can also use a constant step size α instead of 1/N(s):
```
V[s] ← V[s] + α (G_t - V[s])
```

Constant α gives **exponential** averaging — recent returns weigh more.
Useful for non-stationary problems. For stationary policy evaluation either
works.

### Properties

- **Unbiased:** G_t is an unbiased sample of V^π(s_t).
- **High variance:** G_t depends on every random choice in the rest of the
  episode. Long episodes → high variance returns.
- **Requires terminating episodes:** you can't compute G_t until the episode
  ends. (Doesn't work for continuing tasks unless you truncate.)

### Exploring starts

If π is deterministic and the environment is deterministic, a fixed start
state produces the same trajectory every episode — visiting only a tiny
subset of states. To estimate V^π everywhere, the standard fix is
**exploring starts**: each episode begins from a randomly chosen
non-terminal state. Our `mc.py` and `td.py` use this. In Phase 2 we'll
trade exploring starts for ε-greedy exploration, which is more practical.

---

## 3. Temporal-Difference Learning: TD(0)

**Idea:** Don't wait for the episode to end. Use the Bellman expectation
equation as an *update rule* applied to one observed transition:

```
V[s] ← V[s] + α (r + γ V[s'] - V[s])
```

The quantity `δ = r + γ V[s'] - V[s]` is called the **TD error**.

### Where this comes from

The Bellman expectation equation says:
```
V^π(s) = E[r + γ V^π(s') | s, π]
```

We can't compute the expectation, but we can sample it: each observed
(r, s') is one sample of the random variable `r + γ V^π(s')`. So the
TD update nudges V[s] toward this sampled target.

### TD vs MC: same idea, different target

| | Target | Source of randomness |
|---|---|---|
| MC | G_t (full return) | All future rewards in the episode |
| TD(0) | r + γ V[s'] | Just the one immediate transition |

TD's target is a **bootstrapped estimate** — it uses our current V[s']
in the formula, even though that estimate is wrong early in training.

### Properties

- **Biased:** the target uses V[s'], which is wrong (especially early).
- **Low variance:** the target only depends on one transition.
- **Online:** updates happen each step. No need to wait for episode end.
- **Works on continuing tasks:** no episode boundary required.

### TD as sampled DP

Notice the parallel:

```
DP policy evaluation:  V[s] ← Σ_{s',r} P(s',r|s,π(s)) [r + γ V[s']]
TD(0):                 V[s] ← V[s] + α [(r + γ V[s']) - V[s]]
```

DP averages over all possible (s', r) outcomes weighted by P.
TD samples one (s', r) and moves V[s] a bit toward that sampled target.

In the limit of infinite samples and decaying α, TD(0) converges to V^π.
TD is sampled, online, model-free DP.

---

## 4. The Bias-Variance Tradeoff

This is the most important conceptual takeaway of section 1.3.

```
Variance:    MC ████████   |   TD(0) █
Bias:        MC ░░░░░░░░   |   TD(0) ████████ (early)
                                    █████ (mid)
                                    █ (converged)
```

| | MC | TD(0) |
|---|---|---|
| Bias | None — uses true G_t | Yes — uses estimated V[s'] |
| Variance | High — accumulates all future randomness | Low — one-step transition only |
| Speed of learning | Slow (variance dominates early) | Fast (low variance lets α-updates work) |
| Requires episodes? | Yes | No |

**Practical consequence:** TD usually outperforms MC in practice because
its lower variance lets it learn faster, and the bias washes out as
estimates improve. Almost every modern RL algorithm (Q-Learning, SARSA,
DQN, A2C, PPO) uses TD-style bootstrapping for this reason.

This tradeoff doesn't go away — it shows up in deep RL as the choice between
n-step returns, GAE (Generalized Advantage Estimation), and full Monte Carlo
returns. We will return to this.

---

## 5. n-step TD: a spectrum

TD(0) bootstraps after 1 step. MC waits for the full return. There's a
spectrum in between:

```
n-step return: G_t^(n) = r_{t+1} + γ r_{t+2} + ... + γ^(n-1) r_{t+n} + γ^n V[s_{t+n}]
```

- n=1: TD(0)
- n=∞ (or until episode end): MC
- n in between: tradeoff knob

Larger n → less bias, more variance.

Going further, **TD(λ)** weighs n-step returns geometrically using a
parameter λ ∈ [0,1]. λ=0 → TD(0); λ=1 → MC. We won't implement TD(λ)
in this section — it's worth knowing the name and the unifying view.

---

## 6. Pen-and-paper exercises

### Exercise 1 — MC update on one episode

2x2 GridWorld, γ=0.9, V[s]=0 initialized for all s.
Episode trajectory:
```
(0,0) -RIGHT→ r=-1, (0,1) -DOWN→ r=+10, terminal
```

Apply first-visit MC update with α = 1.0 (replace, not average).
What are V[(0,0)] and V[(0,1)] after this episode?

### Exercise 2 — TD(0) update on one step

Same setup, V initialized to 0, α = 0.5.
Apply TD(0) for the first transition:  s=(0,0), r=-1, s'=(0,1).
What is V[(0,0)] after the update? What is V[(0,1)]?

### Exercise 3 — Where MC and TD disagree

Run the same trajectory through both methods step by step.
After **one episode** (with α=0.5 for TD, replace for MC):

| State | MC update | TD update |
|-------|-----------|-----------|
| (0,1) | ? | ? |
| (0,0) | ? | ? |

Notice: TD updates (0,0) using V[(0,1)] = 0 at the time of the update,
even though (0,1)'s "real" value is much higher. This is the bias.
MC updates (0,0) using the actual return, which already accounts for
the +10 received later.

### Exercise 4 — Why does TD still work?

Even though TD's target is biased, it converges to V^π in the long run.
Explain in 2-3 sentences why this is okay (hint: think about how V[s']
changes over many episodes, and how the bias relates to the bootstrapping).

---

## 7. Code tasks

See [`mc.py`](mc.py) and [`td.py`](td.py). Each has one TODO covering the
core update. After implementing both, run [`run_prediction.py`](run_prediction.py)
to compare:

1. The DP ground truth (computed with `policy_evaluation` from 1.2).
2. The MC estimate after N episodes.
3. The TD(0) estimate after N episodes.
4. RMSE of each estimate vs ground truth as N grows.

You should see TD's RMSE drop faster than MC's early on, then both
converge.

**Experiments to try:**
- Fix the seed and watch the same trajectory in both methods.
- Try α = 0.01 vs α = 0.5 for TD — what's the tradeoff?
- Use a stochastic policy (random with prob ε, otherwise greedy) and see
  the variance of MC explode.
- For MC, compare first-visit vs every-visit (a 3-line change).

---

## 8. Skill cards to write

- `monte_carlo_prediction.md` — what it estimates, the update rule, when to use
- `td_learning.md` — TD(0) update, the TD error, why bootstrapping works
- `bias_variance_in_rl.md` — the tradeoff, where it appears, why TD usually wins

---

## 9. Reading

- Sutton & Barto Chapters 5 and 6 — direct overlap with this section.
- David Silver Lectures 4 and 5.
