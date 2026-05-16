# 2.4 — The DQN Improvements Stack: Dueling, PER, and n-step

You've fixed the **target** (Double DQN) and the **scale** (Huber loss). Three improvements remain from the original DQN-improvements line of work — each addresses a different failure mode of the vanilla algorithm:

| Improvement | What it fixes | Where it lives in the code |
|---|---|---|
| **Dueling DQN** | Wasted capacity when action choice doesn't matter | Network architecture |
| **Prioritized Experience Replay (PER)** | Uniform sampling wastes gradient on solved transitions | Replay buffer |
| **N-step returns** | 1-step bootstrap is slow to propagate reward | TD target computation |

These three plus Double DQN are the foundation of the Rainbow paper. We're stacking them all at once. Together they form what the literature later called the "DQN improvements stack" — Rainbow added distributional RL and noisy nets on top.

The thread of this chapter: **each improvement addresses an *orthogonal* failure mode**, which is why they compose cleanly. If you understand why they don't interfere, you also understand exactly what each one is doing.

---

## 1. Dueling DQN — Better Representation

### 1.1 The problem: most timesteps, action choice doesn't matter

Think about a Breakout episode. You're tracking the ball mid-flight, several frames from the next paddle interaction. The agent has three actions (NOOP, LEFT, RIGHT). The value of this state is high — there's a ball in play, a brick wall to demolish — but the *difference* between actions is essentially zero. Whether you move left, right, or do nothing right now doesn't meaningfully change the outcome.

Vanilla DQN spends its entire output layer learning `Q(s, a)` for all three actions independently. That's three numbers it has to keep approximately equal, every gradient step, just so the policy doesn't get distracted.

Wasted capacity.

### 1.2 The decomposition

Any Q-function can be split:

```
Q(s, a) = V(s) + A(s, a)
```

- `V(s)` — the value of being in state `s`, independent of action
- `A(s, a)` — the **advantage** of action `a` over the baseline at `s`

The advantage tells you specifically how each action differs from the average behavior at this state. If the actions don't matter, `A(s, a) ≈ 0` for every `a`, and `Q(s, a) ≈ V(s)`.

Dueling DQN bakes this split into the network architecture:

```
                 ┌────────┐
                 │ shared │
   state ──────► │  trunk │
                 └────┬───┘
                      ├──────► V-head     ──► V(s)        [shape (batch, 1)]
                      └──────► A-head     ──► A(s, ·)     [shape (batch, num_actions)]

                                  ┌─ combine ─┐
                                  ▼           ▼
                              Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
```

The output shape is still `(batch, num_actions)` — the only changed component is how the head computes it.

### 1.3 Why this actually helps — the gradient-flow mechanism

"Most of the time the action doesn't matter" is the *motivation*. The *mechanism* is about how gradient signal reaches the shared baseline.

In vanilla DQN, the output layer produces `num_actions` numbers that are **independently parameterized**. The weights that produce the output for action `a₁` are separate from the weights that produce the output for action `a₂`. When you train on a transition `(s, a, r, s')`, the loss is `huber(target − Q(s, a))`. Gradient flows back into the weights that produced the output for **the action that was actually taken**. The output cells for the other actions appeared nowhere in the loss for this step — they get zero gradient signal.

So if `V(s) = 50` and you'd ideally want all three of `Q(s, a₁)`, `Q(s, a₂)`, `Q(s, a₃)` to learn the value `~50`, vanilla DQN has to discover "they all equal 50" by independently regressing each one — only one of the three gets updated per training step touching state `s`. Three parallel regressions, each updated `1/num_actions` of the time, all trying to converge to the same target.

In Dueling, `V(s)` sits on the gradient path of **every** action's output:

```
Q(s, a) = V(s) + A(s, a) − mean_a' A(s, a')
       └────┘   └─────┘    └──────────────┘
       updates  updates    updates (touches A for ALL actions)
```

When you backpropagate `huber(target − Q(s, a))`, the chain rule sends gradient into `V(s)` regardless of which action was taken. The shared baseline gets updated on **every** transition touching `s`, not just transitions where action `a` happened to be sampled. The part of the value function that's shared across actions learns `~num_actions ×` faster.

The variance-reduction framing — same idea, statistician's language: instead of regressing `Q(s, a)` directly (typical scale 0–100), you regress the bulk `V(s)` and a small correction `A(s, a)` separately. The correction has a much smaller dynamic range, so its estimator has lower variance for the same number of samples. **Exactly the baseline-subtraction argument from REINFORCE** — Phase 3 will redo this in the policy-gradient setting and reach the same conclusion.

This is *why* the architectural prior pays off: it routes gradient signal to the right place. The "most timesteps the action doesn't matter" claim translates into "the shared baseline `V` is doing most of the work, so accelerating its learning accelerates the whole thing."

### 1.4 Why we have to anchor `A` at all

The decomposition `Q(s, a) = V(s) + A(s, a)` is **underdetermined**. Add any constant `c` to `V`, subtract `c` from every `A` — same `Q`. Infinite valid `(V, A)` decompositions produce the same function.

This sounds harmless. It isn't. Here's the worst case: the network learns `V(s) = 0` everywhere, and the A-head absorbs all the Q-value information. This is a valid solution to the loss — `Q(s, a) = 0 + Q(s, a)` is consistent. But the V-head has learned nothing useful; the A-head is doing exactly the same job as a vanilla DQN output layer; the gradient-flow benefit from Section 1.3 is gone. The architectural prior collapsed back to vanilla DQN with extra parameters.

Nothing in the loss tells the network *which* split to pick. The two heads can drift freely as long as their sum is correct. We have to add a constraint that **forces** `V` to mean "state value" and `A` to mean "action-relative correction." That constraint is what anchoring does.

Three reasonable anchorings, all of which make the split well-defined:

```
A must be 0 at argmax_a  →  V(s) = max_a Q(s, a)        ← "value of acting optimally"
A must be 0 on average   →  V(s) = mean_a Q(s, a)       ← "average value over actions"
A must be 0 at action 0  →  V(s) = Q(s, a₀)             ← "value of taking action 0"
```

All three force `V` to learn a specific, meaningful quantity. All three make `(V, A)` unique. The choice between them is about which interpretation is useful and — more importantly — which has a friendly loss landscape.

### 1.5 Why the mean and not the max

This is Exercise 1, but worth tracing here so the choice doesn't feel arbitrary.

Both mean- and max-subtraction make `Q` identifiable and preserve `argmax_a Q(s, a)`. The argument for mean is gradient stability. Concretely:

Suppose your 2-action network is mid-training. Current estimates: `A(a₁) = +0.1`, `A(a₂) = −0.1`. The argmax is `a₁`.

**Under max-subtraction**, `V(s) = max_a Q(s, a) = Q(s, a₁)`. `V`'s interpretation is "the Q-value of action 1."

One small gradient step later, suppose the new estimates are `A(a₁) = −0.05`, `A(a₂) = +0.05`. The argmax just **flipped to `a₂`**. Now `V(s) = Q(s, a₂)`. `V`'s interpretation discontinuously jumped from "Q of action 1" to "Q of action 2." Viewed as a function `V(θ)` of network parameters, it has a **kink** at exactly the parameter values where the argmax flips. The derivative is discontinuous there.

**Under mean-subtraction**, `V(s) = mean_a Q(s, a) = (Q(a₁) + Q(a₂)) / 2`, always. The argmax flip changes nothing about how `V` is computed. `V(θ)` is a smooth function across the entire parameter space.

Smooth gradients → SGD behaves predictably. Kinks → SGD bounces and trains badly near states where multiple actions have similar values. And those near-tie states are *exactly where you most want clean gradient signal*, because they're the states where the policy decision is most sensitive to estimation error.

So mean-subtraction isn't more theoretically correct than max-subtraction — both define valid Q-functions over the same function class. It's **numerically friendlier to optimize**. Same flavor of argument as Huber-over-MSE: both work in principle, one has a dramatically better loss landscape.

### 1.6 When the prior matches and when it doesn't

The architectural prior says: *most of `Q(s, a)`'s variance across actions is shared between actions* (that's `V`), and *only a small per-action correction is needed* (that's `A`). When this prior matches the environment — Breakout being a great example — the gradient-flow advantage from Section 1.3 translates directly into faster learning.

When the prior doesn't match — every action genuinely has a very different value at every state — Dueling is roughly neutral. Not better, not catastrophically worse. The mean-anchoring still couples the heads, so the network can fall back to encoding the bulk of the variance in `A` if it needs to. You get the parameter overhead of two heads without the gradient-flow speedup.

### 1.7 Bridge to Phase 3

`V(s)` and `A(s, a)` are exactly the quantities that show up in actor-critic methods. Phase 3 starts with REINFORCE, then adds a "baseline" `V(s)` to reduce variance — at which point you'll find yourself computing `A(s, a) = Q(s, a) − V(s)` and calling it the *advantage*. Same equation, same intuition. You're meeting it here first as an architectural choice; you'll meet it next as a variance-reduction technique.

---

## 2. Prioritized Experience Replay — Better Samples

### 2.1 The problem: uniform sampling wastes gradient signal

The vanilla replay buffer samples transitions uniformly. A transition where the network is already perfectly correct (`δ = Q_predicted − target = 0`) gets sampled just as often as a transition where the network is wildly wrong (`δ = 5`).

The mechanism this wastes is gradient magnitude. The gradient of the Huber loss with respect to network parameters is:

```
∇_θ huber(δ) ≈ δ · ∇_θ Q_θ(s, a)        (for |δ| < 1, the quadratic regime)
∇_θ huber(δ) ≈ sign(δ) · ∇_θ Q_θ(s, a)  (for |δ| ≥ 1, the linear regime)
```

A batch slot occupied by a `δ = 0` transition contributes essentially zero gradient. The optimizer step is *as if that slot weren't in the batch at all*. Uniform sampling fills the batch with whatever's in the buffer, regardless of how much gradient each transition will produce. With a buffer of 100k transitions and a batch of 1024, on Breakout most of those 1024 slots are spent on transitions the network has already correctly fit — pure compute waste.

PER's pitch: spend each batch slot on a transition that will actually move the parameters.

### 2.2 The fix: sample with priority proportional to TD-error magnitude

Define a priority for each transition:

```
p_i = (|δ_i| + ε)^α
```

- `δ_i` — the latest TD-error of transition `i`
- `α ∈ [0, 1]` — how much to prioritize. `α=0` recovers uniform sampling; `α=1` is "pure priority"
- `ε` — a small constant ensuring no transition has zero probability of being sampled

Then sample transition `i` with probability:

```
P(i) = p_i / sum_j p_j
```

The Schaul et al. (2016) paper recommends `α ≈ 0.6`.

### 2.3 But now sampling is biased — and IS weights fix it

The vanilla DQN loss is the *uniform-sample expectation*:

```
L(θ) = (1/N) · Σ_i huber(δ_i)        ← what we actually want to minimize
```

When you sample uniformly with `1/N` probability, the empirical average over a batch is an unbiased estimate of this. When you sample with priority `P(i)`, the empirical average estimates a **different** quantity:

```
E_{i ~ P}[huber(δ_i)] = Σ_i P(i) · huber(δ_i)        ← weighted, not uniform
```

Gradient descent on this biased estimate converges to a *different minimum* — the one that fits the high-priority transitions best, not the buffer as a whole.

The fix is standard importance sampling. For any sample drawn with probability `P(i)`, reweight the loss by `1/(N · P(i))`:

```
E_{i ~ P}[ (1/(N·P(i))) · huber(δ_i) ] = Σ_i P(i) · (1/(N·P(i))) · huber(δ_i)
                                       = (1/N) · Σ_i huber(δ_i)        ← uniform again
```

The reweighting algebraically cancels the priority weighting in expectation. We've recovered the uniform-sample gradient while only spending compute on high-priority transitions.

In practice, the IS weight is raised to a power `β ∈ [0, 1]`:

```
w_i = (1 / (N · P(i)))^β
```

`β = 1` gives the fully unbiased reweighting derived above. `β = 0` gives `w_i = 1` (no correction; you minimize the biased objective). Intermediate `β` is a partial correction.

Why would you ever choose `β < 1`? **Variance.** Rare transitions (low `P(i)`) get huge IS weights `(1/(N·P(i)))^1`. A single rare sample with a large weight can dominate the batch gradient — high variance per gradient step, even though the expectation is unbiased. Annealing `β` from low to high trades early-training variance reduction for late-training unbiasedness. Section 2.4 makes this concrete.

For numerical stability, the IS weights in a batch are normalized so the largest weight equals 1. This rescales the loss but doesn't change the gradient *direction* — and gradient direction is the only thing the optimizer cares about.

### 2.4 Why anneal β from 0.4 to 1.0

The bias-variance trade from Section 2.3 plays out differently across training:

**Early training.** The network is far from optimal. TD targets are themselves noisy and biased (the bootstrap term is wrong). The gradient signal you're trying to estimate is itself an approximation. Adding *more* variance from full IS correction (`β = 1`, with its huge weights on rare samples) on top of an already-noisy gradient slows everything down. Some sampling bias toward high-error transitions is actually *aligned* with where you want to push the network — toward the transitions it gets most wrong. So early on, low `β` ≈ "trust the priority signal, don't over-correct."

**Late training.** The policy is near-optimal, TD targets are accurate, and the remaining job is to converge the last mile of value estimates. Now the residual sampling bias actually matters — converging to the priority-weighted minimum instead of the uniform minimum is the difference between "Q-values that are correct" and "Q-values that are correct *on high-error transitions and wrong elsewhere*." Full IS correction (`β = 1`) becomes necessary, and the variance cost is acceptable because the gradient signal is otherwise stable.

The linear anneal from `β=0.4` (paper default) to `β=1.0` reflects this: shift the bias-variance trade from "variance dominates" to "bias dominates" as training progresses. Exercise 4 walks through the failure modes of fixing `β` at either endpoint.

### 2.5 The role of ε

The `ε` in `p_i = (|δ_i| + ε)^α` looks like a numerical hygiene constant. It's not. It's about **coverage**. Without it, a transition whose current TD-error happens to be zero gets priority `0` — and is **never sampled again**, no matter how stale that "zero" estimate becomes. The constant `ε` keeps a small but nonzero probability of revisiting every transition, so the buffer can catch silent drift. Exercise 6 makes this concrete.

### 2.6 Stale priorities — PER as a tracking problem

A transition's stored priority reflects its TD-error *at the moment its priority was last updated*. Between updates, every gradient step changes the network's parameters, which changes the Q-value the network would predict for that transition right now, which changes what the transition's *real* TD-error would be if we recomputed it.

But we don't recompute it. The priority just sits there, stale, until the transition is sampled again. The mechanism is asymmetric in a way that matters:

- A transition stored with high priority gets sampled frequently → its priority is refreshed frequently → staleness is small.
- A transition stored with low priority gets sampled rarely → its priority is refreshed rarely → staleness can grow arbitrarily large.

So PER is biased toward keeping its own previous classification correct. A transition that becomes high-error after being mis-classified as low-error will *stay* mis-classified for many gradient steps before it's resampled. Exercise 5 explores how large that gap can get and what sampling pattern produces the worst staleness.

The fundamental compromise: tracking every transition's current TD-error in real-time would require a forward pass on the entire buffer every gradient step — completely impractical at 100k transitions. PER's whole approach is "accept stale priorities; the resampling cycle catches up eventually." This is *why* the `ε` coverage floor from Section 2.5 matters so much — without it, transitions that drift downward in priority can become permanently unreachable, and the resampling cycle never catches them.

---

## 3. N-step Returns — Better Targets

### 3.1 1-step bootstrap is slow — the chain-depth mechanism

The vanilla DQN target uses a single environment step plus a bootstrap:

```
target = r_t + γ · max_a Q_target(s_{t+1}, a)
```

Notice what this update can and cannot do. For the transition `(s_t, a_t, r_t, s_{t+1})`, it nudges `Q(s_t, a_t)` toward `r_t + γ · max_a Q_target(s_{t+1}, a)`. If `r_t = 0` and `Q_target(s_{t+1}, ·) = 0` (because the network hasn't yet learned anything about `s_{t+1}`), the target is `0`. So this transition contributes no signal.

Now imagine a Breakout episode where the reward is earned at time `t+50` (a brick is hit). For that reward to influence `Q(s_t, a_t)`, the credit has to propagate through the chain of bootstrapping targets:

```
gradient round 1:  Q(s_{t+49})  gets nudged toward  r_{t+49}              (real reward, ≠ 0)
gradient round 2:  Q(s_{t+48})  gets nudged toward  γ · Q(s_{t+49})       (now ≠ 0)
gradient round 3:  Q(s_{t+47})  gets nudged toward  γ · Q(s_{t+48})
...
gradient round 50: Q(s_t)       gets nudged toward  γ · Q(s_{t+1})
```

Each round of propagation requires that the transition for the *previous* step has been sampled from the buffer and its Q-value already updated. Reward at `t+50` needs **50 sequential rounds of gradient updates** to reach `Q(s_t)`. In practice these rounds overlap (different transitions are sampled in parallel from the buffer), but the dependency chain is real: information moves one bootstrap-link per gradient update on the corresponding transition.

This is the "credit assignment depth" — how many sequential bootstrap-hops separate a reward from a target state. 1-step DQN sets this depth equal to the temporal distance. With sparse rewards on long horizons, that's painfully slow.

N-step shortens the chain by `n`. With `n = 3`, reward at `t+50` needs `~50/3 = 17` rounds of propagation instead of `50`. Same sample efficiency improvement that motivated TD(λ) and eligibility traces in Phase 1.3 — now in the deep-RL setting.

### 3.2 The n-step target

Use `n` real environment rewards before bootstrapping:

```
R_t^{(n)} = r_t + γ · r_{t+1} + γ² · r_{t+2} + … + γ^{n-1} · r_{t+n-1}
target    = R_t^{(n)} + γ^n · max_a Q_target(s_{t+n}, a)
```

Now a reward at `t+50` can reach `Q(s_t)` after `50/n` gradient updates instead of `50`.

You've seen this exact spectrum before — Phase 1.3's MC ↔ TD comparison. `n=1` is vanilla DQN. `n=∞` (full episode return) is Monte Carlo. Everything in between trades **bias** (long bootstraps are less biased by bad `Q_target`) against **variance** (long real-reward sums accumulate more environment stochasticity).

The Rainbow paper finds `n=3` is roughly the sweet spot on Atari. Exercise 7 walks you through the tradeoff on a small chain MDP.

### 3.3 The off-policy correctness issue

Here's the catch nobody likes to advertise: **n-step Q-learning is not actually correct off-policy.**

The reasoning: the `n-1` intermediate actions `a_{t+1}, …, a_{t+n-1}` were chosen by the *behavior* policy (ε-greedy at that historical moment). But we want to estimate `Q(s_t, a_t)` under the *greedy target policy*. The two policies disagree on roughly `ε` fraction of actions, so the trajectory in your n-step window is contaminated by exploration noise that the greedy policy wouldn't have made.

A fully principled fix is to apply per-step importance-sampling corrections along the window — but the variance is brutal, and in practice nobody does it for small `n`. Exercise 8 explores why we tolerate the bias.

### 3.4 Crossing episode boundaries

The n-step accumulator has to be careful at episode termination. If a terminal state appears partway through the window, the bootstrap term must be zeroed and the window length adjusted. This is a real implementation pitfall, handled by your `done` flag computation in Task C.

---

## 4. Why the Three Compose

The headline claim: **Dueling + PER + n-step + Double DQN all stack without interfering.** Each improvement modifies a *different* part of the algorithm:

```
┌─────────────────┬──────────────────────────────────────────┬──────────────────────────────┐
│ Improvement     │ What it changes                          │ Failure mode it fixes         │
├─────────────────┼──────────────────────────────────────────┼──────────────────────────────┤
│ Dueling         │ Network parameterization                 │ Wasted capacity / slow V      │
│ Double DQN      │ Which network selects max in target      │ Max-operator overestimation   │
│ PER (with IS)   │ Sample distribution + per-sample weight  │ Wasted gradient on solved txn │
│ N-step          │ The TD target itself                     │ Slow credit propagation       │
└─────────────────┴──────────────────────────────────────────┴──────────────────────────────┘
```

Each lives in a different layer of the algorithm. They don't share gradients, don't read each other's outputs, and don't make assumptions about what the others are doing. **That's why they compose.**

The right mental model isn't "each improvement reduces bias and they add up." Each improvement is a **fix for a specific, independent bug** in vanilla DQN. The bugs don't interact, so the fixes don't interact. You can apply any subset of fixes; the unfixed bugs are still there but they don't break the fixes you applied.

The interesting case is what happens when a fix is *incomplete* — for instance, PER without its IS weights. PER without IS weights doesn't introduce a new bias that interacts with Double DQN; it simply *leaves the PER sampling bias in place* (Section 2.3). Double DQN was never going to address that bias — it's solving a completely different problem (overestimation in the target). So "PER without IS + Double DQN" isn't two biases stacking on each other; it's one bug fixed (target overestimation) and another bug ignored (sampling bias). The resulting gradient is biased not because the fixes interfere, but because one of the bugs isn't actually being fixed. Exercise 9 walks through the gradient-signal analysis.

---

## 5. Loss Function When You Combine PER and n-step

The training loss with all four improvements active:

```
δ_i = R_t^{(n)} + γ^n · Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a)) − Q_online(s_t, a_t)

loss = (1/B) · Σ_i  w_i · huber(δ_i)
```

Where:
- `R_t^{(n)}` — the n-step real-reward sum (Section 3.2)
- The `argmax`/evaluate split is Double DQN (Section 6 of `02_double_dqn.md`)
- `w_i` is the IS weight from PER (Section 2.3)
- `huber` is the Huber loss (Section 8 of `02_double_dqn.md`)

The per-sample `|δ_i|` from this computation goes back into the buffer's `update_priorities` call, refreshing the stored priorities for the next sampling round.

---

## 6. Hyperparameters You'll Add

| Hyperparameter | Recommended value | What it controls |
|---|---|---|
| `per_alpha` | 0.6 | How aggressively to prioritize (0 = uniform, 1 = pure) |
| `per_beta_start` | 0.4 | Initial IS-correction strength |
| `per_beta_end` | 1.0 | Final IS-correction strength |
| `per_epsilon` | 1e-6 | Coverage floor on priority |
| `n_step` | 3 | Bootstrap horizon |

`β` anneals linearly from `per_beta_start` to `per_beta_end` over the full training run.

---

## 7. Exercises — Pen and Paper

These exercises are not derivations of material in the text. Each one targets a facet the theory text deliberately leaves underspecified. Do them by hand.

### Exercise 1 — Mean vs Max in Dueling: a gradient-stability story

**Setup:** A 1-state 2-action bandit (no transitions, just immediate rewards). True Q-values: `Q*(a₁) = 10`, `Q*(a₂) = 10 + δ`, where `δ` is a small noise term that the network's current estimate doesn't perfectly capture.

**Part A.** Under **max-subtraction**, write `V` as a function of the network's current estimates `Q_θ(a₁)` and `Q_θ(a₂)`. What does it equal? (Hint: use the relation `Q(s, a) = V(s) + A(s, a) − max_a' A(s, a')` together with the fact that `argmax_a Q = argmax_a A`.)

**Part B.** Under **mean-subtraction**, write `V` as a function of `Q_θ(a₁)` and `Q_θ(a₂)`.

**Part C.** Suppose `δ` oscillates between `+0.1` and `−0.1` over consecutive training batches (i.e., the argmax flips back and forth). Sketch `V` as a function of `δ` under both forms. Which form has a discontinuous derivative at `δ = 0`? Why does that matter for SGD?

---

### Exercise 2 — V vs A across a Breakout episode

**Setup:** Imagine one Breakout rally:
1. Ball spawns at the top, drifts down.
2. Ball is mid-flight, between paddle and bricks.
3. Ball approaches the paddle.
4. Ball hits the paddle.
5. Ball ricochets up, hits a brick.

**Question:** For each of these five moments, predict — qualitatively, no numbers — what `V(s)` looks like and what `max_a |A(s, a)|` looks like. Where in the rally is `A` essentially flat? Where does it spike?

**Why this matters:** The Dueling architectural prior pays off in proportion to how much of the episode is spent in the "`A` is flat" regime. Without numbers, predict whether Dueling will help a lot, a little, or barely on Breakout.

---

### Exercise 3 — Construct a counterexample

The text claims Dueling is "roughly neutral" when actions matter at every state. Construct an MDP — as small as you can make it — where Dueling could plausibly **hurt** learning compared to plain DQN. Argue why.

(You're allowed to lose this argument. The point is to think through what assumptions Dueling makes.)

---

### Exercise 4 — Why β anneals from 0.4 to 1.0

**Question:** Suppose you ran PER with `β` fixed at 1.0 from the start (fully corrected for sampling bias). Compared to the standard annealing schedule, what would you expect to happen to learning in the first 100k gradient updates? In the last 100k?

**Part B.** Now run the same thought experiment with `β` fixed at 0.4 throughout training. What goes wrong at the end?

**Part C.** Articulate, in your own words, the principle: when does sampling bias hurt least, and when does it hurt most?

---

### Exercise 5 — Stale priorities

**Setup:** Transition `T` was pushed into the buffer 10,000 gradient updates ago, with TD-error `2.0`, giving priority `(2.0 + ε)^α`. Since then, training has updated parameters that affect the bootstrap target for `T`. If you recomputed `T`'s TD-error right now, it would be `0.05` — the transition is now well-predicted.

**Part A.** What probability is `T` being sampled with right now (relative to a freshly-pushed transition with TD-error `0.05`)? Express the ratio.

**Part B.** What sampling pattern produces the worst staleness — i.e., the largest gap between stored priority and true TD-error? Describe the conditions.

**Part C.** Why is this a *tracking* problem, not just a *sampling* problem? What would it mean to track priorities perfectly, and why don't we?

---

### Exercise 6 — The ε in (|δ| + ε)^α

**Setup:** Suppose `ε = 0` exactly. Transition `T` is pushed with TD-error 0 (the network already predicts it perfectly given the current target).

**Part A.** What is `T`'s probability of being sampled?

**Part B.** Suppose the network's parameters drift over the next 5,000 updates, such that `T`'s true TD-error is now `1.5`. What is `T`'s probability of being sampled? Why?

**Part C.** State the role of `ε` in one sentence. (It's not "numerical stability" — be precise.)

---

### Exercise 7 — N-step bias-variance on a chain MDP

**Setup:** A 7-state random walk. States `A, B, C, D, E, F, G`. States `A` and `G` are terminal. From any non-terminal state, the agent moves left or right with probability 0.5 each. Reward is +1 on entering `G`, 0 everywhere else. γ = 1.

True values are `V*(B) = 1/6, V*(C) = 2/6, …, V*(F) = 5/6`. Starting state is `D`.

**Part A.** Suppose all Q-values are initialized to `0.5`. After one episode that follows the path `D → E → F → G`, compute the 1-step, 3-step, and 5-step targets for the first transition `D → E`. Use γ = 1.

**Part B.** Now suppose the same starting state but a different rollout: `D → C → D → E → F → G`. Compute the same three targets for the first transition `D → C`.

**Part C.** Across many such episodes, which target has the **highest variance**? Which has the **lowest bias** in expectation? Map this to the MC ↔ TD spectrum from Phase 1.3.

---

### Exercise 8 — Why we tolerate n-step off-policy bias

The text claims n-step Q-learning is "technically incorrect off-policy" but everyone does it anyway. Make the bias quantitative:

**Part A.** With ε = 1.0 (fully random), what fraction of the `n=3` window contains actions that the greedy target policy would *not* have chosen?

**Part B.** With ε = 0.05 (typical late-training value), what's the answer?

**Part C.** Now connect this to when n-step matters most. If reward is sparse and you only learn from reward-bearing episodes, do reward-bearing episodes mostly happen at high ε or low ε? Argue why the bias is small at exactly the moments where n-step matters.

---

### Exercise 9 — Compose or stay broken

For each combination below, decide whether the resulting gradient direction is **unbiased** (matches what vanilla DQN with uniform replay would compute in expectation, given the same true Q-function) or **biased** (systematically tilted from that target).

Argue from the gradient signal, not from intuition. For each biased combination, identify exactly *which* bug isn't being fixed.

| Combination | Unbiased? Which bug remains? |
|---|---|
| Dueling + Double DQN | ? |
| Dueling + n-step | ? |
| PER (with IS weights) + Dueling | ? |
| PER **without** IS weights + Double DQN | ? |
| PER (with IS weights) + n-step | ? |
| Double DQN + n-step | ? |

**Part B.** For each combination you identified as "still biased": is the remaining bias a problem in *all* training regimes, or only specific ones (e.g., only early, only late)? Connect this back to the β-annealing argument in Section 2.4.

---

## 8. Code Tasks

The four tasks are independent — each flag (`--use-dueling`, `--use-per`, `--use-nstep N`) can be turned on alone or in combination with the others. The default recipe you'll train is all four (Double DQN + Dueling + PER + n-step=3) under the variant tag `rainbow_lite`.

---

### Task A — Dueling head

Edit [`dqn_agent.py`](dqn_agent.py).

**Contract.** Add a `dueling: bool = False` parameter to `DQNNetwork.__init__`. When `True`, replace the final linear layer with two heads: a V-head producing shape `(batch, 1)`, and an A-head producing shape `(batch, num_actions)`. The forward pass must still return a tensor of shape `(batch, num_actions)` representing `Q(s, a)`. The combine form must satisfy `Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)`.

Also add a `dueling: bool = False` parameter to `DQNAgent.__init__`, threaded through to the network constructor for both the online and target networks.

**Conceptual hint (one):** Wang et al. (2016) §3 explains the choice of mean-subtraction. Read that section before writing the recombine line, and put a one-line comment in your combine code naming the reason.

**Smoke test for this task:** Build the network with `dueling=True`, `num_actions=4`. Feed a single zeroed `(1, 4, 84, 84)` input. Verify the output shape is `(1, 4)`. Then verify that the relation `Q(s, a) − mean_a Q(s, a) = A(s, a) − mean_a A(s, a)` holds — this is the identifiability check, and it must come from the same formula your forward pass uses.

---

### Task B — Prioritized replay buffer

Create a new file `phase2_dqn/prioritized_replay_buffer.py`.

**Contract.**

- `__init__(capacity, device, alpha=0.6, epsilon=1e-6)` — same observation/action/reward types as `ReplayBuffer`. Internally, also store one priority per transition.
- `push(state, action, reward, next_state, done) → None` — store the transition. New transitions enter at the **maximum priority seen so far** (so every new transition is sampled at least once with high probability). If the buffer is empty, use priority `1.0`.
- `sample(batch_size, beta) → (batch, indices, is_weights)` — sample with probability `P(i) ∝ priority_i^α`, where `α` is the constructor argument. Return the batch tensors (same shapes/dtypes as `ReplayBuffer.sample`), the indices used (so you can update priorities later), and IS weights `w_i = (1 / (N · P(i)))^β` normalized so the max weight in the batch is 1.
- `update_priorities(indices, td_errors) → None` — store `(|δ_i| + ε)^α` at the given indices. (Note: store `(|δ| + ε)^α`, not raw `|δ| + ε` — pre-exponentiating once is cheaper than exponentiating every sample.)
- `__len__()` — number of transitions stored, same as `ReplayBuffer`.

**Conceptual hint (one):** A binary sum-tree gives O(log N) sampling. A flat numpy array gives O(N) sampling via cumulative sums plus a `searchsorted`. For a 100k buffer either approach works — the sampling cost is dwarfed by the GPU forward pass. Pick one, and put a comment at the top of the file justifying the choice. (If you pick the array approach: think about what `np.random.choice(p=...)` does and whether you can do better.)

**Smoke test for this task:** Push 100 transitions, each with reward `i` for `i ∈ [0, 100)`. Call `update_priorities` with TD-errors equal to the reward (so `priority_i ∝ i^α + epsilon^α`). Sample 10,000 times with `β = 0`. The empirical sampling frequency of each transition should rank-correlate strongly with its priority. The expected frequency for each `i` must be computed from the same `(|δ_i| + ε)^α / Σ` formula your code uses — do not hardcode a table.

---

### Task C — N-step accumulator

Add an n-step accumulator to your replay-buffer push path. The simplest design: wrap the buffer's `push` with a small `NStepWrapper` class that buffers the last `n` transitions in a `deque` and emits a synthetic n-step transition once the deque is full.

**Contract.**

- `NStepWrapper(buffer, n_step, gamma)` — wraps any buffer with a `push` method matching `ReplayBuffer.push`.
- `push(state, action, reward, next_state, done)` — accumulates one environment-step transition. When the accumulator holds `n` transitions, emit one synthetic transition `(s_t, a_t, R_n, s_{t+n}, done_n)` into the underlying buffer, where `R_n = Σ_{k=0}^{n-1} γ^k · r_{t+k}` and `s_{t+n}` is the next-state of the most recently pushed transition.
- On episode termination (`done=True`): flush the accumulator. Every remaining partial window should produce a shorter synthetic transition ending in `done_n=True`. After flushing, the accumulator is empty and ready for the next episode.
- Pass-through API: `sample`, `update_priorities`, `__len__` etc. forward to the underlying buffer.

**Conceptual hint (one):** What should `done_n` be when the window crosses an episode boundary? Think about what the bootstrap term `γ^n · max_a Q_target(s_{t+n}, a)` *should* compute when the trajectory terminated before step `t+n`. Your answer determines the flag.

**Smoke test for this task:** Construct an `NStepWrapper(buf, n_step=3, gamma=0.99)`. Push five non-terminal transitions with rewards `[1, 0, 0, 0, 1]`. Assert exactly two synthetic transitions have been emitted to the underlying buffer, and the first has `R_n = 1 + 0·γ + 0·γ² = 1.0`. Then push one more transition with `done=True` and reward `1`. Verify the tail produces the correct flushed transitions (lengths and `done_n` flags). All expected values must be computed from the same `R_n` formula your code uses.

---

### Task D — Train-step integration

Edit [`dqn_agent.py`](dqn_agent.py)'s `train_step`.

**Contract.** Accept two new optional arguments: `is_weights` (a `(batch,)` tensor of IS weights from PER; default `None` meaning uniform weights of 1.0) and `n_step` (an integer; default `1`).

When `n_step > 1`, the rewards in the batch are already n-step accumulated rewards `R_n` (the buffer did this). The bootstrap term in the TD target must use `γ^{n_step}` instead of `γ`:

```
target = R_n + γ^n · max_a Q_target(s_{t+n}, a) · (1 − done_n)
```

The loss becomes:

```
loss = (is_weights · huber(td_target − online_prediction)).mean()
```

Use `F.smooth_l1_loss(..., reduction='none')` to get per-sample Huber losses, then multiply by `is_weights` before the mean. Huber, not MSE — non-negotiable, per the divergence we already debugged.

The Double DQN action-selection (online net argmax, target net evaluate) is **unchanged**. Confirm this by tracing the gradient path before writing the new code.

Return value of `train_step` changes: return a tuple `(loss_value, per_sample_abs_td_errors)`. The training loop uses the per-sample TD-errors to call `buffer.update_priorities`.

**Conceptual hint (one):** When `is_weights=None`, your code must still work — the loss should collapse cleanly to the existing Huber-mean computation. Don't write two code paths; write one that handles both. (A default tensor of ones is fine.)

**Smoke test for this task:** Construct a small batch, run `train_step` with `is_weights=None` and `n_step=1`. Verify the returned loss matches what the old `train_step` would have returned (within float tolerance). Then run with non-trivial `is_weights` and confirm the loss changes in the expected direction.

---

### Task E — Wire it all up in `train.py`

Edit [`train.py`](train.py).

**Contract.** Add three new command-line flags: `--use-dueling`, `--use-per`, `--use-nstep N` (integer with default 1; `N=1` is effectively off). These compose freely with the existing `--use-double-dqn` flag.

The variant tag (used in checkpoint filenames and the results JSON) must be derived from the active flags. Suggested format: a slug like `vanilla`, `double_dqn`, `dueling_per_nstep3_double_dqn`, or `rainbow_lite` when all four are on. Pick a slug scheme and document it in a comment at the top of `train.py`.

The training loop must:
- Construct the appropriate buffer: `ReplayBuffer` if `--use-per` is off, `PrioritizedReplayBuffer` if on. Wrap in `NStepWrapper` if `--use-nstep > 1`.
- Anneal `β` linearly from `per_beta_start` to `per_beta_end` over `total_steps`.
- Pass `is_weights` and `n_step` into `train_step`.
- After each training step with PER on, call `buffer.update_priorities(indices, td_errors)` with the returned per-sample TD-errors.

The existing Q-bias tracking (`q_log`, `eval_states`) must continue to work unchanged.

**Conceptual hint (one):** Each independent flag means a 2^4 = 16-cell ablation space. You don't have to run all of them — but the code must support running any of them without further changes. If you find yourself writing an `if use_per and use_nstep and not use_dueling: ...` branch, you've coupled the flags.

---

### Task F — Analysis script

Create `phase2_dqn/compare_runs.py`. Run as `python -m phase2_dqn.compare_runs`.

**Contract.** Read all `results_*.json` files in `checkpoints/`. Produce three PNG plots in a new `plots/` directory (add `plots/` to `.gitignore`):

1. **Episode reward** vs episode number, one line per variant, with a 50-episode moving average.
2. **Mean-max-Q (bias tracker)** vs environment step, one line per variant.
3. **Final-100-episodes summary** — a bar chart of mean episode reward over the last 100 episodes per variant.

The script must work with however many variants happen to exist in `checkpoints/` — don't hardcode variant names. Sort the legend in a stable order.

---

### Task G — Phase 2 retrospective

Create `phase2_dqn/README.md`. Modeled on `phase1_tabular/README.md`. Contents:

- Phase deliverables checklist (2.1 vanilla DQN, 2.2 stabilization, 2.3 Double DQN, 2.4 improvements stack) — mark each complete.
- One paragraph reflecting on what the tabular Q-learning of Phase 1 looks like with all of this bolted on: the same Bellman update, but with neural function approximation, target stabilization, bias correction, sample reweighting, and longer horizons. Same algorithm, vastly more machinery.
- Explicit pointer to Phase 3, naming `V(s)` and `A(s, a)` from Dueling as the actor-critic split.

---

## 9. Skill Cards to Write

After you've implemented and run training, write your skill cards in `skills/`:

- `dueling_dqn.md`
- `prioritized_replay.md`
- `n_step_returns.md`

Your turn to write these in your own words. I'll polish and point out misunderstandings — I will not write them for you.

---

## 10. Key Takeaways

1. **Each improvement is a fix for a different bug in vanilla DQN.** Representation (Dueling), sample distribution (PER), target horizon (n-step), max-bias (Double DQN). The bugs are independent, so the fixes don't interact — that's why they compose.

2. **Dueling routes gradient signal to the shared baseline.** `V(s)` is updated on every transition touching `s`, not just transitions where a specific action was taken. The variance-reduction argument generalizes: same logic motivates baselines in REINFORCE (Phase 3).

3. **Anchoring `A` isn't optional.** Without it, the network can collapse Dueling back to vanilla DQN by sending all the signal to the A-head and zero to the V-head. Mean-subtraction anchors `V = mean_a Q(s, a)` and gives a smooth `V(θ)` (no argmax-flip kinks).

4. **PER is a sampler with two correction layers.** IS weights (`(1/(N·P(i)))^β`) algebraically cancel the priority weighting in expectation, recovering the uniform-sample gradient. The `ε` coverage floor prevents zero-error transitions from becoming permanently unreachable. Both layers exist because the sampler is biased on purpose.

5. **PER is also a tracking problem.** Priorities go stale between resamplings, and the asymmetry (high priorities refresh often, low priorities don't) means classification errors persist.

6. **N-step is credit-assignment depth.** 1-step DQN takes `~50` gradient rounds to propagate reward across 50 steps; n=3 takes `~17`. Same bias-variance tradeoff as Phase 1.3's MC↔TD axis — just rediscovered in deep RL.

7. **Off-policy correctness is a real concern that we ignore most of the time.** When ε is small, the n-step window is approximately on-policy. When n is small, the off-policy bias window is short. We get away with it. Phase 3 will revisit this honestly.

8. **The right composition model isn't "biases cancel" — it's "fixes don't interact."** Each improvement fixes one independent bug. Remove a fix, the corresponding bug comes back, but the *other* fixes still work correctly. Combinations that look like they "stack bias on bias" are actually just one bug left unfixed.

---

Next: work the exercises by hand. Then implement Tasks A–G in order. When you have a `rainbow_lite` training run completed, write the three skill cards in your own words, and we'll polish them together.
