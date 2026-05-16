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

### 1.3 Why subtract the mean?

`Q(s, a) = V(s) + A(s, a)` is **unidentifiable** as written. You can add any constant `c` to `V` and subtract `c` from every `A`, and `Q` is unchanged. The network would have to choose, somehow, what fraction of `Q` to put in `V` versus `A` — and there's no signal telling it.

So we anchor `A` by forcing it to be zero on average. Two natural choices:

```
Q(s, a) = V(s) + A(s, a) − max_a' A(s, a')     ← max-subtraction
Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)      ← mean-subtraction
```

Both make the decomposition unique. Both preserve `argmax_a Q(s, a)`. The Wang et al. (2016) paper picks **mean-subtraction**, and the reason isn't identifiability — it's gradient stability. Exercise 1 walks you through this.

### 1.4 Why this actually helps

The architectural prior says: *most of `Q(s, a)`'s variance across actions is shared between actions* (that's `V`), and *only a small per-action correction is needed* (that's `A`). When this prior matches the environment — Breakout being a great example — the network learns faster because it doesn't have to redundantly encode `V(s)` `num_actions` times.

When the prior doesn't match — every action genuinely has a very different value — Dueling is roughly neutral. Not better, not catastrophically worse. The mean-subtraction couples the heads enough that the network can still fall back to encoding everything in `A` if it wants.

### 1.5 Bridge to Phase 3

`V(s)` and `A(s, a)` are exactly the quantities that show up in actor-critic methods. Phase 3 starts with REINFORCE, then adds a "baseline" `V(s)` to reduce variance — at which point you'll find yourself computing `A(s, a) = Q(s, a) − V(s)` and calling it the *advantage*. Same equation, same intuition. You're meeting it here first as an architectural choice; you'll meet it next as a variance-reduction technique.

---

## 2. Prioritized Experience Replay — Better Samples

### 2.1 The problem: uniform sampling is dumb

The vanilla replay buffer samples transitions uniformly. So a transition where the network is already perfectly correct (`δ = Q_predicted − target = 0`) gets sampled just as often as a transition where the network is wildly wrong (`δ = 5`).

Gradient descent on the zero-error transition produces zero gradient. It contributes nothing to learning. We are spending compute on it.

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

### 2.3 But now sampling is biased

Uniform sampling produces an *unbiased* estimate of the expected gradient over the buffer. Priority sampling does not — you're systematically overweighting high-error transitions. Gradient descent on a biased sample distribution will converge to the wrong minimum.

The fix is standard importance sampling. Each sampled transition gets an importance-sampling (IS) weight:

```
w_i = (1 / (N · P(i)))^β
```

- `N` — buffer size
- `β ∈ [0, 1]` — how much to correct for the bias

The loss for sample `i` becomes `w_i · huber(δ_i)` instead of just `huber(δ_i)`. The gradient estimate is now unbiased w.r.t. the uniform-sampling gradient — but the *variance* has gone up, because we're using a biased sampler in the first place.

For numerical stability, the IS weights are normalized so the largest weight in a batch is 1.

### 2.4 Why anneal β from 0.4 to 1.0?

This is one of the subtler PER design choices. Early in training, `β` is small (~0.4), meaning you're *only partially correcting* the bias. Late in training, `β = 1.0`, meaning you're fully unbiased.

The justification is in Exercise 4. The short version: early in training, the network is far from optimal anyway, and the biased gradients from PER actually help by directing compute toward the high-error transitions you most need to fix. Late in training, the policy is near-optimal and you need clean unbiased gradients to nail down the last mile. The annealing reflects "what kind of mistake hurts more right now."

### 2.5 The role of ε

The `ε` in `p_i = (|δ_i| + ε)^α` looks like a numerical hygiene constant. It's not. It's about **coverage**. Without it, a transition whose current TD-error happens to be zero gets priority `0` — and is **never sampled again**, no matter how stale that "zero" estimate becomes. The constant `ε` keeps a small but nonzero probability of revisiting every transition, so the buffer can catch silent drift. Exercise 6 makes this concrete.

### 2.6 Stale priorities

A transition's priority reflects its TD-error *at the moment its priority was last updated*. After that, the network keeps training, every parameter shifts a little, and the transition's actual TD-error drifts — but its stored priority doesn't update until it's resampled.

This means PER is a **tracking problem**, not just a sampling problem: the priorities you're sampling from are always slightly out of date. Exercise 5 explores when this stops being a small effect.

---

## 3. N-step Returns — Better Targets

### 3.1 1-step bootstrap is slow

The vanilla DQN target uses a single environment step plus a bootstrap:

```
target = r_t + γ · max_a Q_target(s_{t+1}, a)
```

If a reward is earned at time `t+50`, the only way for that reward to propagate to `Q(s_t, a_t)` is to be relayed through 50 separate gradient updates — each of which slightly improves `Q(s_{t+k})`, which slightly improves `Q(s_{t+k-1})`, and so on. This is **slow**. In Breakout, where the reward is sparse-ish (you only score on brick contact), waiting for credit to crawl backward step-by-step burns sample efficiency.

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
┌─────────────────┬──────────────────────────────────────────┐
│ Improvement     │ What it changes                          │
├─────────────────┼──────────────────────────────────────────┤
│ Dueling         │ Network parameterization                 │
│ Double DQN      │ Which network selects max in target      │
│ PER             │ Sample distribution + per-sample weight  │
│ N-step          │ The TD target itself                     │
└─────────────────┴──────────────────────────────────────────┘
```

Each lives in a different layer of the algorithm. They don't share gradients, don't read each other's outputs, and don't make assumptions about what the other components are doing. That's *why* they compose: orthogonal changes to orthogonal failure modes.

The combinations that *don't* compose cleanly are the ones that touch the same machinery. The classic example: PER without importance-sampling weights, in combination with Double DQN. PER biases sample frequency toward high-error transitions; Double DQN doesn't address that bias (it addresses a different bias on the target). Without IS weights, the two biases stack rather than canceling. Exercise 9 walks through this.

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

### Exercise 9 — Compose-or-cancel

For each combination below, decide whether it composes cleanly (gradient direction is unbiased w.r.t. uniform replay vanilla DQN) or stacks bias on bias. Argue from the *gradient signal*, not from intuition.

| Combination | Composes cleanly? |
|---|---|
| Dueling + Double DQN | ? |
| Dueling + n-step | ? |
| PER (with IS weights) + Dueling | ? |
| PER **without** IS weights + Double DQN | ? |
| PER (with IS weights) + n-step | ? |
| Double DQN + n-step | ? |

For the "stacks bias on bias" cases, identify the bias each component introduces and explain why they don't cancel.

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

1. **Each improvement addresses a different failure mode.** Representation (Dueling), sample distribution (PER), target horizon (n-step). That's why they compose.

2. **Dueling's prior pays off when actions don't matter most of the time.** Breakout fits the prior. So does most of Mario. Most environments do.

3. **PER is a tracking problem.** Priorities go stale. The IS weights and the ε floor are both bias-correction machinery for a sampler that's biased on purpose.

4. **N-step is the Phase 1.3 bias-variance tradeoff, rediscovered.** Same axis. Same answer that 1-step ≠ ∞-step ≠ optimal.

5. **Off-policy correctness is a real concern that we ignore most of the time.** When ε is small, the n-step window is approximately on-policy. When n is small, the off-policy bias window is short. We get away with it. Phase 3 will revisit this honestly.

6. **Composition isn't free, but it's cheaper than you'd guess.** Two improvements that touch different parts of the algorithm stack with no cross-interaction. Two that touch the same machinery (PER without IS + Double DQN) silently corrupt each other's correctness.

---

Next: work the exercises by hand. Then implement Tasks A–G in order. When you have a `rainbow_lite` training run completed, write the three skill cards in your own words, and we'll polish them together.
