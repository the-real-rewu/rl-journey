# 2.3 — Double DQN: Fixing Overestimation Bias

After Vanilla DQN, you have a working agent that learns to play Breakout. But there's a fundamental problem baked into the Q-Learning algorithm itself: **overestimation bias**. This section exposes it and teaches the fix.

---

## 1. The Problem: Max Bias in Q-Learning

Vanilla Q-Learning uses the **Bellman optimality equation** to compute TD targets:

```
Q(s,a) ← r + γ max_a' Q(s', a')
```

This is mathematically sound when Q is the *true* optimal Q-function. But in practice, your network's estimates are **noisy**.

Here's the catch: **the max operator is biased when applied to noisy estimates.**

---

## 2. Understanding the Bias Through Example

Suppose the true optimal values at state s' are:
```
Q*(s', a₁) = 10
Q*(s', a₂) = 8
Q*(s', a₃) = 7
```

The optimal action is a₁ with value 10.

But your network estimates (due to noise from limited data and function approximation):
```
Q_θ(s', a₁) = 12  ← overestimated (true: 10)
Q_θ(s', a₂) = 9   ← slightly overestimated
Q_θ(s', a₃) = 8   ← fairly accurate
```

When you compute `max_a' Q_θ(s', a')`, you get **12**, not 10.

The TD target becomes:
```
target = r + γ × 12
```

But the true TD target should have been:
```
true_target = r + γ × 10
```

You've inflated the target by `γ × 2`. This error **propagates backward** through your episode.

---

## 3. Why the Max Amplifies Noise

The max operator always picks the *largest* value. If all estimates are noisy (some too high, some too low), the max tends to pick an overestimated one.

**Analogy:** If you flip a coin 3 times and keep the maximum number of heads you see, you're more likely to get a number higher than the expected 1.5 heads.

**In RL terms:**
- Actions that happen to be overestimated on this batch will be picked by the max.
- Once picked, they set the target for *future* transitions.
- If that overestimate was noise, you've locked in a wrong target.
- Over 200 steps in an episode, these errors compound.

---

## 4. Does It Always Matter?

No, and this is subtle.

**Case 1: Symmetric noise → it hurts**

If overestimation errors are roughly as common as underestimation, the max operator systematically biases upward. Compounded over many transitions.

**Case 2: Consistent overestimation → it's worse**

If your network has a systematic tendency to overestimate, the bias is consistent and harmful.

**Case 3: When it doesn't matter much**

If all Q-values are overestimated by roughly the *same* amount, the max still picks the right action (relative ordering is preserved). The bias exists but the policy might still be correct.

Example:
```
True values:     Q*(s', a₁)=10, Q*(s', a₂)=8
Network says:    Q_θ(s', a₁)=15, Q_θ(s', a₂)=13  ← both +5

max_a' Q_θ(s', a') picks a₁ ✓ (correct)
But the target is inflated by γ × 5.
```

The *ranking* is preserved but the *scale* is wrong.

---

## 5. The Compounding Effect

Here's what makes overestimation dangerous in DQN:

1. Step t: Network overestimates Q(s_t, a), generates inflated target for s_{t-1}
2. Step t+1: Uses s_t to bootstrap the target for s_{t-2}
3. But s_t's estimate is already inflated from step t
4. This feeds forward through the episode

Over a 200-step episode, compounding errors significantly degrade value estimates.

---

## 6. Double DQN: The Fix

The key insight: **Separate action selection from action evaluation.**

Vanilla Q-Learning uses the *same* network (the target network) for both:
```
Selection:  arg max_a' Q_target(s', a')  ← picks the action
Evaluation: Q_target(s', arg max)         ← evaluates its value
```

If the max picks an overestimated action, we evaluate it with the same overestimated network. Bias is confirmed.

Double DQN decouples these:
```
Selection:  arg max_a' Q_online(s', a')   ← use ONLINE network
Evaluation: Q_target(s', arg max)         ← use TARGET network
```

**Why does this help?**

The online and target networks are trained on the same data but updated at different frequencies. If one accidentally overestimates an action, there's a decent chance the other doesn't overestimate it by the same amount.

**Concrete TD target:**

**Vanilla DQN:**
```python
best_action = argmax_a Q_target(s', a)
target = r + γ × Q_target(s', best_action)
```

**Double DQN:**
```python
best_action = argmax_a Q_online(s', a)    # different network for selection
target = r + γ × Q_target(s', best_action)
```

The two networks provide a "sanity check" on each other.

---

## 7. A Subtle Point: Underestimation

Double DQN can sometimes *reduce* underestimation bias. If the target network is consistently underestimating, using the online network for selection can help.

This is why Double DQN is sometimes called **reduced bias**, not **no bias**. It trades one bias for a (hopefully) smaller one.

---

## 8. Double DQN + Huber Loss

You just fixed Huber loss, which addresses a different stability problem:

- **Huber loss:** Prevents Q-values from *diverging unboundedly*. Caps large gradients.
- **Double DQN:** Prevents Q-values from *systematic bias*. Decouples networks.

Together, they're synergistic:
- Huber keeps the scale reasonable (prevents explosion)
- Double DQN keeps the errors honest (prevents bias)

---

## 9. Exercises — Pen and Paper

Work through these by hand. The goal is to understand the mechanism deeply.

### Exercise 1 — Overestimation Compounds Across Steps

**Setup:**

Your network has overestimated:
- True optimal value Q*(s_hit_paddle) = max_a' Q*(s_hit_paddle, a') = 10
- Network estimate Q_θ(s_hit_paddle) = 15 (overestimated by 5)
  - (This is the max Q-value among all actions at s_hit_paddle)

**Transition sequence (moving backward in time):**
- Time t: agent at s_setup, takes some action, reward r=0, lands in s_hit_paddle (not terminal)
- Time t-1: agent at s_start_sequence, takes some action, reward r=0, lands in s_setup (not terminal)

**Part A:** Using vanilla DQN with γ=0.99, compute the TD target for the transition into s_hit_paddle.

The vanilla DQN TD target is: `target = r + γ × max_a' Q_target(s_hit_paddle, a')`

```
target = ?
```

**Part B:** Now compute the TD target for the transition into s_setup, assuming Q_target(s_setup) has converged to your answer from Part A.

```
target = ?
```

**Part C:** The true value should have been Q*(s_setup) = 0 + 0.99 × 10 = 9.9. By how much has the estimate diverged from truth after two steps?

---

### Exercise 2 — The Uniform Overestimation Edge Case

**Setup:**

True optimal values at s': Q*(s', a₁)=10, Q*(s', a₂)=8

Your network estimates (both actions overestimated by exactly 3):
- Q_θ(s', a₁) = 13
- Q_θ(s', a₂) = 11

γ = 0.99, r = 0.

**Part A:** What is the vanilla DQN target?

```
target_vanilla = ?
```

**Part B:** What *should* the target be (using true values)?

```
target_true = ?
```

**Part C:** Vanilla's target is wrong by how much? Now here's the key insight: despite this error in the *absolute value*, vanilla DQN still learns the correct *policy* (choosing a₁). Why? Under what general condition does vanilla DQN get the policy right even when values are inflated?

(Hint: What matters for policy — the absolute value, or the relative ordering?)

---

### Exercise 3 — When Do Networks Disagree (and Does It Matter)?

**Setup:**

True optimal value at s': max_a' Q*(s', a') = 10

**Two scenarios:**

**Scenario A: Networks disagree on best action**

Online network: Q_online(s', a₁) = 15, Q_online(s', a₂) = 9 ← online thinks a₁ is best
Target network: Q_target(s', a₁) = 8, Q_target(s', a₂) = 11 ← target thinks a₂ is best

**Scenario B: Networks agree on best action**

Online network: Q_online(s', a₁) = 15, Q_online(s', a₂) = 9 ← thinks a₁ is best
Target network: Q_target(s', a₁) = 11, Q_target(s', a₂) = 8 ← also thinks a₁ is best

**Question:** Compute the vanilla DQN and Double DQN targets for both scenarios (γ=0.99, r=0). In which scenario does Double DQN provide an actual benefit? Why?

```
Scenario A:
  target_vanilla = ?
  target_double = ?

Scenario B:
  target_vanilla = ?
  target_double = ?
```

**Part C:** What does this tell you about when Double DQN helps vs when it's neutral?

---

### Exercise 4 — Can Double DQN Overestimate in the Other Direction?

**Setup:**

You have two networks with a persistent bias:
- Online network: consistently **underestimates** all actions by ~2
  - Q_online(s', a₁) = 8, Q_online(s', a₂) = 6 (true values are ~10, ~8)
- Target network: estimates fairly accurately
  - Q_target(s', a₁) = 10, Q_target(s', a₂) = 8

**Question:** Using Double DQN, could your TD target be *systematically lower* than the true value? If so, construct an example. If not, explain why the mechanism prevents it.

(This isn't a trick question—answer honestly. The point is to think through the consequences of decoupling selection and evaluation.)

---

## 10. Code Tasks

### Task 1 — Implement Double DQN in DQNAgent

Edit [`dqn_agent.py`](dqn_agent.py):

**Part A:** Add a `use_double_q: bool = False` parameter to `DQNAgent.__init__`. Store it as `self.use_double_q`.

**Part B:** In the `train_step()` method, modify the line that computes `max_next_state_q`.

Currently it uses the target network for both action selection and evaluation. The change: **decouple them**. When `self.use_double_q=True`, use the online network to decide which action is best, then evaluate that action using the target network. When False, keep the current behavior (target for both).

**Hint:** Look at how you currently gather predictions from the online network in `train_step()` — use a similar pattern to gather from the target network using an action index from the online network.

---

### Task 2 — Run Experiments: Vanilla vs Double DQN

Edit [`dqn_agent.py`](dqn_agent.py) to implement Double DQN from Task 1, then:

**Part A:** Verify that `train.py` can accept `--use-double-q` flag via command line. (It should already be set up.)

**Part B:** Train two agents using the same seed and hyperparameters:

```bash
# Run 1: Vanilla DQN (baseline)
python3 -m phase2_dqn.train

# Run 2: Double DQN (with the flag)
python3 -m phase2_dqn.train --use-double-q
```

Each run will automatically save episode rewards to:
- `checkpoints/results_vanilla.json` (vanilla DQN)
- `checkpoints/results_double_dqn.json` (Double DQN)

---

### Task 3 — Compare Learning Curves

Write a script (`phase2_dqn/plot_comparison.py` or a notebook) that:

1. **Loads episode rewards** from the JSON files created by Task 2:
   - `checkpoints/results_vanilla.json`
   - `checkpoints/results_double_dqn.json`

2. **Plots both on the same graph:**
   - X-axis: episode number
   - Y-axis: episode reward (you can plot raw rewards or 10-episode moving average)
   - Two lines: vanilla vs Double DQN

3. **Computes and reports summary statistics:**
   - Mean reward in episodes 1–25
   - Mean reward in episodes 26–50 (if you have that many)
   - Max episode reward achieved by each variant
   - Does one variant reach higher rewards faster?

**Interpretation:** Compare the curves. Do they differ meaningfully, or are they similar? This tells you whether Double DQN's decoupling helps on this problem at this scale.

---

## 11. Skill Cards to Write

After implementation and comparison, write:

- `double_dqn.md` — the overestimation problem, why it happens, how Double DQN fixes it, when to use it
- (Optional) `overestimation_bias.md` — the general problem of max bias in value-based methods

Your turn to write these in your own words. I'll help polish.

---

## Key Takeaways

1. **Max bias is real:** The max operator picks the largest value among noisy estimates, which tend to be overestimated.

2. **It propagates:** Overestimated targets feed into future transitions, compounding the error.

3. **Double DQN is simple:** One-line fix — use online network for action selection, target network for evaluation.

4. **It's not magic:** Double DQN reduces bias but doesn't eliminate it. Other improvements (Dueling, PER, etc.) address other failure modes.

5. **Combined with Huber:** The two together give you both stability (scale) and correctness (bias).

---

Next: Work the exercises by hand. Come back with your answers.
