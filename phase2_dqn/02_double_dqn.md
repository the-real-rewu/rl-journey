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
- True Q*(s_hit_paddle, a_move_left) = 10
- Network Q_θ(s_hit_paddle, a_move_left) = 15 (overestimated by 5)

**Transition sequence (moving backward in time):**
- Time t: agent at s_setup, takes a_move, reward r=0, lands in s_hit_paddle (not terminal)
- Time t-1: agent at s_start_sequence, takes a_setup, reward r=0, lands in s_setup (not terminal)

**Part A:** Using vanilla DQN with γ=0.99, compute the TD target for (s_setup, a_move).

```
target = ?
```

**Part B:** Now compute the TD target for (s_start_sequence, a_setup), assuming Q_target(s_setup, a_move) has converged to your answer from Part A.

```
target = ?
```

**Part C:** By how much has the error compounded through these two steps?

---

### Exercise 2 — Double DQN vs Vanilla: Same Data

**Setup:**

True values:
- Q*(s', a₁) = 10, Q*(s', a₂) = 8

**Online network (fresh):**
- Q_online(s', a₁) = 12, Q_online(s', a₂) = 9

**Target network (stale):**
- Q_target(s', a₁) = 9, Q_target(s', a₂) = 8

You're computing the TD target for transition: (s, a, r=0, s').

**Part A:** Using vanilla DQN, what is the target?

```
target_vanilla = ?
```

**Part B:** Using Double DQN, what is the target?

```
target_double = ?
```

**Part C:** Which is closer to truth? Why did they differ?

---

### Exercise 3 — When Does Vanilla Accidentally Succeed?

**Scenario A:**

True values: Q*(s', a₁)=20, Q*(s', a₂)=15, Q*(s', a₃)=10
Network: Q_θ(s', a₁)=25, Q_θ(s', a₂)=18, Q_θ(s', a₃)=12

**Part A:** What action does vanilla max pick? Is it optimal?

**Scenario B:**

True values: Q*(s', a₁)=10, Q*(s', a₂)=15, Q*(s', a₃)=8
Network: Q_θ(s', a₁)=20, Q_θ(s', a₂)=12, Q_θ(s', a₃)=9

**Part B:** What action does vanilla max pick? Is it optimal?

**Part C:** Generalize. Under what *conditions* does vanilla Q-Learning pick the right action despite overestimation?

(Hint: Think about *relative ordering*.)

---

### Exercise 4 — Challenge: Max Q-Values

Suppose you train two agents: vanilla DQN and Double DQN (both with Huber loss).

**Question:** Would you expect Double DQN to have lower *maximum Q-value* estimates than vanilla? Why or why not?

---

## 10. Code Tasks

See [`dqn_agent.py`](dqn_agent.py).

### Task 1 — Implement Double DQN

Modify `DQNAgent.__init__` to accept an optional `use_double_q: bool = False` parameter.

In `train_step()`, when `use_double_q=True`:
- Use the **online network** to SELECT the best next action
- Use the **target network** to EVALUATE that action's Q-value
- When `use_double_q=False`, use target network for both (vanilla DQN)

**Hint:** Your current code computes `max_next_state_q` by taking the max of Q_target(s'). Change this to:
1. Compute Q_online(s') to find the best action
2. Use that action index to index into Q_target(s')

This is a **3-line change** in `train_step()`.

### Task 2 — Train Both Variants

Update [`train.py`](train.py):
1. Train an agent with `use_double_q=False` (vanilla, baseline)
2. Train an agent with `use_double_q=True` (Double DQN)
3. Run eval on both and compare learning curves

Use the same seed and hyperparameters for both. The only difference should be this flag.

### Task 3 — Compare Performance

Generate a plot showing both agents' episode rewards over time:
- X-axis: training step
- Y-axis: mean episode reward (smoothed, e.g., moving average)
- Two lines: vanilla vs Double DQN

**Questions to answer:**
- Does Double DQN converge faster?
- Does it reach a higher final score?
- Is training more stable (fewer fluctuations)?

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
