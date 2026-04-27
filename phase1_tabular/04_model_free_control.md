# 1.4 — Model-Free Control

In Phase 1.3 you learned to estimate V^π or Q^π for a *fixed* policy without a
model. That's prediction. **Control** is the harder problem: find the optimal
policy, also without a model.

The twist: everything you know from DP (GPI, policy improvement, Bellman
optimality) still applies — but you can no longer evaluate expectations
analytically. You have to learn from interaction.

---

## 1. The Gap Between Prediction and Control

After Phase 1.3 you can estimate V^π(s) for any fixed policy. But acting
greedily requires:

```
π'(s) = argmax_a  Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
```

Without the transition model P, you can't compute this sum. **V(s) alone is
not enough to act greedily.**

The fix: estimate Q^π(s,a) instead. Then:

```
π'(s) = argmax_a  Q(s, a)
```

No model needed — you just take the max over actions. This is why model-free
control always learns Q, not V.

---

## 2. GPI With Q-Functions

The GPI skeleton from Phase 1.2 still drives everything:

```
Evaluate: estimate Q^π for current π
Improve:  π'(s) = argmax_a Q(s, a)   (greedy w.r.t. Q)
```

Repeat until convergence.

The prediction step now uses sampled experience (TD or MC) instead of Bellman
sweeps. The improvement step is identical — just a greedy argmax.

---

## 3. The Exploration Problem

There's a catch. If π is deterministic and you always act greedily, many
(s, a) pairs are never tried. Q estimates for unvisited pairs stay at their
initial values forever, and the greedy policy never improves past whatever
it found first.

**ε-greedy** is the standard fix:

```
π(a|s) = { 1 - ε + ε/|A|   if a = argmax_a Q(s,a)
          { ε/|A|            otherwise
```

With probability ε, choose uniformly at random. With probability 1-ε, exploit.

This ensures every (s, a) pair is tried infinitely often (as long as the agent
keeps running), while still mostly exploiting good actions.

**The cost:** even after Q converges, ε-greedy takes the suboptimal action ε
fraction of the time. There is no free lunch — exploration always costs
something at execution time.

**GLIE (Greedy in the Limit with Infinite Exploration):** for theoretical
convergence guarantees, ε must decay to 0 as episodes → ∞ (so the policy
eventually becomes greedy) but do so slowly enough that every (s,a) is still
visited infinitely often. In practice a fixed small ε works well enough.

---

## 4. SARSA: On-Policy TD Control

Apply TD(0) to Q instead of V, while following ε-greedy:

```
Q(s,a) ← Q(s,a) + α [r + γ Q(s', a') - Q(s,a)]
```

where a' is the action ε-greedy **actually selects** at s'.

The update depends on the tuple **(S, A, R, S', A')** — hence the name SARSA.

**Why "on-policy":** the bootstrap target `Q(s', a')` uses the action from the
same ε-greedy policy being evaluated. SARSA learns Q^π for the ε-greedy policy
it's following. If ε is small, that's close to the optimal policy.

---

## 5. Q-Learning: Off-Policy TD Control

One change from SARSA: replace `Q(s', a')` with `max_a Q(s', a)`:

```
Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s', a') - Q(s,a)]
```

The bootstrap target always uses the **best action at s'**, regardless of what
ε-greedy actually took.

**Why "off-policy":** the target represents a fully greedy policy, but the
behavior (data collection) follows ε-greedy. The policy being *evaluated* (and
improved toward) is different from the policy being *followed*.

Q-Learning directly approximates Q* — the optimal action-value function — as
long as every (s,a) is visited sufficiently often.

---

## 6. On-Policy vs Off-Policy: When It Matters

Both algorithms converge. The difference is *what* they converge to, and
what that means in risky environments.

Consider a GridWorld with a cliff (row of hazards) between the start and goal.
Two paths exist: a safe route (far from the cliff) and an optimal route
(one step from the cliff edge).

| | SARSA | Q-Learning |
|---|---|---|
| Learns | Q^{ε-greedy} — accounts for accidental cliff steps | Q* — assumes perfect greedy execution |
| Converges to | Safe path (ε occasionally wanders into cliff) | Cliff-edge path (optimal if executed perfectly) |
| Risk during training | Lower — avoids cliff because it "knows" ε can misfire | Higher — walks the cliff edge |
| Deployed policy | Safe and correct | Optimal but fragile to noise |

**Rule of thumb:** if the behavior policy equals the target policy (on-policy),
use SARSA. If you want to learn Q* independently of how you're exploring
(off-policy), use Q-Learning. Most modern deep RL (DQN) uses Q-Learning
because experience replay requires off-policy learning.

---

## 7. Pen-and-Paper Exercises

### Exercise 1 — Why Q, not V?

You are given:
- V(s_L) = 8.0, V(s_R) = 3.0
- From state s, action LEFT deterministically leads to s_L with reward -1
- From state s, action RIGHT deterministically leads to s_R with reward +3
- γ = 0.9

**Part A:** Compute Q(s, LEFT) and Q(s, RIGHT). Which action does greedy
policy improvement select?

**Part B:** Now suppose you don't know *which* action leads to which next state
(you have no model). You only know V(s_L) = 8.0 and V(s_R) = 3.0.
Can you still determine the greedy action? What is missing?

**Part C:** If instead you had Q(s, LEFT) and Q(s, RIGHT) directly
(estimated from experience), do you need the model to act greedily? Why not?

---

### Exercise 2 — The Cost of Exploration

From state s you can take two actions:
- **a1:** always gives reward 0, episode ends.
- **a2:** gives reward +10 with probability 0.5, reward −8 with probability 0.5, episode ends.

You're using Q-Learning with ε = 0.1. Q is initialized to 0 everywhere.

By bad luck, the first 3 times a2 is tried it returns −8. Now Q(s, a2) = −8,
Q(s, a1) = 0.

**Part A:** With ε = 0, what does the agent do from this point? Does it ever
recover? Why or why not?

**Part B:** With ε = 0.1 and Q eventually converged to the true values, what
fraction of episodes does the agent take the suboptimal action? 

**Part C:** What happens to that fraction as ε → 0? As ε → 1? What is the
fundamental tradeoff?

---

### Exercise 3 — SARSA vs Q-Learning on One Transition

An agent in state s takes action a (selected by ε-greedy) and observes
r = −1, landing in s'. At s', ε-greedy selects action a' at random (not
the greedy action). You know:

```
Q(s,  a)  = 5.0
Q(s', a*) = 8.0   ← greedy action at s'
Q(s', a') = 2.0   ← action ε-greedy actually took
γ = 0.9,  α = 0.1
```

**Part A:** Compute the SARSA update to Q(s, a).

**Part B:** Compute the Q-Learning update to Q(s, a).

**Part C:** SARSA decreased Q(s, a) while Q-Learning increased it. In one
sentence each, what does each algorithm "believe" about the future at s'?

**Part D:** If ε → 0 so that ε-greedy becomes nearly deterministic, do the
two updates converge to the same value? Why?

---

### Exercise 4 — The On/Off-Policy Consequence

A 2×4 GridWorld. Start = (1,0), Goal = (0,3). The bottom row (1,1) and (1,2)
are cliffs (hazards, reward −10, episode ends).

```
(0,0) → (0,1) → (0,2) → (0,3) GOAL
  ↑               ↑         ↑
(1,0)   (1,1)   (1,2)   (1,3)
START  CLIFF   CLIFF
```

Two meaningful paths from start to goal:
- **Top path:** (1,0)→(0,0)→(0,1)→(0,2)→(0,3) — 4 steps, always safe
- **Bottom path:** (1,0)→(1,3)→(0,3) — 2 steps, passes adjacent to cliffs

Both algorithms run with ε = 0.1 until convergence.

**Part A:** Q-Learning converges to the bottom path. Why? What does its
update target assume about the agent's behavior at each step?

**Part B:** SARSA converges to the top path. What does SARSA "see" on
episodes where ε-greedy accidentally moves toward a cliff?

**Part C:** You need to deploy one algorithm's learned policy on a physical
robot near a cliff. Which do you choose, and why?

---

## 8. Code Tasks

See [`sarsa.py`](sarsa.py), [`qlearning.py`](qlearning.py), and
[`run_control.py`](run_control.py).

Implement SARSA and Q-Learning, then use `run_control.py` to:

1. Confirm both converge on the standard 4×4 GridWorld (RMSE vs DP solution).
2. Add a hazard and compare the policies learned by SARSA vs Q-Learning —
   do they differ? When does the difference become visible?
3. Try ε = 0.0, 0.05, 0.3. How does convergence speed and final policy quality
   change?
4. Compare α = 0.5 vs α = 0.01. Characterize the tradeoff in one sentence.

---

## 9. Skill Cards to Write

- `epsilon_greedy.md` — the exploration problem, ε-greedy, GLIE, the cost of exploration
- `sarsa.md` — on-policy TD control, the SARSA update, what "on-policy" means
- `qlearning.md` — off-policy TD control, why it learns Q*, the Bellman optimality target
- `on_off_policy.md` — behavior vs target policy, when the distinction matters, cliff walking
