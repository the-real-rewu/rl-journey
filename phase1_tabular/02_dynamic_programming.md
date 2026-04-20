# 1.2 — Dynamic Programming

Dynamic Programming (DP) solves the Bellman equations *exactly*, using the
full environment model `P(s'|s,a)`. This is the only phase where you have
that luxury. Later phases (TD, Q-Learning) will remove it — but having once
seen the exact solution, you'll know what you're approximating.

**Prerequisites:** Section 1.1 — especially the Bellman expectation and
optimality equations. Have those open as you read this.

---

## 1. The Setup

You have a GridWorld and you know everything about it:
- Every state, every action.
- Exactly where each action takes you and what reward you get.

Your goal: find the **optimal policy** `π*` — the action to take in each
state to maximize expected return.

DP does this by working with **value functions** stored as a table
`V[s]` (one float per state). The table starts at zero and gets updated
iteratively until it converges.

---

## 2. Policy Evaluation — Solving for V^π

**Problem:** Given a fixed policy π, compute V^π exactly.

**Algorithm:** Repeatedly sweep through all states and apply the
Bellman expectation equation as an update rule:

```
V_new(s) ← Σ_a π(a|s)  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V_old(s') ]
```

Keep sweeping until the maximum change across all states in one sweep
falls below a small threshold θ (convergence criterion).

**Why does this converge?**
The Bellman expectation operator is a *contraction*: each application
brings V closer to the true V^π. With γ < 1, the contraction factor
is γ — the error shrinks by at least γ each full sweep.

**Important detail: in-place vs two-table updates**

Two options for the sweep:
- **Two-table:** compute `V_new` from `V_old`; swap after each full sweep.
- **In-place:** update `V[s]` immediately; later states in the same sweep
  see already-updated values.

In-place typically converges faster in practice. Both converge to the
same answer. Most implementations use in-place.

---

## 3. Policy Improvement — Getting a Better Policy

**Problem:** Given V^π, find a policy that is at least as good as π.

**Key theorem (Policy Improvement Theorem):**
Define the greedy policy:
```
π'(s) = argmax_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
```
Then `V^{π'}(s) ≥ V^π(s)` for all states s. The new policy is
*at least as good* as the old one everywhere.

**Why?** If the greedy action is different from π(s), it means there's an
action that yields higher immediate reward *plus* higher discounted future
value. So the greedy policy must be at least as good.

**Stability:** If π' = π (no changes), then π is already optimal.
The policy improvement theorem guarantees that we've found π*.

---

## 4. Policy Iteration — Alternating Eval and Improvement

Combine evaluation and improvement into a loop:

```
1. Initialize V arbitrarily, π arbitrarily.
2. Loop:
   a. Policy Evaluation:  compute V^π until convergence
   b. Policy Improvement: π' ← greedy(V^π)
   c. If π' == π: STOP. Return π and V^π.
   d. π ← π'
```

**Convergence:** Guaranteed in finite steps because:
- There are finitely many deterministic policies.
- Each iteration strictly improves V (or we stop).
- So we can never cycle.

**Cost:** Expensive — inner loop (evaluation) runs to convergence before
each improvement step. We can truncate it.

---

## 5. Value Iteration — The Shortcut

Instead of running policy evaluation to full convergence, do *one* sweep
using the Bellman *optimality* equation directly:

```
V_new(s) ← max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V_old(s') ]
```

Note the `max` — we're solving the optimality equation, not the
expectation equation. We've skipped the explicit policy and go
straight for V*.

```
1. Initialize V[s] = 0 for all s.
2. Loop until max_s |V_new(s) - V(s)| < θ:
   a. For each state s: V(s) ← max_a Σ_{s',r} P [ r + γ V(s') ]
3. Extract policy: π*(s) = argmax_a Σ_{s',r} P [ r + γ V(s') ]
```

**Why does this work?** Each application of the `max` Bellman operator
is also a contraction (factor γ). So value iteration converges to V*
in the limit, and we extract π* at the end.

**Value iteration vs policy iteration:**

| | Policy Iteration | Value Iteration |
|---|---|---|
| Inner loop | Policy eval to convergence | One sweep |
| Outer iterations | Few | More |
| Cost per iteration | High | Low |
| Total cost | Often lower | Often comparable |

In practice, value iteration is simpler to implement and commonly preferred.

---

## 6. Generalized Policy Iteration (GPI)

Both algorithms are instances of one unifying pattern:

```
V → better V (evaluation) → better π (improvement) → repeat
```

This is called **Generalized Policy Iteration**. The evaluation and
improvement steps *interact* — the value function drives improvement,
the policy drives evaluation. They compete and cooperate until neither
can improve the other. That fixed point is the optimum.

This pattern appears in every RL algorithm, including DQN and PPO.
The "deep" part of deep RL is just replacing the table with a neural
network. GPI is still the skeleton.

---

## 7. Pen-and-paper exercises

### Exercise 1 — One sweep of policy evaluation

2×2 GridWorld (start (0,0), goal (1,1), step -1, goal +10, γ=0.9).
Policy π: RIGHT everywhere except (0,1) where it goes DOWN.
Initialize V = 0 everywhere.

Do **one full sweep** (update all non-terminal states once, in order
(0,0), (0,1), (1,0)) using in-place updates.

What are V[(0,0)], V[(0,1)], V[(1,0)] after the sweep?

### Exercise 2 — Policy improvement step

Using the V from Exercise 1, apply one policy improvement step.
For each state, compute Q(s,a) for all actions and pick the greedy one.
Does the policy change? What does the new policy look like?

### Exercise 3 — Value iteration converges on 2×2

Convince yourself by tracing through 3 full value iteration sweeps
on the same 2×2 grid. You should see V[(0,0)] increasing toward 8.

### Exercise 4 — Why V*(terminal) = 0 from a DP perspective

In the DP update, terminal states are treated as absorbing (they
transition to themselves with reward 0). Write out the Bellman update
for a terminal state and confirm V*(terminal) = 0 is a fixed point.

---

## 8. Code tasks

See [`dp.py`](dp.py). Implement the four functions marked with `TODO`:

1. **`policy_evaluation`** — the inner sweep. One correct Bellman update
   for each non-terminal state.
2. **`policy_improvement`** — for each state, compute Q(s,a) for all
   actions and return the greedy deterministic policy.
3. **`policy_iteration`** — wire up eval and improvement; stop when
   the policy stabilizes.
4. **`value_iteration`** — one sweep using the `max` Bellman operator.

Then run [`run_dp.py`](run_dp.py) to see the value function and optimal
policy printed on the grid.

**Experiments to try after implementing:**
- Compare the number of iterations for policy iteration vs value iteration.
- Add a wall or a hazard and observe how V* and π* change.
- Try γ = 0.5 vs γ = 0.99 — what does the value function look like?
- What happens with a very tight convergence threshold (θ = 1e-10)?

---

## 9. Skill cards to write

- **`policy_evaluation.md`** — what it solves, the update rule, convergence
- **`policy_iteration.md`** — the two-step loop, the improvement theorem
- **`value_iteration.md`** — why `max` replaces `Σ_a π`, when to prefer it over policy iteration
- **`generalized_policy_iteration.md`** — the unifying pattern; how it shows up in deep RL

---

## 10. Further reading

- Sutton & Barto Chapter 4 — this section maps directly.
- David Silver Lecture 3: https://www.davidsilver.uk/teaching/
