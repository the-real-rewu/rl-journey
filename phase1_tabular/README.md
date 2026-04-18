# Phase 1 — Tabular Reinforcement Learning

**Environment:** GridWorld (see [envs/gridworld.py](../envs/gridworld.py)) — a
small 2D world you can see and reason about by hand.

**Goal of this phase:** Understand the *mathematical skeleton* of RL in a
setting where everything is visible. By the end, you will be able to derive
Q-Learning and SARSA on a napkin and explain why they work.

---

## Why tabular first?

In GridWorld, a 4×4 grid has 16 states and 4 actions — the entire value
function fits in a 64-cell table. You can print it, plot it, inspect it.
Every algorithm you implement becomes a small set of updates to that table.

Deep RL is the same algorithms with neural networks replacing the table.
The hard part is the algorithm, not the network. If you understand the
tabular version, you understand the deep version.

---

## Learning order

Work through these in order. Each section has a theory doc, exercises,
and a code task. Don't skip the pen-and-paper exercises — they are the
cheapest way to catch misunderstandings.

### 1.1 — Foundations *(you are here)*

- Read: [`01_foundations.md`](01_foundations.md)
- Code: [`hello_gridworld.py`](hello_gridworld.py) — run the env, make sense of states, actions, rewards
- Skill cards to write after: `mdp.md`, `bellman_equations.md`, `value_functions.md`

### 1.2 — Dynamic Programming *(next)*

- Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration
- Requires knowing the environment model — which GridWorld exposes via `transitions()`

### 1.3 — Model-Free Prediction

- Monte Carlo, TD(0), TD(λ)
- Evaluate a policy without knowing the transition probabilities

### 1.4 — Model-Free Control

- SARSA, Q-Learning, ε-greedy exploration
- Learn an optimal policy from experience alone

---

## How to work through this phase

1. **Read the theory doc for the current section.** Don't rush — if an
   equation doesn't feel obvious, stop and redo the derivation yourself.
2. **Do the pen-and-paper exercises.** These are designed to surface the
   "I thought I understood this" moments early.
3. **Implement the code task.** Start from the scaffolded file; fill in
   the `TODO` sections.
4. **Run experiments.** Vary γ, the grid layout, the reward structure.
   Watch the value function change and ask why.
5. **Write the skill card(s).** Only after you've implemented and experimented.
   The card should be what you wish you'd been told on day one.
6. **Check in.** Open a discussion — I'll review your skill card, point out
   gaps, and we refine it together before moving on.

---

## Deliverables for Phase 1

- [ ] 1.1 Foundations — understand MDPs, Bellman, value functions
- [ ] 1.2 Dynamic Programming — policy/value iteration working on GridWorld
- [ ] 1.3 Model-Free Prediction — MC and TD(0) estimating V^π
- [ ] 1.4 Model-Free Control — Q-Learning reliably solves GridWorld
- [ ] A set of skill cards in [`skills/`](../skills/) covering the above
