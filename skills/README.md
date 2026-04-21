# Skills — Distilled RL Knowledge

This folder is your personal reference library. Each file is a self-contained
skill card written *after* you've implemented and understood the concept.
The goal is compression: future-you should be able to recall the essence
of any topic in under 5 minutes.

## How to use this folder

- **Write a skill card when:** you finish implementing something and the idea
  has clicked. Not before — understanding before writing keeps the cards honest.
- **Update a card when:** a later phase reveals something that reframes
  an earlier concept (e.g., how TD-learning looks different once you've seen DQN).
- **Read cards before:** starting a new phase, revisiting old code, or
  explaining a concept to someone else.

## Card format

Each card should have:
1. **One-line summary** — the concept in a single sentence
2. **The core idea** — 3–5 sentences, no jargon without definition
3. **The math** — key equation(s), explained in words not just symbols
4. **Intuition** — an analogy or concrete example
5. **Common pitfalls** — mistakes you actually made or nearly made
6. **Connections** — how this concept links to others in the folder

---

## Index

### Foundations

| Skill | Summary | Phase |
|-------|---------|-------|
| [mdp.md](mdp.md) | MDP 5-tuple, stationarity, Markov property, when the framing breaks | 1.1 |
| [bellman_equations.md](bellman_equations.md) | Value = immediate reward + discounted next value; expectation vs optimality forms | 1.1 |
| [value_functions.md](value_functions.md) | V(s) vs Q(s,a) — what they measure, how they relate, why Q wins in model-free settings | 1.1 |

### Dynamic Programming

| Skill | Summary | Phase |
|-------|---------|-------|
| [policy_evaluation.md](policy_evaluation.md) | Iterative Bellman sweeps to compute V^π for a fixed policy | 1.2 |
| [policy_iteration.md](policy_iteration.md) | Alternate eval and greedy improvement; guaranteed finite convergence | 1.2 |
| [value_iteration.md](value_iteration.md) | Solve Bellman optimality directly with one sweep per iteration | 1.2 |
| [generalized_policy_iteration.md](generalized_policy_iteration.md) | The unifying pattern behind every RL algorithm, from DP to DQN to PPO | 1.2 |

### Policy Gradient Methods
*(cards will be added as you work through Phase 3)*

| Skill | Summary | Phase |
|-------|---------|-------|
| — | — | — |

### Advanced Topics
*(cards will be added as you work through Phase 4)*

| Skill | Summary | Phase |
|-------|---------|-------|
| — | — | — |

---

*The best reference is the one you wrote yourself.*
