---
name: Random Insights
description: Cross-cutting mental models about deep learning that don't belong to any single algorithm — inductive bias, architectural priors, and the structure-matching view of learning
type: concept
---

# Random Insights

A place for mental models that are too general to live in a single algorithm card but too useful to forget.

---

## Learning as structure matching (not isomorphism)

The goal of training a neural network is to find a function in your hypothesis class that approximates the target phenomenon well enough. A tempting framing is "isomorphism" — the network's computation graph maps onto the structure of the world — but isomorphisms are exact and invertible. What we actually do is closer to **approximation under an inductive bias**.

The two dimensions of learning efficiency:

1. **Expressive capacity** — the upper bound on what the model *can* represent. Determined by depth, width, and parameterization. Bigger models for more complex phenomena. This is necessary but not sufficient.

2. **Alignment** — how efficiently optimization finds the target function within that capacity. Two levers:
   - **Adaptation dynamics**: how quickly and stably gradient descent traverses the loss landscape. This is where optimizer choice (Adam vs SGD), learning rate schedules, and gradient clipping act.
   - **Starting position and search space**: where you begin and how large the region to search is. Architecture and weight initialization both contribute. A bad starting point in a large space is slow; a constrained space with a good structural prior can be fast even with a mediocre optimizer.

---

## Inductive bias: making the target function cheap to represent

An **inductive bias** is a structural assumption baked into the model that makes certain functions easy to represent and others hard. Good priors make the *target function* the cheap one.

Examples:

| Model | Structural prior | What it makes cheap |
|-------|-----------------|---------------------|
| CNN | Translation invariance | Same filter reused at every position |
| Dueling DQN | V/A decomposition | "State value dominates, actions rarely matter" |
| Transformer | Permutation equivariance + global attention | Long-range dependencies without positional locality |
| BatchNorm | Shift-invariance of activations | Functions that don't depend on the scale of hidden units |

The pattern: **every strong architectural prior encodes a symmetry or invariance of the target phenomenon.** CNNs work on images because a cat in the corner is the same cat as one in the center — translation symmetry. Dueling works on Atari because the value of a state is mostly independent of which action you're about to take — action-independence of the baseline.

When the prior matches the phenomenon, the network doesn't have to discover the structure from data — it's built in. When it doesn't match, the network wastes capacity fighting the mismatch.

---

## Persistent constraints vs. initialization

A common confusion: architectural priors and weight initialization both affect the "starting position," so they seem equivalent. They're not.

- **Weight initialization** gives a starting point in parameter space. The network immediately moves away from it during training. Its influence decays.
- **Architectural constraints** define the *function class* — what the network can express at any point in training, not just at the start. These stay active permanently.

Dueling's `Q = V + (A − mean A)` is a hard constraint on every forward pass. The network cannot learn to ignore it. This is why describing Dueling as "a better initialization" undersells it — it permanently reduces the hypothesis class, not just the starting entropy.

The information-theoretic consequence: a smaller hypothesis class means fewer bits needed to describe the target (MDL principle), better generalization (less capacity for spurious patterns), and faster optimization (smaller search space throughout training).

---

## The "good prior" test

A quick check for whether an architectural choice encodes a useful prior:

> *Does the target function live near the center of the new hypothesis class, or near the edge?*

If the target is easy to represent with the prior (e.g., "V dominates" is the default in Dueling), the prior is good. If the target requires the network to work against the prior to represent it (e.g., a CNN applied to a problem with no translation symmetry), the prior adds overhead without benefit.

This is also why architectural choices are task-specific. There's no universally good inductive bias — it's good relative to the phenomenon (No Free Lunch).

---

## Connections

- [[dueling_dqn.md]] — the V/A decomposition is the concrete example that inspired this framing
- [[generalized_policy_iteration.md]] — GPI is itself an inductive bias: the alternation of policy eval and improvement is a structural assumption about how RL algorithms should work
- **Phase 3 forward**: policy gradient methods encode a different prior — that the policy is a probability distribution parameterized directly, rather than derived from a value function
