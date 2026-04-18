---
name: Markov Decision Process (MDP)
description: The formal framework behind all RL — what it is, its two key assumptions, and when the framing breaks down
type: user
---

# Markov Decision Process (MDP)

An MDP is the mathematical skeleton of a sequential decision problem. Formally a 5-tuple `(S, A, P, R, γ)`:

| Symbol | Meaning |
|--------|---------|
| `S` | Set of all states |
| `A` | Set of all actions |
| `P(s'\|s,a)` | Transition probability: P(land in s' \| was in s, took a) |
| `R(s,a,s')` | Reward received for that transition |
| `γ ∈ [0,1]` | Discount factor — how much future rewards are worth |

## Two core assumptions

**1. Stationarity.** The transition structure `P` and `R` do not change over time. The rules of the game are fixed — the agent's actions affect states, but not the underlying dynamics. (A non-stationary environment, e.g. an opponent that adapts to you, violates this.)

**2. Markov property.** The next state depends only on the current state and action — not on history:
```
P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...)
```
The current state is a sufficient summary of everything that matters.

## When it breaks down

- **History matters:** e.g. a game where the opponent remembers your past moves. Fix: redefine state to include relevant history.
- **Partial observability:** the agent can't see the full state (POMDP). Fix: state = observation history, or use a belief state.
- **Non-stationarity:** the environment itself changes over time. Harder to fix — most RL theory assumes stationarity.

**Key insight:** It is up to the modeler to define what a state is. A bad state representation can make a Markov problem look non-Markov (and vice versa). GridWorld is cleanly Markov — the (row, col) position is all the agent needs.
