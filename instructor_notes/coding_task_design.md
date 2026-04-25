---
name: Coding Task Design Principles
description: How to scaffold coding assignments that test RL understanding, not Python skill
---

# Coding Task Design Principles

## What went wrong in Phase 1.3

### Over-scaffolded TODOs

The TODO comments in `mc.py` and `td.py` wrote the algorithm in pseudocode and asked the student to translate it. Example from `mc.py`:

```
# Strategy:
#   1. Compute returns G_0, G_1, ..., G_{T-1} for the trajectory.
#      G_{T-1} = r_T,  G_{t} = r_{t+1} + γ G_{t+1}
#      Walking backward is the cheapest way to do this.
#   2. Find the first occurrence index of each state in the episode.
#   3. For each first-visit state s with return G:
#         visit_count[s] += 1
#         V[s] += (G - V[s]) / visit_count[s]
```

This tests Python transcription. A student who didn't understand MC at all could implement it correctly. The challenge should be at the conceptual level: what are you computing and why, not how to express it in Python.

### Broken smoke tests

The `mc.py` smoke test assumed a fixed start state but the function uses exploring starts — the expected values were computed under a different assumption than the function's actual behavior. Tests that contradict the implementation teach nothing and erode trust.

### Inflexible experiment harness

`run_prediction.py` was hardcoded for one scenario (optimal policy, α=0.1, no uncertainty, first-visit MC only). The theory doc listed 4 experiments to try, all of which required manually editing the file. This creates friction between "I want to try something" and "I can run it."

## Rules for future chapters

### On TODOs

- Give the **contract**: what the function takes, what it returns, what invariants hold.
- Give one **conceptual hint** that points at the RL idea, not the implementation.
- Give a **failing test** that the student can run to verify correctness.
- Do NOT give the algorithm steps. If the student needs to look them up in the theory doc, that's correct — that's the point.

**Bad TODO:**
```
# Walk backward computing G_t = r + γ G_{t+1}, then update V[s] with running mean.
```

**Good TODO:**
```
# Implement first-visit MC prediction.
# Each episode gives you one unbiased sample of G_t for each state visited.
# How do you turn a sequence of (s, a, r) tuples into those samples?
```

### On smoke tests

- Compute the expected value independently (by hand or with a known-correct method).
- Make sure the test's assumptions exactly match the function's behavior (start state, exploring starts, seed behavior).
- Test the simplest possible case where the answer is analytically derivable.
- The smoke test should fail in an informative way if the implementation is wrong — not just silently produce wrong numbers.

### On experiment harnesses

- Parameterize everything that the theory doc says "try varying": α, num_episodes, action_uncertainty, policy type.
- Use a config dict or argparse at the top so any experiment is a one-line change.
- Structure the output so it's immediately comparable (aligned columns, consistent units).
- The experiments listed in the theory doc should each be runnable by changing one parameter value, not by editing the logic.

## The guiding question

Before writing a TODO comment, ask: *if a student who hasn't read the theory doc reads only this comment, can they implement the function?*

If yes, the comment is too detailed. Strip it back until the student has to consult the theory to know what to do.
