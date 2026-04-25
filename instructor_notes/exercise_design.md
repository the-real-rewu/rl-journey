---
name: Exercise Design Principles
description: What makes a good pen-and-paper exercise for this student — lessons from Phase 1.3 failures
---

# Exercise Design Principles

## What went wrong in Phase 1.3

Exercises 1–4 in `03_model_free_prediction.md` were a guided walkthrough, not a challenge set.

- **Ex 1:** Compute MC update on one episode. (numeric)
- **Ex 2:** Compute TD update on one step. (numeric)
- **Ex 3:** "Run the same trajectory through both." This is just the answers from 1 and 2 side by side. Zero new insight.
- **Ex 4:** "Why does TD work despite bias?" This follows immediately from having done Ex 2 — the student just *saw* that V[s'] was 0 and TD still got nudged in the right direction. It shouldn't be a separate question; it should be a prompt at the end of Ex 2.

**Root cause:** Exercises were designed to confirm understanding of the text, not to reveal something the text didn't make visceral.

## The standard to meet

Each exercise should do exactly one of these things:
1. **Surprise** — the answer is non-obvious or counterintuitive even after reading the theory.
2. **Force a decision** — the student has to choose between two plausible approaches and defend it.
3. **Reveal a failure mode** — what happens when you apply the algorithm wrong or in a corner case.
4. **Expose a connection** — this concept is secretly the same as (or different from) something earlier.

If an exercise just confirms "did you read the section?", it belongs in a quiz, not in a learning sequence.

## Rules for future chapters

- **No redundant exercises.** If Ex N's answer is contained in Ex M's answer, cut one.
- **Each exercise reveals a different facet.** Numeric, edge case, conceptual, connection to prior material — not three numerics in a row.
- **Embed follow-up questions inside the exercise.** Don't separate "compute X" from "explain why X" into two exercises. The explanation is the point; the computation is just the setup.
- **Use exercises to expose what the theory doesn't say.** The theory covers the happy path. Exercises should stress-test edge cases: what if the episode is very long? what if a state is visited twice? what if α is too large?

## Example of a better exercise set for Phase 1.3

Instead of 4 exercises on the same 2-step trajectory:

1. **Variance in returns (MC):** Same policy, three different random episodes in a 3x3 stochastic GridWorld. Compute G_t for state (0,0) in each. How much do the returns vary? What does this tell you about how many episodes you need?

2. **TD lag (one-step propagation):** State s has a neighbor s' adjacent to the goal. V is initialized to 0 everywhere. After one episode visiting s → s' → goal: what is V[s'] after the TD update? What is V[s]? After a *second* episode: what is V[s] now? Why does it take two episodes, not one, for V[s] to reflect the goal reward?

3. **First-visit vs every-visit:** An episode visits s at step 1 (G=8) and again at step 4 (G=3). First-visit uses G=8; every-visit averages G=8 and G=3. Which gives the unbiased estimate of V^π(s)? Why? (Hint: think about whether both G values are independent samples.)

4. **n-step intuition:** You have a 5-step episode. TD(0) makes 5 updates (one per step). MC makes 5 updates (all at the end using the full return). For the state at step 1, which method makes its update first? Which has lower variance? Sketch what the targets look like for n=1, n=2, n=5.
