---
name: ε-Greedy Exploration
description: Injecting randomness into action selection to ensure all (s,a) pairs are visited — the standard solution to the exploration problem
type: concept
---

# ε-Greedy Exploration

## The problem it solves

A greedy policy always takes the action with the highest current Q estimate. In a deterministic environment, this means the agent follows the same trajectory every episode, and Q values for unvisited (s,a) pairs stay at their initial values forever. The agent gets locked into whatever path it found first, even if a better one exists.

ε-greedy injects randomness to prevent this:

```
π(a|s) = { 1 - ε + ε/|A|   if a = argmax Q(s,a)
          { ε/|A|            otherwise
```

With probability ε, choose uniformly at random. With probability 1-ε, exploit.

## The cost of exploration

Even after Q has converged, ε-greedy keeps taking random actions. With ε=0.1 and |A|=4 actions, the suboptimal actions collectively get 10% of the probability. The optimal action gets 0.9 + 0.1/4 = 92.5%. This is the price: exploration never fully stops unless ε decays.

**GLIE (Greedy in the Limit with Infinite Exploration):** for convergence guarantees, ε must decay to 0 so the policy eventually becomes greedy, but slowly enough that every (s,a) is still visited infinitely often. In practice a fixed small ε works well enough.

## The ε-SARSA asymmetry

Larger ε affects SARSA and Q-Learning differently:

- **Q-Learning:** the update target is always `max_a Q(s', a)` — independent of ε. Larger ε means more state coverage (good for convergence speed) and more suboptimal steps at execution time, but the *target* Q-Learning is converging toward stays Q*.
- **SARSA:** the update target is `Q(s', a')` where a' is the ε-greedy action actually taken. Larger ε makes the policy being evaluated more random, so SARSA converges to Q^{π_ε} which is further from Q*. Exploration directly degrades the quality of what SARSA learns.

## When ε=0 fails

Pure greedy (ε=0) with unlucky early experiences can get permanently stuck. If the first few attempts at an action give bad returns, Q drops and greedy never revisits it — even if it's the true optimum. ε > 0 ensures periodic re-exploration regardless of history.
