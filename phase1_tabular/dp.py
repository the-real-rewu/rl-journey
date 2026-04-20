"""Dynamic Programming algorithms for GridWorld.

Run this file from the repo root:
    python -m phase1_tabular.dp

All four functions below have the same signature pattern:
    env      — a GridWorld instance
    gamma    — discount factor
    theta    — convergence threshold (stop when max change < theta)
    policy   — dict mapping state -> action (deterministic, for eval only)

The env.transitions(state, action) method returns:
    list of (prob, next_state, reward, done)

For our deterministic GridWorld this list always has length 1, but writing
the code as if it could be longer makes it correct for stochastic envs too.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs import GridWorld


def policy_evaluation(
    env: GridWorld,
    policy: dict,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> dict:
    """Compute V^π for a fixed policy by iterative Bellman sweeps.

    Args:
        policy: dict mapping state -> action (deterministic policy)

    Returns:
        V: dict mapping state -> float (converged value function)
    """
    V = {s: 0.0 for s in env.states}

    while True:
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue

            v_old = V[s]

            # TODO: Compute the new V[s] using the Bellman expectation equation.
            #
            # For a deterministic policy, π(s) is just policy[s].
            # Use env.transitions(s, action) to get [(prob, s', r, done)].
            # Remember: V[terminal] = 0 (it's already initialized to 0
            # and never updated since we skip terminal states).
            #
            # V[s] = Σ_{s',r} P(s',r | s, π(s)) * [r + γ * V[s']]
            #
            # (For deterministic GridWorld P is always 1.0, so this
            #  simplifies, but write it with the prob multiplication
            #  so it generalises.)
            #
            # Remove the line below and replace with your implementation:
            raise NotImplementedError("implement policy_evaluation update")

            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    return V


def policy_improvement(
    env: GridWorld,
    V: dict,
    gamma: float = 0.9,
) -> dict:
    """Derive a greedy deterministic policy from a value function V.

    For each state, pick the action with the highest one-step lookahead value:
        π'(s) = argmax_a  Σ_{s',r} P(s',r|s,a) [r + γ V(s')]

    Returns:
        policy: dict mapping state -> action
    """
    policy = {}

    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0  # action doesn't matter at terminal states
            continue

        # TODO: For each action a, compute Q(s, a) using env.transitions.
        # Then set policy[s] = the action with the highest Q value.
        #
        # Q(s, a) = Σ_{s',r} P(s',r|s,a) * [r + γ * V[s']]
        #
        # Hint: use the same transitions loop as policy_evaluation, but
        # iterate over all actions and compare.
        #
        # Remove the line below and replace with your implementation:
        raise NotImplementedError("implement policy_improvement")

    return policy


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> tuple[dict, dict]:
    """Find V* and π* by alternating policy evaluation and improvement.

    Returns:
        V:      dict mapping state -> float  (optimal value function)
        policy: dict mapping state -> action (optimal policy)
    """
    # Start with an arbitrary deterministic policy (action 0 everywhere).
    policy = {s: 0 for s in env.states}

    # TODO: Implement the policy iteration loop:
    #   1. Evaluate the current policy with policy_evaluation → V
    #   2. Improve to get a new policy with policy_improvement
    #   3. If the policy didn't change (policy_stable), stop.
    #   4. Otherwise update the policy and repeat.
    #
    # To check if the policy changed, compare old and new policy dicts.
    # They are stable if policy[s] == new_policy[s] for all s.
    #
    # Remove the line below and replace with your implementation:
    raise NotImplementedError("implement policy_iteration")


def value_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> tuple[dict, dict]:
    """Find V* and π* using the Bellman optimality update.

    Returns:
        V:      dict mapping state -> float
        policy: dict mapping state -> action
    """
    V = {s: 0.0 for s in env.states}

    while True:
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue

            v_old = V[s]

            # TODO: Compute the new V[s] using the Bellman OPTIMALITY equation.
            #
            # Unlike policy_evaluation (which averages over π's action),
            # value iteration takes the MAX over all actions:
            #
            # V[s] = max_a  Σ_{s',r} P(s',r|s,a) * [r + γ * V[s']]
            #
            # Compute Q(s,a) for all actions, then set V[s] = max Q.
            #
            # Remove the line below and replace with your implementation:
            raise NotImplementedError("implement value_iteration update")

            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    # Extract the greedy policy from the converged V*.
    policy = policy_improvement(env, V, gamma)
    return V, policy
