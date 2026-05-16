"""Prioritized experience replay buffer.

See Section 2 and Task B of phase2_dqn/03_dqn_improvements.md.

Run from repo root:
    python3 -m phase2_dqn.prioritized_replay_buffer
"""

from __future__ import annotations
import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Fixed-capacity buffer that samples transitions in proportion to TD-error magnitude.

    push(...)                       — add one transition at max-priority-seen-so-far
    sample(batch_size, beta)        — return (batch, indices, is_weights)
    update_priorities(idx, errors)  — refresh stored priorities after a training step

    Priorities are stored already raised to the power alpha:
        priority_i = (|delta_i| + epsilon) ** alpha

    So sampling probability is just  priority_i / sum(priorities) — no per-sample exponent.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.epsilon = epsilon
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0  # used for newly-pushed transitions

        # TODO: allocate storage.
        # Mirror ReplayBuffer's layout for the transition fields:
        #   states, actions, rewards, next_states, dones.
        # Add ONE new array storing the per-transition priority (already ^alpha).
        #
        # Conceptual hint: a binary sum-tree gives O(log N) sampling. A flat numpy
        # array gives O(N) via np.cumsum + np.searchsorted. For our 100k buffer
        # the sampling cost is dwarfed by the GPU forward pass either way.
        # Pick one and put a one-line comment HERE justifying the choice.

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition at the current max priority.

        Newly-pushed transitions take priority = self._max_priority so they are
        guaranteed to be sampled at least once before their real TD-error is known.
        """
        # TODO
        raise NotImplementedError

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
        """Sample a batch with P(i) ∝ priority_i.

        Returns:
            batch       — (states, actions, rewards, next_states, dones), same dtypes
                          and device as ReplayBuffer.sample
            indices     — np.ndarray of sampled indices, shape (batch_size,)
            is_weights  — torch.Tensor on self.device, shape (batch_size,)
                          w_i = (1 / (N * P(i))) ** beta, then divided by max(w) so
                          the largest weight in the batch equals 1.0
        """
        # TODO
        raise NotImplementedError

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: torch.Tensor,
    ) -> None:
        """Store (|delta| + epsilon) ** alpha at the given indices.

        Also update self._max_priority if any new priority exceeds it.
        """
        # TODO
        raise NotImplementedError

    def __len__(self) -> int:
        return self._size


if __name__ == "__main__":
    # Smoke test per Task B:
    #   1. Push 100 transitions with synthetic td_errors = 0..99.
    #   2. Call update_priorities so stored priorities reflect those td_errors.
    #   3. Sample 10_000 times with beta=0.
    #   4. Empirical sampling frequency of each index should rank-correlate with
    #      its priority (Spearman ≳ 0.95 is reasonable for 10k samples).
    #
    # Expected per-index frequency MUST be derived from the same
    # (|delta_i| + epsilon) ** alpha / sum_j ... formula your code uses.
    # Do not hardcode a frequency table.
    pass
