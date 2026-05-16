"""N-step return accumulator that wraps a replay buffer's push path.

See Section 3 and Task C of phase2_dqn/03_dqn_improvements.md.

Run from repo root:
    python3 -m phase2_dqn.nstep_wrapper
"""

from __future__ import annotations
from collections import deque
import numpy as np


class NStepWrapper:
    """Wraps a buffer so that push() accumulates n-step returns before storing.

    Each call to push(...) feeds one environment-step transition. Once n consecutive
    transitions are buffered, one synthetic transition is emitted to the underlying
    buffer:

        (s_t, a_t, R_n, s_{t+n}, done_n)
        R_n = sum_{k=0}^{n-1} gamma^k * r_{t+k}

    On episode termination (done=True), all remaining partial windows are flushed
    so no transitions are lost at episode boundaries.

    Pass-through API: sample, update_priorities, __len__ forward to the underlying
    buffer. Anything else the agent needs from the buffer should be added explicitly.
    """

    def __init__(self, buffer, n_step: int, gamma: float):
        self.buffer = buffer
        self.n_step = n_step
        self.gamma = gamma
        self._window: deque = deque(maxlen=n_step)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Accumulate one transition. Emit synthetic n-step transitions when ready.

        Behaviour:
          - With fewer than n transitions buffered: just accumulate, emit nothing.
          - With exactly n transitions buffered: emit one synthetic transition,
            then drop the oldest from the window.
          - On done=True: flush every remaining partial window so the tail of the
            episode produces shorter synthetic transitions ending in done_n=True.
            After flushing, the window is empty.
        """
        # TODO
        #
        # Conceptual hint: what should done_n be when the synthetic window crosses
        # an episode boundary? Think about what the bootstrap term
        #     gamma^n * max_a Q_target(s_{t+n}, a)
        # should compute when the trajectory already terminated before step t+n.
        # Your answer determines done_n.
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.buffer.update_priorities(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    # Smoke test per Task C:
    #   1. Wrap a ReplayBuffer with NStepWrapper(n_step=3, gamma=0.99).
    #   2. Push five non-terminal transitions with rewards [1, 0, 0, 0, 1].
    #   3. Assert exactly TWO synthetic transitions have reached the underlying
    #      buffer, and the first has R_n = 1 + 0*gamma + 0*gamma^2 = 1.0.
    #   4. Push one terminal transition with reward 1, done=True.
    #   5. Verify the tail flush emits the correct number of shorter transitions,
    #      all with done_n=True.
    #
    # All expected values must be computed from the same R_n formula in push().
    pass
