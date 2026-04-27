"""Experience replay buffer for DQN.

Run from repo root:
    python -m phase2_dqn.replay_buffer
"""

from __future__ import annotations
import random
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.

    push(...)   — add one transition; oldest entry is overwritten when full.
    sample(...) — return a random mini-batch as GPU tensors, ready for training.

    Design question: what dtype and shape should you store observations in?
    Storing float32 uses 4× the memory of uint8. With a 100k-capacity buffer
    and 84×84×4 observations, float32 costs ~11 GB; uint8 costs ~2.8 GB.
    Store as uint8, convert to float32 only when sampling.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self._pos = 0
        self._size = 0

        # TODO: implement ReplayBuffer.
        # Initialise storage arrays here (use numpy for memory efficiency).
        # Think about the shape and dtype of each field before allocating.
        # The observation shape is (4, 84, 84) uint8.
        raise NotImplementedError

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition. Overwrites oldest entry when buffer is full."""
        # TODO: store the transition at self._pos, advance the pointer,
        # and update self._size.
        raise NotImplementedError

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Return (states, actions, rewards, next_states, dones) as GPU tensors.

        states and next_states: float32, values in [0, 1], shape (B, 4, 84, 84)
        actions:                int64,  shape (B,)
        rewards:                float32, shape (B,)
        dones:                  float32, shape (B,)  — 1.0 if terminal, else 0.0
        """
        # TODO: sample batch_size random indices, gather the stored transitions,
        # convert observations from uint8 to float32 / 255.0, and move to device.
        raise NotImplementedError

    def __len__(self) -> int:
        return self._size


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buf = ReplayBuffer(capacity=1000, device=device)

    # Fill with random transitions
    for _ in range(200):
        s  = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        ns = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        buf.push(s, random.randint(0, 3), random.random(), ns, False)

    print(f"Buffer size: {len(buf)}")  # expect 200

    states, actions, rewards, next_states, dones = buf.sample(32)
    print(f"states shape:      {states.shape}")       # (32, 4, 84, 84)
    print(f"states dtype:      {states.dtype}")       # float32
    print(f"states range:      [{states.min():.3f}, {states.max():.3f}]")  # [0, 1]
    print(f"actions shape:     {actions.shape}")      # (32,)
    print(f"device:            {states.device}")
