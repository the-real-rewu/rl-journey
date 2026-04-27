"""DQN agent: CNN architecture and training step.

Run from repo root:
    python -m phase2_dqn.dqn_agent
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """CNN Q-network from Mnih et al. (2015).

    Input:  (batch, 4, 84, 84) float32 frames in [0, 1]
    Output: (batch, num_actions) Q-values, no activation
    """

    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class DQNAgent:
    """Holds the online and target networks, optimizer, and ε schedule."""

    def __init__(
        self,
        num_actions: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.1,
        epsilon_decay_steps: int = 1_000_000,
        target_update_freq: int = 1000,
    ):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.online = DQNNetwork(num_actions).to(device)
        self.target = DQNNetwork(num_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

    @property
    def epsilon(self) -> float:
        progress = min(self.steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_final - self.epsilon_start)

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(state_t).argmax(dim=1).item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> float:
        """One gradient update. Returns the scalar loss value.

        The batch is (states, actions, rewards, next_states, dones) — all GPU tensors.
        Use the TARGET network to compute TD targets, and the ONLINE network to
        compute predictions. The (1 - done) term zeros the bootstrap at episode end.
        See section 6 of the theory doc for the full update equation.
        """
        states, actions, rewards, next_states, dones = batch

        # TODO: implement the DQN training step.
        # 1. Compute TD targets: r + γ * max_{a'} Q(s', a'; θ⁻) * (1 - done)
        #    Use torch.no_grad() around the target network — its gradients
        #    should never be computed.
        # 2. Compute predictions: Q(s, a; θ) for the actions that were taken.
        #    You need to gather the Q-value for each action in the batch.
        # 3. Compute MSE loss between predictions and targets.
        # 4. Zero gradients, backpropagate, clip gradients to norm 10, step.
        # 5. Sync θ⁻ ← θ every self.target_update_freq steps.
        # 6. Increment self.steps and return the loss as a Python float.
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "steps": self.steps}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.steps = ckpt["steps"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net = DQNNetwork(num_actions=4).to(device)
    dummy = torch.zeros(2, 4, 84, 84, device=device)
    out = net(dummy)
    print(f"Network output shape: {out.shape}")  # expect (2, 4)

    total = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total:,}")              # expect ~1.7M
