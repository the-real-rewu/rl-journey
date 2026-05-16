"""DQN agent: CNN architecture and training step.

Run from repo root:
    python3 -m phase2_dqn.dqn_agent
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
        use_double_dqn: bool = False,
    ):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
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
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).div_(255.0)
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
        
        with torch.no_grad():
            if self.use_double_dqn:
                best_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                max_next_state_q = self.target(next_states).gather(1, best_actions).squeeze(1)
            else:
                max_next_state_q = self.target(next_states).max(dim=1)[0]

        td_targets = rewards + self.gamma * max_next_state_q * (1 - dones)
        online_predictions = self.online(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        # Huber loss (smooth L1): caps gradient magnitude for large TD errors,
        # preventing the Q-value overestimation spiral that MSE causes.
        loss = F.smooth_l1_loss(online_predictions, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

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
