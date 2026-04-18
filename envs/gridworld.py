"""GridWorld: a small 2D environment for tabular RL.

Deliberately minimal — no gym dependency, no hidden state. Every method
is short enough to read in one sitting. You should be able to predict
exactly what any call returns.

Layout conventions
------------------
- The grid is `rows x cols`. Position is (row, col) with (0, 0) at top-left.
- Actions are integers 0..3 meaning up, right, down, left.
- Moving into a wall or off the edge keeps the agent in place (but still
  incurs the step reward — standing still isn't free).
- Reaching the goal or a hazard terminates the episode.

Why expose `transitions()`?
---------------------------
Dynamic Programming algorithms need the full model P(s'|s,a) and R(s,a,s').
Model-free algorithms don't. Exposing transitions lets Phase 1.2 (DP) work
on the exact same environment as Phase 1.3-1.4 (model-free). Later you
will simply stop using this method.
"""

from __future__ import annotations
from dataclasses import dataclass, field

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
ACTION_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


@dataclass
class GridWorld:
    rows: int = 4
    cols: int = 4
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (3, 3)
    walls: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    hazards: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    step_reward: float = -1.0
    goal_reward: float = 10.0
    hazard_reward: float = -10.0

    def __post_init__(self):
        self.walls = frozenset(self.walls)
        self.hazards = frozenset(self.hazards)
        assert self.start not in self.walls, "start cannot be a wall"
        assert self.goal not in self.walls, "goal cannot be a wall"
        self._agent = self.start

    # ---- Gym-style interface (for model-free methods) ----

    def reset(self) -> tuple[int, int]:
        self._agent = self.start
        return self._agent

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict]:
        """Take one step. Returns (next_state, reward, done, info)."""
        next_state, reward, done = self._simulate(self._agent, action)
        self._agent = next_state
        return next_state, reward, done, {}

    # ---- Model access (for Dynamic Programming) ----

    @property
    def states(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows)
                       for c in range(self.cols)
                       if (r, c) not in self.walls]

    @property
    def num_actions(self) -> int:
        return 4

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state == self.goal or state in self.hazards

    def transitions(self, state: tuple[int, int], action: int
                    ) -> list[tuple[float, tuple[int, int], float, bool]]:
        """Return list of (prob, next_state, reward, done) tuples.

        For this deterministic GridWorld there's always exactly one outcome,
        so the list has length 1. Keeping the list-of-tuples shape makes
        the DP code easy to extend to stochastic environments later.
        """
        if self.is_terminal(state):
            # Terminal states are absorbing with zero reward.
            return [(1.0, state, 0.0, True)]
        next_state, reward, done = self._simulate(state, action)
        return [(1.0, next_state, reward, done)]

    # ---- Internals ----

    def _simulate(self, state, action):
        dr, dc = ACTION_DELTAS[action]
        nr, nc = state[0] + dr, state[1] + dc
        next_state = (nr, nc)
        # Walls and off-grid: stay put.
        if (not (0 <= nr < self.rows and 0 <= nc < self.cols)
                or next_state in self.walls):
            next_state = state

        if next_state == self.goal:
            return next_state, self.goal_reward, True
        if next_state in self.hazards:
            return next_state, self.hazard_reward, True
        return next_state, self.step_reward, False

    # ---- Rendering ----

    def render(self, agent: tuple[int, int] | None = None) -> str:
        """ASCII rendering. Pass an explicit agent position or use current."""
        pos = agent if agent is not None else self._agent
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == pos:
                    row.append("A")
                elif cell == self.goal:
                    row.append("G")
                elif cell in self.hazards:
                    row.append("H")
                elif cell in self.walls:
                    row.append("#")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
