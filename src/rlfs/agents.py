"""Tabular reinforcement-learning agents, implemented from first principles.

No RL library is used anywhere in this repository. The update rules are written
out directly, because the point of the project is to show the algorithms rather
than to call them.

Two agents are provided:

`QLearningAgent`
    Off-policy temporal-difference control. Learns the value of the *greedy*
    policy while behaving epsilon-greedily:

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

`SarsaAgent`
    The on-policy counterpart, kept for comparison. It bootstraps from the
    action actually taken rather than the best one:

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

The difference is one term, and it changes behaviour in a way that is easy to
show and hard to forget: SARSA learns the value of the policy it is following,
including its exploration mistakes, so it behaves more cautiously near cliffs.
Q-learning learns the optimal policy regardless of how badly it explores.
"""

from __future__ import annotations

import numpy as np


class TabularAgent:
    """Shared machinery: the Q-table, epsilon-greedy action selection, decay."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        seed: int = 42,
        optimistic_init: float = 0.0,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        # Optimistic initialisation is a form of exploration in itself: unvisited
        # actions look good, so the agent tries them before settling.
        self.q = np.full((n_states, n_actions), optimistic_init, dtype=np.float64)

    def act(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action, or purely greedy when evaluating.

        Ties are broken uniformly at random rather than by argmax's first-index
        rule. With a zero-initialised table every action ties at the start, and
        a deterministic argmax would make the agent always pick action 0 and
        explore far more slowly than epsilon alone suggests.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        return self.q.argmax(axis=1)

    def update(self, *args, **kwargs) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class QLearningAgent(TabularAgent):
    """Off-policy TD control."""

    name = "q_learning"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        # A terminal state has no future, so bootstrapping past it would invent
        # value that does not exist. This is the single most common bug in a
        # hand-written TD update.
        future = 0.0 if terminated else self.q[next_state].max()
        target = reward + self.gamma * future
        error = target - self.q[state, action]
        self.q[state, action] += self.alpha * error
        return float(error)


class SarsaAgent(TabularAgent):
    """On-policy TD control."""

    name = "sarsa"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: int | None = None,
    ) -> float:
        if terminated:
            future = 0.0
        else:
            if next_action is None:
                raise ValueError("SARSA needs the action actually taken next")
            future = self.q[next_state, next_action]
        target = reward + self.gamma * future
        error = target - self.q[state, action]
        self.q[state, action] += self.alpha * error
        return float(error)


class RandomAgent:
    """Uniform random policy: the floor every learned policy must clear."""

    name = "random"

    def __init__(self, n_actions: int, seed: int = 42) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0

    def act(self, state: int, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def decay_epsilon(self) -> None:
        pass

    def update(self, *args, **kwargs) -> float:
        return 0.0
