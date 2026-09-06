"""Tests for the learning rules and the state discretiser.

The TD updates are four lines each, which makes them easy to write and easy to
get subtly wrong. These tests pin the arithmetic against values computed by
hand, and pin the one behaviour that separates Q-learning from SARSA.
"""

import numpy as np
import pytest

from rlfs.agents import QLearningAgent, RandomAgent, SarsaAgent
from rlfs.discretize import CARTPOLE_BOUNDS, BoxDiscretizer, cartpole_discretizer


@pytest.fixture
def q_agent():
    return QLearningAgent(n_states=5, n_actions=3, learning_rate=0.5, discount=0.9,
                          epsilon_start=0.0, seed=0)


@pytest.fixture
def sarsa_agent():
    return SarsaAgent(n_states=5, n_actions=3, learning_rate=0.5, discount=0.9,
                      epsilon_start=0.0, seed=0)


class TestQLearningUpdate:
    def test_update_matches_hand_computation(self, q_agent):
        q_agent.q[1] = [0.0, 2.0, 4.0]   # max is 4.0
        q_agent.q[0, 0] = 1.0

        q_agent.update(state=0, action=0, reward=1.0, next_state=1, terminated=False)

        # target = 1 + 0.9 * 4 = 4.6 ; new = 1 + 0.5 * (4.6 - 1) = 2.8
        assert q_agent.q[0, 0] == pytest.approx(2.8)

    def test_returns_the_temporal_difference_error(self, q_agent):
        q_agent.q[1] = [0.0, 2.0, 4.0]
        q_agent.q[0, 0] = 1.0
        error = q_agent.update(0, 0, 1.0, 1, terminated=False)
        assert error == pytest.approx(4.6 - 1.0)

    def test_terminal_transition_does_not_bootstrap(self, q_agent):
        """The most common bug in a hand-written TD update.

        A terminal state has no future. Bootstrapping through it invents value
        that does not exist, and on a sparse-reward task it can make the agent
        confidently wrong.
        """
        q_agent.q[1] = [99.0, 99.0, 99.0]  # would dominate if used
        q_agent.q[0, 0] = 0.0

        q_agent.update(state=0, action=0, reward=1.0, next_state=1, terminated=True)

        # target = 1 + 0 = 1 ; new = 0 + 0.5 * (1 - 0) = 0.5
        assert q_agent.q[0, 0] == pytest.approx(0.5)

    def test_only_the_taken_action_is_updated(self, q_agent):
        before = q_agent.q.copy()
        q_agent.update(0, 1, 1.0, 2, terminated=False)
        changed = np.argwhere(q_agent.q != before)
        assert changed.tolist() == [[0, 1]]

    def test_repeated_updates_converge_to_the_fixed_point(self):
        """With a constant reward and a self-loop, Q converges to r / (1 - gamma)."""
        agent = QLearningAgent(1, 1, learning_rate=0.1, discount=0.9, epsilon_start=0.0)
        for _ in range(2000):
            agent.update(0, 0, reward=1.0, next_state=0, terminated=False)
        assert agent.q[0, 0] == pytest.approx(1.0 / (1 - 0.9), rel=1e-4)


class TestSarsaUpdate:
    def test_update_matches_hand_computation(self, sarsa_agent):
        sarsa_agent.q[1] = [0.0, 2.0, 4.0]
        sarsa_agent.q[0, 0] = 1.0

        # Bootstraps from action 1 (value 2.0), NOT the max.
        sarsa_agent.update(0, 0, 1.0, 1, terminated=False, next_action=1)

        # target = 1 + 0.9 * 2 = 2.8 ; new = 1 + 0.5 * (2.8 - 1) = 1.9
        assert sarsa_agent.q[0, 0] == pytest.approx(1.9)

    def test_sarsa_and_q_learning_differ_on_a_suboptimal_next_action(
        self, q_agent, sarsa_agent
    ):
        """The single line of difference between the two algorithms."""
        for agent in (q_agent, sarsa_agent):
            agent.q[1] = [0.0, 2.0, 4.0]
            agent.q[0, 0] = 1.0

        q_agent.update(0, 0, 1.0, 1, terminated=False, next_action=1)
        sarsa_agent.update(0, 0, 1.0, 1, terminated=False, next_action=1)

        assert q_agent.q[0, 0] > sarsa_agent.q[0, 0]

    def test_they_agree_when_the_next_action_is_greedy(self, q_agent, sarsa_agent):
        for agent in (q_agent, sarsa_agent):
            agent.q[1] = [0.0, 2.0, 4.0]
            agent.q[0, 0] = 1.0

        q_agent.update(0, 0, 1.0, 1, terminated=False, next_action=2)
        sarsa_agent.update(0, 0, 1.0, 1, terminated=False, next_action=2)

        assert q_agent.q[0, 0] == pytest.approx(sarsa_agent.q[0, 0])

    def test_missing_next_action_is_rejected(self, sarsa_agent):
        with pytest.raises(ValueError):
            sarsa_agent.update(0, 0, 1.0, 1, terminated=False, next_action=None)


class TestActionSelection:
    def test_greedy_selection_picks_the_maximum(self, q_agent):
        q_agent.q[0] = [1.0, 5.0, 2.0]
        assert all(q_agent.act(0, greedy=True) == 1 for _ in range(20))

    def test_epsilon_one_explores_every_action(self):
        agent = QLearningAgent(1, 4, epsilon_start=1.0, seed=0)
        agent.q[0] = [10.0, 0.0, 0.0, 0.0]
        chosen = {agent.act(0) for _ in range(300)}
        assert chosen == {0, 1, 2, 3}

    def test_ties_are_broken_randomly_not_by_index(self):
        """With a zero-initialised table every action ties.

        argmax would always return index 0, so the agent would explore far more
        slowly than epsilon alone implies -- a silent bug that just looks like
        slow learning.
        """
        agent = QLearningAgent(1, 4, epsilon_start=0.0, seed=0)
        chosen = {agent.act(0, greedy=True) for _ in range(300)}
        assert len(chosen) > 1

    def test_epsilon_decays_towards_the_floor_and_stops(self):
        agent = QLearningAgent(1, 2, epsilon_start=1.0, epsilon_end=0.1,
                               epsilon_decay=0.5)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.1)

    def test_evaluation_does_not_disturb_the_training_stream(self):
        """Measuring the agent must not change the agent.

        Greedy action selection still needs randomness to break ties. If it
        drew from the training generator, inserting a periodic evaluation would
        shift every subsequent training decision -- and it did: the same seed
        and hyperparameters scored 500.00 on CartPole with periodic evaluation
        and 288.47 without, before the two streams were separated.
        """
        def training_actions(with_evaluation: bool) -> list[int]:
            agent = QLearningAgent(4, 3, epsilon_start=0.5, seed=7)
            actions = []
            for step in range(200):
                actions.append(agent.act(step % 4))
                if with_evaluation and step % 10 == 0:
                    # A stand-in for an evaluation pass.
                    for s in range(4):
                        agent.act(s, greedy=True)
            return actions

        assert training_actions(False) == training_actions(True)

    def test_random_agent_ignores_the_state(self):
        agent = RandomAgent(n_actions=3, seed=0)
        chosen = {agent.act(state) for state in range(50) for _ in range(10)}
        assert chosen == {0, 1, 2}


class TestDiscretiser:
    def test_state_count_is_the_product_of_bins(self):
        assert cartpole_discretizer(6).n_states == 6**4
        assert BoxDiscretizer(CARTPOLE_BOUNDS, (3, 4, 5, 6)).n_states == 360

    def test_indices_are_within_range(self):
        d = cartpole_discretizer(6)
        rng = np.random.default_rng(0)
        for _ in range(2000):
            observation = rng.uniform(-5, 5, size=4)
            assert 0 <= d(observation) < d.n_states

    def test_out_of_range_observations_saturate_rather_than_wrap(self):
        """An index that wrapped would alias a falling pole onto a balanced one."""
        d = cartpole_discretizer(6)
        extreme_low = d([-1e6, -1e6, -1e6, -1e6])
        extreme_high = d([1e6, 1e6, 1e6, 1e6])
        assert extreme_low == 0
        assert extreme_high == d.n_states - 1

    def test_encode_and_decode_round_trip(self):
        d = BoxDiscretizer(CARTPOLE_BOUNDS, (3, 4, 5, 6))
        for index in range(d.n_states):
            digits = d.decode(index)
            rebuilt = 0
            for digit, n in zip(digits, d.bins, strict=True):
                rebuilt = rebuilt * n + digit
            assert rebuilt == index

    def test_nearby_observations_usually_share_a_cell(self):
        d = cartpole_discretizer(6)
        base = np.array([0.0, 0.0, 0.0, 0.0])
        assert d(base) == d(base + 1e-6)

    def test_distinct_regions_get_distinct_cells(self):
        d = cartpole_discretizer(6)
        upright = d([0.0, 0.0, 0.0, 0.0])
        falling = d([0.0, 0.0, 0.19, 2.5])
        assert upright != falling

    def test_mismatched_bin_specification_is_rejected(self):
        with pytest.raises(ValueError):
            BoxDiscretizer(CARTPOLE_BOUNDS, (3, 4))

    def test_zero_bins_is_rejected(self):
        with pytest.raises(ValueError):
            BoxDiscretizer(CARTPOLE_BOUNDS, 0)
