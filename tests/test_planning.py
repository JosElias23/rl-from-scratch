"""Tests for the dynamic-programming ceiling.

Every learned result in this project is reported as a fraction of the exact
optimum. If the optimum is wrong, every headline number is wrong, so these
tests check the planner against closed-form answers and against simulation.
"""

import gymnasium as gym
import numpy as np
import pytest

from rlfs.planning import (
    exact_success_probability,
    finite_horizon_optimum,
    value_iteration,
)

# Published result for slippery 4x4 FrozenLake under an unbounded horizon.
UNBOUNDED_OPTIMUM = 14 / 17  # 0.8235294...


@pytest.fixture(scope="module")
def slippery_env():
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)


@pytest.fixture(scope="module")
def deterministic_env():
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)


@pytest.fixture(scope="module")
def solved(slippery_env):
    return value_iteration(slippery_env, discount=0.99, tolerance=1e-13)


class TestValueIteration:
    def test_converges(self, solved):
        _values, _policy, sweeps = solved
        assert sweeps < 100_000

    def test_values_are_probabilities_when_undiscounted(self, slippery_env):
        """With gamma = 1 and a 0/1 reward, V(s) is a probability of success."""
        values, _, _ = value_iteration(slippery_env, discount=1.0, tolerance=1e-13)
        assert np.all(values >= -1e-9)
        assert np.all(values <= 1 + 1e-9)

    def test_terminal_states_have_zero_value(self, slippery_env, solved):
        """Holes and the goal are absorbing: nothing can be earned from them."""
        values, _, _ = solved
        model = slippery_env.unwrapped.P
        for state in range(slippery_env.observation_space.n):
            is_terminal = all(
                done
                for action in model[state]
                for _p, _s, _r, done in model[state][action]
            )
            if is_terminal:
                assert values[state] == pytest.approx(0.0, abs=1e-9)

    def test_deterministic_lake_is_solved_perfectly(self, deterministic_env):
        """Without slip there is a safe path, so the optimum must be exactly 1."""
        _, policy, _ = value_iteration(deterministic_env, discount=0.99, tolerance=1e-13)
        probability = exact_success_probability(deterministic_env, policy, max_steps=1000)
        assert probability == pytest.approx(1.0, abs=1e-9)

    def test_unbounded_optimum_matches_the_published_value(self, slippery_env, solved):
        _, policy, _ = solved
        probability = exact_success_probability(slippery_env, policy, max_steps=100_000)
        assert probability == pytest.approx(UNBOUNDED_OPTIMUM, abs=1e-6)

    def test_discount_does_not_change_the_optimal_policy_here(self, slippery_env, solved):
        """A useful invariant: on this map the ranking is robust to gamma.

        It is not true in general -- a lower discount would prefer faster,
        riskier routes -- so the test documents that the reported ceiling is not
        an artefact of one particular gamma.
        """
        _, policy_099, _ = solved
        _, policy_1, _ = value_iteration(slippery_env, discount=1.0, tolerance=1e-13)
        p099 = exact_success_probability(slippery_env, policy_099, max_steps=100_000)
        p1 = exact_success_probability(slippery_env, policy_1, max_steps=100_000)
        assert p099 == pytest.approx(p1, abs=1e-6)


class TestHorizonSensitivity:
    """The ceiling depends on the time limit, and the project reports both."""

    def test_success_probability_increases_with_the_horizon(self, slippery_env, solved):
        _, policy, _ = solved
        probabilities = [
            exact_success_probability(slippery_env, policy, max_steps=h)
            for h in (10, 50, 100, 200, 500, 1000)
        ]
        assert all(a <= b + 1e-12 for a, b in zip(probabilities, probabilities[1:], strict=False))

    def test_the_hundred_step_value_is_far_below_the_unbounded_one(
        self, slippery_env, solved
    ):
        """The gap this project exists to point out.

        The widely quoted "~74%" ceiling is the 100-step number, not a property
        of the MDP. Confusing the two makes an agent look either far better or
        far worse than it is.
        """
        _, policy, _ = solved
        capped = exact_success_probability(slippery_env, policy, max_steps=100)
        unbounded = exact_success_probability(slippery_env, policy, max_steps=100_000)
        assert capped == pytest.approx(0.7402, abs=1e-3)
        assert unbounded - capped > 0.07

    def test_horizon_defaults_to_the_environment_time_limit(self, slippery_env, solved):
        _, policy, _ = solved
        default = exact_success_probability(slippery_env, policy)
        explicit = exact_success_probability(
            slippery_env, policy, max_steps=slippery_env.spec.max_episode_steps
        )
        assert default == pytest.approx(explicit)


class TestFiniteHorizonOptimum:
    def test_time_aware_policy_beats_the_stationary_one(self, slippery_env, solved):
        """The reason the ceiling needs backward induction at all.

        A stationary policy cannot know it is running out of time. The
        time-aware optimum is therefore strictly greater, and using the
        stationary value as the denominator would let a learned agent appear to
        exceed 100% of "optimal".
        """
        _, policy, _ = solved
        horizon = slippery_env.spec.max_episode_steps
        stationary = exact_success_probability(slippery_env, policy, max_steps=horizon)
        ceiling, _ = finite_horizon_optimum(slippery_env, horizon)
        assert ceiling > stationary

    def test_ceiling_matches_simulating_the_time_aware_policy(self, slippery_env):
        """Backward induction must agree with actually running the policy."""
        horizon = slippery_env.spec.max_episode_steps
        ceiling, policy = finite_horizon_optimum(slippery_env, horizon)

        wins, n = 0, 20_000
        for i in range(n):
            state, _ = slippery_env.reset(seed=i)
            for step in range(horizon):
                action = int(policy[horizon - 1 - step, state])
                state, reward, terminated, truncated, _ = slippery_env.step(action)
                if terminated or truncated:
                    wins += int(reward > 0)
                    break

        empirical = wins / n
        stderr = np.sqrt(empirical * (1 - empirical) / n)
        assert abs(empirical - ceiling) < 4 * stderr

    def test_longer_horizons_approach_the_unbounded_optimum(self, slippery_env):
        near_infinite, _ = finite_horizon_optimum(slippery_env, 1000)
        assert near_infinite == pytest.approx(UNBOUNDED_OPTIMUM, abs=1e-4)

    def test_a_single_step_horizon_cannot_reach_a_distant_goal(self, slippery_env):
        one_step, _ = finite_horizon_optimum(slippery_env, 1)
        assert one_step == pytest.approx(0.0, abs=1e-12)

    def test_policy_shape_is_time_indexed(self, slippery_env):
        horizon = 20
        _, policy = finite_horizon_optimum(slippery_env, horizon)
        assert policy.shape == (horizon, slippery_env.observation_space.n)
