"""Dynamic programming on the known MDP: the ceiling learning must be measured against.

FrozenLake is small enough that its transition model is available in full, so
the optimal policy can be computed exactly rather than learned. That matters
here for one reason: it turns "the agent reached a 73% success rate" into "the
agent reached 73% against a provable ceiling of X%", and only the second
statement says whether the agent is any good.

The slippery variant is the interesting case. Each intended move succeeds with
probability 1/3, and slides to one of the two perpendicular directions with
probability 1/3 each. Even a perfect policy fails often, so a success rate well
below 100% can still be optimal -- and without computing the ceiling there is
no way to know whether a gap is the agent's fault or the ice's.

Value iteration is used rather than policy iteration; both converge to the same
fixed point, and value iteration is a three-line update:

    V(s) <- max_a  sum_s'  P(s'|s,a) [ r(s,a,s') + gamma * V(s') ]


The ceiling depends on the time limit
-------------------------------------
The figure usually quoted for optimal play on slippery 4x4 FrozenLake is about
74%. That number is not a property of the MDP. It is a property of the MDP
*plus* Gymnasium's default `TimeLimit` of 100 steps per episode.

The optimal policy is slow on purpose. It hugs walls and accepts sideways
slides so that no random slip can push it into a hole, which means it
frequently takes far more than 100 steps to cross a 4x4 grid. Cutting episodes
at 100 truncates a meaningful share of eventual successes:

    horizon    50   ->  0.5356
    horizon   100   ->  0.7402      <- the commonly quoted "~74%"
    horizon   200   ->  0.8164
    horizon   500   ->  0.8235
    unbounded       ->  0.8235      = 14/17

Both numbers are correct answers to different questions, and reporting either
one without naming the horizon is meaningless. `exact_success_probability`
therefore takes the horizon explicitly and defaults to reading it off the
environment's own `TimeLimit`, so the ceiling always matches the conditions the
learned agent was actually evaluated under.
"""

from __future__ import annotations

import numpy as np


def value_iteration(
    env,
    discount: float = 0.99,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve a Gymnasium toy-text MDP exactly.

    Reads the transition model from `env.unwrapped.P`, the nested dict that
    toy-text environments expose:  P[state][action] -> [(prob, next, reward, done), ...]

    Returns the optimal value function, the greedy policy it induces, and the
    number of sweeps taken.
    """
    model = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    values = np.zeros(n_states, dtype=np.float64)

    for sweep in range(1, max_iterations + 1):  # noqa: B007 - returned after the loop
        delta = 0.0
        for state in range(n_states):
            action_values = np.empty(n_actions)
            for action in range(n_actions):
                total = 0.0
                for prob, next_state, reward, done in model[state][action]:
                    # No bootstrapping past a terminal state.
                    future = 0.0 if done else values[next_state]
                    total += prob * (reward + discount * future)
                action_values[action] = total
            best = action_values.max()
            delta = max(delta, abs(best - values[state]))
            values[state] = best
        if delta < tolerance:
            break

    policy = np.empty(n_states, dtype=np.int64)
    for state in range(n_states):
        action_values = np.empty(n_actions)
        for action in range(n_actions):
            total = 0.0
            for prob, next_state, reward, done in model[state][action]:
                future = 0.0 if done else values[next_state]
                total += prob * (reward + discount * future)
            action_values[action] = total
        policy[state] = int(action_values.argmax())

    return values, policy, sweep


def finite_horizon_optimum(env, horizon: int) -> tuple[float, np.ndarray]:
    """The true ceiling for "reach the goal within `horizon` steps".

    This is a different problem from the one `value_iteration` solves, and the
    distinction is easy to miss.

    `value_iteration` finds the policy maximising discounted return over an
    unbounded horizon. Evaluating that policy under a 100-step cap tells you how
    well *it* does when interrupted -- but it is not the best any policy could
    do under that cap, because the discounted-optimal policy does not know it is
    running out of time. A policy that knows step 95 of 100 has arrived should
    abandon the slow, safe, wall-hugging route and gamble on a direct dash.

    The optimum for a time-limited objective is therefore **non-stationary**:
    the right action depends on how many steps remain. Backward induction gives
    it exactly:

        V_0(s)     = 0
        V_{k}(s)   = max_a sum_s' P(s'|s,a) [ r(s,a,s') + V_{k-1}(s') ]

    where V_k(s) is the probability of reaching the goal from s within k steps.
    No discounting is applied: the objective is a probability, not a return, and
    discounting would trade away distant successes that the metric counts in
    full.

    Returns the optimal success probability from the start state and the
    time-indexed policy, shaped (horizon, n_states), where row k is the action
    to take with k+1 steps remaining.
    """
    model = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    values = np.zeros(n_states, dtype=np.float64)
    policy = np.zeros((horizon, n_states), dtype=np.int64)

    for step in range(horizon):
        new_values = np.empty(n_states, dtype=np.float64)
        for state in range(n_states):
            action_values = np.empty(n_actions)
            for action in range(n_actions):
                total = 0.0
                for prob, next_state, reward, done in model[state][action]:
                    future = 0.0 if done else values[next_state]
                    total += prob * (reward + future)
                action_values[action] = total
            best = int(action_values.argmax())
            policy[step, state] = best
            new_values[state] = action_values[best]
        values = new_values

    return float(values[0]), policy


def exact_success_probability(
    env, policy: np.ndarray, max_steps: int | None = None
) -> float:
    """Probability that a policy reaches the goal within `max_steps`, computed exactly.

    Rather than estimating the success rate by sampling episodes -- which has
    its own sampling error and would muddy a comparison against a learned agent
    measured the same way -- the state distribution is propagated forward
    directly under the policy's induced Markov chain, one step at a time. The
    result carries no Monte Carlo noise at all.

    `max_steps` defaults to the environment's own TimeLimit, which is what the
    learned agent is scored under. Pass a large value to get the unconstrained
    MDP optimum instead; the two differ substantially here, and the module
    docstring explains why.
    """
    if max_steps is None:
        max_steps = getattr(env.spec, "max_episode_steps", None) or 1000

    model = env.unwrapped.P
    n_states = env.observation_space.n

    distribution = np.zeros(n_states)
    distribution[0] = 1.0  # FrozenLake always starts at state 0
    success = 0.0

    for _ in range(max_steps):
        if distribution.sum() < 1e-15:
            break
        nxt = np.zeros(n_states)
        for state in range(n_states):
            mass = distribution[state]
            if mass <= 0.0:
                continue
            for prob, next_state, reward, done in model[state][policy[state]]:
                if done:
                    # Reaching the goal pays 1; falling in a hole pays 0.
                    success += mass * prob * reward
                else:
                    nxt[next_state] += mass * prob
        distribution = nxt

    return float(success)
