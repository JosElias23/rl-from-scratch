"""Training and evaluation loops.

Two rules shape everything here.

**Training performance is not evaluation performance.** An agent behaving
epsilon-greedily is deliberately making random moves; its training return
understates what its learned policy can do. Every number reported in the README
comes from `evaluate`, which acts greedily and never updates the table. Quoting
a training curve as a result is the most common way RL projects overstate
themselves.

**Evaluation must be seeded separately from training.** Reusing the training
seed makes the agent look better than it is on stochastic environments, because
it is scored on episodes whose randomness it has already met.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingHistory:
    """Per-episode record, used for the learning-curve figures."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    eval_points: list[int] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    eval_success: list[float] = field(default_factory=list)

    def moving_average(self, window: int = 100) -> np.ndarray:
        r = np.asarray(self.returns, dtype=float)
        if len(r) < window:
            return r
        kernel = np.ones(window) / window
        return np.convolve(r, kernel, mode="valid")


def run_episode(
    env,
    agent,
    discretizer,
    learn: bool = True,
    greedy: bool = False,
    max_steps: int = 10_000,
    seed: int | None = None,
) -> tuple[float, int, bool]:
    """One episode. Returns (undiscounted return, steps, whether it succeeded).

    `success` means the episode ended with a positive reward on the final
    transition, which is FrozenLake's definition of reaching the goal. For
    CartPole every step pays 1 and the metric of interest is the return itself.
    """
    observation, _ = env.reset(seed=seed)
    state = discretizer(observation)
    action = agent.act(state, greedy=greedy)

    total_reward = 0.0
    steps = 0
    success = False

    for steps in range(1, max_steps + 1):  # noqa: B007 - returned after the loop
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = discretizer(next_observation)
        total_reward += float(reward)

        # Only `terminated` cuts the bootstrap. `truncated` means the time
        # limit was reached, and the state still has future value -- treating a
        # timeout as terminal teaches CartPole that balancing for 500 steps is
        # worth nothing beyond step 500.
        next_action = agent.act(next_state, greedy=greedy)
        if learn:
            agent.update(state, action, reward, next_state, terminated, next_action)

        if terminated or truncated:
            success = terminated and reward > 0
            break

        state, action = next_state, next_action

    return total_reward, steps, success


def train(
    env,
    agent,
    discretizer,
    n_episodes: int,
    eval_env=None,
    eval_every: int = 0,
    eval_episodes: int = 200,
    eval_seed: int = 10_000,
    max_steps: int = 10_000,
    log_every: int = 0,
    logger=None,
) -> TrainingHistory:
    """Train for a fixed number of episodes, optionally evaluating along the way."""
    history = TrainingHistory()

    for episode in range(1, n_episodes + 1):
        total_reward, steps, _ = run_episode(
            env, agent, discretizer, learn=True, greedy=False, max_steps=max_steps
        )
        agent.decay_epsilon()

        history.returns.append(total_reward)
        history.lengths.append(steps)
        history.epsilons.append(getattr(agent, "epsilon", 0.0))

        if eval_every and eval_env is not None and episode % eval_every == 0:
            result = evaluate(
                eval_env, agent, discretizer, eval_episodes, seed=eval_seed,
                max_steps=max_steps,
            )
            history.eval_points.append(episode)
            history.eval_returns.append(result["mean_return"])
            history.eval_success.append(result["success_rate"])

        if log_every and logger and episode % log_every == 0:
            recent = np.mean(history.returns[-log_every:])
            logger.info(
                "episode %6d | train return (last %d) %7.3f | epsilon %.3f",
                episode, log_every, recent, getattr(agent, "epsilon", 0.0),
            )

    return history


def evaluate(
    env,
    agent,
    discretizer,
    n_episodes: int = 1000,
    seed: int = 10_000,
    max_steps: int = 10_000,
) -> dict:
    """Score the greedy policy. No exploration, no learning, no table updates.

    Episodes are seeded consecutively from `seed`, so every agent is scored on
    the identical set of environment realisations. On a stochastic environment
    that removes most of the variance from a comparison between agents.
    """
    returns, lengths, successes = [], [], []

    for i in range(n_episodes):
        total_reward, steps, success = run_episode(
            env, agent, discretizer,
            learn=False, greedy=True, max_steps=max_steps, seed=seed + i,
        )
        returns.append(total_reward)
        lengths.append(steps)
        successes.append(success)

    returns_array = np.asarray(returns, dtype=float)
    return {
        "mean_return": float(returns_array.mean()),
        "std_return": float(returns_array.std(ddof=1)) if len(returns) > 1 else 0.0,
        # Standard error, so a difference between agents can be judged rather
        # than eyeballed.
        "stderr_return": (
            float(returns_array.std(ddof=1) / np.sqrt(len(returns)))
            if len(returns) > 1 else 0.0
        ),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
        "eval_seed": seed,
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate.

    Preferred over the normal approximation because success rates here sit far
    from 0.5 on a few hundred episodes, where the naive interval can extend
    past 1 and misstate the uncertainty.
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))
