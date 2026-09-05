"""Train tabular agents on slippery FrozenLake and score them against the exact optimum.

Pipeline:
  1. Solve the MDP exactly with value iteration -- this is the ceiling.
  2. Measure the random policy -- this is the floor.
  3. Train Q-learning and SARSA from scratch.
  4. Evaluate every policy greedily on the same seeded episodes.

The value-iteration step is what makes the learned numbers interpretable. A
success rate of 0.73 means nothing on its own; 0.73 against a provable ceiling
of 0.74 means the agent is essentially optimal.

Usage:
    python scripts/train_frozenlake.py
    python scripts/train_frozenlake.py --episodes 50000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from rlfs.agents import QLearningAgent, RandomAgent, SarsaAgent  # noqa: E402
from rlfs.discretize import IdentityDiscretizer  # noqa: E402
from rlfs.planning import (  # noqa: E402
    exact_success_probability,
    finite_horizon_optimum,
    value_iteration,
)
from rlfs.training import evaluate, train, wilson_interval  # noqa: E402
from rlfs.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402

ACTION_NAMES = ["<", "v", ">", "^"]


class PolicyAgent:
    """Wraps a fixed action table so a planned policy can use the same evaluator."""

    def __init__(self, policy: np.ndarray) -> None:
        self.policy = policy
        self.epsilon = 0.0

    def act(self, state: int, greedy: bool = False) -> int:
        return int(self.policy[state])

    def decay_epsilon(self) -> None:
        pass

    def update(self, *args, **kwargs) -> float:
        return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    cfg = config["frozenlake"]
    n_episodes = args.episodes or cfg["n_episodes"]
    n_eval = args.eval_episodes or cfg["eval_episodes"]

    def make_env():
        return gym.make(
            cfg["env_id"], map_name=cfg["map_name"], is_slippery=cfg["is_slippery"]
        )

    env = make_env()
    eval_env = make_env()
    horizon = env.spec.max_episode_steps
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    discretizer = IdentityDiscretizer(n_states)

    log.info("FrozenLake %s slippery=%s | %d states, %d actions | TimeLimit %d steps",
             cfg["map_name"], cfg["is_slippery"], n_states, n_actions, horizon)

    results: dict[str, dict] = {}

    # --- 1. the exact ceiling --------------------------------------------
    t0 = time.perf_counter()
    values, optimal_policy, sweeps = value_iteration(
        env,
        discount=config["planning"]["discount"],
        tolerance=config["planning"]["tolerance"],
    )
    planning_seconds = time.perf_counter() - t0

    stationary_under_cap = exact_success_probability(
        env, optimal_policy, max_steps=horizon
    )
    ceiling_unbounded = exact_success_probability(env, optimal_policy, max_steps=100_000)

    # The evaluated objective is "reach the goal within `horizon` steps", and its
    # optimum is time-dependent: a policy running out of time should abandon the
    # slow safe route. Backward induction gives that ceiling exactly, and it is
    # the only fair denominator for an agent scored under the same time limit.
    ceiling, time_aware_policy = finite_horizon_optimum(env, horizon)

    log.info("Value iteration: %d sweeps in %.2fs", sweeps, planning_seconds)
    log.info("Unbounded-horizon optimum          : %.4f", ceiling_unbounded)
    log.info("Discounted policy capped at %3d    : %.4f", horizon, stationary_under_cap)
    log.info("TRUE %d-step optimum (time-aware) : %.4f  <- the ceiling", horizon, ceiling)

    # Sanity: the analytic ceiling must agree with simulating the same policy.
    simulated = evaluate(eval_env, PolicyAgent(optimal_policy), discretizer, n_eval,
                         seed=10_000, max_steps=horizon)
    drift = abs(simulated["success_rate"] - stationary_under_cap)
    log.info("Stationary policy simulated (%d eps): %.4f (analytic drift %.4f)",
             n_eval, simulated["success_rate"], drift)
    if drift > 0.02:
        log.error("Analytic value and simulation disagree by %.4f", drift)
        return 1

    results["value_iteration"] = {
        "success_rate": round(ceiling, 4),
        "success_rate_time_aware_optimum": round(ceiling, 4),
        "success_rate_stationary_under_cap": round(stationary_under_cap, 4),
        "success_rate_unbounded_horizon": round(ceiling_unbounded, 4),
        "success_rate_stationary_simulated": round(simulated["success_rate"], 4),
        "gain_from_time_awareness": round(ceiling - stationary_under_cap, 4),
        "value_of_start_state": round(float(values[0]), 6),
        "sweeps": sweeps,
        "seconds": round(planning_seconds, 3),
        "policy": [ACTION_NAMES[a] for a in optimal_policy.tolist()],
    }
    _ = time_aware_policy  # kept for the notebook; not needed downstream

    # --- 2. the floor -----------------------------------------------------
    random_result = evaluate(eval_env, RandomAgent(n_actions, seed=config["seed"]),
                             discretizer, n_eval, seed=10_000, max_steps=horizon)
    log.info("Random policy: %.4f", random_result["success_rate"])
    results["random"] = {
        "success_rate": round(random_result["success_rate"], 4),
        "mean_length": round(random_result["mean_length"], 2),
    }

    # --- 3. learning ------------------------------------------------------
    histories = {}
    for AgentClass in (QLearningAgent, SarsaAgent):
        set_seed(config["seed"])
        agent = AgentClass(n_states, n_actions, seed=config["seed"], **cfg["agent"])

        log.info("Training %s for %d episodes ...", agent.name, n_episodes)
        t0 = time.perf_counter()
        history = train(
            make_env(), agent, discretizer, n_episodes,
            eval_env=eval_env, eval_every=cfg["eval_every"], eval_episodes=1000,
            max_steps=horizon, log_every=max(1, n_episodes // 5), logger=log,
        )
        elapsed = time.perf_counter() - t0

        result = evaluate(eval_env, agent, discretizer, n_eval, seed=10_000,
                          max_steps=horizon)
        wins = int(round(result["success_rate"] * n_eval))
        low, high = wilson_interval(wins, n_eval)

        agreement = float((agent.greedy_policy() == optimal_policy).mean())

        # A stationary learned policy cannot beat the time-aware optimum in
        # expectation, so a point estimate above the ceiling is sampling noise.
        indistinguishable = low <= ceiling <= high
        log.info("%s: success %.4f [%.4f, %.4f] | %.1f%% of ceiling | policy match %.0f%%%s",
                 agent.name, result["success_rate"], low, high,
                 100 * result["success_rate"] / ceiling, 100 * agreement,
                 "  (ceiling inside CI)" if indistinguishable else "")

        results[agent.name] = {
            "success_rate": round(result["success_rate"], 4),
            "ci95_lower": round(low, 4),
            "ci95_upper": round(high, 4),
            "fraction_of_ceiling": round(result["success_rate"] / ceiling, 4),
            "mean_length": round(result["mean_length"], 2),
            "policy_agreement_with_optimal": round(agreement, 4),
            "statistically_indistinguishable_from_optimal": bool(low <= ceiling <= high),
            "train_seconds": round(elapsed, 1),
            "episodes": n_episodes,
            "final_epsilon": round(agent.epsilon, 4),
            "policy": [ACTION_NAMES[a] for a in agent.greedy_policy().tolist()],
        }
        histories[agent.name] = history

    # --- 4. report --------------------------------------------------------
    np.savez(
        Path(__file__).resolve().parents[1] / "reports" / "frozenlake_history.npz",
        **{
            f"{name}_{field}": np.asarray(getattr(h, field))
            for name, h in histories.items()
            for field in ("returns", "eval_points", "eval_success", "epsilons")
        },
    )

    save_json(
        {
            "environment": {
                "id": cfg["env_id"],
                "map": cfg["map_name"],
                "is_slippery": cfg["is_slippery"],
                "time_limit_steps": horizon,
                "n_states": int(n_states),
                "n_actions": int(n_actions),
            },
            "eval_episodes": n_eval,
            "results": results,
        },
        "reports/metrics_frozenlake.json",
    )

    print()
    print("| Agent | Success rate | 95% CI | % of exact optimum |")
    print("|---|---:|:--|---:|")
    for key in ("random", "sarsa", "q_learning", "value_iteration"):
        r = results[key]
        ci = (
            f"[{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}]"
            if "ci95_lower" in r else "exact" if key == "value_iteration" else "-"
        )
        pct = 100 * r["success_rate"] / ceiling
        print(f"| {key} | {r['success_rate']:.4f} | {ci} | {pct:.1f}% |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
