"""Train tabular agents on CartPole by quantising its continuous observation.

CartPole has no finite state space, so a table cannot index it directly. The
four real-valued observations are binned, and the resulting resolution is the
central design choice: too coarse and distinct situations collapse into one
cell, too fine and no cell is visited often enough to converge.

`--sweep` measures that trade-off directly rather than asserting it, training a
fresh agent at each resolution from 3 to 12 bins per dimension.

Usage:
    python scripts/train_cartpole.py
    python scripts/train_cartpole.py --sweep
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
from rlfs.discretize import cartpole_discretizer  # noqa: E402
from rlfs.training import evaluate, train  # noqa: E402
from rlfs.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402

# The classic "solved" bar for CartPole: mean return >= 195 over 100 consecutive
# episodes. CartPole-v1 truncates at 500, so that is the hard maximum.
SOLVED_THRESHOLD = 195.0
MAX_RETURN = 500.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--bins", type=int, default=None)
    p.add_argument("--sweep", action="store_true", help="Train at several resolutions")
    return p.parse_args()


def train_one(cfg, config, log, bins, n_episodes, n_eval, AgentClass, track_history=False):
    """Train a single agent at a given discretisation and return its evaluation."""
    set_seed(config["seed"])
    discretizer = cartpole_discretizer(bins)

    env = gym.make(cfg["env_id"])
    eval_env = gym.make(cfg["env_id"])

    agent = AgentClass(
        discretizer.n_states, env.action_space.n, seed=config["seed"], **cfg["agent"]
    )

    t0 = time.perf_counter()
    history = train(
        env, agent, discretizer, n_episodes,
        eval_env=eval_env if track_history else None,
        eval_every=cfg["eval_every"] if track_history else 0,
        eval_episodes=100,
        max_steps=cfg["max_steps"],
        log_every=max(1, n_episodes // 4) if track_history else 0,
        logger=log,
    )
    elapsed = time.perf_counter() - t0

    result = evaluate(eval_env, agent, discretizer, n_eval, seed=10_000,
                      max_steps=cfg["max_steps"])
    result["bins"] = bins
    result["n_table_states"] = discretizer.n_states
    result["states_visited"] = int((agent.q != cfg["agent"]["optimistic_init"]).any(axis=1).sum())
    result["train_seconds"] = round(elapsed, 1)
    result["solved"] = bool(result["mean_return"] >= SOLVED_THRESHOLD)
    result["fraction_of_max"] = round(result["mean_return"] / MAX_RETURN, 4)
    return result, agent, history


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    cfg = config["cartpole"]
    n_episodes = args.episodes or cfg["n_episodes"]
    n_eval = args.eval_episodes or cfg["eval_episodes"]
    bins = args.bins or cfg["bins"]

    log.info("CartPole | %d episodes | %d bins/dim = %d states | eval on %d episodes",
             n_episodes, bins, bins**4, n_eval)

    results: dict = {}

    # --- floor ------------------------------------------------------------
    eval_env = gym.make(cfg["env_id"])
    discretizer = cartpole_discretizer(bins)
    random_result = evaluate(
        eval_env, RandomAgent(eval_env.action_space.n, seed=config["seed"]),
        discretizer, n_eval, seed=10_000, max_steps=cfg["max_steps"],
    )
    log.info("Random policy: mean return %.2f", random_result["mean_return"])
    results["random"] = {
        "mean_return": round(random_result["mean_return"], 2),
        "std_return": round(random_result["std_return"], 2),
        "solved": False,
    }

    # --- learned agents ---------------------------------------------------
    histories = {}
    for AgentClass in (QLearningAgent, SarsaAgent):
        log.info("Training %s ...", AgentClass.name)
        result, agent, history = train_one(
            cfg, config, log, bins, n_episodes, n_eval, AgentClass, track_history=True
        )
        log.info(
            "%s: mean return %.2f +/- %.2f (sd %.1f) | %d/%d table states visited | %s",
            AgentClass.name, result["mean_return"], 1.96 * result["stderr_return"],
            result["std_return"], result["states_visited"], result["n_table_states"],
            "SOLVED" if result["solved"] else "not solved",
        )
        results[AgentClass.name] = {
            "mean_return": round(result["mean_return"], 2),
            "std_return": round(result["std_return"], 2),
            "stderr_return": round(result["stderr_return"], 3),
            "ci95_lower": round(result["mean_return"] - 1.96 * result["stderr_return"], 2),
            "ci95_upper": round(result["mean_return"] + 1.96 * result["stderr_return"], 2),
            "solved": result["solved"],
            "fraction_of_max": result["fraction_of_max"],
            "bins": bins,
            "n_table_states": result["n_table_states"],
            "states_visited": result["states_visited"],
            "train_seconds": result["train_seconds"],
            "episodes": n_episodes,
        }
        histories[AgentClass.name] = history

    # --- resolution sweep -------------------------------------------------
    if args.sweep:
        log.info("Sweeping discretisation resolution ...")
        sweep = []
        for b in (3, 4, 5, 6, 8, 10, 12):
            result, _, _ = train_one(
                cfg, config, log, b, n_episodes, max(200, n_eval // 2), QLearningAgent
            )
            log.info("  bins=%2d (%6d states, %5d visited): mean return %.2f",
                     b, result["n_table_states"], result["states_visited"],
                     result["mean_return"])
            sweep.append({
                "bins": b,
                "n_table_states": result["n_table_states"],
                "states_visited": result["states_visited"],
                "coverage": round(result["states_visited"] / result["n_table_states"], 4),
                "mean_return": round(result["mean_return"], 2),
                "std_return": round(result["std_return"], 2),
                "solved": result["solved"],
            })
        results["resolution_sweep"] = sweep

    np.savez(
        Path(__file__).resolve().parents[1] / "reports" / "cartpole_history.npz",
        **{
            f"{name}_{field}": np.asarray(getattr(h, field))
            for name, h in histories.items()
            for field in ("returns", "eval_points", "eval_returns", "epsilons")
        },
    )

    save_json(
        {
            "environment": {
                "id": cfg["env_id"],
                "max_steps": cfg["max_steps"],
                "solved_threshold": SOLVED_THRESHOLD,
            },
            "discretisation": {"bins_per_dimension": bins, "n_states": bins**4},
            "eval_episodes": n_eval,
            "results": results,
        },
        "reports/metrics_cartpole.json",
    )

    print()
    print("| Agent | Mean return | 95% CI | Solved (>=195) |")
    print("|---|---:|:--|:--:|")
    for key in ("random", "sarsa", "q_learning"):
        r = results[key]
        ci = f"[{r['ci95_lower']:.1f}, {r['ci95_upper']:.1f}]" if "ci95_lower" in r else "-"
        print(f"| {key} | {r['mean_return']:.2f} | {ci} | {'yes' if r['solved'] else 'no'} |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
