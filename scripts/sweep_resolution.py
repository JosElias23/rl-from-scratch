"""Measure how CartPole performance depends on discretisation resolution.

A first attempt at this ran one seed per resolution and produced nonsense:
3 bins scored 499.8, 4 bins scored 38.7, 5 bins scored 500.0, 6 bins scored
288.5. That is not a bias-variance curve, it is noise with a trend hidden
somewhere underneath.

Tabular Q-learning on CartPole is genuinely high-variance: whether a run
converges depends on which states happened to be visited early, and a single
seed says almost nothing. So every resolution is trained from several seeds and
reported as a mean with a spread. If the error bars overlap, the honest answer
is that the resolutions are indistinguishable -- and saying so is worth more
than a clean-looking line through noise.

Runs are spread across processes because 7 resolutions x 5 seeds is 35 full
training runs.

Usage:
    python scripts/sweep_resolution.py
    python scripts/sweep_resolution.py --seeds 3 --episodes 30000 --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from rlfs.agents import QLearningAgent  # noqa: E402
from rlfs.discretize import cartpole_discretizer  # noqa: E402
from rlfs.training import evaluate, train  # noqa: E402
from rlfs.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402

BIN_COUNTS = (3, 4, 5, 6, 8, 10, 12)
SOLVED_THRESHOLD = 195.0


def run_one(task: tuple) -> dict:
    """One (resolution, seed) training run. Module-level so it can be pickled."""
    bins, seed, n_episodes, n_eval, agent_cfg, env_id, max_steps = task

    set_seed(seed)
    discretizer = cartpole_discretizer(bins)
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    agent = QLearningAgent(
        discretizer.n_states, env.action_space.n, seed=seed, **agent_cfg
    )
    train(env, agent, discretizer, n_episodes, max_steps=max_steps)
    result = evaluate(eval_env, agent, discretizer, n_eval, seed=10_000,
                      max_steps=max_steps)

    visited = int((agent.q != agent_cfg.get("optimistic_init", 0.0)).any(axis=1).sum())
    return {
        "bins": bins,
        "seed": seed,
        "mean_return": round(result["mean_return"], 2),
        "n_table_states": discretizer.n_states,
        "states_visited": visited,
        "coverage": round(visited / discretizer.n_states, 4),
        "solved": bool(result["mean_return"] >= SOLVED_THRESHOLD),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()

    cfg = config["cartpole"]
    n_episodes = args.episodes or cfg["n_episodes"]
    base_seed = config["seed"]
    seeds = [base_seed + 1000 * i for i in range(args.seeds)]

    tasks = [
        (bins, seed, n_episodes, args.eval_episodes, cfg["agent"],
         cfg["env_id"], cfg["max_steps"])
        for bins in BIN_COUNTS
        for seed in seeds
    ]
    log.info("%d runs (%d resolutions x %d seeds), %d episodes each",
             len(tasks), len(BIN_COUNTS), len(seeds), n_episodes)

    from concurrent.futures import ProcessPoolExecutor

    runs: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(run_one, tasks), start=1):
            runs.append(result)
            log.info("[%2d/%2d] bins=%2d seed=%5d -> %7.2f",
                     i, len(tasks), result["bins"], result["seed"],
                     result["mean_return"])

    # Aggregate across seeds.
    summary = []
    for bins in BIN_COUNTS:
        subset = [r for r in runs if r["bins"] == bins]
        returns = np.array([r["mean_return"] for r in subset])
        summary.append({
            "bins": bins,
            "n_table_states": subset[0]["n_table_states"],
            "mean_states_visited": int(np.mean([r["states_visited"] for r in subset])),
            "mean_coverage": round(float(np.mean([r["coverage"] for r in subset])), 4),
            "n_seeds": len(subset),
            "mean_return": round(float(returns.mean()), 2),
            "std_return": round(float(returns.std(ddof=1)), 2),
            "stderr_return": round(float(returns.std(ddof=1) / np.sqrt(len(returns))), 2),
            "min_return": round(float(returns.min()), 2),
            "max_return": round(float(returns.max()), 2),
            "seeds_solved": int(sum(r["solved"] for r in subset)),
        })

    save_json(
        {
            "episodes_per_run": n_episodes,
            "eval_episodes": args.eval_episodes,
            "seeds": seeds,
            "summary": summary,
            "runs": runs,
        },
        "reports/metrics_resolution_sweep.json",
    )

    print()
    print("| Bins | Table states | Visited | Mean return | sd | Range | Seeds solved |")
    print("|---:|---:|---:|---:|---:|:--|---:|")
    for s in summary:
        print(f"| {s['bins']} | {s['n_table_states']:,} | {s['mean_coverage']:.0%} | "
              f"{s['mean_return']:.1f} | {s['std_return']:.1f} | "
              f"{s['min_return']:.0f}-{s['max_return']:.0f} | "
              f"{s['seeds_solved']}/{s['n_seeds']} |")
    print()

    best = max(summary, key=lambda s: s["mean_return"])
    overlapping = [
        s["bins"] for s in summary
        if s["mean_return"] + 1.96 * s["stderr_return"]
        >= best["mean_return"] - 1.96 * best["stderr_return"]
    ]
    print(f"Best mean: {best['bins']} bins ({best['mean_return']:.1f}).")
    print(f"Intervals overlapping it: {overlapping}")
    print()
    print("Overlapping intervals are NOT a test of a difference: they are far too")
    print("conservative, and two estimates whose intervals overlap can still differ")
    print("significantly. Bin count is also an ordered factor, so comparing pairs")
    print("throws the ordering away. Run scripts/analyse_results.py for the trend")
    print("test, which finds an effect this pairwise view cannot localise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
