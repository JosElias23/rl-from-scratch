"""Compare Q-learning and SARSA on CartPole across several seeds.

A single seed cannot separate these two algorithms on this task, and trying to
is how RL projects end up claiming things that do not replicate. Two runs of
this repository, differing only in a random-number-stream bug fix, produced:

    before   Q-learning 500.00   SARSA 294.07
    after    Q-learning 408.41   SARSA 500.00

The ranking flipped. Neither run was wrong; the experiment was underpowered.
`sweep_resolution.py` measures the cause directly -- at six bins per dimension
the standard deviation across seeds is 127 return units, which swamps any
plausible difference between the two update rules.

So this script trains both agents from the same set of seeds and reports the
distribution, plus a paired comparison on the seeds they share. If the interval
covers zero, the honest answer is that this experiment cannot tell them apart,
and that is what the README says.

Usage:
    python scripts/compare_agents.py
    python scripts/compare_agents.py --seeds 10 --workers 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from rlfs.agents import QLearningAgent, SarsaAgent  # noqa: E402
from rlfs.discretize import cartpole_discretizer  # noqa: E402
from rlfs.training import evaluate, train  # noqa: E402
from rlfs.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402

AGENTS = {"q_learning": QLearningAgent, "sarsa": SarsaAgent}
SOLVED_THRESHOLD = 195.0


def run_one(task: tuple) -> dict:
    """One (agent, seed) training run. Module-level so it can be pickled."""
    agent_name, seed, bins, n_episodes, n_eval, agent_cfg, env_id, max_steps = task

    set_seed(seed)
    discretizer = cartpole_discretizer(bins)
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    agent = AGENTS[agent_name](
        discretizer.n_states, env.action_space.n, seed=seed, **agent_cfg
    )
    train(env, agent, discretizer, n_episodes, max_steps=max_steps)
    result = evaluate(eval_env, agent, discretizer, n_eval, seed=10_000,
                      max_steps=max_steps)

    return {
        "agent": agent_name,
        "seed": seed,
        "mean_return": round(result["mean_return"], 2),
        "solved": bool(result["mean_return"] >= SOLVED_THRESHOLD),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=8)
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
    bins = cfg["bins"]
    seeds = [config["seed"] + 1000 * i for i in range(args.seeds)]

    tasks = [
        (name, seed, bins, n_episodes, args.eval_episodes, cfg["agent"],
         cfg["env_id"], cfg["max_steps"])
        for name in AGENTS
        for seed in seeds
    ]
    log.info("%d runs (%d agents x %d seeds) at %d bins, %d episodes each",
             len(tasks), len(AGENTS), len(seeds), bins, n_episodes)

    from concurrent.futures import ProcessPoolExecutor

    runs: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(run_one, tasks), start=1):
            runs.append(result)
            log.info("[%2d/%2d] %-11s seed=%5d -> %7.2f",
                     i, len(tasks), result["agent"], result["seed"],
                     result["mean_return"])

    summary = {}
    by_seed = {}
    for name in AGENTS:
        subset = [r for r in runs if r["agent"] == name]
        values = np.array([r["mean_return"] for r in subset])
        by_seed[name] = {r["seed"]: r["mean_return"] for r in subset}
        summary[name] = {
            "n_seeds": len(values),
            "mean_return": round(float(values.mean()), 2),
            "std_return": round(float(values.std(ddof=1)), 2),
            "stderr_return": round(float(values.std(ddof=1) / np.sqrt(len(values))), 2),
            "min_return": round(float(values.min()), 2),
            "max_return": round(float(values.max()), 2),
            "median_return": round(float(np.median(values)), 2),
            "seeds_solved": int(sum(r["solved"] for r in subset)),
        }

    # Paired comparison, because the design is paired: both agents were run on
    # the same seeds. Note what this does NOT buy here. The usual argument is
    # that pairing removes the shared difficulty of a seed, but a seed only
    # fixes an RNG stream, and two agents that consume randomness differently
    # are on unrelated trajectories after their first differing update. The two
    # arms turn out to be negatively correlated, so pairing widens the interval
    # rather than narrowing it. The paired test is still the right one to report
    # for a paired design -- switching to whichever test gives a smaller p would
    # be choosing the answer -- but the reason for it was wrong, and
    # scripts/analyse_results.py quantifies that.
    shared = sorted(set(by_seed["q_learning"]) & set(by_seed["sarsa"]))
    differences = np.array(
        [by_seed["q_learning"][s] - by_seed["sarsa"][s] for s in shared]
    )
    n = len(differences)
    mean_difference = float(differences.mean())
    stderr = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    # t critical value at 95% for small n, close enough from a normal table.
    t_crit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45,
              8: 2.36, 9: 2.31, 10: 2.26}.get(n - 1, 1.96)
    margin = t_crit * stderr

    comparison = {
        "n_paired_seeds": n,
        "mean_difference_q_minus_sarsa": round(mean_difference, 2),
        "ci95_lower": round(mean_difference - margin, 2),
        "ci95_upper": round(mean_difference + margin, 2),
        "significant": bool(abs(mean_difference) > margin),
        "seeds_q_learning_wins": int((differences > 0).sum()),
        "seeds_sarsa_wins": int((differences < 0).sum()),
    }

    save_json(
        {
            "bins": bins,
            "episodes_per_run": n_episodes,
            "eval_episodes": args.eval_episodes,
            "seeds": seeds,
            "summary": summary,
            "paired_comparison": comparison,
            "runs": runs,
        },
        "reports/metrics_agent_comparison.json",
    )

    print()
    print("| Agent | Mean | sd | Median | Range | Seeds solved |")
    print("|---|---:|---:|---:|:--|---:|")
    for name, s in summary.items():
        print(f"| {name} | {s['mean_return']:.1f} | {s['std_return']:.1f} | "
              f"{s['median_return']:.1f} | {s['min_return']:.0f}-{s['max_return']:.0f} | "
              f"{s['seeds_solved']}/{s['n_seeds']} |")
    print()
    print(f"Paired difference (Q-learning minus SARSA) over {n} seeds: "
          f"{mean_difference:+.1f}, 95% CI [{comparison['ci95_lower']:+.1f}, "
          f"{comparison['ci95_upper']:+.1f}]")
    print("Significant." if comparison["significant"]
          else "NOT significant: this experiment cannot separate the two algorithms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
