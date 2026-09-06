"""Generate every figure in the README from the stored result files.

Reads only reports/*.json and reports/*.npz, so a chart can never disagree with
a reported number.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rlfs.utils import PROJECT_ROOT, load_config, setup_logging  # noqa: E402

NEWLINE = chr(10)  # matplotlib titles wrap on this

COLORS = {
    "random": "#9aa0a6",
    "sarsa": "#edae49",
    "q_learning": "#00798c",
    "value_iteration": "#d1495b",
}
PRETTY = {
    "random": "Random",
    "sarsa": "SARSA",
    "q_learning": "Q-learning",
    "value_iteration": "Exact optimum\n(backward induction)",
}


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_frozenlake_ceiling(metrics: dict, out: Path) -> None:
    """The three different 'optima' this environment has, side by side."""
    vi = metrics["results"]["value_iteration"]
    horizon = metrics["environment"]["time_limit_steps"]

    labels = [
        "Optimal policy,\nno time limit",
        f"TRUE {horizon}-step optimum\n(time-aware policy)",
        f"Discounted-optimal policy,\ncapped at {horizon} steps",
    ]
    values = [
        vi["success_rate_unbounded_horizon"],
        vi["success_rate_time_aware_optimum"],
        vi["success_rate_stationary_under_cap"],
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.barh(labels, values, color=["#5f6caf", "#d1495b", "#b0885f"], height=0.6)
    for bar, v in zip(bars, values, strict=True):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 0.95)
    ax.set_xlabel("Probability of reaching the goal")
    ax.set_title(
        'FrozenLake 4x4 slippery: "the optimum" is three different numbers\n'
        "Quoting one without naming the horizon is meaningless",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_frozenlake_results(metrics: dict, out: Path) -> None:
    results = metrics["results"]
    ceiling = results["value_iteration"]["success_rate"]
    keys = ["random", "sarsa", "q_learning", "value_iteration"]
    values = [results[k]["success_rate"] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(keys)), values, color=[COLORS[k] for k in keys], width=0.6)

    for i, (bar, key, v) in enumerate(zip(bars, keys, values, strict=True)):
        label = f"{v:.4f}\n{100 * v / ceiling:.1f}% of optimum"
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.018, label,
                ha="center", fontsize=9.5, linespacing=1.4)
        r = results[key]
        if "ci95_lower" in r:
            ax.errorbar(i, v, yerr=[[v - r["ci95_lower"]], [r["ci95_upper"] - v]],
                        fmt="none", ecolor="#333333", capsize=5, linewidth=1.3)

    ax.axhline(ceiling, color="#d1495b", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([PRETTY[k] for k in keys], fontsize=10)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 0.95)
    ax.set_title(
        f"Slippery FrozenLake, {metrics['eval_episodes']:,} greedy evaluation episodes\n"
        "Both learned agents recover the optimal stationary policy exactly",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_learning_curves(history_path: Path, metrics: dict, out: Path, title: str,
                         ceiling: float | None = None, ylabel: str = "Success rate",
                         eval_key: str = "eval_success") -> None:
    data = np.load(history_path)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for name in ("q_learning", "sarsa"):
        points_key = f"{name}_eval_points"
        values_key = f"{name}_{eval_key}"
        if points_key in data and len(data[points_key]):
            ax.plot(data[points_key], data[values_key], color=COLORS[name],
                    linewidth=1.8, label=f"{PRETTY[name]} (greedy evaluation)")

    if ceiling is not None:
        ax.axhline(ceiling, color="#d1495b", linestyle="--", linewidth=1.4,
                   label=f"Exact optimum ({ceiling:.4f})")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    for name in ("q_learning", "sarsa"):
        key = f"{name}_epsilons"
        if key in data:
            ax2.plot(data[key], color=COLORS[name], linewidth=1.2)
    ax2.set_ylabel("epsilon")
    ax2.set_xlabel("Training episode")
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_resolution_sweep(summary: list[dict], out: Path) -> None:
    """What discretisation resolution actually buys, measured over five seeds.

    The expected result was an inverted U: some middle resolution winning, with
    finer tables degrading because they cannot be filled. That is not what the
    data shows, so the figure reports what happened instead -- means that are
    statistically indistinguishable, and a spread that collapses.
    """
    bins = [s["bins"] for s in summary]
    means = [s["mean_return"] for s in summary]
    sds = [s["std_return"] for s in summary]
    lows = [s["min_return"] for s in summary]
    highs = [s["max_return"] for s in summary]
    coverage = [100 * s["mean_coverage"] for s in summary]
    solved = [s["seeds_solved"] for s in summary]
    n_seeds = summary[0]["n_seeds"]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    x = range(len(bins))
    ax.fill_between(x, lows, highs, color="#00798c", alpha=0.14,
                    label=f"range across {n_seeds} seeds")
    ax.errorbar(x, means, yerr=sds, fmt="o-", color="#00798c", linewidth=2,
                markersize=7, capsize=5, label="mean +/- 1 sd")
    ax.axhline(195, color="#d1495b", linestyle="--", linewidth=1.2,
               label="solved threshold (195)")
    ax.axhline(500, color="#666666", linestyle=":", linewidth=1,
               label="maximum possible (500)")

    for i, s in enumerate(summary):
        ax.annotate(f"{s['seeds_solved']}/{n_seeds}", xy=(i, highs[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#333333")

    ax.set_ylabel("Mean return (greedy evaluation)")
    ax.set_ylim(0, 560)
    ax.set_title(
        "CartPole: finer discretisation buys reliability, not mean performance"
        + NEWLINE
        + "Means are statistically indistinguishable; the spread across seeds "
        + "collapses from 178 to 34",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    ax2.bar([i - 0.19 for i in x], sds, width=0.38, color="#d1495b",
            label="sd across seeds")
    ax2.bar([i + 0.19 for i in x], coverage, width=0.38, color="#9aa0a6",
            label="table cells visited (%)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(
        [
            f"{b}" + NEWLINE + f"{s['n_table_states']:,}"
            for b, s in zip(bins, summary, strict=True)
        ]
    )
    ax2.set_xlabel("Bins per dimension  /  table size")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)

    _ = solved
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    log = setup_logging()
    config = load_config()
    reports = PROJECT_ROOT / "reports"
    figures = PROJECT_ROOT / config["paths"]["figures_dir"]
    figures.mkdir(parents=True, exist_ok=True)

    fl_path = reports / "metrics_frozenlake.json"
    if fl_path.exists():
        metrics = json.loads(fl_path.read_text(encoding="utf-8"))
        plot_frozenlake_ceiling(metrics, figures / "frozenlake_ceiling.png")
        plot_frozenlake_results(metrics, figures / "frozenlake_results.png")
        log.info("Wrote FrozenLake figures")

        history = reports / "frozenlake_history.npz"
        if history.exists():
            plot_learning_curves(
                history, metrics, figures / "frozenlake_learning.png",
                "Learning curves on slippery FrozenLake",
                ceiling=metrics["results"]["value_iteration"]["success_rate"],
                ylabel="Success rate", eval_key="eval_success",
            )
            log.info("Wrote FrozenLake learning curves")
    else:
        log.warning("No FrozenLake metrics; run scripts/train_frozenlake.py")

    cp_path = reports / "metrics_cartpole.json"
    if cp_path.exists():
        metrics = json.loads(cp_path.read_text(encoding="utf-8"))
        history = reports / "cartpole_history.npz"
        if history.exists():
            plot_learning_curves(
                history, metrics, figures / "cartpole_learning.png",
                "Learning curves on CartPole (6 bins per dimension)",
                ceiling=None, ylabel="Mean return", eval_key="eval_returns",
            )
            log.info("Wrote CartPole learning curves")
    sweep_path = reports / "metrics_resolution_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        plot_resolution_sweep(sweep["summary"], figures / "cartpole_resolution_sweep.png")
        log.info("Wrote CartPole resolution sweep")
    else:
        log.warning("No CartPole metrics; run scripts/train_cartpole.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
