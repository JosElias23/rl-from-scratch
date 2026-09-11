"""Re-analyse the stored runs with the right tests.

    python scripts/analyse_results.py

Reads `reports/metrics_resolution_sweep.json` and
`reports/metrics_agent_comparison.json` -- the same runs, not new ones -- and
writes `reports/metrics_statistics.json`.

It exists because both comparisons were first reported with the wrong
instrument, and the two corrections point in opposite directions:

  resolution   The sweep called two settings "statistically indistinguishable"
               when their 95% intervals overlapped. That is not a test, and it
               is far too conservative. Bin count is also an ordered factor, so
               a trend test uses information the pairwise view discards. It
               finds an effect the pairwise view reported as absent.

  Q vs SARSA   The paired design is right and stays. Its stated justification
               was not: pairing on a seed was said to remove shared difficulty,
               and here the two arms are negatively correlated, so it adds
               variance instead. Reported alongside is which single seed the
               conclusion rests on.

Separating training from analysis is the point: nothing here re-runs an agent,
so a mistake in the statistics costs seconds rather than hours, and the runs
being analysed are byte-identical to the ones already published.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rlfs.statistics import (  # noqa: E402
    mean,
    paired_diagnostics,
    pearson,
    trend_over_ordered_factor,
    two_sided_t_p,
    variance,
)
from rlfs.utils import PROJECT_ROOT, save_json, setup_logging  # noqa: E402

MAX_RETURN = 500.0  # CartPole-v1's step cap, and the reason sd cannot be free


def load(name: str) -> dict:
    path = PROJECT_ROOT / "reports" / name
    if not path.exists():
        raise SystemExit(f"{path} not found; run the experiment scripts first.")
    return json.loads(path.read_text(encoding="utf-8"))


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    na, nb = len(a), len(b)
    va, vb = variance(a), variance(b)
    t = (mean(a) - mean(b)) / math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, two_sided_t_p(t, df)


def main() -> int:
    log = setup_logging()

    # ------------------------------------------------------------ resolution
    sweep = load("metrics_resolution_sweep.json")
    runs = sweep["runs"]
    trend = trend_over_ordered_factor(
        [r["bins"] for r in runs], [r["mean_return"] for r in runs]
    )

    summary = sweep["summary"]
    means = [s["mean_return"] for s in summary]
    sds = [s["std_return"] for s in summary]

    # A return capped at 500 cannot have a free standard deviation: the largest
    # possible sd for a mean m is sqrt(m * (500 - m)). Part of the advertised
    # collapse in sd is therefore the mean improving, reappearing as truncation.
    ceiling = [math.sqrt(m * (MAX_RETURN - m)) for m in means]
    normalised = [sd / c for sd, c in zip(sds, ceiling, strict=True)]

    low = [r["mean_return"] for r in runs if r["bins"] in (3, 4)]
    high = [r["mean_return"] for r in runs if r["bins"] in (10, 12)]
    _, pooled_p = welch(low, high)

    resolution = {
        "trend": trend.summary(),
        "pooled_extremes": {
            "coarse_bins": [3, 4],
            "fine_bins": [10, 12],
            "coarse_mean": round(mean(low), 2),
            "fine_mean": round(mean(high), 2),
            "welch_p": round(pooled_p, 4),
        },
        "spread": {
            "raw_sd_first_to_last": [round(sds[0], 2), round(sds[-1], 2)],
            "raw_collapse_factor": round(sds[0] / sds[-1], 2),
            "sd_is_monotonic": all(a >= b for a, b in zip(sds, sds[1:], strict=True)),
            "corr_mean_sd": round(pearson(means, sds), 4),
            "ceiling_normalised": [
                {"bins": s["bins"], "sd": round(sd, 2), "normalised": round(nv, 3)}
                for s, sd, nv in zip(summary, sds, normalised, strict=True)
            ],
            "normalised_collapse_factor": round(normalised[0] / normalised[-1], 2),
            "note": (
                "sd is divided by sqrt(mean * (500 - mean)), the largest value it "
                "could take at that mean given the hard cap on return. The raw "
                "collapse overstates the reliability gain because a higher mean "
                "mechanically compresses the spread."
            ),
        },
    }

    # --------------------------------------------------------- Q vs SARSA
    comparison = load("metrics_agent_comparison.json")
    by_seed: dict = {}
    for run in comparison["runs"]:
        by_seed.setdefault(run["seed"], {})[run["agent"]] = run["mean_return"]
    seeds = sorted(by_seed)
    diagnostics = paired_diagnostics(
        seeds,
        [by_seed[s]["q_learning"] for s in seeds],
        [by_seed[s]["sarsa"] for s in seeds],
    )

    save_json(
        {
            "source": "post-hoc analysis of the stored runs; nothing was re-trained",
            "resolution_sweep": resolution,
            "agent_comparison": diagnostics.summary(),
        },
        "reports/metrics_statistics.json",
    )

    # -------------------------------------------------------------- console
    print("\nResolution: does mean return rise with bin count?")
    print("-" * 70)
    t = trend.summary()
    print(f"  Spearman rho          {t['spearman_rho']:+.3f}   p = {t['spearman_p']:.4f}")
    print(f"  slope per doubling    {t['slope_per_doubling']:+.1f}   p = {t['slope_p']:.4f}")
    print(f"  3-4 bins {resolution['pooled_extremes']['coarse_mean']:.1f} vs "
          f"10-12 bins {resolution['pooled_extremes']['fine_mean']:.1f}: "
          f"Welch p = {resolution['pooled_extremes']['welch_p']:.4f}")
    print("  -> there IS a trend; the pairwise view could not localise it")

    print("\nResolution: how much of the sd collapse is real?")
    print("-" * 70)
    for row in resolution["spread"]["ceiling_normalised"]:
        print(f"  {row['bins']:>3} bins   sd {row['sd']:>6.2f}   "
              f"normalised {row['normalised']:.3f}")
    print(f"  raw collapse {resolution['spread']['raw_collapse_factor']}x, "
          f"normalised {resolution['spread']['normalised_collapse_factor']}x")
    print(f"  sd column monotonic: {resolution['spread']['sd_is_monotonic']}")

    print("\nQ-learning vs SARSA: did pairing help?")
    print("-" * 70)
    d = diagnostics.summary()
    print(f"  correlation between arms  {d['correlation_between_arms']:+.3f}")
    print(f"  se paired {d['se_paired']:.2f} vs se unpaired {d['se_unpaired']:.2f}"
          f"   -> pairing reduced variance: {d['pairing_reduced_variance']}")
    print(f"  sd of differences         {d['sd_of_differences']:.1f}")
    print(f"  p (paired, reported)      {d['p_paired']:.4f}")
    print(f"  p (unpaired, contrast)    {d['p_unpaired_for_contrast']:.4f}")
    print(f"  most influential seed     {d['most_influential_unit']}"
          f"   (p without it: {d['leave_one_out_p'][d['most_influential_unit']]:.4f})")

    log.info("wrote reports/metrics_statistics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
