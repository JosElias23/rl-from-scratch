"""Tests for the two comparisons this repository actually makes.

Both were originally reported with the wrong instrument, and the corrections
point in opposite directions.

**Resolution.** The sweep compared each bin count against the best one and called
them indistinguishable when their 95% intervals overlapped. Overlapping intervals
are not a test of a difference -- they are far too conservative, and two
estimates whose intervals overlap can still differ significantly. Worse, bin
count is an *ordered* factor with seven levels, and comparing pairs throws that
ordering away. A trend test uses it, and finds an effect the pairwise view
missed.

**Q-learning against SARSA.** The paired design is correct a priori and stays.
What was wrong was the stated reason for it: that pairing on a seed "removes the
shared difficulty of that seed". It does not here, because a seed only fixes an
RNG stream, and two agents that consume randomness differently diverge onto
unrelated trajectories after their first differing update. The correction is to
report that, not to switch to whichever test gives a smaller p.

Implemented from first principles, like everything else here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------- distributions

def _betacf(a: float, b: float, x: float, iterations: int = 300) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iterations):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / (max(abs(1.0 + num * d), tiny) * (1 if 1.0 + num * d >= 0 else -1))
        c = 1.0 + num / (c if abs(c) > tiny else tiny)
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / (max(abs(1.0 + num * d), tiny) * (1 if 1.0 + num * d >= 0 else -1))
        c = 1.0 + num / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sided_t_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with df degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return regularised_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


# ------------------------------------------------------------- descriptive

def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def variance(xs) -> float:
    xs = list(xs)
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _ranks(xs) -> list[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def pearson(xs, ys) -> float:
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def spearman(xs, ys) -> float:
    return pearson(_ranks(list(xs)), _ranks(list(ys)))


# ------------------------------------------------------------------- tests

@dataclass(frozen=True)
class Trend:
    """Does the outcome move monotonically with an ordered factor?"""

    spearman_rho: float
    spearman_p: float
    slope_per_doubling: float
    slope_p: float
    n: int

    def summary(self) -> dict:
        return {
            "spearman_rho": round(self.spearman_rho, 4),
            "spearman_p": round(self.spearman_p, 4),
            "slope_per_doubling": round(self.slope_per_doubling, 2),
            "slope_p": round(self.slope_p, 4),
            "n_runs": self.n,
            "note": ("Bin count is an ordered factor, so a trend test uses "
                     "information that pairwise comparisons discard."),
        }


def trend_over_ordered_factor(levels, values) -> Trend:
    """Spearman on the raw levels, plus an OLS slope on log2(level)."""
    levels, values = list(levels), list(values)
    n = len(levels)
    rho = spearman(levels, values)
    t_rho = rho * math.sqrt((n - 2) / (1 - rho * rho))

    logs = [math.log2(x) for x in levels]
    r = pearson(logs, values)
    slope = r * math.sqrt(variance(values) / variance(logs))
    t_slope = r * math.sqrt((n - 2) / (1 - r * r))

    return Trend(rho, two_sided_t_p(t_rho, n - 2),
                 slope, two_sided_t_p(t_slope, n - 2), n)


@dataclass(frozen=True)
class PairedDiagnostics:
    """Whether pairing actually bought anything, and what carries the result."""

    n: int
    mean_difference: float
    sd_of_differences: float
    correlation_between_arms: float
    se_paired: float
    se_unpaired: float
    pairing_helped: bool
    p_paired: float
    p_unpaired: float
    leave_one_out_p: dict
    most_influential: str

    def summary(self) -> dict:
        return {
            "n_pairs": self.n,
            "mean_difference": round(self.mean_difference, 2),
            "sd_of_differences": round(self.sd_of_differences, 2),
            "correlation_between_arms": round(self.correlation_between_arms, 4),
            "se_paired": round(self.se_paired, 2),
            "se_unpaired": round(self.se_unpaired, 2),
            "pairing_reduced_variance": self.pairing_helped,
            "p_paired": round(self.p_paired, 4),
            "p_unpaired_for_contrast": round(self.p_unpaired, 4),
            "leave_one_out_p": {k: round(v, 4) for k, v in self.leave_one_out_p.items()},
            "most_influential_unit": self.most_influential,
            "note": (
                "The paired test is the one reported: it is the correct analysis "
                "for a paired design, and switching to whichever test gives a "
                "smaller p would be choosing the answer. The unpaired p is shown "
                "only because pairing did not reduce variance here, which is "
                "worth knowing and is not what the design assumed."
            ),
        }


def paired_diagnostics(labels, arm_a, arm_b) -> PairedDiagnostics:
    labels, a, b = list(labels), list(arm_a), list(arm_b)
    d = [x - y for x, y in zip(a, b, strict=True)]
    n = len(d)

    se_paired = math.sqrt(variance(d) / n)
    se_unpaired = math.sqrt(variance(a) / n + variance(b) / n)

    t_paired = mean(d) / se_paired
    p_paired = two_sided_t_p(t_paired, n - 1)

    va, vb = variance(a), variance(b)
    t_unpaired = (mean(a) - mean(b)) / se_unpaired
    df = (va / n + vb / n) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
    p_unpaired = two_sided_t_p(t_unpaired, df)

    loo = {}
    for i, label in enumerate(labels):
        rest = d[:i] + d[i + 1:]
        t = mean(rest) / math.sqrt(variance(rest) / len(rest))
        loo[str(label)] = two_sided_t_p(t, len(rest) - 1)

    # The unit whose removal moves p furthest, measured on the log scale so a
    # drop from 0.06 to 0.002 counts for more than one from 0.06 to 0.13.
    most = max(loo, key=lambda k: abs(math.log10(max(loo[k], 1e-12))
                                      - math.log10(max(p_paired, 1e-12))))

    return PairedDiagnostics(
        n=n,
        mean_difference=mean(d),
        sd_of_differences=math.sqrt(variance(d)),
        correlation_between_arms=pearson(a, b),
        se_paired=se_paired,
        se_unpaired=se_unpaired,
        pairing_helped=se_paired < se_unpaired,
        p_paired=p_paired,
        p_unpaired=p_unpaired,
        leave_one_out_p=loo,
        most_influential=most,
    )
