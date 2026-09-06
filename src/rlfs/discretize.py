"""Turning a continuous observation into a table index.

Tabular Q-learning needs a finite state space. CartPole's observation is four
real numbers -- cart position, cart velocity, pole angle, pole angular velocity
-- so it has to be quantised before a table can hold it.

This is the whole difficulty of applying tabular methods to a continuous
problem, and it is where the interesting engineering lives:

**Bin count was expected to be a bias-variance dial, and it is not — at least
not in the way the textbook argument predicts.** Too few bins and distinct
situations collapse into one cell, so no policy can separate them. Too many
bins and each cell is visited so rarely that its Q-value never converges. That
predicts an inverted U, with some middle resolution winning.

`scripts/sweep_resolution.py` measures it over 3 to 12 bins per dimension with
five seeds each, and the second half of that story does not appear:

    bins  states   visited   mean return   sd     seeds solved
       3      81       94%         246.4   177.7           2/5
       4     256       82%         311.7   193.7           3/5
       5     625       66%         428.3   146.7           4/5
       6   1,296       64%         355.6   126.8           4/5
       8   4,096       46%         433.4    91.8           5/5
      10  10,000       33%         456.6    59.1           5/5
      12  20,736       27%         473.4    33.8           5/5

At 12 bins the table has 20,736 cells and the agent visits 27% of them, yet it
is the *best* configuration tested and the only one that never fails. The
predicted degradation from an unfillable table simply never arrives in this
range.

What resolution actually buys is **reliability, not mean performance**. The
means across 4 to 12 bins are statistically indistinguishable; the standard
deviation across seeds collapses from 177.7 to 33.8. Coarse discretisation does
not produce a worse agent on average, it produces a *lottery* — 3 bins solved
the task in two runs out of five and scored 100 in another.

The reason the visited-cell count can fall while performance rises is that
unvisited cells are unreachable states, not neglected ones. A pole at 20 degrees
with the cart moving hard the other way is a cell the dynamics never produce.
Finer bins mostly subdivide the empty part of the space.

**Bounds matter more than counts.** Gymnasium reports the cart-velocity and
pole-angular-velocity ranges as infinite, so a bound has to be chosen. Setting
it too wide wastes resolution on states the cart never reaches; too narrow and
everything past it saturates into one bin. The values here are the empirical
operating range of a controlled pole, not the theoretical range of the
dynamics.

**Not all dimensions deserve equal resolution.** Pole angle and angular
velocity dominate the control problem; cart position barely matters until the
cart is about to leave the track. Giving them the same bin count is a choice
worth questioning, and `--bins` makes it easy to test.
"""

from __future__ import annotations

import numpy as np

# Gymnasium's CartPole-v1 declares +/- inf for the two velocity dimensions.
# These are the ranges a pole under control actually occupies; see the module
# docstring for why they are set by observation rather than by the spec.
CARTPOLE_BOUNDS = (
    (-2.4, 2.4),      # cart position, the track limit
    (-3.0, 3.0),      # cart velocity, empirical
    (-0.21, 0.21),    # pole angle in radians; termination is at +/-0.2095
    (-3.0, 3.0),      # pole angular velocity, empirical
)


class BoxDiscretizer:
    """Maps a continuous observation vector to a single integer state index.

    Each dimension is split into `bins` equal-width intervals between its
    lower and upper bound, and the resulting per-dimension indices are combined
    into one index by mixed-radix encoding -- the same arithmetic as reading a
    multi-digit number where every digit has base `bins`.
    """

    def __init__(self, bounds: tuple[tuple[float, float], ...], bins: int | tuple[int, ...]):
        self.bounds = tuple(bounds)
        if isinstance(bins, int):
            bins = (bins,) * len(self.bounds)
        if len(bins) != len(self.bounds):
            raise ValueError(f"{len(bins)} bin counts for {len(self.bounds)} dimensions")
        if any(b < 1 for b in bins):
            raise ValueError("Every dimension needs at least one bin")
        self.bins = tuple(bins)

        # Edges exclude the outer boundaries: np.digitize returns 0 below the
        # first edge and len(edges) above the last, which is exactly the
        # saturating behaviour we want for out-of-range observations.
        self.edges = [
            np.linspace(low, high, n + 1)[1:-1]
            for (low, high), n in zip(self.bounds, self.bins, strict=True)
        ]

    @property
    def n_states(self) -> int:
        return int(np.prod(self.bins))

    def __call__(self, observation) -> int:
        return self.encode(observation)

    def encode(self, observation) -> int:
        """Observation vector to a single state index."""
        index = 0
        for value, edges, n in zip(observation, self.edges, self.bins, strict=True):
            digit = int(np.digitize(value, edges))
            index = index * n + digit
        return index

    def decode(self, index: int) -> tuple[int, ...]:
        """State index back to per-dimension bin indices. Used by the figures."""
        digits = []
        for n in reversed(self.bins):
            digits.append(index % n)
            index //= n
        return tuple(reversed(digits))


def cartpole_discretizer(bins: int | tuple[int, ...] = 6) -> BoxDiscretizer:
    return BoxDiscretizer(CARTPOLE_BOUNDS, bins)


class IdentityDiscretizer:
    """Pass-through for environments whose observations are already discrete."""

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states

    def __call__(self, observation) -> int:
        return int(observation)

    def encode(self, observation) -> int:
        return int(observation)
