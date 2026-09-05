"""Turning a continuous observation into a table index.

Tabular Q-learning needs a finite state space. CartPole's observation is four
real numbers -- cart position, cart velocity, pole angle, pole angular velocity
-- so it has to be quantised before a table can hold it.

This is the whole difficulty of applying tabular methods to a continuous
problem, and it is where the interesting engineering lives:

**Bin count is a bias-variance dial.** Too few bins and distinct situations
collapse into one cell, so no policy can separate them: the agent is not
failing to learn, it is being asked to answer with insufficient resolution.
Too many bins and each cell is visited so rarely that its Q-value never
converges. With 6 bins per dimension the table has 6^4 = 1,296 states; with 12
it has 20,736, and the agent would need far more episodes to fill them.

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
