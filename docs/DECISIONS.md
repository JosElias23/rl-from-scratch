# Decision log

What was built, in what order, why each choice was made, and what the numbers
turned out to be. Every figure here comes from a file in `reports/`, produced by
running the code. Nothing is typed from memory.

---

## 1. The question

Q-learning and SARSA are implemented from scratch, no RL library. The obvious
way to report that is "the agent reaches 74% success on FrozenLake". That number
is meaningless on its own, because nobody reading it knows whether 74% is
excellent or terrible.

So the actual question of this repository is the one underneath: **what is the
best score obtainable, and how close did the agent get?**

On FrozenLake the environment is small enough that the answer can be *computed*
rather than estimated, which is the whole reason it was chosen. That is a rare
luxury in RL and it should be spent.

---

## 2. Decisions about measurement

### 2.1 Value iteration is a measuring instrument, not a baseline

`src/rlfs/planning.py` solves the FrozenLake MDP exactly with the transition
model, and the learned agents never see that model. Its job is to produce a
denominator.

FrozenLake 4x4 slippery, γ = 1, tolerance 1e−12: **516 sweeps, 0.022 s,
V(start) = 0.542026.**

### 2.2 There is not one ceiling, there are three, and picking the wrong one is
the easiest mistake in the project

| Ceiling | Success rate | What it assumes |
|---|---:|---|
| Unbounded horizon | **0.8235** | Infinite time, no step limit |
| Time-aware optimum, 100 steps | **0.7442** | Policy may depend on steps remaining |
| Best stationary policy under the cap | 0.7402 | One fixed action per state |

The commonly quoted figure for this environment is 0.8235, exactly 14/17. It is
the right answer to a question nobody is asking, because Gymnasium wraps
FrozenLake in a `TimeLimit` of 100 steps by default. The optimal policy on
slippery ice spends most of its time deliberately *not* moving toward the goal,
pressing into walls so the slip probability carries it sideways rather than into
a hole. That is patient, and patience runs out at 100 steps.

Scoring the agents against 0.8235 would have made a near-optimal agent look like
it was leaving 10 points on the table. **The environment the agent is graded on
must be the environment it is actually run in**, and finding that out required
writing a finite-horizon backward-induction solver rather than reusing the
infinite-horizon one.

The third row is the interesting one. A time-aware policy, which is allowed to
change its mind as the clock runs down, is worth only **+0.004** over the best
stationary policy. The tabular agents can only learn stationary policies, so
this quantifies exactly what that restriction costs them: almost nothing, here.
Without the number it would be an assumption.

### 2.3 Evaluation draws from a separate random stream

`eval_rng` is distinct from the training RNG in `src/rlfs/agents.py`. Evaluating
mid-training with the training stream consumes draws from it, so *how often you
measure* changes *what the agent learns*. The learning curve and the final score
then depend on the evaluation schedule, which is absurd and very hard to notice.

This is the kind of bug that does not raise, does not fail a test written the
obvious way, and quietly makes runs irreproducible.

### 2.4 Success rates get confidence intervals

5,000 evaluation episodes per agent, with Wilson intervals rather than the
normal approximation, because the normal interval misbehaves near 0 and 1 and
the random baseline sits at 0.012.

---

## 3. Results on FrozenLake

5,000 evaluation episodes, ε = 0.01 at the end of 200,000 training episodes.

| Agent | Success rate | 95% CI | Mean episode length |
|---|---:|:--|---:|
| Random | 0.0120 | — | 7.8 |
| **Q-learning** | **0.7378** | [0.7254, 0.7498] | 44.3 |
| **SARSA** | **0.7378** | [0.7254, 0.7498] | 44.3 |
| _Time-aware optimum_ | _0.7442_ | — | — |

**Both agents recover a policy that agrees with the optimal one in all 16
states, and reach 99.14% of the attainable ceiling.**

Two details are worth more than the headline.

The identical scores are not a copy-paste error and not a coincidence: both
agents converged to the *same* policy, and identical policies evaluated on
identical seeds produce identical episodes. Reporting "SARSA 0.7378, Q-learning
0.7378" without that explanation would look like a bug, which is why the policy
agreement is reported alongside.

The 0.0064 shortfall against 0.7442 is not a learning failure. Simulating the
optimal *stationary* policy on the same 5,000 seeded episodes gives **0.7378**,
the agents' exact score. The gap is the +0.004 time-awareness term from 2.2 plus
sampling noise, not something more training would fix.

Mean episode length, 44.3 steps against the random agent's 7.8, is the patience
described in 2.2 showing up in the data.

---

## 4. Decisions about CartPole

FrozenLake has 16 states. CartPole has four continuous dimensions, so a tabular
method needs the state space cut up first, and that cut is the entire modelling
decision.

### 4.1 The bounds are hand-set, and that choice matters more than the bin count

**Correction.** An earlier version of this section claimed the edges were fitted
on percentiles of states visited by random rollouts. They are not, and never
were. `src/rlfs/discretize.py` builds equal-width edges with
`np.linspace(low, high, n + 1)[1:-1]` over the hard-coded `CARTPOLE_BOUNDS`, and
a search for `percentile`, `quantile` or `rollout` across the repository returns
nothing. The README and the module's own docstring both described this correctly
throughout; only this file was wrong. Leaving it would have been the exact
failure the rest of this portfolio is about, so it is stated rather than quietly
edited.

What the code actually does, and why:

Gymnasium declares the cart-velocity and pole-angular-velocity ranges as
infinite, so a finite bound has to come from somewhere. The four ranges in
`CARTPOLE_BOUNDS` are the operating region a pole under control actually
occupies — ±2.4 for cart position because that is the track limit, ±0.21 rad for
pole angle because termination is at ±0.2095, and ±3.0 for the two velocities by
observation.

The bound, not the bin count, is the decision that moves the numbers. Too wide
and resolution is spent on states the cart never reaches; too narrow and
everything past it saturates into the outermost bin. Fitting percentiles would
be a defensible way to set them and is listed in section 6 as not done.

The honest consequence is stated in the README: a different choice of bounds
would move every CartPole number in this repository.

### 4.2 Resolution is swept, not chosen

Five seeds per setting, 60,000 episodes each, 300 evaluation episodes:

| Bins/dim | States | Mean return | sd | Seeds solved | Coverage |
|---:|---:|---:|---:|:--:|---:|
| 3 | 81 | 246.43 | 177.70 | 2 / 5 | 93.6% |
| 4 | 256 | 311.73 | 193.73 | 3 / 5 | 82.3% |
| 5 | 625 | 428.33 | 146.72 | 4 / 5 | 66.3% |
| 6 | 1,296 | 355.55 | 126.79 | 4 / 5 | 64.1% |
| 8 | 4,096 | 433.35 | 91.80 | 5 / 5 | 46.3% |
| 10 | 10,000 | 456.58 | 59.08 | 5 / 5 | 33.1% |
| **12** | **20,736** | **473.45** | **33.78** | **5 / 5** | **27.2%** |

The mean return is **not monotonic**: 5 bins beats 6. Anyone tuning this on a
single seed would conclude that 6 bins is worse than 5 and stop there. Across
five seeds the standard deviations are 146.72 and 126.79, so that reversal is
comfortably inside the noise and means nothing.

**Correction.** An earlier version of this section called the standard-deviation
column "the one that is monotonic". It is not: 3 bins to 4 bins is 177.70 to
193.73, an increase — the same kind of reversal dismissed as noise in the mean
column two sentences earlier. It also claimed finer bins do not mainly make the
agent better. A trend test says otherwise (section 4.2.1).

What holds: **spread falls from 177.70 to 33.78 as resolution rises, and seeds
solved goes from 2/5 to 5/5.** Finer discretisation makes the agent both better
and more reliable. Coarse bins force distinct states to share a Q-value, so whether
the run works depends on whether that collision happened somewhere harmful, and
that is a coin flip decided by the seed.

The classic objection is that finer bins are slower to fill: at 12 bins only
27.2% of the 20,736 cells are ever visited. That turns out not to matter, since
the unvisited cells are unreachable configurations rather than gaps in
knowledge. Within the budget tested, finer wins on every axis that counts.

### 4.2.1 The right test for an ordered factor

The sweep originally concluded "the means are statistically indistinguishable"
from a rule in `scripts/sweep_resolution.py`: two settings were called
indistinguishable when their 95% intervals overlapped. That rule is wrong twice.

Overlapping intervals are not a test of a difference. Two estimates whose
intervals overlap can still differ significantly, and the overlap rule is far
more conservative than the test it stands in for. And bin count is an **ordered**
factor with seven levels, so comparing each level against the best one discards
the ordering, which is most of the information in the design.

`scripts/analyse_results.py` runs the tests that use it, over the same 35 stored
runs — nothing is re-trained:

| Test | Result |
|---|---|
| Spearman ρ(bins, return) | +0.364, p = 0.0315 |
| OLS slope on log₂(bins) | +106.3 return per doubling, p = 0.0020 |
| 3–4 bins (279.1) vs 10–12 bins (465.0) | Welch p = 0.0095 |

So there **is** a mean effect of resolution. The original analysis could not
localise it to any adjacent pair and reported that as evidence of no effect —
which is exactly the move section 5 correctly refuses to make for Q-learning
against SARSA ninety lines later. The repository was applying its own standard
inconsistently.

The reliability claim also needed deflating. Return is capped at 500, so the
largest standard deviation attainable at mean *m* is √(m(500−m)). Across the
seven resolutions the correlation between mean and sd is −0.853, and normalising
by that ceiling turns the advertised 5.26× collapse into **2.36×** — and stops it
being monotone (0.711, 0.800, 0.837, 0.559, 0.540, 0.420, 0.301). A real part of
what was sold as "reliability, not mean performance" is the mean improving and
reappearing as a truncation effect.

`sweep_resolution.py` still prints which intervals overlap, because that is what
it computes, but it now says plainly that this is not a test and points here.

---

## 5. The result that had to be corrected

The first CartPole run, one seed each, 6 bins, 60,000 episodes:

| Agent | Mean return | 95% CI | States visited |
|---|---:|:--|---:|
| Random | 21.74 | — | — |
| Q-learning | 408.41 | [400.60, 416.22] | 803 / 1,296 |
| **SARSA** | **500.00** | [500.00, 500.00] | 840 / 1,296 |

SARSA hits the 500-step ceiling on every one of 500 evaluation episodes, sd
exactly 0. That is a clean, quotable story: the on-policy agent learns a safer
policy and never drops the pole.

It does not survive eight seeds.

| Agent | Mean | sd | Median | Range | Solved |
|---|---:|---:|---:|:--|:--:|
| **Q-learning** | **432.14** | 87.45 | 477.74 | 276.20 – 500.00 | **8 / 8** |
| SARSA | 302.33 | 125.58 | 325.17 | 121.46 – 500.00 | 6 / 8 |

Paired over the same eight seeds: **+129.80 in Q-learning's favour, 95% CI
[−15.01, +274.62], not significant.** Q-learning wins on 7 of 8 seeds, SARSA on
1.

So the ordering reverses, and the reversal is *still* not statistically
significant. Both statements need saying. Seed 42 was not representative, and
one run at 500.00 with zero variance was a property of that seed, not of SARSA.

Reporting the single-seed table would have produced a confident claim in the
wrong direction, supported by an interval of width zero. The interval was
narrow because 500 evaluation episodes of *one* policy were being measured; the
uncertainty that mattered was across training runs, and nothing in the
single-seed experiment could see it. **The variance that matters in tabular RL
is between seeds, not within an evaluation.**

Both tables are kept in the repository. The wrong one is more interesting than
the right one.

---

## 6. What was not done

- **Two environments, both classic control.** FrozenLake and CartPole. Nothing
  here says anything about function approximation, continuous actions, or
  anything at Atari scale.
- **No neural networks.** Tabular by design: the point was to see the algorithms
  work without a library, and to be able to compute the optimum. DQN would
  remove the exact ceiling that makes section 2 possible.
- **Hyperparameters are not tuned.** Learning rate, discount and the ε schedule
  are standard values applied identically to both agents. Fair for comparison,
  certainly not optimal for either. The Q-learning versus SARSA comparison in
  section 5 is a comparison *at these settings*, and a tuned SARSA might well
  win.
- **The 8-seed comparison is underpowered**, which is exactly what the interval
  [−15.01, +274.62] says. Distinguishing the agents at this variance needs
  roughly an order of magnitude more seeds, and the honest report is "not
  significant", not "no difference".
- **No eligibility traces, no double Q-learning, no prioritised replay.** The
  maximisation bias that double Q-learning fixes is a plausible contributor to
  Q-learning's behaviour here and is not measured.
- **CartPole's ceiling is the environment's step limit, not a computed
  optimum.** There is no equivalent of section 2 for CartPole, so 500.00 means
  "hit the cap", not "solved optimally".

---

## 7. Reproducing

```bash
pip install -e ".[dev]"
python -m pytest                          # 37 tests
python scripts/train_frozenlake.py        # agents plus the three ceilings
python scripts/train_cartpole.py          # single-seed run, ~30 minutes
python scripts/sweep_resolution.py        # 7 resolutions x 5 seeds
python scripts/compare_agents.py          # 8 paired seeds
python scripts/make_figures.py
```

Every number in this document lives in `reports/metrics_frozenlake.json`,
`reports/metrics_cartpole.json`, `reports/metrics_resolution_sweep.json` and
`reports/metrics_agent_comparison.json`.
