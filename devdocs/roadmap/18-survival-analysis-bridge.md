# 18. Survival analysis bridge

Add the machinery that turns fitted parametric survival models into inputs the
existing engines accept: time-varying transition probabilities for `MarkovModel`,
competing event-time samplers for `MicrosimModel.continuous`, and the survival
curves the partitioned survival engine (item 19) integrates. It also carries the
uncertainty in the fitted parameters onto the iteration index, so survival
estimates flow through `run_psa` into cost-effectiveness and value-of-information
analysis unchanged.

This is the single clearest gap the package comparison identified
([`../package-comparison.md`](../package-comparison.md)). A health economist
fitting survival curves to patient data has no way to bring the fit, or its
uncertainty, into a heormodel model. The engines can already simulate the
resulting process; the estimation-to-model step is what is missing.

## Sequencing: reproduce hesim first, then settle the architecture

The first deliverable is a bespoke replication that reproduces the target `hesim`
results with example-local functions, validated against closed forms. Only once
the numbers match do we extract a public `heormodel.survival` layer, letting the
replication code determine its shape. The API sketches below are candidates to
inform that second step, not commitments. This ordering applies across items 18,
19, and 20: reproduce all three survival-driven `hesim` tutorials with bespoke
code, confirm parity, then design the shared abstractions once, knowing what they
actually have to carry. Designing the survival layer before the replication risks
an API fitted to a guess rather than to three worked models.

## Phase 1: bespoke replication (do this first)

Reproduce the `hesim` "time inhomogeneous Markov individual-level models" tutorial
with throwaway helpers under `examples/`. Fit the parametric survival curves,
sample event times conditional on time already elapsed (the clock-forward case),
run the population through `MicrosimModel.continuous`, and reproduce the state
probabilities, quality-adjusted life-years, costs, and cost-effectiveness results.
Also validate the survival-to-cohort path: parameterize a `MarkovModel` from
fitted survival curves and reproduce a time-inhomogeneous cohort trace. Nothing in
this phase is a public module; the fitting, sampling, and conversion are example
functions, the way the `examples/mdm_*` replications keep bespoke logic local.

Phase 1 also produces the bespoke survival helpers that items 19 and 20 use in
their own phase-1 replications, so the survival curve code is written once and
exercised by all three before any of it is promoted to the package.

Acceptance for phase 1 is numeric: the replication matches `hesim` within Monte
Carlo error and matches the closed forms below exactly at fixed seeds.

## Phase 2: extract the architecture (after parity)

Once the replications of items 18, 19, and 20 all pass, promote the recurring
machinery into `heormodel.survival`. What follows is the candidate shape, to be
confirmed or revised by what the replications needed.

### Survival distributions (candidate)

Distribution objects for the families the field relies on: exponential, Weibull
(accelerated-failure-time and proportional-hazards parameterizations), Gompertz,
lognormal, log-logistic, generalized gamma, restricted cubic spline (the flexible
parametric form), and piecewise exponential. Each answers four questions on a time
argument: `survival(t)`, `hazard(t)` and `cumhazard(t)`, `sample_time(rng, size)`
by inverse-transform sampling, and `transition_probability(t0, t1)`, the
conditional probability of the event in `(t0, t1]` given survival to `t0`.

### Fitting and its uncertainty (candidate)

Fitting stays in a dedicated survival package rather than being reimplemented. The
default adapter reads a fit from `lifelines`, chosen because it covers the families
above with extrapolation and carries the same permissive license as heormodel, so
it can be an optional dependency (`heormodel[survival]`). Two constructors:
`from_fit(fitter)` returns a distribution at the point estimate; `sample_params(
fitter, n, seed)` draws coefficient vectors from the fit's asymptotic multivariate
normal (with a bootstrap alternative) onto the canonical iteration index, so
survival uncertainty and the other parameters share one index, the guarantee
`evppi` depends on.

### From hazards to transitions (candidate)

`to_transition_matrix(dists, cycle_length)` builds a per-cycle transition array
from cause-specific hazards using the transition-intensity matrix and its matrix
exponential, so probabilities are consistent and sum to one. Stacking these across
cycles is the age-varying input `MarkovModel` already accepts. For the individual
engine, `sample_time` is the competing-times sampler `MicrosimModel.continuous`
already calls, and clock-reset (semi-Markov) falls out of the engine redrawing at
state entry.

### Curve algebra (candidate)

`apply_hazard_ratio`, `apply_acceleration_factor`, `mix(dists, weights)`, and
`splice(dist_early, dist_late, cutpoint)` return new distributions. `splice` is the
extrapolation operation: the observed curve to the cutpoint, then a parametric
tail. These compose, so a treatment arm is the comparator curve under a sampled
hazard ratio, and a long-term model is a spline fit spliced onto a
background-mortality tail.

## Coherence with the engine architecture

The layer adds no engine and touches none of the three commitments. It produces
two things the engines already consume, per-cycle transition arrays and
per-iteration draws, and uncertainty enters through the shared iteration index, so
`run_psa`, `icer_table`, `ceac`, `evpi`, and `evppi` need no special case.

## Validation (acceptance)

- Phase 1: the bespoke replication of the `hesim` time-inhomogeneous
  individual-level tutorial matches within Monte Carlo error, and the
  survival-parameterized cohort matches a time-inhomogeneous `MarkovModel` trace.
- Closed forms, at fixed seeds: an all-exponential competing-risks model is a
  continuous-time Markov chain whose expected discounted costs and effects solve
  `(discount * I - Q) v = r` for generator `Q`, which the sampler plus the
  intensity-matrix path must match (reusing the check that anchors item 13);
  `transition_probability` integrates to the analytic transition matrix; and
  `sample_params` recovers the fitted mean and covariance as the draw count grows.
- Phase 2 does not change any number; it re-expresses the passing replication
  through the extracted layer, and the same tests pass against the public API.

## Deliverables

- Phase 1: `examples/hesim_indiv_timedep.py` reproducing the individual-level
  tutorial with bespoke survival helpers, its closed-form tests, and a replication
  gallery entry.
- Phase 2: `heormodel.survival` (distribution families, `lifelines` adapter,
  `to_transition_matrix`, curve algebra), the `survival` optional extra, docstring
  worked examples, tests, a website tutorial, and API reference and changelog
  entries.

## Reasonable-extent boundaries

- Performance. `hesim`'s benchmarks tutorial measures its C++ throughput.
  heormodel reproduces the models and their numbers on vectorized NumPy, not the
  raw speed; parity means the same answers, not the same wall-clock time.
- Multinomial-logistic-regression transitions (the `hesim` "Markov models with
  multinomial logistic regression" tutorial) need a regression-to-transition path
  for a different fitting family. It is a sibling of this survival bridge, kept in
  the backlog rather than folded in here.
- The website tutorials cite the published clinical model each example is drawn
  from, not the R package, matching the existing replications and keeping the
  user-facing documentation free of external package names.
