# 19. Partitioned survival model engine

Add a partitioned survival model that derives health-state occupancy from a set of
survival curves and emits the standard `Outcomes` structure. It is the standard
oncology model type heormodel lacks. It uses the survival curves of item 18 and
reuses `heormodel.models._accrual` for discounting.

## Why partitioned survival is its own model

A partitioned survival model does not simulate transitions. It reads state
occupancy off survival curves: for an N-state model ordered from best to worst
health, curve `k` is the probability of being in state `k` or better. State 1
occupancy is `S_1(t)`, each middle state is `S_k(t) - S_{k-1}(t)`, and the worst
state (death) is `1 - S_{N-1}(t)`. Costs and utilities accrue on occupancy, and the
discounted area under each curve gives the state's contribution to life-years,
quality-adjusted life-years, and cost. Expressing this as a Markov model would
require inverting the curves into transition probabilities, the assumption a
partitioned survival model declines to make, so it warrants its own engine.

## Sequencing: reproduce the reference model first, then settle the architecture

As in item 18, the first deliverable is a bespoke replication of the reference
model below using example-local functions and the survival helpers from item 18's
phase 1. Only after the numbers match do we promote the logic to a `PartSurvModel`
engine. The sketch below is a candidate to inform that step. Holding the engine
design until the replication passes keeps the `Outcomes` reduction, the
crossing-curve handling, and the discounting grounded in a worked model.

## The reference model and its expected results

A three-state model, stable, progressed, and dead, with two curves: progression-free
survival and overall survival, both Weibull in the accelerated-failure-time
parameterization `S(t) = exp(-(t / scale) ** shape)`.

```
Progression-free survival:  shape = 1.2, scale = 3.0 years
Overall survival:           shape = 1.2, scale = 6.0 years
```

State occupancy is `stable(t) = S_PFS(t)`, `progressed(t) = S_OS(t) - S_PFS(t)`,
`dead(t) = 1 - S_OS(t)`. Two strategies: the comparator, and a new intervention that
applies an acceleration factor of 1.30 to both scales (`scale_new = 1.30 * scale`,
longer survival). State utilities are 0.80 stable and 0.60 progressed; state costs
are 1000 and 2000 per year; the annual discount rate is 0.03. The expected results,
which the replication must reproduce:

| Strategy | Discounted stable life-years | Discounted progressed life-years | Discounted quality-adjusted life-years | Discounted cost |
|---|---|---|---|---|
| Comparator | 2.6315 | 2.2956 | 3.4826 | 7222.7 |
| New | 3.3526 | 2.8146 | 4.3709 | 8981.9 |

The incremental cost-effectiveness ratio of the new intervention against the
comparator is 1980 per quality-adjusted life-year. These are the exact
area-under-the-curve values (the model is deterministic given the curves), so the
replication must match them to within numerical integration error, not Monte Carlo
error.

## Phase 1: bespoke replication (do this first)

Reproduce the table with example-local functions under `examples/`: evaluate the two
Weibull curves with item 18's bespoke helpers, build occupancy from the
successive-curve differences on a time grid, accrue the discounted rewards, assemble
`Outcomes`, and compute the incremental cost-effectiveness ratio. The occupancy
construction, the integration, and the reward attachment are all example code at
this stage.

## Phase 2: extract the engine (after parity)

Once phase 1 passes (and items 18 and 20 pass their replications), promote the
occupancy-and-accrual logic to an engine. Candidate shape:

```python
PartSurvModel(
    states=("Stable", "Progressed", "Death"),   # ordered best to worst
    interventions=("Comparator", "New"),
    curves=fn,                                    # fn(params, intervention) -> PartSurvSpec
    horizon=40.0,
    discount_rate=0.03,
)
```

`PartSurvSpec` carries the ordered survival curves (the `N - 1` from item 18's
survival layer), the per-state cost and utility rates accrued on occupancy, and
optional one-time costs on entry to a state. The engine is deterministic given the
curves, so it draws no random numbers and satisfies `ModelEngine` alone, in the
shape `MarkovModel` and `ODEModel` take; parameter uncertainty enters through the
draws, including the sampled survival parameters from item 18. Discounting and the
`Outcomes` reduction come from `heormodel.models._accrual`, whose `integrate_flow`
already integrates a piecewise trajectory.

### Crossing curves

The construction requires `S_k(t) <= S_{k-1}(t)`. Fitted curves can cross and give a
negative occupancy. Clamp each middle state to be non-negative and report the
largest clamp so the user sees when it happened, the standard handling for this
model type; document that a large clamp signals mis-specified curves.

## Validation (acceptance)

- Phase 1 reproduces the reference table to numerical-integration tolerance, and the
  incremental cost-effectiveness ratio of 1980 per quality-adjusted life-year.
- Closed form: with exponential progression-free and overall survival at constant
  rates, the discounted life-years in each state have a closed form (for a single
  exponential curve at rate `r` with discount `d`, the discounted area is
  `1 / (r + d)`), which the occupancy construction and integration must match; the
  occupancy curves sum to one at every grid time and each is non-negative after
  clamping.
- Phase 2 changes no number; the extracted engine reproduces the phase-1 result,
  and contract tests identical in shape to the `MarkovModel` ones confirm the
  returned iteration index matches the draws.

## Deliverables

- Phase 1: `examples/partitioned_survival.py` reproducing the reference table with
  bespoke functions, its closed-form tests, and a replication gallery entry.
- Phase 2: `heormodel.models.PartSurvModel` and `PartSurvSpec` exported from
  `heormodel.models`, a `trace` parallel to `MarkovModel.trace` returning the
  occupancy curves, docstring worked examples, tests, a website tutorial, a
  `docs/concepts/engines.qmd` update describing the `curves` callback, and API
  reference and changelog entries.
