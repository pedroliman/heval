# 19. Partitioned survival model engine

Add a partitioned survival model that derives health-state occupancy from a set of
survival curves and emits the standard `Outcomes` structure. It is the standard
oncology model type both `hesim` and `heemod` provide and heormodel lacks. It uses
the survival curves of item 18 and reuses `heormodel.models._accrual` for
discounting.

## Why partitioned survival is its own model

A partitioned survival model does not simulate transitions. It reads state
occupancy off survival curves: for an N-state model ordered from best to worst
health, curve `k` is the probability of being in state `k` or better. State 1
occupancy is `S_1(t)`, each middle state is `S_k(t) - S_{k-1}(t)`, and the worst
state (death) is `1 - S_{N-1}(t)`. Costs and utilities accrue on occupancy, and
the discounted area under each curve gives the state's contribution to life-years,
quality-adjusted life-years, and cost. Expressing this as a Markov model would
require inverting the curves into transition probabilities, the assumption a
partitioned survival model declines to make, so it warrants its own engine.

## Sequencing: reproduce hesim first, then settle the architecture

As in item 18, the first deliverable is a bespoke replication of the `hesim`
partitioned survival results using example-local functions and the bespoke
survival helpers from item 18's phase 1. Only after the numbers match do we
promote the logic to a `PartSurvModel` engine. The sketch below is a candidate to
inform that step. Holding the engine design until the replication passes keeps the
`Outcomes` reduction, the crossing-curve handling, and the discounting choices
grounded in a worked model rather than guessed.

## Phase 1: bespoke replication (do this first)

Reproduce the `hesim` "Partitioned survival models" tutorial with example-local
functions under `examples/`: a four-state oncology model of a two-line sequential
treatment strategy, three strategies across patient profiles varying by age and
sex, three Weibull curves (progression on first line, progression on second line,
and mortality) with age, sex, and strategy covariates, state utilities, drug costs
fixed by strategy, and medical costs by state. Fit the curves with item 18's
bespoke helpers, build occupancy from the successive-curve differences on a time
grid, accrue discounted rewards, assemble `Outcomes`, and reproduce the state
probabilities and cost-effectiveness results. The occupancy construction, the
integration, and the reward attachment are all example code at this stage.

Acceptance for phase 1 is numeric: the replication matches `hesim` within Monte
Carlo error and matches the closed forms below exactly.

## Phase 2: extract the engine (after parity)

Once phase 1 passes (and items 18 and 20 pass their replications), promote the
occupancy-and-accrual logic to an engine. Candidate shape:

```python
PartSurvModel(
    states=("PF", "PD", "Death"),   # ordered best to worst
    interventions=("SoC", "New"),
    curves=fn,                       # fn(params, intervention) -> PartSurvSpec
    horizon=30.0,
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
already integrates a piecewise trajectory, which is what the occupancy curves are.

### Crossing curves

The construction requires `S_k(t) <= S_{k-1}(t)`. Fitted curves can cross and give
a negative occupancy. Clamp each middle state to be non-negative and report the
largest clamp so the user sees when it happened, the standard handling for this
model type; document that a large clamp signals mis-specified curves.

## Validation (acceptance)

- Phase 1: the bespoke replication of the `hesim` partitioned survival tutorial
  matches within Monte Carlo error.
- Closed forms: with exponential progression-free and overall survival at constant
  rates, the discounted life-years in each state have a closed form (for a single
  exponential curve at rate `r` with discount `d`, the discounted area is
  `1 / (r + d)`), which the occupancy construction and integration must match; the
  occupancy curves sum to one at every grid time and each is non-negative after
  clamping.
- Phase 2 changes no number; the extracted engine reproduces the phase-1 result,
  and contract tests identical in shape to the `MarkovModel` ones confirm the
  returned iteration index matches the draws.

## Deliverables

- Phase 1: `examples/hesim_psm.py` reproducing the partitioned survival tutorial
  with bespoke functions, its closed-form tests, and a replication gallery entry.
- Phase 2: `heormodel.models.PartSurvModel` and `PartSurvSpec` exported from
  `heormodel.models`, a `trace` parallel to `MarkovModel.trace` returning the
  occupancy curves, docstring worked examples, tests, a website tutorial, a
  `docs/concepts/engines.qmd` update describing the `curves` callback, and API
  reference and changelog entries.
