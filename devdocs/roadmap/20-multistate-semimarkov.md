# 20. Multi-state and semi-Markov models from fitted transition hazards

Add first-class support for multi-state models whose transitions are parameterized
by fitted survival hazards, one distribution per allowed transition, simulated on
the continuous clock. This is the individual-level modeling style at the center of
the `hesim` tutorials: a transition structure, a survival model per transition, and
a choice between the clock-forward (Markov) and clock-reset (semi-Markov) timing of
those hazards. It builds on the survival layer of item 18 and runs on the existing
`MicrosimModel.continuous`.

## Why a multi-state layer, given the continuous engine already exists

`MicrosimModel.continuous` already samples a time to each competing destination,
takes the earliest, and redraws at state entry, so a semi-Markov process is
expressible today. What a multi-state model adds is structure that recurs and is
error-prone to hand-assemble each time:

1. An allowed-transition matrix over the states, so the model function samples only
   permitted destinations and the state probabilities are read off a known
   structure.
2. One survival distribution per transition, with per-transition covariates, from
   item 18's layer.
3. An explicit clock choice. Clock-forward measures each hazard from time zero
   (model time); clock-reset measures it from the last state entry. The two give
   materially different hazards in a state a patient can dwell in, the point the
   `hesim` multi-state tutorial makes, so the choice must be a named, tested option
   rather than an implementation detail.
4. State probabilities over time, the multi-state analog of a cohort trace, which
   `heormodel.models.state_occupancy` already derives from an event log.

## Sequencing: reproduce hesim first, then settle the architecture

As with items 18 and 19, the bespoke replication comes first. Reproduce the `hesim`
multi-state results with example-local functions on `MicrosimModel.continuous`
using item 18's bespoke survival helpers, confirm parity for both clock variants,
and only then decide whether the structure above deserves a dedicated
`MultiStateModel` convenience or is better left as a documented pattern plus a few
helpers. Building the abstraction only after the replication avoids committing to a
new engine surface before the worked model shows what it needs to carry.

## Phase 1: bespoke replication (do this first)

Reproduce the `hesim` "Markov and semi-Markov multi-state models" tutorial with
example-local functions under `examples/`: the reversible illness-death model with
three states (healthy, sick, death) and four transitions (healthy to sick, sick to
healthy, healthy to death, sick to death), a Weibull distribution fitted per
transition with a treatment covariate, simulated over a population that varies by
age and sex, comparing standard of care with a new intervention. Reproduce both the
clock-reset (semi-Markov) and clock-forward (Markov) variants, and the state
probabilities, quality-adjusted life-years, costs, and cost-effectiveness summary.
The two clocks must differ in the sick state, the result the source tutorial
highlights, which is also the phase's headline validation that the clock handling
is correct.

Acceptance for phase 1 is numeric: the replication matches `hesim` within Monte
Carlo error, the clock-forward and clock-reset results differ in the sick state as
the source shows, and the all-Weibull-shape-one case matches the continuous-time
Markov chain closed form.

## Phase 2: extract the architecture (after parity)

Once phase 1 passes (with items 18 and 19), decide the shape. Candidate: a
`MultiStateModel` in `heormodel/models/multistate.py` that takes the states, an
allowed-transition matrix, an `interventions` sequence, a `transitions` callback
returning one survival distribution per allowed transition, and a `clock`
argument (`"forward"` or `"reset"`). It assembles the competing-times sampler the
continuous engine expects and reuses `MicrosimModel.continuous` underneath rather
than being a separate simulation kernel, so there is one continuous-time engine and
this layer is a structured way to build its model function. State probabilities come
from `state_occupancy` on the event log, already shipped. The alternative, if the
replication shows the assembly is thin, is to keep it as a documented pattern with a
transition-structure helper rather than a new class. The replication decides.

## Coherence with the engine architecture

No new simulation kernel and no new output structure. The layer builds a model
function for `MicrosimModel.continuous`, whose randomness already comes from the run
loop's per-iteration streams, so results stay invariant to worker count and batch
size. Survival uncertainty enters through item 18's `sample_params` on the shared
iteration index, and `state_occupancy` supplies the state-probability trace.

## Validation (acceptance)

- Phase 1: the bespoke replication of the `hesim` multi-state tutorial matches
  within Monte Carlo error for both clocks, with the clock-forward and clock-reset
  results differing in the sick state.
- Closed form: an all-exponential (or Weibull shape one) illness-death model is a
  continuous-time Markov chain whose expected discounted costs and effects solve
  `(discount * I - Q) v = r` for generator `Q`, which the simulation must match,
  exercising the transition assembly and both clocks (the clocks coincide when the
  hazard is constant, a useful check that clock-reset reduces correctly).
- Phase 2 changes no number; whatever shape is chosen reproduces the phase-1
  result, with contract tests confirming the iteration index matches the draws.

## Deliverables

- Phase 1: `examples/hesim_mstate.py` reproducing the multi-state tutorial for both
  clocks with bespoke functions, its closed-form test, and a replication gallery
  entry.
- Phase 2: whichever the replication justifies, a `MultiStateModel` convenience or a
  documented pattern with a transition-structure helper, with docstring worked
  examples, tests, a website tutorial, and API reference and changelog entries.

## Relationship to items 18 and 19

The three items share one survival layer (item 18) and one continuous-time engine.
Item 20 is the individual-level multi-state modeling style, item 19 is the
partitioned survival style, and item 18 is the estimation machinery both rest on.
Their phase-1 replications together reproduce the survival-driven `hesim` tutorials;
the cohort, time-inhomogeneous cohort, and cost-effectiveness tutorials already
reproduce with the shipped engines. The phase-2 architecture for all three is
settled together, after the replications confirm what the abstractions must carry.
