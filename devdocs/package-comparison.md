# Feature comparison with health economic modeling packages

This note positions `heormodel` against the established R packages a health
economist would otherwise reach for. It exists to guide the roadmap: it shows
where `heormodel` already matches the field, where it leads (one Python package
spanning model building through value of information), and where the R ecosystem
still does more.

The four comparators, chosen because they are the packages `heormodel` overlaps
with most, split into two groups. `hesim` builds and simulates decision models,
as `heormodel` does. `dampack`, `BCEA`, and `voi` analyze the outputs of a model
someone else built: they consume a probabilistic sensitivity analysis (PSA)
sample of costs and effects and never simulate a state-transition or event model
themselves. Reading each cell against that split explains most of the pattern
below.

The comparison reflects the documentation of each package read in July 2026:
the reference index, README, and package description of `dampack`
(version 1.0.2), `hesim`, `BCEA` (version 2.4.83), and `voi`. Where a package
delegates a capability to another (for example `BCEA` calls `voi` for the
expected value of partial perfect information), the cell says so.

## Scope and foundations

| Feature | heormodel | dampack | hesim | BCEA | voi |
|---|---|---|---|---|---|
| Language | Python 3.11+ | R | R | R | R |
| Primary role | Build models and analyze their outputs, end to end | Analyze model outputs (PSA and deterministic) | Build and simulate models, then analyze | Analyze Bayesian model outputs | Analyze model outputs (value of information only) |
| Builds decision models | Yes, four engines (below) | No, consumes a user model function or PSA table | Yes, four model classes (below) | No, consumes posterior cost and effect samples | No, consumes PSA samples or a model function |
| Performance backend | NumPy and SciPy, parallel over cores | Base R and `ggplot2` | C++ via `Rcpp`, `data.table`, built for large individual-level runs | Base R, vectorized over MCMC draws | Base R, regression backends (`mgcv`, others) |
| License | MIT | GPL-3 | GPL-3 | GPL-3 | GPL-3 |

## Model-building engines

`heormodel` and `hesim` are the only two that build models. The others analyze
whatever costs and effects you hand them, so their cells here are "no" by design,
not by omission.

| Feature | heormodel | dampack | hesim | BCEA | voi |
|---|---|---|---|---|---|
| Markov cohort state-transition | Yes, `MarkovModel`: constant or per-cycle transition arrays, per-state and per-transition rewards, `trace()` for occupancy | No | Yes, `CohortDtstm`: discrete-time cohort transitions, time-homogeneous or time-inhomogeneous | No | No |
| Microsimulation (discrete-time individual) | Yes, `MicrosimModel.discrete`: individual population per iteration, common random numbers, `duration_groups` | No | Partly, individual simulation is continuous-time (`IndivCtstm`) rather than a discrete cycle grid | No | No |
| Individual continuous-time state transition | Yes, `MicrosimModel.continuous`: continuous clock | No | Yes, `IndivCtstm`: continuous-time, Markov and semi-Markov, the package's flagship engine | No | No |
| Discrete-event simulation | Yes, `DESModel`: wraps SimPy, event log, `queue_waits` for queueing reports, per-entity discounted accrual | No | No | No | No |
| Compartmental transmission (ordinary differential equations) | Yes, `ODEModel`: integrates a user system with `solve_ivp`, force-of-infection coupling, flow-event costs, susceptible-exposed-infectious-recovered example | No | No | No | No |
| Stochastic compartmental (Gillespie) | Planned (roadmap item 16), not yet shipped | No | No | No | No |
| Partitioned survival model | No | No | Yes, `Psm` and `PsmCurves`: N-state partitioned survival from fitted survival curves | No | No |
| Decision tree | No | No | No | No | No |
| Survival model integration (parametric fitting) | No, parameters come from distributions, data, or calibration, not from fitting survival curves in-package | No | Yes, integrates fitted parametric survival models and multinomial logit models | No | No |
| Life table / age-dependent mortality | Yes, `LifeTable` samples age-dependent mortality | No | Partly, via time-inhomogeneous transitions | No | No |
| Within-cycle correction | Yes, Simpson's 1/3, half-cycle, or none | No | Yes, via time steps and `time_intervals()` | No | No |
| Bring your own model outputs | Yes, `Outcomes.from_tidy` / `from_wide` / `as_outcomes` accept an external results table | Yes, `make_psa_obj` wraps an external PSA table | Partly, model classes expect hesim's own structures | Yes, `bcea()` takes external cost and effect matrices | Yes, `evppi()` and `evsi()` take an external PSA sample |

## Parameters and uncertainty

| Feature | heormodel | dampack | hesim | BCEA | voi |
|---|---|---|---|---|---|
| Probability distributions for parameters | Yes, `Beta`, `Gamma`, `LogNormal`, `Normal`, `Uniform`, `Dirichlet`, `Fixed` | Yes, samples via `gen_psa_samp`; parameter helpers `beta_params`, `gamma_params`, `lnorm_params`, `dirichlet_params` | Yes, `define_rng` with `beta_rng`, `gamma_rng`, `dirichlet_rng`, `lognormal_rng`, `multi_normal_rng`, and others | No, consumes posterior draws produced upstream | No, consumes PSA draws produced upstream |
| Method-of-moments constructors (mean and standard error) | Yes, mean/SE constructors on distributions | Yes, `*_params` helpers convert moments to distribution parameters | Yes, `mom_beta`, `mom_gamma` | No | No |
| Correlated sampling | Yes, correlated draws in `ParameterSet.sample` | Partly, multivariate normal only | Yes, `multi_normal_rng` and bootstrapping | No | No |
| Probabilistic sensitivity analysis execution | Yes, `run_psa` is the single execution point | Yes, `run_psa` runs a user function over a PSA sample | Yes, propagates PSA through the simulation | No, expects the PSA already run | No, expects the PSA already run |
| Deterministic sensitivity analysis | Yes, `dsa.one_way`, `one_at_a_time`, `grid`, feeding the same runner | Yes, `run_owsa_det`, `run_twsa_det`, `owsa`, `twsa` (also as PSA metamodels) | No dedicated deterministic sensitivity analysis functions | Partly, `struct.psa` for structural uncertainty | No |
| Tornado diagram | Yes, `tornado_data` and `plot_tornado` | Yes, `owsa_tornado` | No | Yes, `info.rank` is an information-value tornado, not a one-way tornado | No |
| Two-parameter grid / heatmap | Yes, `dsa.grid` and `heatmap_data` | Yes, `twsa` and `plot.twsa` | No | No | No |
| Parallel execution | Yes, `run_psa` uses all cores by default, results invariant to worker count | No | Yes, C++ backend built for speed on large runs | No | Partly, some regression methods parallelize |
| Progress and time-remaining display | Yes, `run_psa` reports completed work and estimated time remaining | No | No | No | No |
| Calibration | Yes, `abc_calibrate`: approximate Bayesian computation, posterior returned as a draw matrix that flows into `run_psa` | No | No | No | No |
| Reproducible seeding across parallel runs | Yes, `SeedManager` keys streams by iteration so results do not depend on `n_jobs` | No | Partly, standard R seeding | No | No |

## Cost-effectiveness analysis

Every package covers the core of cost-effectiveness analysis. The differences
are at the edges: risk aversion, efficiency frontiers, and expected loss.

| Feature | heormodel | dampack | hesim | BCEA | voi |
|---|---|---|---|---|---|
| Incremental cost-effectiveness ratio table | Yes, `icer_table` | Yes, `calculate_icers` and `calculate_icers_psa` | Yes, `icer` and `cea` | Yes, `compute_ICER`, `ce_table` | No |
| Simple and extended dominance | Yes, marked in the ICER table status column | Yes, `calculate_icers` flags dominated and extendedly dominated | Yes | Yes, via the efficiency frontier | No |
| Net monetary and net health benefit | Yes, `nmb`, `nhb`, `expected_nmb` | Yes, within the PSA summary and metamodels | Yes, within `cea` | Yes, incremental benefit `compute_IB` and `compute_EIB` | No |
| Cost-effectiveness frontier | Yes, `frontier` | Yes, from `calculate_icers` | Yes | Yes, `ceef.plot` cost-effectiveness efficiency frontier | No |
| Cost-effectiveness acceptability curve and frontier | Yes, `ceac`, `ceaf` | Yes, `ceac`, `summary.ceac` | Yes, `cea` output, `plot_ceac`, `plot_ceaf` | Yes, `ceac.plot`, `ceaf.plot`, `multi.ce` | No |
| Cost-effectiveness plane | Yes, `ce_plane` and `plot_ce_plane` | Yes, `plot.psa` and ICER plots | Yes, `plot_ceplane` | Yes, `ceplane.plot`, `contour`, `contour2` | No |
| Expected loss curves | Yes, `expected_loss` and `plot_expected_loss` | Yes, `calc_exp_loss` and `plot.exp_loss` | No | Yes, opportunity loss via `compute_ol` | No |
| Multiple comparators | Yes, any number of interventions | Yes | Yes | Yes, `multi.ce`, `setComparisons` | No |
| Risk aversion | No | No | No | Yes, `CEriskav` adds a risk-aversion parameter | No |
| Mixed or portfolio strategies | No | No | No | Yes, `mixedAn` for a mix of interventions in the market | No |

## Value of information

This is where the R ecosystem is deepest, and where `voi` is the specialist:
`BCEA` calls it for the expected value of partial perfect information.
`heormodel` implements the same three quantities natively in Python.

| Feature | heormodel | dampack | hesim | BCEA | voi |
|---|---|---|---|---|---|
| Expected value of perfect information | Yes, `evpi` | Yes, `calc_evpi` | Yes, from `cea` and `plot_evpi` | Yes, `compute_EVI`, `evi.plot` | Yes, `evpi` |
| Expected value of partial perfect information | Yes, `evppi` (spline and Gaussian process), `evppi_ranking` | Yes, `calc_evppi` (generalized additive model metamodel) | No | Yes, `evppi` (delegates to `voi`) | Yes, `evppi` with many methods (below) |
| EVPPI estimation methods | Two, spline and Gaussian process | One, generalized additive model | Not applicable | Inherited from `voi` | Generalized additive model, Gaussian process, multivariate adaptive regression splines (earth), integrated nested Laplace approximation, Bayesian additive regression trees, and single-parameter methods |
| Expected value of sample information | Yes, `evsi_regression`, `evsi_moment_matching`, `evsi_importance_sampling`, `simulate_summaries` | Yes, `calc_evsi` | No | Yes, via `voi` | Yes, `evsi` |
| EVSI estimation methods | Three, nonparametric regression, moment matching, importance sampling | Nonparametric regression | Not applicable | Inherited from `voi` | Nonparametric regression, moment matching, importance sampling |
| Value of information for an estimation problem | No | No | No | No | Yes, `evppivar`, `evsivar` |
| Expected net benefit of sampling and population value | No | No | No | Partly, population EVPI in the report | Yes, `enbs`, `enbs_opt`, `pop_voi` |
| Information-rank plot | No | No | No | Yes, `info.rank` | No |

## Reporting, provenance, and documentation

| Feature | heormodel | dampack | hesim | BCEA | voi |
|---|---|---|---|---|---|
| Publication-style plots | Yes, cost-effectiveness plane, acceptability curve and frontier, frontier, tornado, expected loss, with a shared palette | Yes, `ggplot2` plots for each analysis | Yes, `ggplot2` and `autoplot` methods | Yes, base R, `ggplot2`, and interactive `plotly` engines | Yes, `plot.evppi` and tidy output for `ggplot2` |
| Automated report generation | Partly, `capture_run` and run records | No | No | Yes, a report combining the analysis into one document | No |
| Provenance and run records | Yes, `capture_run`, `RunRecord`, records draw sources | No | No | Partly, `sim_table` summarizes simulations | No |
| Reproducible seeding as a guarantee | Yes, the shared iteration index ties draws to outcomes for value-of-information regression | No | No | No | No |
| Tutorials and worked examples | Yes, executable Quarto tutorials with Colab notebooks for every engine and analysis | Yes, six vignettes | Yes, extensive vignettes and articles | Yes, vignettes and a companion book | Yes, vignettes for each method |

## Reading the table

Three patterns stand out.

`heormodel` is the only package that spans the full workflow in one language.
`hesim` builds models but stops at the expected value of perfect information for
value of information and has no calibration or deterministic sensitivity
analysis. `dampack`, `BCEA`, and `voi` analyze outputs but build no models. A
Python user assembling the R equivalent of `heormodel` would combine a modeling
package, a cost-effectiveness package, and `voi`, then move data between them.

`heormodel` carries engines the R packages do not: discrete-event simulation and
compartmental ordinary-differential-equation transmission models sit outside all
four comparators. The stochastic compartmental engine on the roadmap would widen
that gap.

The R ecosystem still leads in three places. `hesim` fits parametric survival
models and partitioned survival models and runs individual continuous-time
simulation at a scale its C++ backend is built for. `voi` offers more estimation
methods for the expected value of partial perfect information and adds the
expected net benefit of sampling and population value of information. `BCEA`
adds risk aversion, mixed-strategy analysis, and an automated report. These are
candidate roadmap items where the need is demonstrated.
