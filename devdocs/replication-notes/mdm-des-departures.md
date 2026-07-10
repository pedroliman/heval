# Departures of the discrete-event Sick-Sicker replication

This note records every place where `examples/mdm_des.py` and its tutorial differ from the companion R code of Lopez-Mendez, Goldhaber-Fiebert, and Alarid-Escudero (Medical Decision Making 2026;46(5):533-548), and what exact reproduction of their published figures would take. The design note [`roadmap/13-des-sicksicker-replication.md`](roadmap/13-des-sicksicker-replication.md) states the modeling intent; this note is the reconciliation against the source code, read from the authors' public repository. Two of the departures are deliberate corrections of what the companion code does relative to its own Table 1, and one follows from a design choice in `heormodel.run`.

The base-case cost-effectiveness analysis and every epidemiological outcome (survival, prevalence, dwell times) match the companion code, because those depend only on the transition dynamics and the state flows, which the two implementations share. The departures below affect the probabilistic analysis and the absolute cost level, not the base-case frontier ordering.

## Departures that move the published figures

### 1. Transition rewards: paid once, not accrued over the preceding sojourn

Table 1 defines `ic_HS1`, `ic_D`, and `du_HS1` as one-time amounts ("increase in cost when transitioning"). The companion `cea_fn` adds each to the annual flow of the sojourn that ends in the transition, then multiplies by the discounted length of that sojourn:

```r
dt_all[, Total_cost_ep := Annual_Cost_ep + Annual_Cost_trt + Cost_trans]
dt_all[, Disc_total_cost := Total_cost_ep * v_dwc_t12]   # v_dwc_t12 = discounted sojourn length
```

So a $2,000 cost of dying enters as $2,000 per year over the final sojourn, which averages about 14 years in Sicker, and a $1,000 onset cost enters as $1,000 per year over the Healthy sojourn before onset. The replication pays each once at the event, discounted at the event time, through `transition_payoffs`, the convention of the sibling cohort replication `examples/mdm_cohort_timedep.py`.

Effect: the companion's expected costs sit about $20,000 per strategy above the replication's (the dominant term is the death cost accrued over the long Sicker sojourn). Quality-adjusted life-years are almost unchanged, since `du_HS1` is 0.01 over a roughly 6-year Healthy sojourn, about 0.05 QALY. The published figure 4A cost axis runs about $100,000 to $380,000; the replication's runs about $107,000 to $265,000.

### 2. The probabilistic analysis holds six Table 1 parameters at their base case

The companion draws its parameter sets with `generate_psa_params_DES`, then merges each row into the simulation and cost lists with `update_param_list`, which is `modifyList` (exact-name matching). Six drawn columns have names the target lists do not use, so `modifyList` adds them as unread entries and leaves the base-case values in force:

| Drawn column (PSA) | Name the model reads | Base value held |
|---|---|---|
| `r_S1S2_scale` | `r_S1S2_scale_ph` (simulation) | 0.08 |
| `c_trtA` | `dc_trtA` (cost list) | $12,000 |
| `c_trtB` | `dc_trtB` (cost list) | $13,000 |
| `ic_HS1` | `dc_HS1` (cost list) | $1,000 |
| `ic_D` | `dc_D` (cost list) | $2,000 |
| `u_trtA` | `du_trtA` (cost list) | 0.20 increment |

The Weibull scale is the consequential one: it carries the progression uncertainty that the bivariate lognormal in `generate_psa_params_DES` was built to express, correlated with the shape. Because only the shape varies, the progression hazard barely moves across draws, and the treatment-B comparison that hinges on it carries little parameter uncertainty. The treatment-A utility is a subtler case: the companion applies the treated Sick utility as `u_S1 + du_trtA` with `du_trtA` fixed at 0.20, so the drawn `u_trtA` is updated in the list but never read, and the treated utility varies only through the shared `u_S1`. The replication draws every Table 1 distribution and applies `u_trtA` directly as the treated Sick utility.

Effect: the published acceptability curves are nearly step functions and the published expected value of perfect information (EVPI) peaks are small ($2,600 and $750 per person, figure 4D), because the strategy comparisons carry little uncertainty. With all Table 1 distributions active, more iterations flip their preferred strategy near the two thresholds, so the replication's decision uncertainty and EVPI are several times larger (peaks near $5,600 and $7,600 per person at 1,000 draws).

### 3. Common random numbers across all parameter sets, not per iteration

The companion resets to one fixed seed (`set.seed(2)`) inside every probabilistic iteration, so all 1,000 parameter sets are driven by the same random-number stream. This is a strong variance-reduction choice: it correlates the Monte Carlo error across parameter draws, which sharpens the cost-effectiveness scatter and the acceptability curves. `heormodel.run.run_psa` instead seeds each iteration from its index through the `SeedManager`, so iteration results do not depend on how a run is chunked across workers, and applies common random numbers across strategies within an iteration only. The replication follows the framework's design.

Effect: the two probabilistic analyses have different Monte Carlo error structure. Neither is biased; the companion's clouds are visually tighter across the parameter draws at a given simulation size.

## Departures that do not materially move the figures

### 4. Mortality sampling: exact inversion, not discrete-year with a uniform jitter

The companion samples the age at death by binning the cumulative hazard to integer ages (`nps_nhppp` with `correction = "uniform"`), then adds a uniform draw within the year. `LifeTable.sample_time_to_death` inverts the piecewise-constant cumulative hazard exactly. Both target the same continuous distribution, since the life table is piecewise constant by integer age; the companion's is a within-year approximation of the exact inversion. The difference in sampled ages at death is under a year and averages out.

### 5. Correlated Weibull draw: Gaussian copula, not a direct bivariate lognormal

Table 1 specifies the Weibull scale and shape as a bivariate lognormal with means (0.08, 1.10), standard deviations (0.02, 0.05), and correlation 0.5. The replication draws two lognormal marginals with Spearman rank correlation 0.5 through the Gaussian-copula sampler in `heormodel.params`, which reproduces the bivariate lognormal up to the rank-to-linear correlation conversion. This departure is moot in the companion's own analysis, where the scale is held fixed (departure 2), so the correlation never enters.

### 6. Latin hypercube sampling is in the prose, not the code

The article text describes Latin hypercube sampling for the parameter sets. The companion `generate_psa_params_DES` draws each parameter independently with `rgamma`, `rbeta`, `rlnorm`, and a bivariate-lognormal routine, with no stratification. The replication also uses plain Monte Carlo through `ParameterSet.sample`, so it matches the companion code rather than the article's prose. `heormodel.params` has no Latin hypercube sampler.

## What exact reproduction of the published figures would take

To reproduce the companion code's figures 4A through 4D (rather than the Table 1 specification), each substantive departure has a concrete change, none of which needs an engine change beyond what already ships:

1. Transition rewards over the sojourn. Drop `transition_payoffs` and instead post-process the event history returned by `evaluate(trace="events")`: for each event row, multiply the transition reward by the discounted sojourn integral over `[T_start, T_stop]` and sum per individual, the exact arithmetic of `cea_fn`. The event trace carries the `from_state`, `to_state`, and event `time` this needs; the sojourn start is the individual's previous event time.

2. Held-fixed parameters. Build the `ParameterSet` with `r_S1S2_scale`, `c_trtA`, `c_trtB`, `ic_HS1`, and `ic_D` as `Fixed` at their base values, and model the treated Sick utility as `u_S1 + 0.20` rather than drawing `u_trtA`. This makes the acceptability curves near-step and shrinks the EVPI peaks to the published scale.

3. Single random-number stream. Drive every iteration from one fixed seed instead of the per-index seeding in `run_psa`. This means bypassing `run_psa` with a short custom loop that calls the engine's `evaluate` on each one-row draw under a constant `SeedManager`, since keying the seed by iteration index is deliberate in `heormodel.run` and not a parameter.

The minor departures (4 through 6) need no change: the mortality-sampling and correlated-draw differences are within Monte Carlo error, and the Latin hypercube description does not match the companion code either.

Reproducing all three substantive departures would recover the published figures. The replication does not, because departures 1 and 2 are corrections of the companion code relative to its own Table 1, and departure 3 is the framework's reproducibility guarantee. The tutorial states each difference and its effect so a reader can see both what the article reports and what the specification implies.
