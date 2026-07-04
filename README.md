# heval

**Health economic evaluation in Python** — parameter specification and
probabilistic sampling, simulation across multiple model types, full
cost-effectiveness analysis (CEA), and value-of-information (VoI) analysis
for model-based HEOR/HTA work.

## The core idea

One parameter object flows through swappable model engines into a shared
analysis layer. Engines differ radically internally and do **not** share an
implementation API — they share a **contract on their outputs**. Every
engine, given a matrix of parameter draws, returns costs and effects per
strategy per iteration in one standardized structure (`Outcomes`, a tidy
frame indexed by `(strategy, iteration)`). Once outputs are standardized,
CEA and VoI become engine-agnostic.

Two invariants hold everywhere:

1. **The outcome schema is the integration point.** Every engine targets
   it; every analysis consumes it; no analysis reaches into engine
   internals.
2. **The parameter matrix and the outcome matrix share the iteration
   index.** EVPPI and EVSI trace which parameter draw produced which
   outcome through that index; `run_psa` enforces it.

## Bring your own outputs

You don't need an engine to use `heval`. A costs/effects PSA table from
*any* source — an external simulator, a spreadsheet export, a legacy
model — enters the pipeline through one call:

```python
import numpy as np
from heval.run import as_outcomes
from heval.cea import icer_table, ceac, ceaf
from heval.voi import evpi

outcomes = as_outcomes("my_psa.csv")        # columns: strategy, iteration, cost, qaly

print(icer_table(outcomes))                 # ICERs, dominance, extended dominance
curves = ceac(outcomes, np.linspace(0, 150_000, 61))
print(evpi(outcomes, wtp=50_000))
```

A complete runnable walkthrough — external CSV through CEA, VoI, plots, and
a model card — is in [`examples/byoo_example.py`](examples/byoo_example.py):

```bash
uv run python examples/byoo_example.py
```

## Full pipeline example

```python
import pandas as pd
from heval.params import Beta, Gamma, ParameterSet
from heval.models import Outcomes
from heval.run import SeedManager, run_psa
from heval.voi import evppi_ranking

params = ParameterSet(
    {
        "p_event": Beta.from_mean_se(0.20, 0.04),     # from published mean/SE
        "c_event": Gamma.from_mean_se(8_000, 1_200),
    },
    correlation={("p_event", "c_event"): 0.4},        # Spearman, Gaussian copula
)

sm = SeedManager(42)                                   # logged for reproducibility
draws = params.sample(10_000, seed=sm.generator())     # index name: "iteration"

def my_model(d: pd.DataFrame) -> Outcomes:            # any callable draws -> Outcomes
    costs = pd.DataFrame({"SoC": d["c_event"] * d["p_event"],
                          "Tx":  d["c_event"] * d["p_event"] * 0.6 + 2_000}, index=d.index)
    effects = pd.DataFrame({"SoC": 10 - d["p_event"],
                            "Tx":  10 - d["p_event"] * 0.6}, index=d.index)
    return Outcomes.from_wide(costs, effects)

outcomes = run_psa(my_model, draws, n_jobs=4)          # iteration index preserved
print(evppi_ranking(outcomes, draws, wtp=50_000))      # research prioritisation
```

## Package layout

| Subpackage | Status | Contents |
|---|---|---|
| `heval.params` | ✅ implemented | Distribution specs (`Beta`, `Gamma`, `LogNormal`, `Normal`, `Uniform`, `Fixed`, `Dirichlet`) with method-of-moments constructors from published means/SEs; correlated PSA sampling (Gaussian copula on Spearman targets) producing the iteration-indexed draw matrix. |
| `heval.models` | ✅ contract; engines stubbed | The `Outcomes` schema and the `ModelEngine` protocol. Cohort state-transition, discrete/continuous-time microsimulation, and a SimPy-wrapping DES engine are documented stubs for later phases. |
| `heval.run` | ✅ implemented | `SeedManager` (SeedSequence-based spawning), `run_psa` (serial or joblib-parallel, output contract enforced), `as_outcomes` (bring-your-own-outputs), running-mean convergence diagnostics. |
| `heval.cea` | ✅ implemented | `icer_table` (dominance, extended dominance, frontier ICERs), `frontier`, `nmb`/`nhb`/`expected_nmb`, `ceac`, `ceaf`, `ce_plane`. |
| `heval.voi` | ✅ EVPI, EVPPI, EVSI (regression) | `evpi` directly from the PSA; `evppi` via metamodeling (additive spline or Gaussian-process regression, scikit-learn); `evsi_regression` (nonparametric regression on simulated study summaries) plus `simulate_summaries`. Moment-matching and importance-sampling EVSI are stubbed. |
| `heval.calibrate` | ✅ implemented (optional extra) | ABC-SMC calibration via `pyabc`: `heval` priors translate to `pyabc` priors, and the posterior comes back as an iteration-indexed draw matrix that flows through the same pipeline. |
| `heval.report` | ✅ implemented | CE plane, CEAC/CEAF, frontier, and tornado plots (colorblind-safe fixed palette); `capture_run`/`RunRecord` for seed logging, parameter provenance, and markdown model cards. |

## Installation

Requires Python 3.11+. Use [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e .                    # core
uv pip install -e ".[calibration]"     # + pyabc for ABC calibration
uv pip install -e ".[dev]"             # + pytest, ruff, mypy, pyabc
```

## Development

```bash
uv run pytest                          # test suite incl. validation checks
uv run pytest --doctest-modules src    # every docstring example runs
uv run ruff check . && uv run ruff format --check .
uv run mypy                            # clean on the public API
```

### Validation checks (the phase-1 acceptance bar)

Both live in the test suite and pass:

- **Incremental analysis** (`tests/test_cea.py::TestValidationIncrementalAnalysis`):
  a five-strategy example with strong dominance, extended dominance, and
  hand-computed ICERs; the frontier, statuses, and ICERs must match exactly.
- **EVPI/EVPPI/EVSI** (`tests/test_voi.py::TestValidationAnalyticGaussian`):
  a two-strategy model with Gaussian costs/effects, where EVPI, EVPPI, and
  EVSI have closed-form values via the unit normal loss integral; the
  estimators must agree within Monte Carlo error (5% relative tolerance at
  80,000 iterations).

## Design notes and deviations

- **Metamodeling uses scikit-learn only.** The spec allowed `pygam` and/or
  scikit-learn; `pygam` does not currently install cleanly against
  numpy ≥ 2.4, so EVPPI/EVSI use an additive cubic-spline basis
  (`SplineTransformer` + linear regression, GAM-like, the default) or
  Gaussian-process regression. The `method=` argument leaves room to add a
  `pygam` backend later without API changes.
- **`heval.calibrate` is a seventh subpackage** beyond the original six,
  added to support calibration via `pyabc` (optional dependency). Its
  output re-enters the pipeline as a standard draw matrix, so nothing
  downstream knows whether draws came from priors or a calibrated posterior.
- **Method-of-moments constructors are kept deliberately.** Published
  evidence usually arrives as a point estimate with a standard error;
  `Beta.from_mean_se(0.2, 0.05)` turns that directly into a sampling
  distribution without hand-solving for shape parameters. Direct
  parameterisation (`Beta(a, b)`) remains available.
- **Dirichlet parameters and correlation:** Dirichlet vectors are sampled as
  normalised independent Gammas so they coexist with the copula machinery;
  correlation targets should be left at zero for Dirichlet component columns.
- **Tornado diagrams are PSA-based** (univariate linear fits of NMB on each
  parameter, evaluated at the parameter's 2.5th/97.5th percentiles) rather
  than deterministic one-way analyses, since phase 1 has no deterministic
  engine to re-run.
- **`run` caching** is deferred to the engine phases; seed management,
  parallel execution, contract enforcement, and running-mean diagnostics are
  in place.
