# heval

Health economic evaluation in Python: parameter specification and
probabilistic sampling, simulation across multiple model types,
cost-effectiveness analysis (CEA), and value-of-information (VoI) analysis
for model-based HEOR/HTA work.

## The core idea

One parameter object flows through swappable model engines into a shared
analysis layer. Engines differ internally and do not share an
implementation API. They share a contract on their outputs: given a matrix
of parameter draws, every engine returns costs and effects per strategy
per iteration in one standardized structure (`Outcomes`, a tidy frame
indexed by `(strategy, iteration)`). Once outputs are standardized, CEA
and VoI are engine-agnostic.

Two invariants hold everywhere:

1. The outcome schema is the integration point. Every engine targets it,
   every analysis consumes it, and no analysis reaches into engine
   internals.
2. The parameter matrix and the outcome matrix share the iteration index.
   EVPPI and EVSI trace which parameter draw produced which outcome
   through that index; `run_psa` enforces it.

## Bring your own outputs

You do not need an engine to use `heval`. A costs/effects PSA table from
any source enters the pipeline through one call:

```python
import numpy as np
from heval.run import as_outcomes
from heval.cea import icer_table, ceac
from heval.voi import evpi

outcomes = as_outcomes("my_psa.csv")   # columns: strategy, iteration, cost, qaly

print(icer_table(outcomes))            # ICERs, dominance, extended dominance
curves = ceac(outcomes, np.linspace(0, 150_000, 61))
print(evpi(outcomes, wtp=50_000))
```

A runnable walkthrough (external CSV through CEA, VoI, plots, and a model
card) is in [`examples/byoo_example.py`](examples/byoo_example.py):

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
print(evppi_ranking(outcomes, draws, wtp=50_000))      # research prioritization
```

## Package layout

| Subpackage | Status | Contents |
|---|---|---|
| `heval.params` | done | Distribution specs (`Beta`, `Gamma`, `LogNormal`, `Normal`, `Uniform`, `Fixed`, `Dirichlet`) with mean/SE constructors; correlated PSA sampling producing the iteration-indexed draw matrix. |
| `heval.models` | contract done, engines stubbed | The `Outcomes` schema and the `ModelEngine` protocol. Cohort state-transition, microsimulation, and DES engines are documented stubs. |
| `heval.run` | done | `SeedManager`, `run_psa` (serial or joblib-parallel, contract enforced), `as_outcomes`, running-mean diagnostics. |
| `heval.cea` | done | `icer_table` (dominance, extended dominance, frontier ICERs), `frontier`, `nmb`/`nhb`/`expected_nmb`, `ceac`, `ceaf`, `ce_plane`. |
| `heval.voi` | done (EVSI: regression method) | `evpi`; `evppi` via spline or Gaussian-process metamodels (scikit-learn); `evsi_regression` plus `simulate_summaries`. Moment matching and importance sampling stubbed. |
| `heval.calibrate` | done (optional extra) | ABC-SMC via `pyabc`; the posterior returns as an iteration-indexed draw matrix. |
| `heval.report` | done | CE plane, CEAC/CEAF, frontier, and tornado plots; `capture_run`/`RunRecord` for seed logging, parameter provenance, and model cards. |

Next steps are prioritized in [`roadmap/`](roadmap/README.md). Writing
follows [`guidance/writing_style.md`](guidance/writing_style.md).

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

### Validation checks

Both live in the test suite and pass:

- Incremental analysis
  (`tests/test_cea.py::TestValidationIncrementalAnalysis`): a
  five-strategy example with strong dominance, extended dominance, and
  hand-computed ICERs. The frontier, statuses, and ICERs match exactly.
- EVPI/EVPPI/EVSI (`tests/test_voi.py::TestValidationAnalyticGaussian`): a
  two-strategy model with Gaussian costs and effects, where EVPI, EVPPI,
  and EVSI have closed-form values via the unit normal loss integral. The
  estimators agree within Monte Carlo error (5% relative tolerance at
  80,000 iterations).

## Design notes and deviations

- Metamodeling uses scikit-learn only. `pygam` does not install against
  numpy 2.4+, so EVPPI/EVSI use an additive cubic-spline basis (default)
  or Gaussian-process regression. The `method=` argument leaves room for a
  `pygam` backend later.
- `heval.calibrate` is a seventh subpackage beyond the original six, added
  for calibration via `pyabc` (optional dependency). Calibrated draws
  re-enter the pipeline as a standard draw matrix.
- Mean/SE constructors are kept deliberately. Published evidence usually
  arrives as a point estimate with a standard error;
  `Beta.from_mean_se(0.2, 0.05)` turns that into a sampling distribution
  without hand-solving for shape parameters. Direct parameterisation
  (`Beta(a, b)`) remains available.
- Dirichlet vectors are sampled as normalised independent Gammas so they
  coexist with the copula machinery. Leave correlation targets at zero for
  Dirichlet component columns.
- Tornado diagrams are PSA-based (univariate linear fits of NMB on each
  parameter, evaluated at the 2.5th and 97.5th percentiles) because phase
  1 has no deterministic engine to re-run.
- Run-loop caching is deferred to the engine phases; seed management,
  parallel execution, contract enforcement, and running-mean diagnostics
  are in place.
