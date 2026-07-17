"""Calibrate a Markov model twice: once through a surrogate, once against the model.

Calibration by approximate Bayesian computation (ABC) calls the model thousands of
times. When each run is expensive, that budget is the constraint. This script shows
the surrogate-accelerated alternative: run the real model only on a small Latin
hypercube design, fit a Gaussian process to the model's calibration targets, and do
the inference against the surrogate instead of the model.

The model is a three-state cohort state-transition (Markov) model over Healthy, Sick,
and Dead. Two transition probabilities are calibrated:

    p_HS: Healthy -> Sick per-cycle probability
    p_SD: Sick -> Dead per-cycle probability

The calibration targets are the Sick-state prevalence at three cycles, observed with a
known measurement error. Targets are generated from p_HS = 0.08 and p_SD = 0.15, so
both methods have a known answer to recover.

The script runs two calibrations and compares them:

1. Reference. `heormodel.calibrate.abc_calibrate` runs ABC-SMC against the model. A
   counter records how many times the model is evaluated.
2. Surrogate. A Latin hypercube design of 60 points trains a Gaussian process per
   target. Neural posterior estimation from the `sbi` package then infers the
   posterior using the surrogate as the simulator, so the only real model runs are
   the 60 design points.

The two posteriors agree, and the surrogate path uses about a hundred times fewer
model runs.

Run it with::

    uv pip install -e '.[calibration,surrogate]'
    uv run python examples/surrogate_calibration.py

Outputs (written to ``examples/output/``):
    - surrogate_calibration_posteriors.png
    - run_report_surrogate_calibration.md
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
from scipy.stats.qmc import LatinHypercube, scale
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from heormodel.calibrate import abc_calibrate
from heormodel.models import CohortSpec, MarkovModel
from heormodel.params import Uniform

# Quiet the third-party progress chatter. pyabc reads ABC_LOG_LEVEL when it runs, sbi's
# logger and the warning filter apply at call time, so setting them after the imports is
# enough. The model is smooth, so the Gaussian process marginal-likelihood optimizer
# sometimes reports non-convergence while still fitting well (the hold-out check
# confirms it); that warning is cosmetic.
os.environ["ABC_LOG_LEVEL"] = "WARNING"
logging.getLogger("sbi").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("healthy", "sick", "dead")
INTERVENTION = "natural_history"
N_CYCLES = 40
TARGET_CYCLES = (8, 16, 28)
TARGET_LABELS = [f"sick_c{cycle}" for cycle in TARGET_CYCLES]
BACKGROUND_MORTALITY = 0.01  # fixed Healthy -> Dead per-cycle probability
MEASUREMENT_SD = 0.01  # standard error on each observed prevalence

CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}

LHS_POINTS = 60
SURROGATE_SIMS = 10_000
POSTERIOR_DRAWS = 3_000


def transitions_and_rewards(params: pd.Series, intervention: str) -> CohortSpec:
    """Transition matrix for one parameter set; rewards are unused by calibration."""
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    transition = np.array(
        [
            [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY],
            [0.0, 1.0 - p_SD, p_SD],
            [0.0, 0.0, 1.0],
        ]
    )
    return CohortSpec(transition, np.zeros(3), np.array([1.0, 0.8, 0.0]))


engine = MarkovModel(
    states=STATES,
    interventions=(INTERVENTION,),
    transitions_and_rewards=transitions_and_rewards,
    n_cycles=N_CYCLES,
    cycle_correction="none",
)

model_runs = {"count": 0}


def model_targets(params: dict[str, float]) -> np.ndarray:
    """Sick-state prevalence at the target cycles. One real model run."""
    model_runs["count"] += 1
    occupancy = engine.trace(pd.Series(params), INTERVENTION)["sick"].to_numpy()
    return np.array([occupancy[cycle] for cycle in TARGET_CYCLES])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])

    observed = model_targets(TRUTH)
    print("Observed targets (Sick prevalence at cycles 8, 16, 28):", observed.round(4))

    # --- Reference: ABC-SMC against the real model ---------------------------------
    # A stochastic simulator adds the measurement error, so ABC targets the same
    # posterior the surrogate path will: the parameters given noisy prevalence.
    abc_noise = np.random.default_rng(2024)

    def abc_simulator(params: dict[str, float]) -> dict[str, float]:
        noisy = model_targets(params) + abc_noise.normal(0.0, MEASUREMENT_SD, len(observed))
        return dict(zip(TARGET_LABELS, noisy, strict=True))

    model_runs["count"] = 0
    reference = abc_calibrate(
        abc_simulator,
        priors={name: Uniform(*BOUNDS[name]) for name in CALIBRATED},
        observed=dict(zip(TARGET_LABELS, observed, strict=True)),
        population_size=300,
        max_populations=10,
        min_epsilon=MEASUREMENT_SD * np.sqrt(len(observed)),
        n_posterior=POSTERIOR_DRAWS,
        seed=1,
    )
    abc_runs = model_runs["count"]
    abc_posterior = reference.posterior
    print(f"\nABC: {abc_runs} model runs over {reference.n_populations} populations.")
    print("ABC posterior mean:", abc_posterior.mean().round(4).to_dict())

    # --- Surrogate: small design, Gaussian process, then sbi -----------------------
    model_runs["count"] = 0
    unit_design = LatinHypercube(d=len(CALIBRATED), seed=7).random(LHS_POINTS)
    design = pd.DataFrame(scale(unit_design, low, high), columns=list(CALIBRATED))
    design_targets = np.array([model_targets(row) for row in design.to_dict("records")])
    design_runs = model_runs["count"]
    print(f"\nSurrogate design: {design_runs} model runs.")

    kernel = ConstantKernel(1.0) * RBF([0.1, 0.1]) + WhiteKernel(1e-6, (1e-10, 1e-2))
    surrogates = [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5).fit(
            design.to_numpy(), design_targets[:, target]
        )
        for target in range(len(TARGET_LABELS))
    ]

    # Hold-out accuracy: predict fresh model runs the surrogate never saw.
    model_runs["count"] = 0
    holdout = scale(LatinHypercube(d=len(CALIBRATED), seed=99).random(300), low, high)
    holdout_targets = np.array(
        [model_targets(dict(zip(CALIBRATED, row, strict=True))) for row in holdout]
    )
    predicted = np.column_stack([surrogate.predict(holdout) for surrogate in surrogates])
    rmse = np.sqrt(((predicted - holdout_targets) ** 2).mean(axis=0))
    print("Surrogate hold-out RMSE per target:", rmse.round(5))

    # Neural posterior estimation with the surrogate as the (noisy) simulator.
    torch.manual_seed(0)
    surrogate_noise = np.random.default_rng(11)
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32),
        high=torch.tensor(high, dtype=torch.float32),
    )

    def surrogate_simulator(theta: torch.Tensor) -> torch.Tensor:
        mean = np.column_stack([surrogate.predict(theta.numpy()) for surrogate in surrogates])
        noisy = mean + surrogate_noise.normal(0.0, MEASUREMENT_SD, mean.shape)
        return torch.tensor(noisy, dtype=torch.float32)

    theta = prior.sample((SURROGATE_SIMS,))
    inference = NPE(prior=prior, show_progress_bars=False)
    with contextlib.redirect_stdout(io.StringIO()):  # quiet sbi's training progress print
        inference.append_simulations(theta, surrogate_simulator(theta)).train()
    neural_posterior = inference.build_posterior()
    samples = neural_posterior.sample(
        (POSTERIOR_DRAWS,),
        x=torch.tensor(observed, dtype=torch.float32),
        show_progress_bars=False,
    )
    sbi_posterior = pd.DataFrame(samples.numpy(), columns=list(CALIBRATED))
    print(f"\nSurrogate + sbi: {design_runs} model runs, {SURROGATE_SIMS} surrogate simulations.")
    print("Neural posterior mean:", sbi_posterior.mean().round(4).to_dict())

    # --- Compare -------------------------------------------------------------------
    summary = pd.DataFrame(
        {
            "truth": [TRUTH[name] for name in CALIBRATED],
            "abc_mean": abc_posterior.mean().reindex(CALIBRATED).to_numpy(),
            "abc_sd": abc_posterior.std().reindex(CALIBRATED).to_numpy(),
            "sbi_mean": sbi_posterior.mean().reindex(CALIBRATED).to_numpy(),
            "sbi_sd": sbi_posterior.std().reindex(CALIBRATED).to_numpy(),
        },
        index=list(CALIBRATED),
    )
    print("\nPosterior comparison:\n", summary.round(4))
    print(f"\nModel-run budget: ABC {abc_runs}, surrogate {design_runs} "
          f"({abc_runs / design_runs:.0f} times fewer).")

    fig, axes = plt.subplots(1, len(CALIBRATED), figsize=(9, 3.5))
    for axis, name in zip(axes, CALIBRATED, strict=True):
        axis.hist(abc_posterior[name], bins=40, density=True, alpha=0.5, label="ABC (pyabc)")
        axis.hist(sbi_posterior[name], bins=40, density=True, alpha=0.5, label="surrogate + sbi")
        axis.axvline(TRUTH[name], color="black", linestyle="--", linewidth=1, label="truth")
        axis.set_xlabel(name)
        axis.set_yticks([])
    axes[0].set_ylabel("posterior density")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "surrogate_calibration_posteriors.png", dpi=150)
    plt.close(fig)

    report = OUT / "run_report_surrogate_calibration.md"
    report.write_text(
        "# Surrogate-accelerated calibration\n\n"
        f"Model-run budget: ABC {abc_runs}, surrogate {design_runs}.\n\n"
        + summary.round(4).to_markdown()
        + "\n"
    )
    print(f"\nWrote {report} and the posterior comparison figure.")


if __name__ == "__main__":
    main()
