"""Surrogate-accelerated calibration matches an ABC reference (skipped without deps).

The tutorial's claim is twofold: a Gaussian process trained on a small Latin
hypercube design reproduces the model's calibration targets, and neural posterior
estimation against that surrogate recovers the same posterior an ABC run recovers
against the model, at far fewer model runs. Both checks run here at reduced sizes,
on the same three-state Markov model as ``examples/surrogate_calibration.py``.
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyabc")
pytest.importorskip("sbi")

os.environ["ABC_LOG_LEVEL"] = "WARNING"
logging.getLogger("sbi").setLevel(logging.WARNING)

from heormodel.calibrate import abc_calibrate  # noqa: E402
from heormodel.models import CohortSpec, MarkovModel  # noqa: E402
from heormodel.params import Uniform  # noqa: E402

STATES = ("healthy", "sick", "dead")
INTERVENTION = "natural_history"
N_CYCLES = 40
TARGET_CYCLES = (8, 16, 28)
TARGET_LABELS = [f"sick_c{cycle}" for cycle in TARGET_CYCLES]
BACKGROUND_MORTALITY = 0.01
MEASUREMENT_SD = 0.01
CALIBRATED = ("p_HS", "p_SD")
BOUNDS = {"p_HS": (0.02, 0.20), "p_SD": (0.05, 0.35)}
TRUTH = {"p_HS": 0.08, "p_SD": 0.15}


def _transitions_and_rewards(params, intervention):
    p_HS, p_SD = params["p_HS"], params["p_SD"]
    transition = np.array([
        [1.0 - p_HS - BACKGROUND_MORTALITY, p_HS, BACKGROUND_MORTALITY],
        [0.0, 1.0 - p_SD, p_SD],
        [0.0, 0.0, 1.0],
    ])
    return CohortSpec(transition, np.zeros(3), np.array([1.0, 0.8, 0.0]))


def _engine():
    return MarkovModel(
        states=STATES, interventions=(INTERVENTION,),
        transitions_and_rewards=_transitions_and_rewards,
        n_cycles=N_CYCLES, cycle_correction="none",
    )


def _targets(engine, params):
    occupancy = engine.trace(pd.Series(params), INTERVENTION)["sick"].to_numpy()
    return np.array([occupancy[cycle] for cycle in TARGET_CYCLES])


def test_surrogate_matches_model_and_abc():
    import warnings

    from scipy.stats.qmc import LatinHypercube, scale
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    engine = _engine()
    low = np.array([BOUNDS[name][0] for name in CALIBRATED])
    high = np.array([BOUNDS[name][1] for name in CALIBRATED])
    observed = _targets(engine, TRUTH)

    # Gaussian process on a small design reproduces held-out model runs.
    design = pd.DataFrame(
        scale(LatinHypercube(d=2, seed=7).random(60), low, high), columns=list(CALIBRATED)
    )
    design_targets = np.array([_targets(engine, row) for row in design.to_dict("records")])
    kernel = ConstantKernel(1.0) * RBF([0.1, 0.1]) + WhiteKernel(1e-6, (1e-10, 1e-2))
    surrogates = [
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
        .fit(design.to_numpy(), design_targets[:, target])
        for target in range(len(TARGET_LABELS))
    ]
    holdout = scale(LatinHypercube(d=2, seed=99).random(100), low, high)
    holdout_targets = np.array(
        [_targets(engine, dict(zip(CALIBRATED, row, strict=True))) for row in holdout]
    )
    predicted = np.column_stack([surrogate.predict(holdout) for surrogate in surrogates])
    rmse = np.sqrt(((predicted - holdout_targets) ** 2).mean(axis=0))
    assert rmse.max() < MEASUREMENT_SD / 5  # surrogate error well below measurement error

    # ABC reference against the model.
    abc_noise = np.random.default_rng(2024)

    def abc_simulator(params):
        noisy = _targets(engine, params) + abc_noise.normal(0.0, MEASUREMENT_SD, len(observed))
        return dict(zip(TARGET_LABELS, noisy, strict=True))

    reference = abc_calibrate(
        abc_simulator,
        priors={name: Uniform(*BOUNDS[name]) for name in CALIBRATED},
        observed=dict(zip(TARGET_LABELS, observed, strict=True)),
        population_size=150,
        max_populations=8,
        min_epsilon=MEASUREMENT_SD * np.sqrt(len(observed)),
        n_posterior=1_000,
        seed=1,
    )
    abc_mean = reference.posterior.mean()

    # Neural posterior estimation against the surrogate.
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    torch.manual_seed(0)
    surrogate_noise = np.random.default_rng(11)
    prior = BoxUniform(
        low=torch.tensor(low, dtype=torch.float32), high=torch.tensor(high, dtype=torch.float32)
    )

    def surrogate_simulator(theta):
        mean = np.column_stack([surrogate.predict(theta.numpy()) for surrogate in surrogates])
        noisy = mean + surrogate_noise.normal(0.0, MEASUREMENT_SD, mean.shape)
        return torch.tensor(noisy, dtype=torch.float32)

    theta = prior.sample((4_000,))
    inference = NPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, surrogate_simulator(theta)).train()
    posterior = inference.build_posterior()
    samples = posterior.sample(
        (1_000,), x=torch.tensor(observed, dtype=torch.float32), show_progress_bars=False
    ).numpy()
    sbi_mean = pd.DataFrame(samples, columns=list(CALIBRATED)).mean()

    for name in CALIBRATED:
        assert abs(abc_mean[name] - TRUTH[name]) < 0.01  # ABC recovers truth
        assert abs(sbi_mean[name] - TRUTH[name]) < 0.02  # surrogate path recovers truth
        assert abs(abc_mean[name] - sbi_mean[name]) < 0.015  # the two posteriors agree
