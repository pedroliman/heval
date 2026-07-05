"""Tests for ABC calibration (skipped when pyabc is not installed)."""

import numpy as np
import pytest

pyabc = pytest.importorskip("pyabc")

from heormodel.calibrate import abc_calibrate, to_pyabc_prior  # noqa: E402
from heormodel.params import Beta, Fixed, Normal, Uniform  # noqa: E402


class TestPriorTranslation:
    def test_supported_distributions_translate(self):
        prior = to_pyabc_prior({"p": Beta(2, 8), "mu": Normal(0, 1), "u": Uniform(0, 2)})
        sample = prior.rvs()
        assert set(sample.keys()) == {"p", "mu", "u"}
        assert 0.0 <= sample["p"] <= 1.0

    def test_unsupported_distribution_raises(self):
        with pytest.raises(TypeError, match="not supported"):
            to_pyabc_prior({"k": Fixed(1.0)})


class TestAbcCalibrate:
    def test_posterior_concentrates_on_target(self, tmp_path):
        rng = np.random.default_rng(0)

        def simulator(params: dict[str, float]) -> dict[str, float]:
            # noisy observation of the parameter itself
            return {"prevalence": params["risk"] + rng.normal(0.0, 0.01)}

        result = abc_calibrate(
            simulator,
            priors={"risk": Uniform(0.0, 1.0)},
            observed={"prevalence": 0.30},
            population_size=100,
            max_populations=4,
            n_posterior=500,
            seed=1,
            db_path=tmp_path / "abc.db",
        )
        posterior = result.posterior
        assert posterior.index.name == "iteration"
        assert len(posterior) == 500
        assert posterior["risk"].mean() == pytest.approx(0.30, abs=0.05)
        assert posterior["risk"].std() < 0.15  # tighter than the U(0,1) prior (sd .29)
        assert result.n_populations >= 1
        assert "weight" in result.weighted.columns
