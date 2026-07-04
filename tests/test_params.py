"""Tests for distribution specs and correlated PSA sampling."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from heval.params import Beta, Dirichlet, Fixed, Gamma, LogNormal, Normal, ParameterSet, Uniform


class TestMethodOfMoments:
    def test_beta_matches_mean_and_se(self):
        d = Beta.from_mean_se(0.3, 0.05)
        assert d.mean() == pytest.approx(0.3)
        assert d.sd() == pytest.approx(0.05)

    def test_gamma_matches_mean_and_se(self):
        d = Gamma.from_mean_se(1200.0, 250.0)
        assert d.mean() == pytest.approx(1200.0)
        assert d.sd() == pytest.approx(250.0)

    def test_lognormal_matches_mean_and_se(self):
        d = LogNormal.from_mean_se(2.5, 0.8)
        assert d.mean() == pytest.approx(2.5)
        assert d.sd() == pytest.approx(0.8)

    def test_beta_rejects_impossible_se(self):
        with pytest.raises(ValueError, match="se too large"):
            Beta.from_mean_se(0.5, 0.6)

    def test_beta_rejects_mean_outside_unit_interval(self):
        with pytest.raises(ValueError):
            Beta.from_mean_se(1.2, 0.1)


class TestDistributions:
    def test_sampling_is_reproducible(self):
        d = Gamma.from_mean_se(100.0, 20.0)
        assert np.array_equal(d.sample(10, rng=5), d.sample(10, rng=5))

    def test_fixed_is_constant(self):
        assert np.all(Fixed(2.5).sample(4) == 2.5)
        assert Fixed(2.5).sd() == 0.0

    def test_dirichlet_mean(self):
        d = Dirichlet((8.0, 1.0, 1.0))
        np.testing.assert_allclose(d.mean(), [0.8, 0.1, 0.1])

    def test_dirichlet_component_labels(self):
        d = Dirichlet((1.0, 2.0), names=("a", "b"))
        assert d.component_labels("p") == ["p[a]", "p[b]"]

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            Gamma(-1.0, 2.0)
        with pytest.raises(ValueError):
            Normal(0.0, 0.0)
        with pytest.raises(ValueError):
            Uniform(1.0, 1.0)
        with pytest.raises(ValueError):
            Dirichlet((1.0,))


class TestParameterSet:
    def test_draw_matrix_shape_and_index(self):
        ps = ParameterSet({"a": Normal(0, 1), "b": Uniform(0, 1)})
        draws = ps.sample(100, seed=1)
        assert draws.shape == (100, 2)
        assert draws.index.name == "iteration"
        assert list(draws.columns) == ["a", "b"]

    def test_reproducible_with_same_seed(self):
        ps = ParameterSet({"a": Normal(0, 1)})
        pd.testing.assert_frame_equal(ps.sample(50, seed=3), ps.sample(50, seed=3))

    def test_marginals_are_exact_under_correlation(self):
        ps = ParameterSet(
            {"p": Beta.from_mean_se(0.2, 0.04), "c": Gamma.from_mean_se(500, 80)},
            correlation={("p", "c"): 0.6},
        )
        draws = ps.sample(50_000, seed=7)
        assert draws["p"].mean() == pytest.approx(0.2, abs=0.005)
        assert draws["p"].std() == pytest.approx(0.04, abs=0.005)
        assert draws["c"].mean() == pytest.approx(500, rel=0.02)
        # Kolmogorov-Smirnov against the target marginal
        target = stats.gamma((500 / 80) ** 2, scale=80**2 / 500)
        ks = stats.kstest(draws["c"], target.cdf)
        assert ks.pvalue > 0.01

    def test_spearman_correlation_matches_target(self):
        ps = ParameterSet(
            {"a": Normal(0, 1), "b": Gamma.from_mean_se(10, 3)},
            correlation={("a", "b"): 0.5},
        )
        draws = ps.sample(50_000, seed=11)
        rho = stats.spearmanr(draws["a"], draws["b"]).statistic
        assert rho == pytest.approx(0.5, abs=0.02)

    def test_dirichlet_expands_and_sums_to_one(self):
        ps = ParameterSet({"p": Dirichlet((70, 20, 10), names=("s", "p", "d"))})
        draws = ps.sample(20_000, seed=2)
        assert list(draws.columns) == ["p[s]", "p[p]", "p[d]"]
        np.testing.assert_allclose(draws.sum(axis=1), 1.0)
        np.testing.assert_allclose(draws.mean(), [0.7, 0.2, 0.1], atol=0.01)

    def test_unknown_parameter_in_correlation_raises(self):
        with pytest.raises(KeyError):
            ParameterSet({"a": Normal(0, 1)}, correlation={("a", "zzz"): 0.2})

    def test_correlation_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ParameterSet({"a": Normal(0, 1), "b": Normal(0, 1)}, correlation={("a", "b"): 1.5})

    def test_means_and_spec(self):
        ps = ParameterSet({"a": Fixed(3.0), "p": Dirichlet((1.0, 1.0))})
        assert ps.means()["a"] == 3.0
        assert ps.means()["p[0]"] == pytest.approx(0.5)
        assert "Fixed" in ps.spec()["a"]
