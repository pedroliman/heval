"""Value of information end to end, on the Sick-Sicker cohort model.

Runs the full value-of-information workflow, expected value of perfect
information (EVPI), of partial perfect information (EVPPI), and of sample
information (EVSI), on the four-state, four-intervention Sick-Sicker cohort
state-transition model of Alarid-Escudero and others, "An Introductory
Tutorial on Cohort State-Transition Models in R Using a Cost-Effectiveness
Analysis Example," Medical Decision Making 2023;43(1):3-20 (the same model
``examples/mdm_cohort.py`` reproduces and validates against the published
base case).

The workflow is the standard one: a ``ParameterSet``, ``run_psa``, then the
``heormodel.voi`` estimators on the outcomes. ``evppi_ranking`` finds that
the Sick-state utility under standard of care, ``u_S1``, drives the decision
uncertainty, so the EVSI section proposes a preference-based utility survey
of Sick patients to resolve it, and the closing section prices the survey
size that maximizes its expected net benefit of sampling.

Run it with::

    uv run python examples/voi_tutorial.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heormodel.cea import icer_table
from heormodel.models import CohortSpec, MarkovModel
from heormodel.params import Beta, Gamma, LogNormal, ParameterSet
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi, evppi_ranking, evsi_regression, simulate_summaries

STATES = ("H", "S1", "S2", "D")
INTERVENTIONS = ("Standard of care", "Intervention A", "Intervention B", "Intervention AB")
N_CYCLES = 75  # ages 25 to 100, annual cycles
WTP = 100_000.0
N_SIM = 10_000  # value-of-information estimates need more draws than a mean estimate does
PATIENT_SD = 0.20  # per-patient sd of a preference-based utility instrument


def rate_to_prob(rate: float | np.ndarray) -> np.ndarray:
    """Convert an instantaneous rate to a per-cycle probability."""
    return 1.0 - np.exp(-np.asarray(rate))


def sick_sicker_model(params: pd.Series, intervention: str) -> CohortSpec:
    """Transition matrix and per-state payoffs for one intervention and draw."""
    p_HS1 = rate_to_prob(params["r_HS1"])
    p_S1H = rate_to_prob(params["r_S1H"])
    p_S1S2 = rate_to_prob(params["r_S1S2"])
    p_HD = rate_to_prob(params["r_HD"])
    p_S1D = rate_to_prob(params["r_HD"] * params["hr_S1"])
    p_S2D = rate_to_prob(params["r_HD"] * params["hr_S2"])
    treats_b = intervention in ("Intervention B", "Intervention AB")
    p_prog = rate_to_prob(params["r_S1S2"] * params["hr_S1S2_trtB"]) if treats_b else p_S1S2

    P = np.zeros((4, 4))
    P[0, 0] = (1 - p_HD) * (1 - p_HS1)
    P[0, 1] = (1 - p_HD) * p_HS1
    P[0, 3] = p_HD
    P[1, 0] = (1 - p_S1D) * p_S1H
    P[1, 1] = (1 - p_S1D) * (1 - (p_S1H + p_prog))
    P[1, 2] = (1 - p_S1D) * p_prog
    P[1, 3] = p_S1D
    P[2, 2] = 1 - p_S2D
    P[2, 3] = p_S2D
    P[3, 3] = 1.0

    add = {
        "Standard of care": 0.0,
        "Intervention A": params["c_trtA"],
        "Intervention B": params["c_trtB"],
        "Intervention AB": params["c_trtA"] + params["c_trtB"],
    }[intervention]
    cost = np.array([params["c_H"], params["c_S1"] + add, params["c_S2"] + add, 0.0])
    treats_a = intervention in ("Intervention A", "Intervention AB")
    u_s1 = params["u_trtA"] if treats_a else params["u_S1"]
    effect = np.array([params["u_H"], u_s1, params["u_S2"], 0.0])
    return CohortSpec(P, cost, effect)


def parameters() -> ParameterSet:
    """PSA distributions from the tutorial's supplementary code."""
    return ParameterSet(
        {
            "r_HD": Gamma(shape=20, scale=1 / 10000),
            "r_HS1": Gamma(shape=30, scale=1 / 200),
            "r_S1H": Gamma(shape=60, scale=1 / 120),
            "r_S1S2": Gamma(shape=84, scale=1 / 800),
            "hr_S1": LogNormal(mu=np.log(3.0), sigma=0.01),
            "hr_S2": LogNormal(mu=np.log(10.0), sigma=0.02),
            "hr_S1S2_trtB": LogNormal(mu=np.log(0.6), sigma=0.02),
            "c_H": Gamma(shape=100, scale=20.0),
            "c_S1": Gamma(shape=177.8, scale=22.5),
            "c_S2": Gamma(shape=225, scale=66.7),
            "c_trtA": Gamma(shape=73.5, scale=163.3),
            "c_trtB": Gamma(shape=86.2, scale=150.8),
            "u_H": Beta(200, 3),
            "u_S1": Beta(130, 45),
            "u_S2": Beta(230, 230),
            "u_trtA": Beta(300, 15),
        }
    )


def main() -> None:
    engine = MarkovModel(
        states=STATES, interventions=INTERVENTIONS, transitions_and_rewards=sick_sicker_model,
        n_cycles=N_CYCLES, initial_state="H", cycle_correction="simpson",
    )
    draws = parameters().sample(N_SIM, seed=SeedManager(20260705).generator())
    outcomes = run_psa(engine, draws).outcomes

    # --- decision -----------------------------------------------------------
    print(f"Willingness to pay: {WTP:,.0f} per QALY\n")
    print("ICER table:")
    print(icer_table(outcomes).round(1).to_string())

    # --- EVPI: the ceiling on research value --------------------------------
    print(f"\nEVPI (value of resolving all uncertainty), per person: {evpi(outcomes, WTP):,.0f}")

    # --- EVPPI: which parameters that value attaches to ---------------------
    ranking = evppi_ranking(outcomes, draws, WTP)
    print("\nEVPPI by parameter (top 5 of 16):")
    for p in ranking.index[:5]:
        print(f"  {p:>8}  {ranking[p]:>10,.0f}")

    # --- EVSI: the value of a utility survey targeting the top parameter, ---
    # --- by sample size -------------------------------------------------------
    top_param = str(ranking.index[0])
    sizes = (25, 50, 100, 200, 400, 800, 1_600, 3_200, 6_400)
    est_by_size = {}
    print(f"\nEVSI of a {top_param!r} utility survey, per person:")
    print(f"  {'patients':>8}  {'estimate':>10}")
    for n_survey in sizes:
        tau = PATIENT_SD / np.sqrt(n_survey)
        rng = np.random.default_rng(n_survey)

        def survey(row: pd.Series, r: np.random.Generator, tau: float = tau) -> dict[str, float]:
            return {"ubar": float(row[top_param]) + r.normal(0.0, tau)}

        summaries = simulate_summaries(draws, survey, seed=rng)
        est = evsi_regression(outcomes, summaries, WTP)
        est_by_size[n_survey] = est
        print(f"  {n_survey:>8}  {est:>10,.0f}")

    # --- expected net benefit of sampling, with the survey size to fund -----
    evsi = pd.Series(est_by_size)
    years = np.arange(10)
    beneficiaries = 2_000 * (1.03**-years).sum()  # discounted future patients
    cost = 100_000 + 300 * evsi.index  # fixed cost plus a per-participant cost
    enbs = beneficiaries * evsi - cost  # population EVSI minus survey cost
    best = enbs.idxmax()
    print(
        "\nENBS: 2,000 patients enter the Sick state a year over 10 years, "
        "100,000 fixed + 300 per participant."
        f"\nBest size {best} patients, ENBS {enbs[best]:,.0f}."
    )


if __name__ == "__main__":
    main()
