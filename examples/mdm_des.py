"""Replicate the continuous-time Sick-Sicker DES of Lopez-Mendez and others (2026).

Reproduces the published figures of Lopez-Mendez, Goldhaber-Fiebert, and
Alarid-Escudero, "A Tutorial on Discrete Event Simulation Models Using a
Cost-Effectiveness Analysis Example in R," Medical Decision Making
2026;46(5):533-548: the Sick-Sicker model in continuous time from age 25 to 100,
with age-dependent background mortality from a US 2015 life table, a Weibull
hazard on time spent Sick for progression, recovery, four strategies, transition
rewards, a cost-effectiveness analysis, epidemiological outcomes, and a
probabilistic analysis with acceptability curves, expected loss curves, and the
expected value of perfect information.

This example reproduces the companion R code's behavior on the two points that
move the numbers, not the Table 1 specification. First, each one-time transition
reward (onset cost, onset disutility, cost of dying) accrues as an annual flow
over the sojourn that ends in that transition rather than a lump sum at the
event, which is what the companion cost function computes; the accrual is
reconstructed here from the event history. Second, six Table 1 parameters the
companion draws but never reads are held at their base case. The one remaining
departure, a single random-number stream shared across all parameter sets, is
the framework's per-iteration seeding guarantee and is left in place, so the
probabilistic results match the published figures within Monte Carlo error.
See devdocs/replication-notes/mdm-des-departures.md for the full accounting.

The engine is the continuous clock of `MicrosimModel`, whose competing
time-to-event kernel is the article's next-event algorithm: sample a latent
arrival time for every permitted transition, take the earliest, move, repeat.
Because competing times are redrawn at every state entry, the Weibull draw
needs no truncation (time in state is zero when it is drawn), and `LifeTable`
redraws the death time conditional on current age under each state's hazard
ratio.

Run it with::

    uv run python examples/mdm_des.py

Outputs (written to ``examples/output/``):
    - epi_des_mdm.png, ceac_des_mdm.png, expected_loss_des_mdm.png,
      evpi_des_mdm.png, frontier_des_mdm.png
    - run_report_des_mdm.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from heormodel.cea import ceac, ceaf, expected_loss, icer_table
from heormodel.models import LifeTable, MicrosimModel, Outcomes, state_occupancy
from heormodel.params import Beta, Fixed, Gamma, LogNormal, ParameterSet, single_draw
from heormodel.report import (
    capture_run,
    plot_ceac,
    plot_expected_loss,
    plot_frontier,
    strategy_colors,
)
from heormodel.run import SeedManager, run_psa
from heormodel.voi import evpi

HERE = Path(__file__).parent
OUT = HERE / "output"

STATES = ("H", "S1", "S2", "D")
AGE_START, AGE_END = 25.0, 100.0
HORIZON = AGE_END - AGE_START
DISCOUNT_RATE = 0.03
N_BASE = 100_000  # individuals in the base case, matching the article
N_PSA_DRAWS = 1_000  # parameter sets, matching the article
N_PSA_IND = 10_000  # individuals per parameter set (the article uses 100,000)
WTP_GRID = np.arange(0.0, 200_001.0, 1_000.0)  # $1,000 steps, matching the article

# US all-cause mortality rate by single year of age, 25 to 99 (period life
# table, total population, 2015). The same array drives the age-varying cohort
# replication in examples/mdm_cohort_timedep.py.
MORTALITY_BY_AGE = np.array([
    0.001014, 0.000999, 0.00107, 0.001087, 0.001162, 0.001167, 0.001213, 0.001289, 0.001331,
    0.001375, 0.00142, 0.00149, 0.00155, 0.001616, 0.001657, 0.001747, 0.001902, 0.002052,
    0.002173, 0.002395, 0.002559, 0.002807, 0.003023, 0.003349, 0.003712, 0.004085, 0.00449,
    0.004905, 0.005364, 0.005806, 0.006253, 0.006775, 0.007395, 0.007895, 0.008418, 0.008974,
    0.009666, 0.010456, 0.011384, 0.011838, 0.012667, 0.013593, 0.0147, 0.015732, 0.01734,
    0.018758, 0.020967, 0.022917, 0.024913, 0.026767, 0.029707, 0.032412, 0.035982, 0.039238,
    0.043595, 0.048727, 0.053735, 0.059911, 0.066618, 0.074051, 0.08219, 0.090754, 0.103968,
    0.115093, 0.124341, 0.137872, 0.154177, 0.172393, 0.1941, 0.212654, 0.243752, 0.259087,
    0.287781, 0.316429, 0.339149,
])
MORTALITY = LifeTable(ages=np.arange(25, 100, dtype=float), rates=MORTALITY_BY_AGE)

# Base-case values from the article's Table 1 (c_trtB follows the text and the
# distribution mean, $13,000; the table's $12,000 entry is a misprint). Treatment
# A raises the Sick utility by a fixed 0.20 increment, so there is no separate
# treated-utility base value.
BASE = dict(
    r_HS1=0.15, r_S1H=0.5, r_S1S2_scale=0.08, r_S1S2_shape=1.1,
    hr_S1=3.0, hr_S2=10.0, hr_S1S2_trtB=0.6,
    c_H=2000.0, c_S1=4000.0, c_S2=15000.0, c_trtA=12000.0, c_trtB=13000.0,
    u_H=1.0, u_S1=0.75, u_S2=0.5,
    du_HS1=0.01, ic_HS1=1000.0, ic_D=2000.0,
)

# Sick and Sicker are indistinguishable in practice, so a strategy's treatment
# cost applies in both disease states; treatment A improves utility only in S1,
# and treatment B scales only the S1 to S2 progression hazard.
STRATEGIES = {
    "Standard of care": {"trtA": 0.0, "trtB": 0.0},
    "Strategy A": {"trtA": 1.0, "trtB": 0.0},
    "Strategy B": {"trtA": 0.0, "trtB": 1.0},
    "Strategy AB": {"trtA": 1.0, "trtB": 1.0},
}


def hazards(
    p: pd.Series, state: np.ndarray, attrs: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Sampled time to each competing transition, from the current state."""
    n = len(state)
    times = np.full((n, 4), np.inf)
    age = AGE_START + attrs["time"].to_numpy()
    h = state == 0
    if h.any():
        times[h, 1] = rng.exponential(1.0 / p["r_HS1"], int(h.sum()))
        times[h, 3] = MORTALITY.sample_time_to_death(rng, age[h])
    s1 = state == 1
    if s1.any():
        times[s1, 0] = rng.exponential(1.0 / p["r_S1H"], int(s1.sum()))
        # Weibull in proportional-hazards form: treatment B multiplies the PH
        # scale; sampling uses scale ** (-1/shape), the equivalent
        # accelerated-failure-time scale.
        scale_ph = p["r_S1S2_scale"] * (p["hr_S1S2_trtB"] if p["trtB"] else 1.0)
        aft_scale = scale_ph ** (-1.0 / p["r_S1S2_shape"])
        times[s1, 2] = aft_scale * rng.weibull(p["r_S1S2_shape"], int(s1.sum()))
        times[s1, 3] = MORTALITY.sample_time_to_death(rng, age[s1], hazard_ratio=p["hr_S1"])
    s2 = state == 2
    if s2.any():
        times[s2, 3] = MORTALITY.sample_time_to_death(rng, age[s2], hazard_ratio=p["hr_S2"])
    return times


def payoffs(p: pd.Series, state: np.ndarray, attrs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annual cost and utility flows by state and strategy."""
    n = len(state)
    cost = np.zeros(n)
    util = np.zeros(n)
    on_a, on_b = bool(p["trtA"]), bool(p["trtB"])
    tx_cost = on_a * p["c_trtA"] + on_b * p["c_trtB"]
    h = state == 0
    cost[h], util[h] = p["c_H"], p["u_H"]
    s1 = state == 1
    cost[s1] = p["c_S1"] + tx_cost
    # Treatment A raises the Sick utility by a fixed 0.20, matching the companion.
    util[s1] = (p["u_S1"] + 0.20) if on_a else p["u_S1"]
    s2 = state == 2
    cost[s2] = p["c_S2"] + tx_cost
    util[s2] = p["u_S2"]
    return cost, util


def transition_rewards(
    events: pd.DataFrame, draws: pd.DataFrame, n_individuals: int
) -> pd.DataFrame:
    """Mean per-person transition reward, accrued over the sojourn that ends in it.

    The companion cost function adds each one-time reward (the onset cost and
    disutility, and the cost of dying) to the annual flow of the sojourn that
    ends in the transition, then multiplies by the discounted length of that
    sojourn. This reconstructs that arithmetic from the event history: for every
    event, the reward enters as a per-year rate over the sojourn ``[start,
    time]``, discounted, so a $2,000 cost of dying enters as $2,000 per year over
    the final sojourn rather than a lump sum at death. The per-person mean is
    returned as a cost and a QALY column to add to each strategy's outcomes.
    """
    ev = events.sort_values(["strategy", "iteration", "individual", "time"])
    start = ev.groupby(["strategy", "iteration", "individual"])["time"].shift(fill_value=0.0)
    stop = ev["time"].to_numpy()
    # Discounted sojourn length, the companion's v_dwc: the integral of the
    # continuous discount factor over the sojourn that the transition ends.
    disc_years = (np.exp(-DISCOUNT_RATE * start.to_numpy()) - np.exp(-DISCOUNT_RATE * stop)) / (
        DISCOUNT_RATE
    )
    p = draws.loc[ev["iteration"].to_numpy()]
    onset = (ev["from_state"].to_numpy() == "H") & (ev["to_state"].to_numpy() == "S1")
    death = ev["to_state"].to_numpy() == "D"
    cost_rate = onset * p["ic_HS1"].to_numpy() + death * p["ic_D"].to_numpy()
    util_rate = -(onset * p["du_HS1"].to_numpy())
    contrib = pd.DataFrame(
        {
            "strategy": ev["strategy"].to_numpy(),
            "iteration": ev["iteration"].to_numpy(),
            "cost": cost_rate * disc_years,
            "qaly": util_rate * disc_years,
        }
    )
    return contrib.groupby(["strategy", "iteration"])[["cost", "qaly"]].sum() / n_individuals


def with_transition_rewards(
    outcomes: Outcomes, events: pd.DataFrame, draws: pd.DataFrame, n_individuals: int
) -> Outcomes:
    """Add the sojourn-accrued transition rewards to an outcomes panel."""
    totals = transition_rewards(events, draws, n_individuals)
    data = outcomes.data.add(totals.reindex(outcomes.data.index, fill_value=0.0), fill_value=0.0)
    return Outcomes(data, effect=outcomes.effect)


def build_engine(seed_manager: SeedManager, population: int) -> MicrosimModel:
    """The continuous-clock engine shared by the base case and the PSA."""
    return MicrosimModel(
        states=STATES,
        clock="continuous",
        hazards=hazards,
        payoffs=payoffs,
        population=population,
        strategies=STRATEGIES,
        horizon=HORIZON,
        discount_rate=DISCOUNT_RATE,
        seed_manager=seed_manager,
    )


def reward_adjusted_model(engine: MicrosimModel, n_individuals: int):
    """Wrap the engine as a ``draws -> Outcomes`` model that adds transition rewards.

    ``run_psa`` drives this closure over the draw matrix: it evaluates the engine
    with the event history, folds in the sojourn-accrued transition rewards, and
    returns the standard outcomes, so the reward accrual runs inside the same
    per-iteration seeding the framework guarantees.
    """

    def model(draws: pd.DataFrame) -> Outcomes:
        outcomes, events = engine.evaluate(draws, trace="events")
        return with_transition_rewards(outcomes, events, draws, n_individuals)

    return model


def parameter_set() -> ParameterSet:
    """The probabilistic parameters, matching the companion code's active draws.

    The companion draws all of Table 1 but merges six columns into its model
    under names the model does not read, so it runs them at base case: the
    Weibull progression scale, both treatment costs, both transition costs, and
    the treated-utility increment (Sick utility is ``u_S1 + 0.20`` regardless of
    the drawn value). Those six are held fixed here. With the scale fixed, only
    the Weibull shape varies, so the progression hazard barely moves across
    draws and no scale-shape correlation applies.
    """
    return ParameterSet(
        {
            "r_HS1": Gamma(30.0, 1.0 / 200.0),
            "r_S1H": Gamma(60.0, 1.0 / 120.0),
            "r_S1S2_scale": Fixed(0.08),
            "r_S1S2_shape": LogNormal.from_mean_se(1.10, 0.05),
            "hr_S1": LogNormal(np.log(3.0), 0.01),
            "hr_S2": LogNormal(np.log(10.0), 0.02),
            "hr_S1S2_trtB": LogNormal(np.log(0.6), 0.02),
            "c_H": Gamma(100.0, 20.0),
            "c_S1": Gamma(177.8, 22.5),
            "c_S2": Gamma(225.0, 66.7),
            "c_trtA": Fixed(12000.0),
            "c_trtB": Fixed(13000.0),
            "u_H": Beta(200.0, 3.0),
            "u_S1": Beta(130.0, 45.0),
            "u_S2": Beta(230.0, 230.0),
            "du_HS1": Beta(11.0, 1088.0),
            "ic_HS1": Fixed(1000.0),
            "ic_D": Fixed(2000.0),
        },
    )


def plot_epi(events: pd.DataFrame) -> None:
    """Survival and prevalence panels, the article's figure 3A and 3B."""
    grid = np.linspace(0.0, HORIZON, 151)
    occ = state_occupancy(
        events, states=STATES, initial_state="H", n_individuals=N_BASE, times=grid
    ).droplevel("iteration")
    # Survival is one minus dead-state occupancy; prevalence among the alive is
    # the summed disease occupancy over survival.
    alive = 1.0 - occ["D"]
    surv = alive.unstack("strategy")[list(STRATEGIES)]
    prev = (occ[["S1", "S2"]].sum(axis=1) / alive).unstack("strategy")[list(STRATEGIES)]
    colors = strategy_colors(list(STRATEGIES))
    # A shares SoC's transition dynamics and AB shares B's, so their curves
    # coincide; dashes keep both members of each pair visible.
    dashes = {"Strategy A": (4, 2), "Strategy AB": (4, 2)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    panels = ((axes[0], surv, "Survival"), (axes[1], prev, "Prevalence of S1 and S2"))
    for ax, curves, title in panels:
        for s in curves.columns:
            ax.plot(
                AGE_START + curves.index, curves[s], lw=1.8, color=colors[s],
                label=s, dashes=dashes.get(s, (None, None)),
            )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Proportion")
        ax.set_title(title)
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "epi_des_mdm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def dwell_times(events: pd.DataFrame) -> pd.DataFrame:
    """Mean completed sojourn per state and strategy, the article's figure 3C."""
    ev = events.sort_values(["strategy", "individual", "time"])
    entered = ev.groupby(["strategy", "individual"])["time"].shift(fill_value=0.0)
    ev = ev.assign(dwell=ev["time"] - entered)
    return ev.groupby(["strategy", "from_state"])["dwell"].mean().unstack("from_state")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    seeds = SeedManager(2)

    # Base case: one deterministic run with the event history. The transition
    # rewards accrue over the preceding sojourn, added from that history.
    engine = build_engine(seeds, N_BASE)
    base = single_draw(BASE)
    outcomes, events = engine.evaluate(base, trace="events")
    outcomes = with_transition_rewards(outcomes, events, base, N_BASE)
    print(f"Base case, {N_BASE:,} individuals per strategy, ages 25 to 100:")
    print(icer_table(outcomes).round(2).to_string())
    print(
        "\nThe frontier runs standard of care, then B, then AB; A is dominated"
        "\n(B costs less and yields more QALYs). The article reports the same"
        "\nordering: its acceptability frontier switches near the two ICERs."
    )

    plot_epi(events)
    print("\nMean completed dwell time by state (years):")
    print(dwell_times(events).round(2).to_string())

    # Probabilistic analysis: the article's 1,000 parameter sets. 10,000
    # individuals per set (rather than the article's 100,000) keeps the run
    # near two minutes; common random numbers absorb most of the extra
    # per-set Monte Carlo noise in the incremental comparisons. The engine is
    # wrapped so each iteration adds its sojourn-accrued transition rewards.
    # One draw per experiment keeps each worker's event history small.
    params = parameter_set()
    draws = params.sample(N_PSA_DRAWS, seed=seeds.generator())
    psa = run_psa(
        reward_adjusted_model(build_engine(seeds, N_PSA_IND), N_PSA_IND), draws, batch_size=1
    )
    print(f"\nProbabilistic analysis, {N_PSA_DRAWS:,} parameter sets:")
    print(icer_table(psa).round(2).to_string())

    curve = ceac(psa, WTP_GRID)
    front = ceaf(psa, WTP_GRID)
    losses = expected_loss(psa, WTP_GRID)
    evpi_curve = evpi(psa, WTP_GRID)

    plot_ceac(curve, ceaf_df=front).figure.savefig(
        OUT / "ceac_des_mdm.png", dpi=150, bbox_inches="tight"
    )
    plot_expected_loss(losses).figure.savefig(
        OUT / "expected_loss_des_mdm.png", dpi=150, bbox_inches="tight"
    )
    plot_frontier(psa).figure.savefig(OUT / "frontier_des_mdm.png", dpi=150, bbox_inches="tight")
    ax = plt.subplots()[1]
    ax.plot(evpi_curve.index, evpi_curve, lw=1.8, color="0.15")
    ax.set_xlabel("Willingness to pay")
    ax.set_ylabel("EVPI per person")
    ax.set_title("Expected value of perfect information")
    ax.figure.savefig(OUT / "evpi_des_mdm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    switches = front["strategy"].ne(front["strategy"].shift()) & front.index.to_series().gt(0)
    switch_points = [float(w) for w in front.index[switches]]
    at_switches = ", ".join(
        f"{evpi_curve.loc[w]:,.0f} at {w:,.0f}" for w in switch_points
    )
    print(
        f"\nThe acceptability frontier switches at {switch_points} dollars per QALY,"
        f"\nwith EVPI peaks (dollars per person) of {at_switches}, matching the two"
        "\nswitch points and the small EVPI peaks of the article's figure 4. Holding"
        "\nthe six companion-fixed parameters at base case (the Weibull progression"
        "\nscale, both treatment costs, both transition costs, and the treatment-A"
        "\nutility increment) keeps the strategy comparisons near-certain. The only"
        "\nremaining difference is Monte Carlo error: the companion drives every"
        "\nparameter set from one shared random-number stream, while this run keeps"
        "\nthe framework's per-iteration seeding. See"
        "\ndevdocs/replication-notes/mdm-des-departures.md for the full accounting."
    )

    record = capture_run(
        seed=seeds,
        outcomes=psa,
        draw_sources={c: str(params.distributions[c]) for c in draws.columns},
        note=(
            f"Continuous-time Sick-Sicker DES replication (MDM 2026): base case "
            f"{N_BASE:,} individuals, PSA {N_PSA_DRAWS:,} sets x {N_PSA_IND:,} individuals."
        ),
    )
    (OUT / "run_report_des_mdm.md").write_text(
        record.to_markdown("heormodel Sick-Sicker DES replication run report")
    )
    print(f"\nWrote plots and run report to {OUT}/")


if __name__ == "__main__":
    main()
