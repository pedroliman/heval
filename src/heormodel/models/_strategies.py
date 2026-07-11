"""Shared strategy handling for the engines.

A strategy is a name, optionally carrying parameter overrides merged into each
draw. `Strategy` is the canonical spelling; a bare string is shorthand for a
strategy with no overrides. Every engine accepts ``strategies`` as a sequence of
either, normalises it here, and passes the strategy name to its user-supplied
model functions, so branch-on-name and override styles work identically across
engines.

Branch on the name when the arms differ in structure or in which model function
runs (a treatment that changes a utility, a comparator that skips a step). Use
overrides for numeric scenario knobs the model already reads as parameters (a
server count, a capacity). Overrides written as flags (``{"on_treatment": 1.0}``
read back as a boolean) are the case branch-on-name replaces: a flag standing in
for a float is invisible to `heormodel.voi.evppi_ranking` and to the
deterministic sensitivity builders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Strategy:
    """A named strategy, optionally carrying parameter overrides.

    Args:
        name: The strategy label, as it appears in `Outcomes`.
        overrides: Parameter values written over each draw for this strategy.
            Empty by default; reach for it only for numeric scenario knobs, not
            to encode which arm is which (branch on ``name`` for that).

    Example:
        >>> from heormodel.models import Strategy
        >>> Strategy("Expanded capacity", {"n_servers": 2}).overrides
        {'n_servers': 2}
        >>> Strategy("Standard care").overrides
        {}
    """

    name: str
    overrides: Mapping[str, Any] = field(default_factory=dict)


#: A sequence of strategy names or `Strategy` objects.
StrategySpec = Sequence[str | Strategy]


def normalize_strategies(strategies: StrategySpec) -> dict[str, dict[str, Any]]:
    """Coerce a strategy sequence into an ordered name-to-overrides mapping.

    Args:
        strategies: A sequence whose items are strategy names (no overrides) or
            `Strategy` objects.

    Returns:
        A dict keyed by strategy name, preserving order, whose values are the
        override dicts (empty for the bare-name form).

    Example:
        >>> from heormodel.models import Strategy
        >>> from heormodel.models._strategies import normalize_strategies
        >>> normalize_strategies(["A", "B"])
        {'A': {}, 'B': {}}
        >>> normalize_strategies([Strategy("Treatment", {"n_servers": 2})])
        {'Treatment': {'n_servers': 2}}
    """
    result: dict[str, dict[str, Any]] = {}
    for item in strategies:
        if isinstance(item, Strategy):
            name, overrides = item.name, dict(item.overrides)
        elif isinstance(item, str):
            name, overrides = item, {}
        else:
            raise TypeError("strategies must be a sequence of names or Strategy objects.")
        if name in result:
            raise ValueError(f"Duplicate strategy name {name!r}.")
        result[name] = overrides
    if not result:
        raise ValueError("Provide at least one strategy.")
    return result


def merge_overrides(params: pd.Series, overrides: Mapping[str, Any]) -> pd.Series:
    """Return ``params`` with the strategy's overrides written over it.

    The original series is left untouched when there are overrides; when there
    are none the same series is returned so the common case allocates nothing.

    Args:
        params: One draw-matrix row.
        overrides: The strategy's parameter overrides.
    """
    if not overrides:
        return params
    merged = params.copy()
    for key, value in overrides.items():
        merged[key] = value
    return merged
