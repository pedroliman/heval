"""Shared strategy handling for the engines.

Every engine accepts ``strategies`` in one of two forms: a sequence of names,
or a mapping of name to a parameter-override dict merged into each draw. Both
normalise to an ordered mapping here, and every engine passes the strategy name
to its user-supplied model functions, so branch-on-name and override styles
work identically across engines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

#: A sequence of strategy names, or a mapping of name to parameter overrides.
StrategySpec = Sequence[str] | Mapping[str, Mapping[str, Any]]


def normalize_strategies(strategies: StrategySpec) -> dict[str, dict[str, Any]]:
    """Coerce either strategy form into an ordered name-to-overrides mapping.

    Args:
        strategies: A sequence of names (each carries no overrides) or a
            mapping of name to a parameter-override dict.

    Returns:
        A dict keyed by strategy name, preserving order, whose values are the
        override dicts (empty for the sequence form).

    Example:
        >>> from heormodel.models._strategies import normalize_strategies
        >>> normalize_strategies(["A", "B"])
        {'A': {}, 'B': {}}
        >>> normalize_strategies({"Treatment": {"on_treatment": 1.0}})
        {'Treatment': {'on_treatment': 1.0}}
    """
    if isinstance(strategies, Mapping):
        result = {str(name): dict(overrides) for name, overrides in strategies.items()}
    else:
        result = {}
        for name in strategies:
            if not isinstance(name, str):
                raise TypeError(
                    "strategies must be a sequence of names or a name-to-overrides mapping."
                )
            result[name] = {}
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
