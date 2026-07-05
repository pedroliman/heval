"""Expected value of perfect information, directly from the PSA."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from heormodel.cea.nb import nmb
from heormodel.models.outcomes import Outcomes


def evpi(
    outcomes: Outcomes,
    wtp: float | ArrayLike | Sequence[float],
    *,
    effect: str | None = None,
) -> float | pd.Series:
    """Expected value of perfect information per decision.

    EVPI is the expected NMB gain from resolving **all** uncertainty:
    ``E[max_d NMB_d] - max_d E[NMB_d]``, estimated directly from the PSA
    iterations.

    Args:
        outcomes: Standard PSA outcomes.
        wtp: A willingness-to-pay value or grid.
        effect: Effect column (default: the primary effect).

    Returns:
        A float for scalar ``wtp``; a Series indexed by ``wtp`` for a grid.

    Example:
        >>> import pandas as pd
        >>> from heormodel.models import Outcomes
        >>> from heormodel.voi import evpi
        >>> c = pd.DataFrame({"A": [0.0, 0.0], "B": [5.0, 5.0]})
        >>> e = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, -1.0]})
        >>> evpi(Outcomes.from_wide(c, e), wtp=10.0)
        2.5
    """
    grid = np.atleast_1d(np.asarray(wtp, dtype=np.float64))
    values = []
    for lam in grid:
        nb = nmb(outcomes, float(lam), effect=effect).to_numpy(dtype=np.float64)
        values.append(float(nb.max(axis=1).mean() - nb.mean(axis=0).max()))
    if np.isscalar(wtp) or (hasattr(wtp, "ndim") and getattr(wtp, "ndim", 1) == 0):
        return values[0]
    return pd.Series(values, index=pd.Index(grid, name="wtp"), name="evpi")
