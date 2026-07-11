"""Tests for the shared `Strategy` value object and strategy normalisation."""

import pandas as pd
import pytest

from heormodel.models import Strategy
from heormodel.models._strategies import merge_overrides, normalize_strategies


def test_strategy_defaults_to_no_overrides():
    assert Strategy("care").overrides == {}
    assert Strategy("care", {"n": 2}).overrides == {"n": 2}


def test_normalize_mixes_names_and_strategy_objects():
    got = normalize_strategies(["A", Strategy("B", {"scale": 2.0})])
    assert got == {"A": {}, "B": {"scale": 2.0}}


def test_normalize_rejects_a_non_strategy_item():
    with pytest.raises(TypeError, match="names or Strategy objects"):
        normalize_strategies([object()])


def test_normalize_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate strategy name"):
        normalize_strategies(["A", Strategy("A", {"x": 1})])


def test_normalize_rejects_empty_sequence():
    with pytest.raises(ValueError, match="at least one strategy"):
        normalize_strategies([])


def test_merge_overrides_leaves_the_original_row_untouched():
    params = pd.Series({"a": 1.0, "b": 2.0})
    merged = merge_overrides(params, {"b": 5.0})
    assert merged["b"] == 5.0
    assert params["b"] == 2.0  # original unchanged


def test_merge_overrides_returns_the_same_row_when_empty():
    params = pd.Series({"a": 1.0})
    assert merge_overrides(params, {}) is params
