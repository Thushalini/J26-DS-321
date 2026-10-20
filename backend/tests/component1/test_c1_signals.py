"""Tests for component1/agent/signals.py (no Pydantic needed)."""
from component1.agent.signals import CONFLICT, GAP, SAVED, Signal


def test_the_same_signal_has_the_same_key():
    assert Signal(GAP, "dataset").key == Signal(GAP, "dataset").key


def test_the_order_of_the_values_does_not_change_the_key():
    a = Signal(CONFLICT, "target_variable", ("x", "y"))
    b = Signal(CONFLICT, "target_variable", ("y", "x"))
    assert a.key == b.key


def test_different_values_give_a_different_key():
    a = Signal(CONFLICT, "target_variable", ("x", "y"))
    b = Signal(CONFLICT, "target_variable", ("x", "y", "z"))
    assert a.key != b.key


def test_the_detail_matters_for_saved_signals_but_not_for_gaps():
    assert Signal(SAVED, "metric", ("RMSE",), detail="marked uncertain").key != Signal(SAVED, "metric", ("RMSE",), detail="now confirmed").key
    assert Signal(GAP, "dataset", detail="a").key == Signal(GAP, "dataset", detail="b").key