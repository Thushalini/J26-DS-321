"""Shared test setup for Component 1: keep the NLI model off so tests run offline and fast.
Tests that need NLI install a pretend model with conflicts.nli.set_relation."""
import pytest

from component1.conflicts import nli


@pytest.fixture(autouse=True)
def _nli_off(monkeypatch):
    monkeypatch.setenv("C1_NLI", "off")
    nli.set_relation(None)
    yield
    nli.set_relation(None)
