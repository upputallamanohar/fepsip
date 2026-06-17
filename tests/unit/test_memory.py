"""Unit tests for Financial Memory Store."""
import pytest


def test_memory_store_seeds():
    from src.memory.financial_memory import FinancialMemoryStore
    store = FinancialMemoryStore()
    assert store._fallback._entries or store._use_qdrant


def test_search_returns_results():
    from src.memory.financial_memory import FinancialMemoryStore
    store = FinancialMemoryStore()
    results = store.search("oil price increase 20%", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_scores_bounded():
    from src.memory.financial_memory import FinancialMemoryStore
    store = FinancialMemoryStore()
    results = store.search("financial crisis bank collapse")
    for r in results:
        assert -1.0 <= r["score"] <= 1.1  # cosine can be slightly above 1 due to float


def test_query_returns_string():
    from src.memory.financial_memory import FinancialMemoryStore
    store = FinancialMemoryStore()
    answer = store.query("What happened during the 2008 financial crisis?")
    assert isinstance(answer, str)
    assert len(answer) > 10


def test_store_and_retrieve():
    from src.memory.financial_memory import FinancialMemoryStore, MemoryEntry
    from datetime import datetime
    store = FinancialMemoryStore()
    entry = MemoryEntry(
        id="test_event_001",
        event_type="test",
        description="Unique test event for unit testing purposes",
        outcome="Market recovered",
        affected_sectors=["Technology"],
        impact_magnitude=0.5,
        timestamp=datetime(2024, 1, 1),
    )
    store.store(entry)
    results = store.search("unique test event unit testing")
    assert len(results) >= 1
