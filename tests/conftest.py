"""Pytest configuration and shared fixtures."""
import sys
import os
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging for tests."""
    from src.utils.logger import setup_logger
    setup_logger(level="WARNING")


@pytest.fixture
def sample_tickers():
    return ["AAPL", "MSFT", "JPM", "XOM", "TSLA"]


@pytest.fixture
def knowledge_graph():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    return FinancialKnowledgeGraph()


@pytest.fixture
def prediction_engine():
    from src.prediction.predictor import PredictionEngine
    return PredictionEngine()


@pytest.fixture
def event_classifier():
    from src.events.event_intelligence import EventClassifier
    return EventClassifier(use_finbert=False)
