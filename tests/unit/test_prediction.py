"""Unit tests for Prediction Engine."""
import pytest
import pandas as pd
import numpy as np


def _make_price_series(n=100, seed=42):
    np.random.seed(seed)
    prices = 100 * (1 + np.random.randn(n) * 0.02).cumprod()
    return pd.Series(prices)


def test_stock_prediction_heuristic():
    from src.prediction.predictor import PredictionEngine
    engine = PredictionEngine()
    pred = engine.predict_stock("AAPL", price_series=_make_price_series(), sentiment=0.3)
    assert pred.ticker == "AAPL"
    assert pred.direction in ["UP", "DOWN", "NEUTRAL"]
    assert abs(pred.up_prob + pred.down_prob + pred.neutral_prob - 1.0) < 1e-4
    assert 0 <= pred.confidence <= 1


def test_stock_prediction_probs_sum_to_one():
    from src.prediction.predictor import PredictionEngine
    engine = PredictionEngine()
    for ticker in ["AAPL", "MSFT", "XOM"]:
        pred = engine.predict_stock(ticker, sentiment=np.random.uniform(-1, 1))
        total = pred.up_prob + pred.down_prob + pred.neutral_prob
        assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1 for {ticker}: {total}"


def test_negative_sentiment_increases_down_prob():
    from src.prediction.predictor import PredictionEngine
    engine = PredictionEngine()
    pred_neg = engine.predict_stock("TSLA", sentiment=-0.9, event_severity=0.8)
    pred_pos = engine.predict_stock("TSLA", sentiment=0.9, event_severity=0.0)
    assert pred_neg.down_prob > pred_pos.down_prob


def test_ripple_effect():
    from src.prediction.predictor import PredictionEngine
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    engine = PredictionEngine()
    graph = FinancialKnowledgeGraph()
    ripple = engine.predict_ripple_effects("XOM", "Oil price shock", 0.8, graph)
    assert ripple.source_ticker == "XOM"
    assert len(ripple.affected) >= 0


def test_systemic_risk_report():
    from src.prediction.predictor import PredictionEngine
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    engine = PredictionEngine()
    graph = FinancialKnowledgeGraph()
    tickers = ["AAPL", "MSFT", "JPM", "XOM"]
    report = engine.compute_systemic_risk(graph, tickers)
    assert 0 <= report.overall_score <= 100
    assert report.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(report.node_scores) == len(tickers)


def test_prediction_to_dict():
    from src.prediction.predictor import PredictionEngine
    engine = PredictionEngine()
    pred = engine.predict_stock("GS", sentiment=0.1)
    d = pred.to_dict()
    assert "ticker" in d
    assert "direction" in d
    assert "up_prob" in d
    assert "down_prob" in d
    assert "neutral_prob" in d
    assert "confidence" in d


def test_risk_report_to_dict():
    from src.prediction.predictor import PredictionEngine
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    engine = PredictionEngine()
    graph = FinancialKnowledgeGraph()
    report = engine.compute_systemic_risk(graph, ["AAPL", "MSFT"])
    d = report.to_dict()
    assert "overall_score" in d
    assert "risk_level" in d
    assert "node_scores" in d
