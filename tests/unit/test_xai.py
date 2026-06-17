"""Unit tests for XAI Explainer."""
import pytest


def test_explain_prediction_up():
    from src.xai.explainer import FeatureImportanceExplainer
    exp = FeatureImportanceExplainer()
    expl = exp.explain_prediction(
        ticker="AAPL",
        prediction="UP",
        confidence=0.75,
        features={"sentiment": 0.6, "momentum": 0.4, "volatility": -0.2, "sector_weakness": -0.1},
    )
    assert expl.ticker == "AAPL"
    assert expl.prediction == "UP"
    assert len(expl.top_factors) > 0
    assert len(expl.narrative) > 0


def test_explain_prediction_down():
    from src.xai.explainer import FeatureImportanceExplainer
    exp = FeatureImportanceExplainer()
    expl = exp.explain_prediction(
        ticker="TSLA",
        prediction="DOWN",
        confidence=0.68,
        features={"sentiment": -0.8, "event_severity": 0.9, "supply_chain_risk": 0.7},
    )
    assert "DOWN" in expl.narrative or "bearish" in expl.narrative.lower()


def test_feature_contributions_sum_to_one():
    from src.xai.explainer import FeatureImportanceExplainer
    exp = FeatureImportanceExplainer()
    expl = exp.explain_prediction(
        ticker="MSFT",
        prediction="NEUTRAL",
        confidence=0.45,
        features={"sentiment": 0.1, "momentum": 0.3, "macro_factor": 0.2},
    )
    total = sum(expl.feature_contributions.values())
    assert abs(total - 1.0) < 1e-4


def test_explain_systemic_risk():
    from src.xai.explainer import FeatureImportanceExplainer
    exp = FeatureImportanceExplainer()
    result = exp.explain_systemic_risk(
        ticker="JPM",
        risk_score=72.5,
        graph_centrality=0.6,
        volatility=0.3,
        sector_avg_risk=60.0,
        macro_stress=0.4,
    )
    assert result["ticker"] == "JPM"
    assert "breakdown" in result
    assert "narrative" in result
    assert result["risk_level"] == "HIGH"


def test_explanation_to_dict():
    from src.xai.explainer import FeatureImportanceExplainer
    exp = FeatureImportanceExplainer()
    expl = exp.explain_prediction("GS", "UP", 0.8, {"sentiment": 0.5, "earnings_surprise": 0.8})
    d = expl.to_dict()
    assert "ticker" in d
    assert "prediction" in d
    assert "feature_contributions" in d
    assert "top_factors" in d
    assert "narrative" in d
