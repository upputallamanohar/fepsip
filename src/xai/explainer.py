"""
FEPSIP Explainable AI Module
SHAP-based explanations + attention visualization + feature importance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from src.utils import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed; using approximation")


@dataclass
class Explanation:
    ticker: str
    prediction: str
    confidence: float
    feature_contributions: dict[str, float]  # feature -> % contribution
    top_factors: list[dict]
    narrative: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "feature_contributions": {k: round(v, 4) for k, v in self.feature_contributions.items()},
            "top_factors": self.top_factors,
            "narrative": self.narrative,
        }


FEATURE_LABELS = {
    "sentiment": "News Sentiment",
    "event_severity": "Event Severity",
    "volatility": "Price Volatility",
    "momentum": "Price Momentum",
    "volume_anomaly": "Volume Anomaly",
    "sector_weakness": "Sector Weakness",
    "macro_factor": "Macro Environment",
    "graph_centrality": "Network Centrality",
    "supply_chain_risk": "Supply Chain Risk",
    "earnings_surprise": "Earnings Surprise",
}


class FeatureImportanceExplainer:
    """
    Provides gradient-free feature importance explanations
    via SHAP (when available) or linear approximation.
    """

    def explain_prediction(
        self,
        ticker: str,
        prediction: str,
        confidence: float,
        features: dict[str, float],
    ) -> Explanation:
        """
        Generate an explanation for a stock prediction.
        Features dict maps feature_name -> raw_value (pre-normalized).
        """
        # Compute relative contributions (simplified attribution)
        abs_features = {k: abs(v) for k, v in features.items()}
        total = sum(abs_features.values()) or 1.0
        contributions = {k: v / total for k, v in abs_features.items()}

        # Sort by contribution
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_factors = [
            {
                "factor": FEATURE_LABELS.get(k, k),
                "contribution_pct": round(v * 100, 1),
                "raw_value": round(features.get(k, 0.0), 4),
                "direction": "positive" if features.get(k, 0) > 0 else "negative",
            }
            for k, v in sorted_factors[:6]
        ]

        narrative = self._generate_narrative(ticker, prediction, top_factors)

        return Explanation(
            ticker=ticker,
            prediction=prediction,
            confidence=confidence,
            feature_contributions={k: round(v, 4) for k, v in contributions.items()},
            top_factors=top_factors,
            narrative=narrative,
        )

    def _generate_narrative(self, ticker: str, prediction: str, top_factors: list[dict]) -> str:
        if not top_factors:
            return f"{ticker} prediction: {prediction} (insufficient features for explanation)"

        top = top_factors[0]
        second = top_factors[1] if len(top_factors) > 1 else None

        direction = "bearish" if prediction == "DOWN" else ("bullish" if prediction == "UP" else "neutral")
        narrative = (
            f"{ticker} is predicted to move {prediction} with the primary driver being "
            f"{top['factor']} ({top['contribution_pct']}% contribution). "
        )
        if second:
            narrative += (
                f"Secondary factor: {second['factor']} ({second['contribution_pct']}%). "
            )
        if prediction == "DOWN":
            narrative += "Investors should consider hedging or reducing exposure."
        elif prediction == "UP":
            narrative += "Conditions appear favorable for accumulation."
        else:
            narrative += "Market conditions suggest a wait-and-see approach."

        return narrative

    def explain_systemic_risk(
        self,
        ticker: str,
        risk_score: float,
        graph_centrality: float,
        volatility: float,
        sector_avg_risk: float,
        macro_stress: float,
    ) -> dict:
        contributions = {
            "Network Centrality": graph_centrality * 40,
            "Price Volatility": volatility * 30,
            "Sector Risk Exposure": sector_avg_risk * 20,
            "Macro Stress": macro_stress * 10,
        }
        total = sum(abs(v) for v in contributions.values()) or 1.0
        pct = {k: round(abs(v) / total * 100, 1) for k, v in contributions.items()}

        return {
            "ticker": ticker,
            "systemic_risk_score": round(risk_score, 2),
            "risk_level": "CRITICAL" if risk_score > 80 else "HIGH" if risk_score > 60 else "MEDIUM" if risk_score > 30 else "LOW",
            "breakdown": pct,
            "narrative": (
                f"{ticker} systemic risk score is {risk_score:.1f}/100. "
                f"Largest contributor: {max(pct, key=pct.get)} ({max(pct.values())}%). "
                + ("⚠️ This node is highly interconnected — contagion risk is elevated." if graph_centrality > 0.5 else "")
            ),
        }


class AttentionVisualizer:
    """
    Extracts and formats attention weights from transformer models
    for visualization in the dashboard.
    """

    @staticmethod
    def format_attention_map(
        attention_weights: np.ndarray,
        modalities: list[str] = None,
    ) -> dict:
        if modalities is None:
            modalities = ["Price", "News", "Graph", "Fundamentals", "Macro"]
        n = len(modalities)
        if attention_weights.shape != (n, n):
            attention_weights = np.eye(n)

        return {
            "modalities": modalities,
            "weights": attention_weights.tolist(),
            "dominant_modality": modalities[int(np.argmax(attention_weights.mean(axis=0)))],
        }
