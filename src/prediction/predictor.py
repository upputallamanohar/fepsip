"""
FEPSIP Prediction Engine
Stock-level, sector-level, and systemic risk predictions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from src.utils import logger

# Torch optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available — using heuristic prediction mode")


@dataclass
class StockPrediction:
    ticker: str
    direction: str
    up_prob: float
    down_prob: float
    neutral_prob: float
    expected_return: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    explanation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "up_prob": round(self.up_prob, 4),
            "down_prob": round(self.down_prob, 4),
            "neutral_prob": round(self.neutral_prob, 4),
            "expected_return": round(self.expected_return, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "explanation": self.explanation,
        }


@dataclass
class RippleEffect:
    source_ticker: str
    source_event: str
    affected: list[dict]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "source_ticker": self.source_ticker,
            "source_event": self.source_event,
            "affected": self.affected,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SystemicRiskReport:
    overall_score: float
    risk_level: str
    node_scores: dict[str, float]
    contagion_paths: dict[str, list]
    top_risks: list[dict]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 2),
            "risk_level": self.risk_level,
            "node_scores": {k: round(v, 2) for k, v in self.node_scores.items()},
            "top_risks": self.top_risks,
            "timestamp": self.timestamp.isoformat(),
        }


class PredictionEngine:
    """Orchestrates stock, ripple-effect, and systemic risk predictions."""

    DIRECTION_LABELS = ["UP", "NEUTRAL", "DOWN"]

    def __init__(self, fused_dim: int = 256) -> None:
        self._model = None
        if TORCH_AVAILABLE:
            try:
                import torch.nn as nn
                self._model = nn.Linear(fused_dim, 3)
                self._model.eval()
            except Exception:
                pass

    def predict_stock(
        self,
        ticker: str,
        fused_embedding: Optional[np.ndarray] = None,
        price_series: Optional[pd.Series] = None,
        sentiment: float = 0.0,
        event_severity: float = 0.0,
    ) -> StockPrediction:
        if fused_embedding is not None and TORCH_AVAILABLE and self._model is not None:
            import torch
            with torch.no_grad():
                t = torch.tensor(fused_embedding, dtype=torch.float32).unsqueeze(0)
                logits = self._model(t)
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            up_p, neutral_p, down_p = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            momentum = 0.0
            if price_series is not None and len(price_series) >= 5:
                recent = price_series.pct_change().tail(5).mean()
                momentum = float(recent) if not np.isnan(recent) else 0.0

            base_up = 0.33 + 0.20 * sentiment + 0.15 * momentum
            base_down = 0.33 - 0.15 * sentiment + 0.10 * event_severity
            base_up = max(0.05, min(0.90, base_up))
            base_down = max(0.05, min(0.90, base_down))
            base_neutral = max(0.05, 1.0 - base_up - base_down)
            total = base_up + base_down + base_neutral
            up_p, down_p, neutral_p = base_up / total, base_down / total, base_neutral / total

        direction = self.DIRECTION_LABELS[int(np.argmax([up_p, neutral_p, down_p]))]
        expected_return = up_p * 0.05 - down_p * 0.05
        confidence = float(max(up_p, down_p, neutral_p))

        return StockPrediction(
            ticker=ticker,
            direction=direction,
            up_prob=up_p,
            down_prob=down_p,
            neutral_prob=neutral_p,
            expected_return=expected_return,
            confidence=confidence,
            explanation={
                "sentiment_contribution": round(sentiment * 0.25, 3),
                "event_severity": round(event_severity, 3),
            },
        )

    def predict_ripple_effects(
        self, source_ticker: str, event_description: str,
        event_severity: float, graph, contagion_depth: int = 3,
    ) -> RippleEffect:
        paths = graph.get_contagion_path(source_ticker, depth=contagion_depth)
        affected = []
        for node_id, path in paths.items():
            if node_id == source_ticker:
                continue
            depth = len(path) - 1
            decay = 0.6 ** depth
            node_data = graph.G.nodes.get(node_id, {})
            impact_pct = float(event_severity * decay * 0.08)
            affected.append({
                "ticker": node_id,
                "sector": node_data.get("sector", "Unknown"),
                "node_type": node_data.get("node_type", "Unknown"),
                "impact_pct": round(-impact_pct * 100, 2),
                "time_to_impact_days": depth,
                "path": " → ".join(path),
            })
        affected.sort(key=lambda x: abs(x["impact_pct"]), reverse=True)
        return RippleEffect(source_ticker=source_ticker, source_event=event_description, affected=affected[:20])

    def compute_systemic_risk(
        self, graph, tickers: list[str], market_volatility: float = 0.2,
    ) -> SystemicRiskReport:
        centrality = graph.compute_centrality_scores()
        node_scores: dict[str, float] = {}

        for ticker in tickers:
            pagerank = centrality["pagerank"].get(ticker, 0.0)
            degree = centrality["degree"].get(ticker, 0.0)
            node_data = graph.G.nodes.get(ticker, {})
            volatility = min(node_data.get("volatility", market_volatility) or market_volatility, 1.0)
            raw = (0.4 * pagerank + 0.3 * degree + 0.3 * volatility) * 100
            node_scores[ticker] = round(min(raw * 3, 100), 2)

        overall = float(np.mean(list(node_scores.values()))) if node_scores else 0.0
        overall = round(min(overall * (1 + market_volatility), 100), 2)

        if overall >= 80: risk_level = "CRITICAL"
        elif overall >= 60: risk_level = "HIGH"
        elif overall >= 30: risk_level = "MEDIUM"
        else: risk_level = "LOW"

        top_risks = sorted(
            [{"ticker": t, "score": s} for t, s in node_scores.items()],
            key=lambda x: x["score"], reverse=True
        )[:10]

        contagion_paths = {}
        for item in top_risks[:3]:
            t = item["ticker"]
            contagion_paths[t] = graph.get_contagion_path(t, depth=2)

        return SystemicRiskReport(
            overall_score=overall, risk_level=risk_level,
            node_scores=node_scores, contagion_paths=contagion_paths, top_risks=top_risks,
        )
