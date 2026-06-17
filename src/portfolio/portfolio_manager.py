"""
FEPSIP Portfolio Manager
Risk-aware portfolio optimization using Modern Portfolio Theory
and RL-style action recommendations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils import logger


@dataclass
class PortfolioState:
    tickers: list[str]
    weights: dict[str, float]
    total_value: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    risk_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "tickers": self.tickers,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "total_value": round(self.total_value, 2),
            "expected_return": round(self.expected_return, 4),
            "volatility": round(self.volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "risk_score": round(self.risk_score, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeAction:
    ticker: str
    action: str      # BUY / SELL / HOLD
    current_weight: float
    target_weight: float
    reason: str
    urgency: str     # LOW / MEDIUM / HIGH

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "weight_delta": round(self.target_weight - self.current_weight, 4),
            "reason": self.reason,
            "urgency": self.urgency,
        }


class MeanVarianceOptimizer:
    """Markowitz mean-variance portfolio optimizer."""

    def __init__(self, risk_free_rate: float = 0.05, max_weight: float = 0.20) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight

    def optimize(
        self,
        returns_matrix: pd.DataFrame,
        objective: str = "sharpe",
        risk_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        tickers = list(returns_matrix.columns)
        n = len(tickers)
        if n == 0:
            return {}

        mu = returns_matrix.mean().values * 252
        cov = returns_matrix.cov().values * 252

        # Penalize high-risk assets
        if risk_scores:
            risk_penalty = np.array([risk_scores.get(t, 50) / 100 for t in tickers])
            mu = mu - 0.05 * risk_penalty  # reduce expected return for risky assets

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(port_ret - self.risk_free_rate) / port_vol

        def min_variance(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        obj_fn = neg_sharpe if objective == "sharpe" else min_variance

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, min(self.max_weight, 1.0))] * n
        w0 = np.ones(n) / n

        result = minimize(obj_fn, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 500})

        if result.success:
            weights = {t: float(w) for t, w in zip(tickers, result.x)}
        else:
            logger.warning("Optimization failed: {} — using equal weight", result.message)
            weights = {t: 1.0 / n for t in tickers}

        return weights

    def compute_portfolio_stats(
        self, weights: dict[str, float], returns_matrix: pd.DataFrame
    ) -> tuple[float, float, float]:
        """Returns (expected_return, volatility, sharpe_ratio)."""
        tickers = [t for t in weights if t in returns_matrix.columns]
        w = np.array([weights[t] for t in tickers])
        R = returns_matrix[tickers]
        mu = R.mean().values * 252
        cov = R.cov().values * 252
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 1e-10 else 0.0
        return ret, vol, sharpe

    @staticmethod
    def compute_max_drawdown(weights: dict[str, float], returns_matrix: pd.DataFrame) -> float:
        tickers = [t for t in weights if t in returns_matrix.columns]
        w = np.array([weights[t] for t in tickers])
        port_returns = returns_matrix[tickers].values @ w
        cumulative = (1 + port_returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / rolling_max - 1
        return float(drawdowns.min())


class PortfolioManager:
    """
    Manages portfolio state, generates trade recommendations,
    and tracks risk-adjusted performance.
    """

    def __init__(
        self,
        tickers: list[str],
        initial_capital: float = 1_000_000,
        risk_free_rate: float = 0.05,
        max_position: float = 0.20,
    ) -> None:
        self.tickers = tickers
        self.capital = initial_capital
        self.optimizer = MeanVarianceOptimizer(risk_free_rate, max_position)
        self._current_weights: dict[str, float] = {t: 1.0 / len(tickers) for t in tickers}
        self._history: list[PortfolioState] = []

    def optimize(
        self,
        returns_matrix: pd.DataFrame,
        risk_scores: Optional[dict[str, float]] = None,
        objective: str = "sharpe",
    ) -> PortfolioState:
        target_weights = self.optimizer.optimize(returns_matrix, objective, risk_scores)
        ret, vol, sharpe = self.optimizer.compute_portfolio_stats(target_weights, returns_matrix)
        max_dd = self.optimizer.compute_max_drawdown(target_weights, returns_matrix)

        avg_risk = float(np.mean(list(risk_scores.values()))) if risk_scores else 40.0

        state = PortfolioState(
            tickers=self.tickers,
            weights=target_weights,
            total_value=self.capital,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            risk_score=avg_risk,
        )
        self._current_weights = target_weights
        self._history.append(state)
        logger.info("Portfolio optimized: Sharpe={:.3f}, Vol={:.3f}, Ret={:.3f}", sharpe, vol, ret)
        return state

    def generate_trade_actions(
        self,
        target_weights: dict[str, float],
        risk_scores: Optional[dict[str, float]] = None,
        predictions: Optional[dict[str, str]] = None,
    ) -> list[TradeAction]:
        actions = []
        for ticker in self.tickers:
            current = self._current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            delta = target - current
            risk = risk_scores.get(ticker, 50) if risk_scores else 50
            direction = predictions.get(ticker, "NEUTRAL") if predictions else "NEUTRAL"

            if abs(delta) < 0.01:
                action, reason = "HOLD", "Weight within tolerance"
                urgency = "LOW"
            elif delta > 0:
                action = "BUY"
                reason = f"Optimizer suggests +{delta:.1%} allocation"
                urgency = "HIGH" if direction == "UP" else "MEDIUM"
            else:
                action = "SELL"
                reason = f"Reduce exposure by {abs(delta):.1%}"
                urgency = "HIGH" if risk > 70 or direction == "DOWN" else "MEDIUM"

            actions.append(TradeAction(
                ticker=ticker,
                action=action,
                current_weight=current,
                target_weight=target,
                reason=reason,
                urgency=urgency,
            ))

        return sorted(actions, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x.urgency])

    @property
    def current_state(self) -> dict[str, float]:
        return self._current_weights.copy()
