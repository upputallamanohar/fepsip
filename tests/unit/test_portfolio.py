"""Unit tests for Portfolio Manager."""
import pytest
import pandas as pd
import numpy as np


def _make_returns(tickers, n=252, seed=42):
    np.random.seed(seed)
    return pd.DataFrame(np.random.randn(n, len(tickers)) * 0.015, columns=tickers)


def test_optimizer_returns_valid_weights():
    from src.portfolio.portfolio_manager import MeanVarianceOptimizer
    tickers = ["AAPL", "MSFT", "JPM", "XOM"]
    returns = _make_returns(tickers)
    opt = MeanVarianceOptimizer()
    weights = opt.optimize(returns)
    assert len(weights) == len(tickers)
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-3
    for w in weights.values():
        assert 0.0 <= w <= 1.0  # fallback may be equal weight


def test_weights_sum_to_one():
    from src.portfolio.portfolio_manager import MeanVarianceOptimizer
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    returns = _make_returns(tickers, n=300)
    opt = MeanVarianceOptimizer(max_weight=0.30)
    weights = opt.optimize(returns)
    assert abs(sum(weights.values()) - 1.0) < 1e-3


def test_portfolio_stats_computed():
    from src.portfolio.portfolio_manager import MeanVarianceOptimizer
    tickers = ["AAPL", "MSFT", "JPM"]
    returns = _make_returns(tickers)
    opt = MeanVarianceOptimizer()
    weights = {t: 1/3 for t in tickers}
    ret, vol, sharpe = opt.compute_portfolio_stats(weights, returns)
    assert isinstance(ret, float)
    assert isinstance(vol, float)
    assert vol > 0


def test_max_drawdown_computed():
    from src.portfolio.portfolio_manager import MeanVarianceOptimizer
    tickers = ["AAPL", "MSFT"]
    returns = _make_returns(tickers)
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    dd = MeanVarianceOptimizer.compute_max_drawdown(weights, returns)
    assert dd <= 0


def test_portfolio_manager_optimize():
    from src.portfolio.portfolio_manager import PortfolioManager
    tickers = ["AAPL", "MSFT", "JPM", "XOM"]
    returns = _make_returns(tickers)
    pm = PortfolioManager(tickers, initial_capital=500_000)
    state = pm.optimize(returns)
    assert state.sharpe_ratio is not None
    assert state.total_value == 500_000
    assert len(state.weights) > 0


def test_trade_actions_generated():
    from src.portfolio.portfolio_manager import PortfolioManager
    tickers = ["AAPL", "MSFT", "JPM"]
    returns = _make_returns(tickers)
    pm = PortfolioManager(tickers)
    state = pm.optimize(returns)
    actions = pm.generate_trade_actions(state.weights)
    assert len(actions) == len(tickers)
    for a in actions:
        assert a.action in ["BUY", "SELL", "HOLD"]
        assert a.urgency in ["LOW", "MEDIUM", "HIGH"]


def test_portfolio_state_to_dict():
    from src.portfolio.portfolio_manager import PortfolioManager
    tickers = ["AAPL", "MSFT"]
    returns = _make_returns(tickers)
    pm = PortfolioManager(tickers)
    state = pm.optimize(returns)
    d = state.to_dict()
    assert "weights" in d
    assert "sharpe_ratio" in d
    assert "expected_return" in d


def test_min_variance_objective():
    from src.portfolio.portfolio_manager import MeanVarianceOptimizer
    tickers = ["AAPL", "MSFT", "XOM"]
    returns = _make_returns(tickers, n=200)
    opt = MeanVarianceOptimizer()
    weights = opt.optimize(returns, objective="min_variance")
    assert abs(sum(weights.values()) - 1.0) < 1e-3
