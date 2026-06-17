"""Unit tests for Anomaly Detection."""
import pytest
import pandas as pd
import numpy as np


def _make_ohlcv(n=200, seed=42, inject_anomaly=True):
    np.random.seed(seed)
    close = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
    if inject_anomaly:
        close[100] *= 0.85  # flash crash
        close[150] *= 1.12  # spike
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.003),
        "high": close * (1 + abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
    })
    if inject_anomaly:
        df.loc[100, "volume"] *= 10  # volume spike
    return df


def test_isolation_forest_fits():
    from src.anomaly.anomaly_detector import IsolationForestDetector
    df = _make_ohlcv()
    det = IsolationForestDetector()
    det.fit(df)
    assert det._fitted


def test_isolation_forest_detects():
    from src.anomaly.anomaly_detector import IsolationForestDetector
    df = _make_ohlcv(inject_anomaly=True)
    det = IsolationForestDetector(contamination=0.1)
    alerts = det.detect(df, "TEST")
    assert len(alerts) > 0


def test_statistical_detector():
    from src.anomaly.anomaly_detector import StatisticalAnomalyDetector
    np.random.seed(0)
    returns = pd.Series(np.random.randn(200) * 0.01)
    returns.iloc[100] = 0.15   # inject outlier
    det = StatisticalAnomalyDetector(z_threshold=3.0)
    alerts = det.detect_price_anomalies(returns, "TEST")
    assert len(alerts) >= 1


def test_engine_fit_detect():
    from src.anomaly.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    df = _make_ohlcv(inject_anomaly=True)
    engine.fit_ticker("TEST", df)
    alerts = engine.detect_all("TEST", df)
    assert isinstance(alerts, list)


def test_anomaly_to_dict():
    from src.anomaly.anomaly_detector import StatisticalAnomalyDetector
    np.random.seed(1)
    returns = pd.Series(np.random.randn(100) * 0.01)
    returns.iloc[50] = 0.20
    det = StatisticalAnomalyDetector()
    alerts = det.detect_price_anomalies(returns, "AAPL")
    if alerts:
        d = alerts[0].to_dict()
        assert "anomaly_type" in d
        assert "ticker" in d
        assert "severity" in d


def test_alert_severity_bounded():
    from src.anomaly.anomaly_detector import StatisticalAnomalyDetector
    np.random.seed(2)
    returns = pd.Series(np.concatenate([np.random.randn(50)*0.01, [0.5], np.random.randn(49)*0.01]))
    det = StatisticalAnomalyDetector()
    alerts = det.detect_price_anomalies(returns, "MSFT")
    for a in alerts:
        assert 0.0 <= a.severity <= 1.0
