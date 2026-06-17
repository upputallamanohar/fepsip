"""
FEPSIP Anomaly Detection
Detects flash crashes, manipulation, and contagion spikes
using Isolation Forest and statistical methods.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils import logger


class AnomalyType(str, Enum):
    FLASH_CRASH = "flash_crash"
    VOLUME_SPIKE = "volume_spike"
    SENTIMENT_SURGE = "sentiment_surge"
    PRICE_MANIPULATION = "price_manipulation"
    CONTAGION_SPIKE = "contagion_spike"
    VOLATILITY_EXPLOSION = "volatility_explosion"


@dataclass
class AnomalyAlert:
    anomaly_type: AnomalyType
    ticker: str
    timestamp: datetime
    severity: float           # 0.0 – 1.0
    description: str
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type.value,
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "severity": round(self.severity, 4),
            "description": self.description,
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
        }


class IsolationForestDetector:
    """Isolation Forest for multivariate price anomaly detection."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100) -> None:
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats["returns"] = df["close"].pct_change()
        feats["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        feats["volatility"] = feats["returns"].rolling(5).std()
        feats["price_range"] = (df["high"] - df["low"]) / df["close"]
        feats["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        return feats.dropna()

    def fit(self, price_df: pd.DataFrame) -> "IsolationForestDetector":
        feats = self._build_features(price_df)
        X = self.scaler.fit_transform(feats.values)
        self.model.fit(X)
        self._fitted = True
        logger.info("IsolationForest fitted on {} samples", len(X))
        return self

    def detect(self, price_df: pd.DataFrame, ticker: str = "UNKNOWN") -> list[AnomalyAlert]:
        if not self._fitted:
            self.fit(price_df)

        feats = self._build_features(price_df)
        X = self.scaler.transform(feats.values)
        scores = self.model.decision_function(X)
        labels = self.model.predict(X)  # -1 = anomaly

        alerts = []
        for i, (ts, row) in enumerate(feats.iterrows()):
            if labels[i] == -1:
                severity = min((-scores[i] - 0.1) * 2, 1.0)
                feats_dict = row.to_dict()

                # Classify anomaly type
                if abs(row.get("returns", 0)) > 0.05:
                    atype = AnomalyType.FLASH_CRASH
                    desc = f"Abnormal price move: {row.get('returns', 0):.2%}"
                elif row.get("volume_ratio", 0) > 3:
                    atype = AnomalyType.VOLUME_SPIKE
                    desc = f"Volume {row.get('volume_ratio', 0):.1f}x above average"
                elif row.get("volatility", 0) > 0.04:
                    atype = AnomalyType.VOLATILITY_EXPLOSION
                    desc = f"Volatility explosion: {row.get('volatility', 0):.4f}"
                else:
                    atype = AnomalyType.PRICE_MANIPULATION
                    desc = "Unusual multi-feature deviation"

                alerts.append(AnomalyAlert(
                    anomaly_type=atype,
                    ticker=ticker,
                    timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                    severity=float(severity),
                    description=desc,
                    features=feats_dict,
                ))

        logger.info("Detected {} anomalies for {}", len(alerts), ticker)
        return alerts


class StatisticalAnomalyDetector:
    """Z-score and rolling-window statistical anomaly detection."""

    def __init__(self, z_threshold: float = 3.0, window: int = 30) -> None:
        self.z_threshold = z_threshold
        self.window = window

    def detect_price_anomalies(self, returns: pd.Series, ticker: str = "UNKNOWN") -> list[AnomalyAlert]:
        rolling_mean = returns.rolling(self.window).mean()
        rolling_std = returns.rolling(self.window).std()
        z_scores = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        alerts = []
        anomaly_mask = z_scores.abs() > self.z_threshold
        for ts, z in z_scores[anomaly_mask].items():
            severity = min(abs(float(z)) / 6.0, 1.0)
            ret = returns.get(ts, 0.0)
            alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.FLASH_CRASH if abs(float(ret)) > 0.05 else AnomalyType.VOLATILITY_EXPLOSION,
                ticker=ticker,
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                severity=severity,
                description=f"Z-score={float(z):.2f}, return={float(ret):.2%}",
                features={"z_score": float(z), "return": float(ret)},
            ))
        return alerts

    def detect_sentiment_anomalies(
        self, sentiment_series: pd.Series, ticker: str = "UNKNOWN"
    ) -> list[AnomalyAlert]:
        rolling_mean = sentiment_series.rolling(self.window).mean()
        rolling_std = sentiment_series.rolling(self.window).std()
        z_scores = (sentiment_series - rolling_mean) / rolling_std.replace(0, np.nan)
        alerts = []
        for ts, z in z_scores[z_scores.abs() > self.z_threshold].items():
            alerts.append(AnomalyAlert(
                anomaly_type=AnomalyType.SENTIMENT_SURGE,
                ticker=ticker,
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                severity=min(abs(float(z)) / 5.0, 1.0),
                description=f"Sentiment Z-score={float(z):.2f}",
                features={"z_score": float(z)},
            ))
        return alerts


class AnomalyDetectionEngine:
    """Unified anomaly detection orchestrator."""

    def __init__(self) -> None:
        self.if_detectors: dict[str, IsolationForestDetector] = {}
        self.stat_detector = StatisticalAnomalyDetector()

    def fit_ticker(self, ticker: str, price_df: pd.DataFrame) -> None:
        detector = IsolationForestDetector()
        detector.fit(price_df)
        self.if_detectors[ticker] = detector

    def detect_all(self, ticker: str, price_df: pd.DataFrame) -> list[AnomalyAlert]:
        alerts = []
        if ticker in self.if_detectors:
            alerts.extend(self.if_detectors[ticker].detect(price_df, ticker))
        returns = price_df["close"].pct_change().dropna()
        alerts.extend(self.stat_detector.detect_price_anomalies(returns, ticker))
        # Deduplicate by timestamp
        seen = set()
        unique = []
        for a in alerts:
            key = (a.ticker, a.timestamp, a.anomaly_type)
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return sorted(unique, key=lambda x: x.severity, reverse=True)
