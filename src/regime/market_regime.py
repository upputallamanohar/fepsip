"""
FEPSIP Market Regime Detection
Detects Bull/Bear/Crisis/High-Volatility/Recovery regimes
using Hidden Markov Models and clustering.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from src.utils import logger

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available; using GMM fallback for regime detection")


class MarketRegime(str, Enum):
    BULL = "Bull Market"
    BEAR = "Bear Market"
    CRISIS = "Crisis"
    HIGH_VOLATILITY = "High Volatility"
    RECOVERY = "Recovery"
    NEUTRAL = "Neutral"


REGIME_COLORS = {
    MarketRegime.BULL: "#00C853",
    MarketRegime.BEAR: "#D50000",
    MarketRegime.CRISIS: "#AA00FF",
    MarketRegime.HIGH_VOLATILITY: "#FF6D00",
    MarketRegime.RECOVERY: "#0091EA",
    MarketRegime.NEUTRAL: "#9E9E9E",
}


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    start_date: Optional[datetime]
    features: dict
    risk_multiplier: float

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "risk_multiplier": round(self.risk_multiplier, 3),
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
            "color": REGIME_COLORS.get(self.regime, "#9E9E9E"),
        }


def _build_features(returns: pd.Series, vix: Optional[pd.Series] = None) -> pd.DataFrame:
    """Build feature matrix for regime detection."""
    df = pd.DataFrame(index=returns.index)
    df["returns"] = returns
    df["volatility_21d"] = returns.rolling(21).std() * np.sqrt(252)
    df["volatility_5d"] = returns.rolling(5).std() * np.sqrt(252)
    df["momentum_21d"] = returns.rolling(21).mean() * 252
    df["momentum_63d"] = returns.rolling(63).mean() * 252
    df["drawdown"] = (returns + 1).cumprod() / (returns + 1).cumprod().cummax() - 1
    if vix is not None:
        df["vix"] = vix.reindex(returns.index, method="ffill")
    return df.dropna()


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector.
    States are mapped to financial regime labels post-hoc
    based on feature statistics.
    """

    N_STATES = 5
    FEATURE_COLS = ["returns", "volatility_21d", "momentum_21d", "drawdown"]

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model: Optional[hmm.GaussianHMM] = None
        self._state_to_regime: dict[int, MarketRegime] = {}

    def fit(self, returns: pd.Series, vix: Optional[pd.Series] = None) -> "HMMRegimeDetector":
        if not HMM_AVAILABLE:
            raise RuntimeError("hmmlearn not installed")

        features = _build_features(returns, vix)[self.FEATURE_COLS]
        X = self.scaler.fit_transform(features.values)

        self.model = hmm.GaussianHMM(
            n_components=self.N_STATES, covariance_type="full",
            n_iter=100, random_state=42
        )
        self.model.fit(X)
        states = self.model.predict(X)

        # Map states to regimes by feature statistics
        state_means = {}
        for s in range(self.N_STATES):
            mask = states == s
            if mask.sum() == 0:
                continue
            state_means[s] = {
                "returns": features["returns"].values[mask].mean(),
                "volatility": features["volatility_21d"].values[mask].mean(),
                "drawdown": features["drawdown"].values[mask].mean(),
            }

        self._state_to_regime = self._assign_regimes(state_means)
        logger.info("HMM regime detector fitted with {} states", self.N_STATES)
        return self

    def _assign_regimes(self, state_means: dict) -> dict[int, MarketRegime]:
        mapping = {}
        if not state_means:
            return mapping

        returns_vals = {s: m["returns"] for s, m in state_means.items()}
        vol_vals = {s: m["volatility"] for s, m in state_means.items()}
        dd_vals = {s: m["drawdown"] for s, m in state_means.items()}

        sorted_by_return = sorted(returns_vals, key=lambda s: returns_vals[s], reverse=True)
        sorted_by_vol = sorted(vol_vals, key=lambda s: vol_vals[s], reverse=True)

        for i, s in enumerate(sorted_by_return):
            vol = vol_vals[s]
            ret = returns_vals[s]
            dd = dd_vals[s]
            if dd < -0.15 and vol > 0.3:
                mapping[s] = MarketRegime.CRISIS
            elif vol > 0.25:
                mapping[s] = MarketRegime.HIGH_VOLATILITY
            elif ret > 0.1:
                mapping[s] = MarketRegime.BULL
            elif ret < -0.05:
                mapping[s] = MarketRegime.BEAR
            else:
                mapping[s] = MarketRegime.NEUTRAL if i < len(sorted_by_return) - 1 else MarketRegime.RECOVERY

        return mapping

    def predict(self, returns: pd.Series, vix: Optional[pd.Series] = None) -> list[MarketRegime]:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        features = _build_features(returns, vix)[self.FEATURE_COLS]
        X = self.scaler.transform(features.values)
        states = self.model.predict(X)
        return [self._state_to_regime.get(s, MarketRegime.NEUTRAL) for s in states]

    def predict_current(self, returns: pd.Series, vix: Optional[pd.Series] = None) -> RegimeState:
        regimes = self.predict(returns, vix)
        current = regimes[-1] if regimes else MarketRegime.NEUTRAL
        features = _build_features(returns, vix)
        last_feats = features.iloc[-1].to_dict()

        risk_multipliers = {
            MarketRegime.BULL: 0.8,
            MarketRegime.NEUTRAL: 1.0,
            MarketRegime.RECOVERY: 1.1,
            MarketRegime.BEAR: 1.5,
            MarketRegime.HIGH_VOLATILITY: 1.8,
            MarketRegime.CRISIS: 2.5,
        }

        # Confidence from posterior probabilities
        feat_last = _build_features(returns, vix)[self.FEATURE_COLS].values[-1:]
        X_last = self.scaler.transform(feat_last)
        posteriors = self.model.predict_proba(X_last)[0]
        state_idx = self.model.predict(X_last)[0]
        confidence = float(posteriors[state_idx])

        return RegimeState(
            regime=current,
            confidence=confidence,
            start_date=datetime.now(),
            features=last_feats,
            risk_multiplier=risk_multipliers.get(current, 1.0),
        )


class GMMRegimeDetector:
    """GMM-based fallback regime detector."""

    N_COMPONENTS = 4
    FEATURE_COLS = ["returns", "volatility_21d", "momentum_21d"]

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=self.N_COMPONENTS, random_state=42)
        self._fitted = False

    def fit(self, returns: pd.Series) -> "GMMRegimeDetector":
        feats = _build_features(returns)[self.FEATURE_COLS]
        X = self.scaler.fit_transform(feats.values)
        self.gmm.fit(X)
        self._fitted = True
        logger.info("GMM regime detector fitted")
        return self

    def predict_current(self, returns: pd.Series) -> RegimeState:
        if not self._fitted:
            self.fit(returns)
        feats = _build_features(returns)[self.FEATURE_COLS]
        X = self.scaler.transform(feats.values)
        label = self.gmm.predict(X)[-1]
        proba = self.gmm.predict_proba(X[-1:]).max()

        regime_map = {0: MarketRegime.BULL, 1: MarketRegime.NEUTRAL,
                      2: MarketRegime.BEAR, 3: MarketRegime.HIGH_VOLATILITY}
        regime = regime_map.get(int(label), MarketRegime.NEUTRAL)
        last = feats.iloc[-1].to_dict()

        return RegimeState(
            regime=regime, confidence=float(proba), start_date=datetime.now(),
            features=last, risk_multiplier={
                MarketRegime.BULL: 0.8, MarketRegime.NEUTRAL: 1.0,
                MarketRegime.BEAR: 1.5, MarketRegime.HIGH_VOLATILITY: 1.8,
            }.get(regime, 1.0),
        )


def get_regime_detector(returns: pd.Series, vix: Optional[pd.Series] = None) -> RegimeState:
    """Auto-select best available regime detector and return current regime."""
    if HMM_AVAILABLE and len(returns) >= 100:
        detector = HMMRegimeDetector()
        detector.fit(returns, vix)
        return detector.predict_current(returns, vix)
    else:
        detector = GMMRegimeDetector()
        return detector.predict_current(returns)
