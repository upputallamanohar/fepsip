"""
FEPSIP - Core Data Models & Schemas
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Regime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CRISIS = "CRISIS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    RECOVERY = "RECOVERY"
    SIDEWAYS = "SIDEWAYS"


class EventType(str, Enum):
    EARNINGS = "EARNINGS"
    MA = "M&A"
    BANKRUPTCY = "BANKRUPTCY"
    SUPPLY_CHAIN_DISRUPTION = "SUPPLY_CHAIN_DISRUPTION"
    PRODUCT_RECALL = "PRODUCT_RECALL"
    REGULATORY_ACTION = "REGULATORY_ACTION"
    INTEREST_RATE_DECISION = "INTEREST_RATE_DECISION"
    COMMODITY_SHOCK = "COMMODITY_SHOCK"
    GEOPOLITICAL_EVENT = "GEOPOLITICAL_EVENT"
    EARNINGS_BEAT = "EARNINGS_BEAT"
    EARNINGS_MISS = "EARNINGS_MISS"
    FACTORY_SHUTDOWN = "FACTORY_SHUTDOWN"
    LEADERSHIP_CHANGE = "LEADERSHIP_CHANGE"
    CYBERSECURITY = "CYBERSECURITY"
    NATURAL_DISASTER = "NATURAL_DISASTER"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ─── Market Data ──────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: str


class MarketSnapshot(BaseModel):
    ticker: str
    timestamp: datetime
    price: float
    change_pct: float
    volume: float
    volatility: float
    sentiment_score: float = 0.0
    risk_score: float = 0.0


# ─── Events ───────────────────────────────────────────────────────────────────

class FinancialEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{datetime.utcnow().timestamp():.0f}")
    event_type: EventType = EventType.UNKNOWN
    company: Optional[str] = None
    ticker: Optional[str] = None
    sector: Optional[str] = None
    severity: float = Field(0.5, ge=0.0, le=1.0)
    region: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    headline: Optional[str] = None
    description: Optional[str] = None
    affected_entities: list[str] = Field(default_factory=list)
    sentiment_score: float = 0.0
    confidence: float = 0.5
    source: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NewsArticle(BaseModel):
    title: str
    summary: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[datetime] = None
    ticker: Optional[str] = None
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    event: Optional[FinancialEvent] = None


# ─── Predictions ──────────────────────────────────────────────────────────────

class StockPrediction(BaseModel):
    ticker: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    direction: Direction
    probabilities: dict[str, float]  # {"UP": 0.68, "NEUTRAL": 0.20, "DOWN": 0.12}
    confidence: float
    horizon_days: int = 5
    explanation: Optional[dict[str, Any]] = None


class RippleEffect(BaseModel):
    source_ticker: str
    source_event: Optional[str] = None
    affected_ticker: str
    affected_sector: Optional[str] = None
    impact_magnitude: float  # % change expected
    time_to_impact_days: float
    propagation_path: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class SystemicRiskScore(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_score: float = Field(0.0, ge=0.0, le=100.0)
    level: RiskLevel = RiskLevel.LOW
    sector_scores: dict[str, float] = Field(default_factory=dict)
    ticker_scores: dict[str, float] = Field(default_factory=dict)
    top_risks: list[str] = Field(default_factory=list)
    contagion_probability: float = 0.0
    regime: Regime = Regime.SIDEWAYS


class PropagationPath(BaseModel):
    source: str
    target: str
    path: list[str]
    total_impact: float
    hop_count: int
    edge_types: list[str] = Field(default_factory=list)


# ─── Portfolio ────────────────────────────────────────────────────────────────

class PortfolioPosition(BaseModel):
    ticker: str
    weight: float
    shares: float = 0.0
    current_price: float = 0.0
    risk_exposure: float = 0.0


class Portfolio(BaseModel):
    name: str = "Default Portfolio"
    positions: list[PortfolioPosition] = Field(default_factory=list)
    cash: float = 0.0
    total_value: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    risk_score: float = 0.0


# ─── Scenarios ────────────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    scenario_description: str
    affected_ticker: Optional[str] = None
    affected_sector: Optional[str] = None
    magnitude: float = 0.1  # 10% default shock
    direction: str = "negative"
    simulate_portfolio: bool = False
    portfolio: Optional[Portfolio] = None


class ScenarioResult(BaseModel):
    scenario: ScenarioRequest
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ripple_effects: list[RippleEffect] = Field(default_factory=list)
    systemic_risk_delta: float = 0.0
    regime_change_probability: float = 0.0
    top_affected_sectors: list[dict[str, Any]] = Field(default_factory=list)
    propagation_paths: list[PropagationPath] = Field(default_factory=list)
    portfolio_impact: Optional[dict[str, float]] = None


# ─── Anomalies ────────────────────────────────────────────────────────────────

class Anomaly(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: f"anm_{datetime.utcnow().timestamp():.0f}")
    ticker: Optional[str] = None
    anomaly_type: str
    score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: str
    features: dict[str, float] = Field(default_factory=dict)


# ─── Graph ────────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    node_id: str
    node_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)
