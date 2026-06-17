"""
FEPSIP Event Intelligence Engine
Converts raw news text into structured financial events
using rule-based NLP + optional FinBERT sentiment scoring.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
from src.utils import logger

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed – NLP scoring disabled")


class EventType(str, Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    BANKRUPTCY = "bankruptcy"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    PRODUCT_RECALL = "product_recall"
    REGULATORY_ACTION = "regulatory_action"
    INTEREST_RATE_DECISION = "interest_rate_decision"
    COMMODITY_SHOCK = "commodity_shock"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    FACTORY_SHUTDOWN = "factory_shutdown"
    LEADERSHIP_CHANGE = "leadership_change"
    CYBER_ATTACK = "cyber_attack"
    GENERAL = "general"


class EventSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FinancialEvent:
    event_type: EventType
    company: Optional[str]
    severity: float          # 0.0 – 1.0
    sentiment: float         # -1.0 to +1.0
    title: str
    summary: str
    timestamp: datetime
    region: Optional[str] = None
    affected_sectors: list[str] = field(default_factory=list)
    affected_tickers: list[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.5

    @property
    def severity_label(self) -> EventSeverity:
        if self.severity >= 0.8:
            return EventSeverity.CRITICAL
        elif self.severity >= 0.6:
            return EventSeverity.HIGH
        elif self.severity >= 0.4:
            return EventSeverity.MEDIUM
        return EventSeverity.LOW

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "company": self.company,
            "severity": self.severity,
            "severity_label": self.severity_label.value,
            "sentiment": self.sentiment,
            "title": self.title,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "region": self.region,
            "affected_sectors": self.affected_sectors,
            "affected_tickers": self.affected_tickers,
            "confidence": self.confidence,
        }


# ─────────────────────────────────────────────────────────
# Rule-based event classifier
# ─────────────────────────────────────────────────────────

EVENT_PATTERNS: list[tuple[EventType, list[str], float]] = [
    (EventType.EARNINGS, ["earnings", "revenue", "profit", "EPS", "quarterly results", "beat", "miss"], 0.6),
    (EventType.MERGER_ACQUISITION, ["acqui", "merger", "takeover", "buyout", "deal", "acquisition"], 0.75),
    (EventType.BANKRUPTCY, ["bankrupt", "chapter 11", "insolvency", "default", "liquidat"], 0.95),
    (EventType.SUPPLY_CHAIN_DISRUPTION, ["supply chain", "shortage", "disruption", "factory halt", "production stop"], 0.8),
    (EventType.PRODUCT_RECALL, ["recall", "defect", "safety concern", "withdrawn from market"], 0.7),
    (EventType.REGULATORY_ACTION, ["SEC", "fine", "regulatory", "investigation", "antitrust", "sanction", "penalty"], 0.75),
    (EventType.INTEREST_RATE_DECISION, ["interest rate", "federal reserve", "Fed", "rate hike", "rate cut", "FOMC", "basis points"], 0.85),
    (EventType.COMMODITY_SHOCK, ["oil price", "commodity", "crude", "OPEC", "gas price", "energy crisis"], 0.8),
    (EventType.GEOPOLITICAL_EVENT, ["war", "sanction", "geopolit", "invasion", "conflict", "trade war", "tariff"], 0.85),
    (EventType.FACTORY_SHUTDOWN, ["factory", "plant", "shutdown", "closure", "halt production", "facility"], 0.75),
    (EventType.LEADERSHIP_CHANGE, ["CEO", "CFO", "resign", "appointed", "new chief", "executive change"], 0.6),
    (EventType.CYBER_ATTACK, ["cyber", "hack", "breach", "ransomware", "data leak", "attack"], 0.85),
]

REGION_PATTERNS: list[tuple[str, list[str]]] = [
    ("USA", ["United States", "U.S.", "America", "Washington"]),
    ("Europe", ["Europe", "EU", "Germany", "France", "UK", "Britain"]),
    ("China", ["China", "Beijing", "Chinese"]),
    ("Asia", ["Asia", "Japan", "Korea", "India", "Southeast Asia"]),
    ("Middle East", ["Middle East", "Saudi", "Iran", "OPEC", "Gulf"]),
]

SECTOR_EXPOSURE: dict[str, list[str]] = {
    EventType.COMMODITY_SHOCK.value: ["Energy", "Industrial", "Consumer"],
    EventType.INTEREST_RATE_DECISION.value: ["Finance", "RealEstate", "Utilities"],
    EventType.SUPPLY_CHAIN_DISRUPTION.value: ["Technology", "Industrial", "Consumer"],
    EventType.MERGER_ACQUISITION.value: ["Finance"],
    EventType.REGULATORY_ACTION.value: ["Finance", "Technology"],
    EventType.GEOPOLITICAL_EVENT.value: ["Energy", "Industrial", "Finance"],
}

KNOWN_TICKERS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "Tesla": "TSLA", "NVIDIA": "NVDA", "Meta": "META",
    "JPMorgan": "JPM", "Goldman": "GS", "ExxonMobil": "XOM", "Boeing": "BA",
    "Pfizer": "PFE", "Johnson": "JNJ", "Walmart": "WMT",
}


class EventClassifier:
    """Rule-based + NLP event classification engine."""

    def __init__(self, use_finbert: bool = True) -> None:
        self.sentiment_pipe = None
        if use_finbert and TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipe = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    return_all_scores=True,
                    device=-1,  # CPU
                )
                logger.info("FinBERT sentiment pipeline loaded")
            except Exception as e:
                logger.warning("FinBERT load failed: {}", e)

    def _classify_event_type(self, text: str) -> tuple[EventType, float]:
        text_lower = text.lower()
        for etype, keywords, base_conf in EVENT_PATTERNS:
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            if matches > 0:
                conf = min(base_conf + 0.05 * (matches - 1), 0.99)
                return etype, conf
        return EventType.GENERAL, 0.4

    def _extract_region(self, text: str) -> Optional[str]:
        for region, patterns in REGION_PATTERNS:
            if any(p.lower() in text.lower() for p in patterns):
                return region
        return None

    def _extract_company(self, text: str) -> tuple[Optional[str], Optional[str]]:
        for company, ticker in KNOWN_TICKERS.items():
            if company.lower() in text.lower():
                return company, ticker
        # Try uppercase ticker detection
        tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        if tickers:
            return None, tickers[0]
        return None, None

    def _get_sentiment_score(self, text: str) -> float:
        """Returns -1.0 (negative) to +1.0 (positive)."""
        if self.sentiment_pipe:
            try:
                results = self.sentiment_pipe(text[:512])[0]
                scores = {r["label"]: r["score"] for r in results}
                return scores.get("positive", 0) - scores.get("negative", 0)
            except Exception:
                pass
        # Fallback keyword sentiment
        neg = ["crash", "bankrupt", "shutdown", "loss", "decline", "warning", "fear", "risk"]
        pos = ["growth", "profit", "beat", "surge", "record", "strong", "rally", "gain"]
        text_lower = text.lower()
        score = sum(1 for w in pos if w in text_lower) - sum(1 for w in neg if w in text_lower)
        return max(-1.0, min(1.0, score * 0.2))

    def _compute_severity(self, event_type: EventType, sentiment: float, confidence: float) -> float:
        base = {
            EventType.BANKRUPTCY: 0.95,
            EventType.GEOPOLITICAL_EVENT: 0.85,
            EventType.COMMODITY_SHOCK: 0.80,
            EventType.SUPPLY_CHAIN_DISRUPTION: 0.75,
            EventType.INTEREST_RATE_DECISION: 0.70,
            EventType.REGULATORY_ACTION: 0.70,
            EventType.FACTORY_SHUTDOWN: 0.70,
            EventType.MERGER_ACQUISITION: 0.60,
            EventType.PRODUCT_RECALL: 0.65,
            EventType.EARNINGS: 0.50,
            EventType.LEADERSHIP_CHANGE: 0.45,
            EventType.CYBER_ATTACK: 0.80,
            EventType.GENERAL: 0.30,
        }.get(event_type, 0.40)

        # Negative sentiment increases severity
        sentiment_factor = (1.0 - sentiment) * 0.5 if sentiment < 0 else 0.0
        severity = base + sentiment_factor * 0.2
        return round(min(severity * confidence, 1.0), 3)

    def classify(self, title: str, summary: str = "", timestamp: Optional[datetime] = None) -> FinancialEvent:
        text = f"{title} {summary}".strip()
        event_type, confidence = self._classify_event_type(text)
        region = self._extract_region(text)
        company, ticker = self._extract_company(text)
        sentiment = self._get_sentiment_score(text)
        severity = self._compute_severity(event_type, sentiment, confidence)
        affected_sectors = SECTOR_EXPOSURE.get(event_type.value, [])
        affected_tickers = [ticker] if ticker else []

        return FinancialEvent(
            event_type=event_type,
            company=company or ticker,
            severity=severity,
            sentiment=sentiment,
            title=title,
            summary=summary,
            timestamp=timestamp or datetime.now(),
            region=region,
            affected_sectors=affected_sectors,
            affected_tickers=affected_tickers,
            raw_text=text,
            confidence=confidence,
        )

    def classify_batch(self, articles: list[dict]) -> list[FinancialEvent]:
        events = []
        for article in articles:
            try:
                ev = self.classify(
                    title=article.get("title", ""),
                    summary=article.get("summary", ""),
                    timestamp=article.get("timestamp"),
                )
                events.append(ev)
            except Exception as e:
                logger.warning("Event classification failed: {}", e)
        logger.info("Classified {} events from {} articles", len(events), len(articles))
        return events
