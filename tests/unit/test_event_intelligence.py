"""Unit tests for Event Intelligence Engine."""
import pytest
from datetime import datetime


def test_event_classifier_basic():
    from src.events.event_intelligence import EventClassifier, EventType
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Tesla factory in Berlin halts production", "Production has stopped")
    assert event.event_type in [EventType.FACTORY_SHUTDOWN, EventType.SUPPLY_CHAIN_DISRUPTION]
    assert 0.0 <= event.severity <= 1.0
    assert -1.0 <= event.sentiment <= 1.0


def test_earnings_classification():
    from src.events.event_intelligence import EventClassifier, EventType
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Apple reports record quarterly earnings, beats EPS estimates")
    assert event.event_type == EventType.EARNINGS


def test_bankruptcy_classification():
    from src.events.event_intelligence import EventClassifier, EventType
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Company files for Chapter 11 bankruptcy protection")
    assert event.event_type == EventType.BANKRUPTCY
    assert event.severity >= 0.7


def test_fed_rate_classification():
    from src.events.event_intelligence import EventClassifier, EventType
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Federal Reserve raises interest rates by 25 basis points")
    assert event.event_type == EventType.INTEREST_RATE_DECISION


def test_geopolitical_classification():
    from src.events.event_intelligence import EventClassifier, EventType
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Trade war escalates with new tariffs on imports")
    assert event.event_type == EventType.GEOPOLITICAL_EVENT


def test_batch_classification():
    from src.events.event_intelligence import EventClassifier
    clf = EventClassifier(use_finbert=False)
    articles = [
        {"title": "Apple earnings beat expectations", "summary": "", "timestamp": datetime.now()},
        {"title": "Oil prices surge on OPEC cuts", "summary": "", "timestamp": datetime.now()},
        {"title": "Bank files for bankruptcy", "summary": "", "timestamp": datetime.now()},
    ]
    events = clf.classify_batch(articles)
    assert len(events) == 3


def test_region_extraction():
    from src.events.event_intelligence import EventClassifier
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Factory shutdown in Germany affects European supply chains")
    assert event.region == "Europe"


def test_company_extraction():
    from src.events.event_intelligence import EventClassifier
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Tesla announces new Gigafactory in Texas")
    assert event.company is not None or event.affected_tickers


def test_event_to_dict():
    from src.events.event_intelligence import EventClassifier
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("NVIDIA chip shortage impacts tech sector")
    d = event.to_dict()
    assert "event_type" in d
    assert "severity" in d
    assert "sentiment" in d
    assert "timestamp" in d


def test_sentiment_negative():
    from src.events.event_intelligence import EventClassifier
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Market crashes, stocks plunge on recession fears")
    assert event.sentiment < 0.0


def test_severity_bankruptcy_high():
    from src.events.event_intelligence import EventClassifier
    clf = EventClassifier(use_finbert=False)
    event = clf.classify("Major bank declares insolvency, triggers systemic risk")
    assert event.severity >= 0.5
