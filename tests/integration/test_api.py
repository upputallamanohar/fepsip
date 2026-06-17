"""
Integration tests for FastAPI endpoints.
Uses TestClient (no real network calls needed).
"""
import pytest
import json


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "FEPSIP"


def test_predict_endpoint(client):
    r = client.post("/predict", json={"ticker": "AAPL", "sentiment": 0.3, "event_severity": 0.2})
    assert r.status_code == 200
    d = r.json()
    assert "direction" in d
    assert "up_prob" in d
    assert d["ticker"] == "AAPL"


def test_predict_probs_sum_to_one(client):
    r = client.post("/predict", json={"ticker": "MSFT", "sentiment": -0.5})
    d = r.json()
    total = d["up_prob"] + d["down_prob"] + d["neutral_prob"]
    assert abs(total - 1.0) < 1e-3


def test_risk_endpoint(client):
    r = client.get("/risk")
    assert r.status_code == 200
    d = r.json()
    assert "overall_score" in d
    assert "risk_level" in d


def test_ticker_risk_endpoint(client):
    r = client.get("/risk/AAPL")
    assert r.status_code == 200
    d = r.json()
    assert d["ticker"] == "AAPL"
    assert 0 <= d["risk_score"] <= 100


def test_graph_endpoint(client):
    r = client.get("/graph")
    assert r.status_code == 200
    d = r.json()
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) > 0


def test_contagion_endpoint(client):
    r = client.get("/graph/contagion/AAPL?depth=2")
    assert r.status_code == 200
    d = r.json()
    assert d["source"] == "AAPL"
    assert "paths" in d


def test_simulate_endpoint(client):
    r = client.post("/simulate", json={
        "scenario_type": "factory_shutdown",
        "source_ticker": "TSLA",
        "magnitude": 1.0,
    })
    assert r.status_code == 200
    d = r.json()
    assert "scenario_name" in d
    assert "steps" in d


def test_simulate_text_endpoint(client):
    r = client.post("/simulate/text", json={
        "text": "Tesla factory in Berlin halts production",
        "source_ticker": "TSLA",
    })
    assert r.status_code == 200


def test_scenarios_list_endpoint(client):
    r = client.get("/scenarios")
    assert r.status_code == 200
    d = r.json()
    assert "scenarios" in d
    assert len(d["scenarios"]) > 0


def test_explain_endpoint(client):
    r = client.post("/explain", json={
        "ticker": "NVDA",
        "prediction": "UP",
        "confidence": 0.72,
        "features": {"sentiment": 0.6, "momentum": 0.4, "event_severity": 0.1},
    })
    assert r.status_code == 200
    d = r.json()
    assert "narrative" in d
    assert "top_factors" in d


def test_research_endpoint(client):
    r = client.post("/research", json={"query": "What happened during the 2008 financial crisis?"})
    assert r.status_code == 200
    d = r.json()
    assert "answer" in d
    assert len(d["answer"]) > 10


def test_regime_endpoint(client):
    r = client.get("/regime")
    assert r.status_code == 200


def test_events_endpoint(client):
    r = client.get("/events?limit=10")
    assert r.status_code == 200
    d = r.json()
    assert "events" in d


def test_invalid_scenario_returns_400(client):
    r = client.post("/simulate", json={
        "scenario_type": "nonexistent_scenario",
        "source_ticker": "AAPL",
    })
    assert r.status_code == 400


def test_unknown_contagion_ticker(client):
    r = client.get("/graph/contagion/ZZZNOTREAL?depth=2")
    assert r.status_code == 404
