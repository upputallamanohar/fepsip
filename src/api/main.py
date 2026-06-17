"""
FEPSIP FastAPI Backend
Production-grade REST API with OpenAPI docs.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils import logger, setup_logger
from src.ingestion.market_data import DataPipeline
from src.graph.knowledge_graph import FinancialKnowledgeGraph
from src.graph.temporal_graph import TemporalGraphStore
from src.events.event_intelligence import EventClassifier
from src.prediction.predictor import PredictionEngine
from src.regime.market_regime import get_regime_detector
from src.anomaly.anomaly_detector import AnomalyDetectionEngine
from src.xai.explainer import FeatureImportanceExplainer
from src.simulation.scenario_simulator import ScenarioSimulator, SCENARIO_TEMPLATES
from src.portfolio.portfolio_manager import PortfolioManager
from src.agents.agents import AgentCoordinator
from src.memory.financial_memory import FinancialMemoryStore
from src.alerts.alert_system import AlertDispatcher, AlertRuleEngine

# ─────────────────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "GS", "XOM", "BA"]

class AppState:
    pipeline: Optional[DataPipeline] = None
    graph: Optional[FinancialKnowledgeGraph] = None
    temporal_store: Optional[TemporalGraphStore] = None
    event_classifier: Optional[EventClassifier] = None
    prediction_engine: Optional[PredictionEngine] = None
    anomaly_engine: Optional[AnomalyDetectionEngine] = None
    explainer: Optional[FeatureImportanceExplainer] = None
    portfolio_manager: Optional[PortfolioManager] = None
    memory_store: Optional[FinancialMemoryStore] = None
    agent_coordinator: Optional[AgentCoordinator] = None
    alert_dispatcher: Optional[AlertDispatcher] = None
    alert_engine: Optional[AlertRuleEngine] = None
    data_cache: dict = {}

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    setup_logger()
    logger.info("🚀 FEPSIP API starting up...")

    state.graph = FinancialKnowledgeGraph()
    state.temporal_store = TemporalGraphStore()
    state.event_classifier = EventClassifier(use_finbert=False)
    state.prediction_engine = PredictionEngine()
    state.anomaly_engine = AnomalyDetectionEngine()
    state.explainer = FeatureImportanceExplainer()
    state.portfolio_manager = PortfolioManager(DEFAULT_TICKERS)
    state.memory_store = FinancialMemoryStore()
    state.alert_dispatcher = AlertDispatcher()
    state.alert_engine = AlertRuleEngine(state.alert_dispatcher)

    state.agent_coordinator = AgentCoordinator(
        graph=state.graph,
        event_classifier=state.event_classifier,
        prediction_engine=state.prediction_engine,
        portfolio_manager=state.portfolio_manager,
        memory_store=state.memory_store,
        alert_engine=state.alert_engine,
    )

    # Fetch initial data
    try:
        state.pipeline = DataPipeline(DEFAULT_TICKERS, lookback_days=180)
        data = await state.pipeline.run_all()
        state.data_cache = data
        # Update graph correlations
        returns = state.pipeline.market.get_returns_matrix()
        if not returns.empty:
            state.graph.update_correlations(returns)
        state.temporal_store.save_snapshot(state.graph)
        logger.info("✅ Initial data load complete")
    except Exception as e:
        logger.warning("Initial data load failed: {} — API will use mock data", e)

    yield

    logger.info("FEPSIP API shutting down")


# ─────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="FEPSIP – Financial Event Propagation & Systemic Risk Intelligence Platform",
    description="AI-powered financial event propagation, contagion modeling, and systemic risk prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker: str
    sentiment: float = Field(0.0, ge=-1.0, le=1.0)
    event_severity: float = Field(0.0, ge=0.0, le=1.0)

class SimulateRequest(BaseModel):
    scenario_type: str
    source_ticker: str
    magnitude: float = Field(1.0, ge=0.1, le=50.0)
    company: Optional[str] = None
    region: Optional[str] = None

class SimulateTextRequest(BaseModel):
    text: str
    source_ticker: str

class ExplainRequest(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    features: dict[str, float]

class ResearchQuery(BaseModel):
    query: str

class AlertThresholdRequest(BaseModel):
    risk_threshold: float = Field(80.0, ge=0.0, le=100.0)
    contagion_threshold: float = Field(0.7, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "FEPSIP", "version": "1.0.0", "timestamp": datetime.now().isoformat()}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "components": {
            "graph": state.graph is not None,
            "prediction_engine": state.prediction_engine is not None,
            "memory_store": state.memory_store is not None,
            "data_loaded": bool(state.data_cache),
        },
    }


@app.post("/predict", tags=["Prediction"])
async def predict(req: PredictRequest):
    """Predict stock direction for a single ticker."""
    if state.prediction_engine is None:
        raise HTTPException(503, "Prediction engine not initialized")

    market_data = state.data_cache.get("market", {})
    price_series = None
    if req.ticker in market_data:
        price_series = market_data[req.ticker]["close"]

    pred = state.prediction_engine.predict_stock(
        ticker=req.ticker,
        price_series=price_series,
        sentiment=req.sentiment,
        event_severity=req.event_severity,
    )
    return pred.to_dict()


@app.get("/predict/batch", tags=["Prediction"])
async def predict_batch():
    """Predict directions for all tracked tickers."""
    if state.prediction_engine is None:
        raise HTTPException(503, "Prediction engine not initialized")

    results = []
    for ticker in DEFAULT_TICKERS:
        market_data = state.data_cache.get("market", {})
        price_series = market_data.get(ticker, {}).get("close") if ticker in market_data else None
        pred = state.prediction_engine.predict_stock(ticker=ticker, price_series=price_series)
        results.append(pred.to_dict())

    return {"predictions": results, "count": len(results), "timestamp": datetime.now().isoformat()}


@app.get("/risk", tags=["Risk"])
async def get_systemic_risk():
    """Compute systemic risk report for all tracked tickers."""
    if state.prediction_engine is None or state.graph is None:
        raise HTTPException(503, "Risk engine not initialized")

    market_data = state.data_cache.get("market", {})
    volatility = 0.2
    if market_data:
        import numpy as np
        vols = [df["volatility_20d"].dropna().iloc[-1] if "volatility_20d" in df.columns else 0.2
                for df in market_data.values() if hasattr(df, 'columns')]
        volatility = float(np.mean(vols)) if vols else 0.2

    report = state.prediction_engine.compute_systemic_risk(
        state.graph, DEFAULT_TICKERS, market_volatility=volatility
    )
    return report.to_dict()


@app.get("/risk/{ticker}", tags=["Risk"])
async def get_ticker_risk(ticker: str):
    """Get systemic risk score for a single ticker."""
    if state.graph is None:
        raise HTTPException(503, "Graph not initialized")
    score = state.graph.get_systemic_risk_score(ticker)
    return {"ticker": ticker, "risk_score": score, "timestamp": datetime.now().isoformat()}


@app.get("/graph", tags=["Graph"])
async def get_graph(max_nodes: int = 50):
    """Get the financial knowledge graph structure."""
    if state.graph is None:
        raise HTTPException(503, "Graph not initialized")
    data = state.graph.to_dict()
    # Limit nodes for performance
    data["nodes"] = data["nodes"][:max_nodes]
    return data


@app.get("/graph/contagion/{ticker}", tags=["Graph"])
async def get_contagion(ticker: str, depth: int = 3):
    """Get contagion propagation paths from a source ticker."""
    if state.graph is None:
        raise HTTPException(503, "Graph not initialized")
    if not state.graph.G.has_node(ticker):
        raise HTTPException(404, f"Ticker {ticker} not found in graph")
    paths = state.graph.get_contagion_path(ticker, depth=depth)
    return {
        "source": ticker,
        "depth": depth,
        "paths": {k: v for k, v in list(paths.items())[:30]},
        "affected_count": len(paths) - 1,
    }


@app.post("/simulate", tags=["Simulation"])
async def simulate(req: SimulateRequest):
    """Run a named scenario simulation."""
    if state.prediction_engine is None or state.graph is None:
        raise HTTPException(503, "Simulation engine not initialized")

    if req.scenario_type not in SCENARIO_TEMPLATES:
        raise HTTPException(400, f"Unknown scenario type. Available: {list(SCENARIO_TEMPLATES.keys())}")

    simulator = ScenarioSimulator(state.graph, state.prediction_engine)
    result = simulator.simulate(
        scenario_type=req.scenario_type,
        source_ticker=req.source_ticker,
        magnitude=req.magnitude,
        company=req.company,
        region=req.region,
    )
    return result.to_dict()


@app.post("/simulate/text", tags=["Simulation"])
async def simulate_from_text(req: SimulateTextRequest):
    """Auto-detect scenario from free text and simulate."""
    if state.prediction_engine is None or state.graph is None:
        raise HTTPException(503, "Simulation engine not initialized")

    simulator = ScenarioSimulator(state.graph, state.prediction_engine)
    result = simulator.simulate_from_text(req.text, req.source_ticker)
    return result.to_dict()


@app.get("/events", tags=["Events"])
async def get_events(limit: int = 20):
    """Get latest classified financial events."""
    news_df = state.data_cache.get("news")
    if news_df is None or (hasattr(news_df, 'empty') and news_df.empty):
        return {"events": [], "count": 0}

    articles = news_df.head(limit).to_dict(orient="records")
    if state.event_classifier:
        events = state.event_classifier.classify_batch(articles)
        return {"events": [e.to_dict() for e in events], "count": len(events)}

    return {"events": [], "count": 0}


@app.get("/anomalies", tags=["Anomaly Detection"])
async def get_anomalies(ticker: str = "AAPL"):
    """Detect anomalies for a ticker."""
    market_data = state.data_cache.get("market", {})
    if ticker not in market_data:
        return {"anomalies": [], "ticker": ticker, "message": "No data available"}

    df = market_data[ticker]
    if state.anomaly_engine is None:
        raise HTTPException(503, "Anomaly engine not initialized")

    state.anomaly_engine.fit_ticker(ticker, df)
    alerts = state.anomaly_engine.detect_all(ticker, df)
    return {
        "ticker": ticker,
        "anomalies": [a.to_dict() for a in alerts[:20]],
        "count": len(alerts),
    }


@app.post("/explain", tags=["Explainability"])
async def explain_prediction(req: ExplainRequest):
    """Generate an XAI explanation for a prediction."""
    if state.explainer is None:
        raise HTTPException(503, "Explainer not initialized")

    explanation = state.explainer.explain_prediction(
        ticker=req.ticker,
        prediction=req.prediction,
        confidence=req.confidence,
        features=req.features,
    )
    return explanation.to_dict()


@app.get("/portfolio", tags=["Portfolio"])
async def get_portfolio():
    """Get optimized portfolio state."""
    if state.portfolio_manager is None:
        raise HTTPException(503, "Portfolio manager not initialized")

    market_data = state.data_cache.get("market", {})
    if not market_data:
        return {"message": "No market data loaded", "weights": {}}

    import pandas as pd
    returns = pd.concat(
        {t: df["returns"] for t, df in market_data.items() if "returns" in df.columns}, axis=1
    ).dropna()

    if returns.empty:
        return {"message": "Insufficient return data"}

    state_obj = state.portfolio_manager.optimize(returns)
    return state_obj.to_dict()


@app.get("/regime", tags=["Market Regime"])
async def get_regime():
    """Detect current market regime."""
    market_data = state.data_cache.get("market", {})
    if not market_data:
        return {"regime": "Unknown", "confidence": 0.0, "message": "No data loaded"}

    import pandas as pd
    sp500_returns = None
    if "AAPL" in market_data:
        sp500_returns = market_data["AAPL"]["returns"].dropna()
    else:
        first = next(iter(market_data.values()))
        sp500_returns = first["returns"].dropna()

    regime = get_regime_detector(sp500_returns)
    return regime.to_dict()


@app.post("/research", tags=["Research"])
async def research_query(req: ResearchQuery):
    """Answer a research question using financial memory RAG."""
    if state.memory_store is None:
        raise HTTPException(503, "Memory store not initialized")

    answer = state.memory_store.query(req.query)
    results = state.memory_store.search(req.query, top_k=3)
    return {
        "query": req.query,
        "answer": answer,
        "sources": [r["entry"] for r in results],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/agents/run", tags=["Agents"])
async def run_agents(research_query: Optional[str] = None):
    """Run the full multi-agent pipeline."""
    if state.agent_coordinator is None:
        raise HTTPException(503, "Agent coordinator not initialized")

    market_data = state.data_cache.get("market", {})
    returns_matrix = None
    latest_prices = {}
    if market_data:
        import pandas as pd
        returns_matrix = pd.concat(
            {t: df["returns"] for t, df in market_data.items() if "returns" in df.columns}, axis=1
        ).dropna()
        latest_prices = {t: float(df["close"].iloc[-1]) for t, df in market_data.items()
                         if not df.empty and "close" in df.columns}

    context = {
        "tickers": DEFAULT_TICKERS,
        "news_df": state.data_cache.get("news"),
        "returns_matrix": returns_matrix,
        "latest_prices": latest_prices,
        "research_query": research_query or "",
    }

    results = await state.agent_coordinator.run_pipeline(context)
    return {k: v.to_dict() for k, v in results.items()}


@app.post("/data/refresh", tags=["Data"])
async def refresh_data(background_tasks: BackgroundTasks):
    """Trigger a background data refresh."""
    async def _refresh():
        logger.info("Background data refresh started...")
        try:
            data = await state.pipeline.run_all()
            state.data_cache.update(data)
            returns = state.pipeline.market.get_returns_matrix()
            if not returns.empty:
                state.graph.update_correlations(returns)
            state.temporal_store.save_snapshot(state.graph)
            logger.info("Background data refresh complete")
        except Exception as e:
            logger.error("Data refresh failed: {}", e)

    if state.pipeline:
        background_tasks.add_task(_refresh)
        return {"status": "refresh_scheduled", "timestamp": datetime.now().isoformat()}
    return {"status": "pipeline_not_initialized"}


@app.get("/scenarios", tags=["Simulation"])
async def list_scenarios():
    """List all available scenario templates."""
    return {
        "scenarios": [
            {"name": k, "description": v["description"], "severity": v["severity"],
             "affected_sectors": v["affected_sectors"]}
            for k, v in SCENARIO_TEMPLATES.items()
        ]
    }
