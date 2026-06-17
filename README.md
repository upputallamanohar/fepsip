# FEPSIP — Financial Event Propagation & Systemic Risk Intelligence Platform

> AI-powered multimodal platform for modeling, explaining, and predicting how financial events propagate through markets, sectors, and macroeconomic networks.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FEPSIP Platform                              │
├──────────────┬──────────────────────────────┬───────────────────────┤
│  Data Layer  │       Intelligence Layer      │    Delivery Layer     │
├──────────────┼──────────────────────────────┼───────────────────────┤
│ Market Data  │  Financial Knowledge Graph   │   FastAPI REST API    │
│ (YFinance)   │  (NetworkX / Neo4j)          │   /predict /risk      │
│              │                              │   /simulate /explain  │
│ News Feeds   │  Temporal Graph Network      │                       │
│ (Yahoo News) │  (TGN / EvolveGCN)          │  Streamlit Dashboard  │
│              │                              │  Risk Heatmaps        │
│ Fundamentals │  Event Intelligence          │  Knowledge Graph Viz  │
│ (YFinance)   │  (FinBERT + Rules)           │  Scenario Simulator   │
│              │                              │                       │
│ Macro Data   │  Multimodal Fusion           │  Alert System         │
│ (YFinance)   │  (Cross-Modal Attention)     │  Telegram/Email/Slack │
└──────────────┴──────────────────────────────┴───────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| 📊 **Event Intelligence** | Classifies news into 13 structured event types using FinBERT + rules |
| 🕸️ **Knowledge Graph** | Dynamic financial graph with 50+ nodes, 10+ edge types, correlation edges |
| ⏱️ **Temporal GNN** | TGN architecture captures time-varying graph dynamics |
| 🎯 **Multimodal Prediction** | Fuses price, NLP, graph, fundamentals, macro via cross-attention |
| 🌊 **Ripple Effect Modeling** | BFS contagion propagation through supply chains & correlations |
| ⚠️ **Systemic Risk Scores** | PageRank + centrality + volatility composite (0-100) |
| 🔭 **Market Regime Detection** | HMM / GMM detects Bull/Bear/Crisis/High-Vol/Recovery regimes |
| 🔍 **Anomaly Detection** | Isolation Forest + Z-score for flash crashes & volume spikes |
| 🔬 **Scenario Simulation** | 7 scenario templates + free-text event injection |
| 💡 **Explainability** | SHAP + feature attribution + attention maps |
| 💼 **Portfolio Optimization** | Markowitz MVO with risk penalization & trade actions |
| 📚 **Financial Memory** | RAG over historical crises using Qdrant vector DB |
| 🤖 **Multi-Agent Pipeline** | 6 specialized agents collaborate via async message passing |
| 🚨 **Alert System** | Telegram, Email, Webhook alerts on threshold breaches |

---

## Project Structure

```
fepsip/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── market_data.py         # DataPipeline, MarketDataIngester, NewsIngester
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py     # FinancialKnowledgeGraph (NetworkX)
│   │   └── temporal_graph.py      # TemporalGraphNetwork, TemporalGraphStore
│   ├── events/
│   │   ├── __init__.py
│   │   └── event_intelligence.py  # EventClassifier, FinancialEvent taxonomy
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py            # PriceEncoder, NLPEncoder, MultimodalFusion
│   │   └── training.py            # ModelTrainer, PriceDataset
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py           # PredictionEngine, RippleEffect, SystemicRisk
│   ├── regime/
│   │   ├── __init__.py
│   │   └── market_regime.py       # HMMRegimeDetector, GMMRegimeDetector
│   ├── anomaly/
│   │   ├── __init__.py
│   │   └── anomaly_detector.py    # IsolationForest, StatisticalAnomalyDetector
│   ├── xai/
│   │   ├── __init__.py
│   │   └── explainer.py           # FeatureImportanceExplainer, AttentionVisualizer
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── scenario_simulator.py  # ScenarioSimulator, 7 scenario templates
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── portfolio_manager.py   # MeanVarianceOptimizer, PortfolioManager
│   ├── agents/
│   │   ├── __init__.py
│   │   └── agents.py              # AgentCoordinator + 6 specialized agents
│   ├── memory/
│   │   ├── __init__.py
│   │   └── financial_memory.py    # FinancialMemoryStore (Qdrant / in-memory)
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── alert_system.py        # AlertDispatcher (Telegram/Email/Webhook)
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI app, 18+ endpoints
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                 # Streamlit dashboard, 9 pages
│   └── utils/
│       ├── __init__.py
│       ├── settings.py            # Pydantic settings
│       └── logger.py              # Loguru logging
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_knowledge_graph.py
│   │   ├── test_event_intelligence.py
│   │   ├── test_prediction.py
│   │   ├── test_anomaly.py
│   │   ├── test_portfolio.py
│   │   ├── test_simulation.py
│   │   ├── test_xai.py
│   │   └── test_memory.py
│   └── integration/
│       └── test_api.py
├── config/
│   └── config.yaml                # Full platform configuration
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── k8s/
│   └── deployment.yaml            # Kubernetes manifests
├── notebooks/
│   └── 01_quickstart.py           # Full demo script
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourorg/fepsip.git
cd fepsip
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
```

### 2. Run the Quickstart Demo

```bash
cd fepsip
python notebooks/01_quickstart.py
```

### 3. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
# OpenAPI docs at http://localhost:8000/docs
```

### 4. Start the Dashboard

```bash
streamlit run src/dashboard/app.py
# Dashboard at http://localhost:8501
```

### 5. Docker Compose (Full Stack)

```bash
docker-compose up -d
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Neo4j:     http://localhost:7474
# Qdrant:    http://localhost:6333
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict` | Predict stock direction |
| GET | `/predict/batch` | Batch predict all tickers |
| GET | `/risk` | Systemic risk report |
| GET | `/risk/{ticker}` | Single ticker risk score |
| GET | `/graph` | Knowledge graph structure |
| GET | `/graph/contagion/{ticker}` | Contagion propagation paths |
| POST | `/simulate` | Run named scenario |
| POST | `/simulate/text` | Auto-detect & simulate from text |
| GET | `/scenarios` | List available scenarios |
| GET | `/events` | Latest classified events |
| GET | `/anomalies` | Anomaly detection for ticker |
| POST | `/explain` | XAI explanation for prediction |
| GET | `/portfolio` | Optimized portfolio state |
| GET | `/regime` | Current market regime |
| POST | `/research` | RAG financial memory query |
| POST | `/agents/run` | Run full multi-agent pipeline |
| POST | `/data/refresh` | Trigger background data refresh |

---

## Scenario Templates

| Scenario | Description | Severity |
|----------|-------------|----------|
| `factory_shutdown` | Company factory halts production | 0.75 |
| `chip_shortage` | Semiconductor supply shortage | 0.80 |
| `oil_price_shock` | Oil price surge (e.g. OPEC cut) | 0.85 |
| `fed_rate_hike` | Federal Reserve rate decision | 0.70 |
| `bank_failure` | Bank collapse with contagion | 0.95 |
| `geopolitical_crisis` | War, sanctions, trade conflict | 0.85 |
| `pandemic_outbreak` | Global health crisis | 0.95 |

---

## Event Types

```
EARNINGS · MERGER_ACQUISITION · BANKRUPTCY · SUPPLY_CHAIN_DISRUPTION
PRODUCT_RECALL · REGULATORY_ACTION · INTEREST_RATE_DECISION
COMMODITY_SHOCK · GEOPOLITICAL_EVENT · FACTORY_SHUTDOWN
LEADERSHIP_CHANGE · CYBER_ATTACK · GENERAL
```

---

## Model Training

```python
from src.models.training import ModelTrainer
from src.ingestion.market_data import MarketDataIngester
import asyncio

async def train():
    ingester = MarketDataIngester(["AAPL","MSFT","GOOGL","JPM","XOM"], lookback_days=730)
    data = await ingester.fetch_all()
    trainer = ModelTrainer(checkpoint_dir="checkpoints")
    model = trainer.train_price_encoder(data, num_epochs=30)
    metrics = trainer.evaluate(model, data)
    print(f"Accuracy: {metrics['accuracy']:.3f}")

asyncio.run(train())
```

---

## Running Tests

```bash
# Unit tests (fast, no API)
pytest tests/unit/ -v --cov=src

# Integration tests (starts full API)
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Configuration

All settings in `config/config.yaml`. Override with environment variables in `.env`:

```bash
FINNHUB_API_KEY=your_key      # Live market data
NEO4J_URI=bolt://localhost:7687  # Graph DB
QDRANT_HOST=localhost          # Vector DB
TELEGRAM_TOKEN=your_token      # Alerts
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.1, Transformers 4.35 |
| NLP | FinBERT (ProsusAI/finbert) |
| Graph | NetworkX 3.2, PyVis |
| Temporal GNN | Custom TGN (PyTorch) |
| ML | scikit-learn, hmmlearn |
| API | FastAPI 0.104, Uvicorn |
| Dashboard | Streamlit 1.28, Plotly 5.18 |
| Graph DB | Neo4j 5.13 (optional) |
| Vector DB | Qdrant 1.7 (optional) |
| Alerts | httpx, python-telegram-bot |
| Logging | Loguru |
| Testing | pytest, pytest-asyncio, pytest-cov |
| CI/CD | GitHub Actions |
| Deployment | Docker, Kubernetes |

---

## License

MIT License – see [LICENSE](LICENSE)

---

## Roadmap

- [ ] PyTorch Geometric integration (full PyG TGN/TGAT)
- [ ] FinRL reinforcement learning portfolio agent
- [ ] LangGraph multi-agent orchestration
- [ ] Real-time WebSocket streaming
- [ ] Bloomberg/Refinitiv data connector
- [ ] Options pricing risk integration
- [ ] Federated learning across data sources
