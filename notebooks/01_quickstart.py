"""
FEPSIP Quickstart Notebook
Run this script to demonstrate the full FEPSIP pipeline.
"""
import asyncio
import sys
sys.path.insert(0, "..")

from src.utils.logger import setup_logger
setup_logger(level="INFO")

# ── 1. Build Knowledge Graph ──────────────────────────────
from src.graph.knowledge_graph import FinancialKnowledgeGraph

print("\n=== 1. FINANCIAL KNOWLEDGE GRAPH ===")
graph = FinancialKnowledgeGraph()
stats = graph.to_dict()["stats"]
print(f"  Nodes: {stats['num_nodes']}  |  Edges: {stats['num_edges']}")

# Contagion paths
paths = graph.get_contagion_path("XOM", depth=2)
print(f"\n  Contagion from XOM (depth=2): {len(paths)-1} affected nodes")
for node, path in list(paths.items())[:5]:
    print(f"    {' → '.join(path)}")

# ── 2. Event Intelligence ─────────────────────────────────
from src.events.event_intelligence import EventClassifier

print("\n=== 2. EVENT INTELLIGENCE ===")
clf = EventClassifier(use_finbert=False)
test_headlines = [
    "Tesla factory in Berlin halts production due to supply chain issues",
    "Federal Reserve raises interest rates by 50 basis points",
    "NVIDIA beats Q4 earnings estimates by 18%, revenue at record high",
    "Oil prices surge 12% after OPEC announces production cuts",
    "Major bank files for Chapter 11 bankruptcy protection",
]
for headline in test_headlines:
    ev = clf.classify(headline)
    print(f"  [{ev.event_type.value.upper():<28}] severity={ev.severity:.2f}  sent={ev.sentiment:+.2f}  → {headline[:60]}")

# ── 3. Predictions ────────────────────────────────────────
from src.prediction.predictor import PredictionEngine

print("\n=== 3. STOCK PREDICTIONS ===")
engine = PredictionEngine()
tickers = ["AAPL", "MSFT", "XOM", "JPM", "TSLA"]
for ticker in tickers:
    pred = engine.predict_stock(ticker, sentiment=0.2, event_severity=0.3)
    bar = "█" * int(max(pred.up_prob, pred.down_prob, pred.neutral_prob) * 20)
    print(f"  {ticker:<6} {pred.direction:<8} UP={pred.up_prob:.2f} NEU={pred.neutral_prob:.2f} DWN={pred.down_prob:.2f}  conf={pred.confidence:.2f}")

# ── 4. Systemic Risk ──────────────────────────────────────
print("\n=== 4. SYSTEMIC RISK ===")
risk_report = engine.compute_systemic_risk(graph, tickers, market_volatility=0.25)
print(f"  Overall Risk Score: {risk_report.overall_score:.1f}/100  [{risk_report.risk_level}]")
print("  Top 5 Risky Nodes:")
for item in risk_report.top_risks[:5]:
    print(f"    {item['ticker']:<8} {item['score']:.1f}")

# ── 5. Market Regime ──────────────────────────────────────
import numpy as np, pandas as pd
from src.regime.market_regime import get_regime_detector

print("\n=== 5. MARKET REGIME ===")
np.random.seed(42)
fake_returns = pd.Series(np.random.randn(300) * 0.012 + 0.0005)
regime = get_regime_detector(fake_returns)
print(f"  Current Regime: {regime.regime.value}  (confidence={regime.confidence:.2%})")
print(f"  Risk Multiplier: {regime.risk_multiplier:.2f}x")

# ── 6. Scenario Simulation ────────────────────────────────
from src.simulation.scenario_simulator import ScenarioSimulator

print("\n=== 6. SCENARIO SIMULATION: Oil Price Shock ===")
sim = ScenarioSimulator(graph, engine, steps=3)
result = sim.simulate("oil_price_shock", "XOM", magnitude=5.0)
print(f"  Severity: {result.severity:.2f}  |  Portfolio Impact: {result.portfolio_impact_pct:+.1f}%")
for step in result.steps:
    print(f"\n  Wave {step.step}: {step.description}")
    for node in step.affected_nodes[:4]:
        print(f"    {node['ticker']:<8} {node['impact_pct']:+.2f}%  [{node['sector']}]")

# ── 7. Explainability ─────────────────────────────────────
from src.xai.explainer import FeatureImportanceExplainer

print("\n=== 7. EXPLAINABILITY ===")
exp = FeatureImportanceExplainer()
explanation = exp.explain_prediction(
    ticker="TSLA",
    prediction="DOWN",
    confidence=0.72,
    features={"sentiment": -0.7, "event_severity": 0.85,
               "supply_chain_risk": 0.6, "sector_weakness": 0.4,
               "macro_factor": 0.3, "volatility": 0.5},
)
print(f"  {explanation.narrative}")
print("  Top Factors:")
for f in explanation.top_factors[:4]:
    print(f"    {f['factor']:<28} {f['contribution_pct']:.1f}%")

# ── 8. Financial Memory ───────────────────────────────────
from src.memory.financial_memory import FinancialMemoryStore

print("\n=== 8. FINANCIAL MEMORY (RAG) ===")
memory = FinancialMemoryStore()
query = "What happened when oil prices increased dramatically?"
answer = memory.query(query)
print(f"  Query: {query}")
print(f"  {answer[:300]}")

# ── 9. Portfolio Optimization ─────────────────────────────
from src.portfolio.portfolio_manager import PortfolioManager

print("\n=== 9. PORTFOLIO OPTIMIZATION ===")
np.random.seed(7)
all_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "TSLA"]
returns_df = pd.DataFrame(
    np.random.randn(252, len(all_tickers)) * 0.015,
    columns=all_tickers
)
pm = PortfolioManager(all_tickers, initial_capital=1_000_000)
state = pm.optimize(returns_df, risk_scores={t: 40.0 for t in all_tickers})
print(f"  Sharpe Ratio:    {state.sharpe_ratio:.3f}")
print(f"  Expected Return: {state.expected_return:.2%}")
print(f"  Volatility:      {state.volatility:.2%}")
print(f"  Max Drawdown:    {state.max_drawdown:.2%}")
print("  Weights:")
for t, w in sorted(state.weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    {t:<8} {w:.1%}  {'█' * int(w * 40)}")

print("\n✅ FEPSIP Quickstart Complete!")
