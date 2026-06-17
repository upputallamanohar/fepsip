"""
FEPSIP Streamlit Dashboard
Interactive financial intelligence dashboard.
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "GS", "XOM", "BA"]

st.set_page_config(
    page_title="FEPSIP – Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=20)
        return r.json() if r.ok else None
    except Exception:
        return None


def risk_color(score: float) -> str:
    if score >= 80: return "#D50000"
    if score >= 60: return "#FF6D00"
    if score >= 30: return "#FFD600"
    return "#00C853"


def generate_mock_prices(ticker: str, days: int = 90) -> pd.DataFrame:
    np.random.seed(hash(ticker) % 1000)
    price = np.random.uniform(50, 500)
    prices = [price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.02)))
    dates = pd.date_range(end=datetime.today(), periods=days)
    return pd.DataFrame({"date": dates, "close": prices, "ticker": ticker})


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 FEPSIP")
    st.caption("Financial Event Propagation & Systemic Risk Intelligence")

    page = st.radio(
        "Navigation",
        ["📊 Market Overview", "🕸️ Knowledge Graph", "⚡ Event Intelligence",
         "🎯 Predictions", "⚠️ Risk Monitor", "🔬 Scenario Simulator",
         "💼 Portfolio", "🔍 Anomaly Detection", "📚 Research Memory"],
        label_visibility="collapsed",
    )

    st.divider()
    selected_ticker = st.selectbox("Focus Ticker", TICKERS)

    if st.button("🔄 Refresh Data", use_container_width=True):
        api_post("/data/refresh", {})
        st.success("Refresh scheduled")

    health = api_get("/health")
    if health:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline – showing mock data")


# ─────────────────────────────────────────────────────────
# Page: Market Overview
# ─────────────────────────────────────────────────────────

if page == "📊 Market Overview":
    st.title("📊 Market Overview")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    # Regime detection
    regime = api_get("/regime")
    col1, col2, col3, col4 = st.columns(4)
    if regime:
        col1.metric("Market Regime", regime.get("regime", "Unknown"))
        col2.metric("Regime Confidence", f"{regime.get('confidence', 0):.1%}")
        col3.metric("Risk Multiplier", f"{regime.get('risk_multiplier', 1.0):.2f}x")
        col4.metric("Status", "🟢 Live" if health else "🔴 Offline")
    else:
        col1.metric("Market Regime", "Bull Market")
        col2.metric("Regime Confidence", "72.3%")
        col3.metric("Risk Multiplier", "0.80x")
        col4.metric("Status", "🔴 Mock Data")

    st.subheader("Price History")
    if PLOTLY:
        fig = go.Figure()
        for ticker in TICKERS[:5]:
            df = generate_mock_prices(ticker)
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["close"],
                mode="lines", name=ticker, line=dict(width=1.5)
            ))
        fig.update_layout(
            height=350, template="plotly_dark",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk heatmap
    st.subheader("Systemic Risk Heatmap")
    risk_data = api_get("/risk")
    if risk_data and PLOTLY:
        node_scores = risk_data.get("node_scores", {})
        if node_scores:
            tickers_list = list(node_scores.keys())[:20]
            scores = [node_scores[t] for t in tickers_list]
            colors = [risk_color(s) for s in scores]

            fig2 = go.Figure(go.Bar(
                x=tickers_list, y=scores,
                marker_color=colors,
                text=[f"{s:.1f}" for s in scores],
                textposition="auto",
            ))
            fig2.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Critical")
            fig2.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="High")
            fig2.update_layout(
                height=300, template="plotly_dark",
                yaxis_range=[0, 100], yaxis_title="Risk Score",
                margin=dict(l=40, r=20, t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
            col_a, col_b = st.columns(2)
            col_a.metric("Overall Risk Score", f"{risk_data.get('overall_score', 0):.1f}/100")
            col_b.metric("Risk Level", risk_data.get("risk_level", "UNKNOWN"))
    else:
        st.info("Connect API to see live risk heatmap")


# ─────────────────────────────────────────────────────────
# Page: Knowledge Graph
# ─────────────────────────────────────────────────────────

elif page == "🕸️ Knowledge Graph":
    st.title("🕸️ Financial Knowledge Graph")

    try:
        from pyvis.network import Network
        PYVIS = True
    except ImportError:
        PYVIS = False

    graph_data = api_get("/graph", {"max_nodes": 40})

    if graph_data and PYVIS:
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        net = Network(height="600px", bgcolor="#0E1117", font_color="white", directed=True)
        net.barnes_hut(spring_length=200)

        color_map = {
            "Company": "#4FC3F7", "Sector": "#81C784", "Commodity": "#FFB74D",
            "EconomicIndicator": "#CE93D8", "Currency": "#F48FB1", "Country": "#80DEEA",
        }

        for node in nodes:
            ntype = node.get("node_type", "Company")
            risk = node.get("risk_score", 0)
            color = "#D50000" if risk > 80 else "#FF6D00" if risk > 60 else color_map.get(ntype, "#4FC3F7")
            net.add_node(node["id"], label=node["id"], color=color,
                        title=f"Type: {ntype}\nRisk: {risk:.1f}", size=15)

        node_ids = {node["id"] for node in nodes}
        for edge in edges[:100]:
            etype = edge.get("edge_type", "")
            src = edge["source"]
            tgt = edge["target"]
            if src not in node_ids:
                net.add_node(src, label=src, color="#9E9E9E", size=10)
                node_ids.add(src)
            if tgt not in node_ids:
                net.add_node(tgt, label=tgt, color="#9E9E9E", size=10)
                node_ids.add(tgt)
            net.add_edge(src, tgt, title=etype, width=1)

        html = net.generate_html()
        st.components.v1.html(html, height=620, scrolling=False)

        stats = graph_data.get("stats", {})
        col1, col2 = st.columns(2)
        col1.metric("Total Nodes", stats.get("num_nodes", 0))
        col2.metric("Total Edges", stats.get("num_edges", 0))

    elif graph_data:
        stats = graph_data.get("stats", {})
        st.metric("Nodes", stats.get("num_nodes", 0))
        st.metric("Edges", stats.get("num_edges", 0))
        st.info("Install pyvis for interactive graph visualization: `pip install pyvis`")

    # Contagion explorer
    st.subheader("Contagion Path Explorer")
    depth = st.slider("Contagion Depth", 1, 5, 3)
    if st.button("🔍 Trace Contagion"):
        contagion = api_get(f"/graph/contagion/{selected_ticker}", {"depth": depth})
        if contagion:
            st.success(f"Found {contagion.get('affected_count', 0)} affected nodes")
            paths = contagion.get("paths", {})
            for node, path in list(paths.items())[:10]:
                st.code(" → ".join(path), language=None)
        else:
            st.warning("Could not compute contagion paths")


# ─────────────────────────────────────────────────────────
# Page: Event Intelligence
# ─────────────────────────────────────────────────────────

elif page == "⚡ Event Intelligence":
    st.title("⚡ Event Intelligence")

    events_data = api_get("/events", {"limit": 30})
    events = events_data.get("events", []) if events_data else []

    if events:
        for ev in events[:15]:
            severity = ev.get("severity", 0)
            color = "red" if severity >= 0.8 else "orange" if severity >= 0.6 else "blue"
            with st.expander(f"[{ev.get('event_type', 'GENERAL').upper()}] {ev.get('title', 'Event')[:80]}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Severity", f"{severity:.2f}")
                col2.metric("Sentiment", f"{ev.get('sentiment', 0):.2f}")
                col3.metric("Company", ev.get("company") or "—")
                st.write(f"**Type:** {ev.get('event_type', '')}")
                st.write(f"**Region:** {ev.get('region') or 'Unknown'}")
                st.write(f"**Affected Sectors:** {', '.join(ev.get('affected_sectors', []))}")
                if ev.get("summary"):
                    st.write(ev["summary"][:300])
    else:
        st.info("No events loaded. Connect API and refresh data.")

        # Show mock events
        mock_events = [
            {"type": "SUPPLY_CHAIN_DISRUPTION", "title": "TSMC production halt due to earthquake", "severity": 0.82},
            {"type": "INTEREST_RATE_DECISION", "title": "Fed raises rates by 25bps to 5.50%", "severity": 0.70},
            {"type": "EARNINGS", "title": "NVIDIA beats Q4 estimates by 15%", "severity": 0.55},
        ]
        for ev in mock_events:
            st.markdown(f"**[{ev['type']}]** {ev['title']} — Severity: {ev['severity']}")


# ─────────────────────────────────────────────────────────
# Page: Predictions
# ─────────────────────────────────────────────────────────

elif page == "🎯 Predictions":
    st.title("🎯 Stock Predictions")

    col1, col2 = st.columns([1, 2])
    with col1:
        sentiment = st.slider("News Sentiment", -1.0, 1.0, 0.0, 0.05)
        event_sev = st.slider("Event Severity", 0.0, 1.0, 0.2, 0.05)

    with col2:
        if st.button("🔮 Run Predictions for All Tickers"):
            predictions = api_get("/predict/batch")
            if predictions:
                preds = predictions.get("predictions", [])
                rows = []
                for p in preds:
                    rows.append({
                        "Ticker": p["ticker"],
                        "Direction": p["direction"],
                        "↑ Up %": f"{p['up_prob']:.1%}",
                        "→ Neutral %": f"{p['neutral_prob']:.1%}",
                        "↓ Down %": f"{p['down_prob']:.1%}",
                        "Confidence": f"{p['confidence']:.1%}",
                        "Expected Return": f"{p['expected_return']:+.2%}",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df.style.applymap(
                    lambda v: "background-color: #1B5E20" if v == "UP" else
                              "background-color: #B71C1C" if v == "DOWN" else "",
                    subset=["Direction"]
                ), use_container_width=True)

    # Single ticker predict
    st.subheader(f"Predict: {selected_ticker}")
    if st.button("Predict This Ticker"):
        result = api_post("/predict", {
            "ticker": selected_ticker,
            "sentiment": sentiment,
            "event_severity": event_sev
        })
        if result:
            c1, c2, c3 = st.columns(3)
            direction = result.get("direction", "NEUTRAL")
            icon = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "🟡"
            c1.metric("Direction", f"{icon} {direction}")
            c2.metric("Confidence", f"{result.get('confidence', 0):.1%}")
            c3.metric("Expected Return", f"{result.get('expected_return', 0):+.2%}")

            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=["UP", "NEUTRAL", "DOWN"],
                    y=[result["up_prob"], result["neutral_prob"], result["down_prob"]],
                    marker_color=["#00C853", "#FFD600", "#D50000"],
                ))
                fig.update_layout(height=250, template="plotly_dark",
                                  yaxis_title="Probability", margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        else:
            st.warning("No prediction returned. Is the API running?")


# ─────────────────────────────────────────────────────────
# Page: Risk Monitor
# ─────────────────────────────────────────────────────────

elif page == "⚠️ Risk Monitor":
    st.title("⚠️ Systemic Risk Monitor")

    risk_data = api_get("/risk")
    if risk_data:
        overall = risk_data.get("overall_score", 0)
        level = risk_data.get("risk_level", "UNKNOWN")
        colors = {"LOW": "normal", "MEDIUM": "off", "HIGH": "inverse", "CRITICAL": "inverse"}

        st.metric("Overall Systemic Risk", f"{overall:.1f}/100", delta=None,
                  delta_color=colors.get(level, "normal"))
        st.progress(min(overall / 100, 1.0))
        st.markdown(f"**Risk Level:** :{'red' if level in ('HIGH','CRITICAL') else 'orange' if level == 'MEDIUM' else 'green'}[{level}]")

        top_risks = risk_data.get("top_risks", [])
        if top_risks:
            st.subheader("Top Risk Nodes")
            for item in top_risks[:10]:
                col1, col2 = st.columns([3, 1])
                col1.write(item["ticker"])
                score = item["score"]
                color = "🔴" if score >= 80 else "🟠" if score >= 60 else "🟡"
                col2.write(f"{color} {score:.1f}")
    else:
        st.info("Connect the API to see live risk monitoring")
        st.metric("Overall Risk", "45.2/100")
        st.progress(0.45)

    # Ticker risk
    st.subheader(f"Risk Details: {selected_ticker}")
    ticker_risk = api_get(f"/risk/{selected_ticker}")
    if ticker_risk:
        st.metric(f"{selected_ticker} Risk Score", f"{ticker_risk.get('risk_score', 0):.1f}/100")


# ─────────────────────────────────────────────────────────
# Page: Scenario Simulator
# ─────────────────────────────────────────────────────────

elif page == "🔬 Scenario Simulator":
    st.title("🔬 Scenario Simulator")

    scenarios_data = api_get("/scenarios")
    scenario_names = (
        [s["name"] for s in scenarios_data.get("scenarios", [])]
        if scenarios_data else list([
            "factory_shutdown", "chip_shortage", "oil_price_shock",
            "fed_rate_hike", "bank_failure", "geopolitical_crisis"
        ])
    )

    col1, col2 = st.columns(2)
    with col1:
        scenario = st.selectbox("Scenario Type", scenario_names)
        source = st.selectbox("Source Ticker", TICKERS)
        magnitude = st.slider("Magnitude", 0.5, 20.0, 1.0, 0.5)
    with col2:
        company = st.text_input("Company Name (optional)", "")
        region = st.text_input("Region (optional)", "")
        st.markdown("Or describe a scenario:")
        free_text = st.text_area("Free Text Scenario", "Tesla factory in Berlin halts production")

    col_a, col_b = st.columns(2)
    if col_a.button("🚀 Run Named Scenario"):
        result = api_post("/simulate", {
            "scenario_type": scenario,
            "source_ticker": source,
            "magnitude": magnitude,
            "company": company or None,
            "region": region or None,
        })
        if result:
            st.success(f"Simulation complete: {result.get('scenario_name')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Severity", f"{result.get('severity', 0):.2f}")
            c2.metric("Portfolio Impact", f"{result.get('portfolio_impact_pct', 0):+.1f}%")
            c3.metric("Peak Systemic Risk", f"{result.get('peak_systemic_risk', 0):.1f}")

            for step in result.get("steps", []):
                with st.expander(f"Wave {step['step']}: {step['description']}"):
                    for node in step["affected_nodes"][:8]:
                        st.write(f"**{node['ticker']}** ({node['sector']}) → {node['impact_pct']:+.2f}% | {node['path']}")
        else:
            st.error("Simulation failed. Is the API running?")

    if col_b.button("📝 Run Text Scenario"):
        result = api_post("/simulate/text", {"text": free_text, "source_ticker": source})
        if result:
            st.success(f"Detected: {result.get('scenario_name')} | Impact: {result.get('portfolio_impact_pct', 0):+.1f}%")
        else:
            st.error("Text simulation failed")


# ─────────────────────────────────────────────────────────
# Page: Portfolio
# ─────────────────────────────────────────────────────────

elif page == "💼 Portfolio":
    st.title("💼 Portfolio Optimization")

    portfolio = api_get("/portfolio")
    if portfolio and "weights" in portfolio:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Return", f"{portfolio.get('expected_return', 0):.2%}")
        col2.metric("Volatility", f"{portfolio.get('volatility', 0):.2%}")
        col3.metric("Sharpe Ratio", f"{portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Max Drawdown", f"{portfolio.get('max_drawdown', 0):.2%}")

        weights = portfolio.get("weights", {})
        if weights:
            try:
                import plotly.express as px
                df_w = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
                df_w["Weight_pct"] = df_w["Weight"] * 100
                fig = px.pie(df_w, values="Weight_pct", names="Ticker",
                             title="Optimal Portfolio Allocation", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"]))
    else:
        st.info("Connect API for live portfolio optimization")
        st.markdown("""
        **Portfolio Optimizer** uses:
        - Markowitz Mean-Variance Optimization
        - Risk-adjusted expected returns
        - Maximum position size constraints (20%)
        - Systemic risk penalization
        """)


# ─────────────────────────────────────────────────────────
# Page: Anomaly Detection
# ─────────────────────────────────────────────────────────

elif page == "🔍 Anomaly Detection":
    st.title("🔍 Anomaly Detection")

    anomaly_data = api_get("/anomalies", {"ticker": selected_ticker})

    if anomaly_data:
        anomalies = anomaly_data.get("anomalies", [])
        count = anomaly_data.get("count", 0)
        st.metric("Anomalies Detected", count, delta=None)

        if anomalies:
            for a in anomalies[:10]:
                sev = a.get("severity", 0)
                icon = "🔴" if sev >= 0.8 else "🟠" if sev >= 0.5 else "🟡"
                with st.expander(f"{icon} [{a.get('anomaly_type', '').upper()}] — Severity: {sev:.2f}"):
                    st.write(f"**Ticker:** {a.get('ticker')}")
                    st.write(f"**Time:** {a.get('timestamp', '')}")
                    st.write(f"**Description:** {a.get('description', '')}")
                    if a.get("features"):
                        st.json(a["features"])
        else:
            st.success("No anomalies detected for this ticker")
    else:
        st.info(f"Connect API to detect anomalies for {selected_ticker}")


# ─────────────────────────────────────────────────────────
# Page: Research Memory
# ─────────────────────────────────────────────────────────

elif page == "📚 Research Memory":
    st.title("📚 Financial Research Memory (RAG)")

    st.markdown("Ask questions about historical financial events and crises:")
    query = st.text_input("Research Query", "What happened last time oil prices increased 20%?")

    if st.button("🔍 Search Memory"):
        result = api_post("/research", {"query": query})
        if result:
            st.markdown("### Answer")
            st.write(result.get("answer", "No answer found"))

            st.markdown("### Source Events")
            for src in result.get("sources", []):
                with st.expander(src.get("description", "")[:80]):
                    st.write(f"**Outcome:** {src.get('outcome', '')}")
                    st.write(f"**Sectors:** {', '.join(src.get('affected_sectors', []))}")
                    st.write(f"**Impact:** {src.get('impact_magnitude', 0):.2f}")
                    st.write(f"**Date:** {src.get('timestamp', '')}")
        else:
            st.warning("Connect API for live memory search")
            st.info("The Financial Memory system contains pre-seeded historical crises: 2008 GFC, COVID-19, Dot-com bubble, 1973 Oil Embargo, SVB 2023, Ukraine War 2022")

    st.divider()
    st.markdown("**Pre-seeded Historical Events:**")
    for crisis in ["2008 Global Financial Crisis", "COVID-19 Pandemic", "Dot-com Bubble", "1973 Oil Embargo", "SVB Collapse 2023"]:
        st.markdown(f"• {crisis}")
