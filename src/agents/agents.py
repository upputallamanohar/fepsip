"""
FEPSIP Agentic AI Layer
Specialized financial agents that collaborate via message passing.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from src.utils import logger


class AgentRole(str, Enum):
    NEWS = "news_agent"
    EVENT = "event_agent"
    GRAPH = "graph_agent"
    RISK = "risk_agent"
    PORTFOLIO = "portfolio_agent"
    RESEARCH = "research_agent"
    COORDINATOR = "coordinator"


@dataclass
class AgentMessage:
    sender: AgentRole
    recipient: AgentRole
    content: dict
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp():.0f}")


@dataclass
class AgentResult:
    agent: AgentRole
    status: str
    data: dict
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent.value,
            "status": self.status,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseAgent:
    """Base class for all FEPSIP agents."""

    def __init__(self, role: AgentRole) -> None:
        self.role = role
        self._inbox: list[AgentMessage] = []
        self._outbox: list[AgentMessage] = []

    def receive(self, message: AgentMessage) -> None:
        self._inbox.append(message)

    def send(self, recipient: AgentRole, content: dict) -> AgentMessage:
        msg = AgentMessage(sender=self.role, recipient=recipient, content=content)
        self._outbox.append(msg)
        return msg

    async def run(self, context: dict) -> AgentResult:
        raise NotImplementedError


class NewsAgent(BaseAgent):
    """Fetches and preprocesses financial news."""

    def __init__(self, news_ingester=None) -> None:
        super().__init__(AgentRole.NEWS)
        self.ingester = news_ingester

    async def run(self, context: dict) -> AgentResult:
        logger.info("[NewsAgent] Fetching news...")
        tickers = context.get("tickers", [])
        articles = context.get("news_df")

        if articles is not None and not articles.empty:
            recent = articles.sort_values("timestamp", ascending=False).head(50)
            news_list = recent.to_dict(orient="records")
        else:
            news_list = []

        return AgentResult(
            agent=self.role,
            status="success",
            data={"articles": news_list, "count": len(news_list)},
        )


class EventAgent(BaseAgent):
    """Classifies news into structured financial events."""

    def __init__(self, event_classifier=None) -> None:
        super().__init__(AgentRole.EVENT)
        self.classifier = event_classifier

    async def run(self, context: dict) -> AgentResult:
        logger.info("[EventAgent] Classifying events...")
        articles = context.get("articles", [])

        if self.classifier and articles:
            events = self.classifier.classify_batch(articles)
            event_dicts = [e.to_dict() for e in events]
        else:
            event_dicts = []

        # Filter high-severity events
        high_severity = [e for e in event_dicts if e.get("severity", 0) >= 0.7]

        return AgentResult(
            agent=self.role,
            status="success",
            data={
                "events": event_dicts,
                "high_severity_events": high_severity,
                "event_count": len(event_dicts),
            },
        )


class GraphAgent(BaseAgent):
    """Updates the knowledge graph and computes centrality."""

    def __init__(self, graph=None) -> None:
        super().__init__(AgentRole.GRAPH)
        self.graph = graph

    async def run(self, context: dict) -> AgentResult:
        logger.info("[GraphAgent] Updating graph...")
        if self.graph is None:
            return AgentResult(agent=self.role, status="skipped", data={})

        # Update prices
        price_data = context.get("latest_prices", {})
        if price_data:
            self.graph.update_node_prices(price_data)

        # Update correlations
        returns_matrix = context.get("returns_matrix")
        if returns_matrix is not None and not returns_matrix.empty:
            self.graph.update_correlations(returns_matrix)

        centrality = self.graph.compute_centrality_scores()
        top_pagerank = sorted(centrality["pagerank"].items(), key=lambda x: x[1], reverse=True)[:10]

        return AgentResult(
            agent=self.role,
            status="success",
            data={
                "graph_stats": self.graph.to_dict()["stats"],
                "top_central_nodes": [{"node": n, "pagerank": round(s, 4)} for n, s in top_pagerank],
            },
        )


class RiskAgent(BaseAgent):
    """Computes systemic risk scores and fires alerts."""

    def __init__(self, prediction_engine=None, alert_engine=None, graph=None) -> None:
        super().__init__(AgentRole.RISK)
        self.engine = prediction_engine
        self.alert_engine = alert_engine
        self.graph = graph

    async def run(self, context: dict) -> AgentResult:
        logger.info("[RiskAgent] Computing systemic risk...")
        tickers = context.get("tickers", [])

        if self.engine and self.graph and tickers:
            risk_report = self.engine.compute_systemic_risk(
                self.graph, tickers,
                market_volatility=context.get("market_volatility", 0.2)
            )
            report_dict = risk_report.to_dict()

            # Fire alerts for critical nodes
            if self.alert_engine:
                for node in risk_report.top_risks[:3]:
                    if node["score"] >= 80:
                        await self.alert_engine.check_risk_score(node["ticker"], node["score"])
        else:
            report_dict = {"overall_score": 40.0, "risk_level": "MEDIUM", "node_scores": {}}

        return AgentResult(agent=self.role, status="success", data=report_dict)


class PortfolioAgent(BaseAgent):
    """Generates portfolio optimization recommendations."""

    def __init__(self, portfolio_manager=None) -> None:
        super().__init__(AgentRole.PORTFOLIO)
        self.manager = portfolio_manager

    async def run(self, context: dict) -> AgentResult:
        logger.info("[PortfolioAgent] Optimizing portfolio...")
        returns_matrix = context.get("returns_matrix")
        risk_scores = context.get("risk_scores", {})
        predictions = context.get("predictions", {})

        if self.manager and returns_matrix is not None and not returns_matrix.empty:
            state = self.manager.optimize(returns_matrix, risk_scores)
            actions = self.manager.generate_trade_actions(
                state.weights, risk_scores, predictions
            )
            return AgentResult(
                agent=self.role,
                status="success",
                data={
                    "portfolio_state": state.to_dict(),
                    "trade_actions": [a.to_dict() for a in actions],
                },
            )

        return AgentResult(agent=self.role, status="skipped", data={})


class ResearchAgent(BaseAgent):
    """Answers research queries via the financial memory RAG system."""

    def __init__(self, memory_store=None) -> None:
        super().__init__(AgentRole.RESEARCH)
        self.memory = memory_store

    async def run(self, context: dict) -> AgentResult:
        query = context.get("research_query", "")
        if not query or not self.memory:
            return AgentResult(agent=self.role, status="skipped", data={})

        answer = self.memory.query(query)
        results = self.memory.search(query, top_k=3)

        return AgentResult(
            agent=self.role,
            status="success",
            data={
                "query": query,
                "answer": answer,
                "sources": [r["entry"] for r in results],
            },
        )


class AgentCoordinator:
    """
    Orchestrates all agents in a collaborative pipeline.
    Runs agents in parallel where possible, sequentially where dependencies exist.
    """

    def __init__(
        self,
        graph=None,
        event_classifier=None,
        prediction_engine=None,
        portfolio_manager=None,
        memory_store=None,
        alert_engine=None,
    ) -> None:
        self.news_agent = NewsAgent()
        self.event_agent = EventAgent(event_classifier)
        self.graph_agent = GraphAgent(graph)
        self.risk_agent = RiskAgent(prediction_engine, alert_engine, graph)
        self.portfolio_agent = PortfolioAgent(portfolio_manager)
        self.research_agent = ResearchAgent(memory_store)

    async def run_pipeline(self, context: dict) -> dict[str, AgentResult]:
        results: dict[str, AgentResult] = {}

        # Phase 1: News + Graph in parallel
        news_result, graph_result = await asyncio.gather(
            self.news_agent.run(context),
            self.graph_agent.run(context),
        )
        results["news"] = news_result
        results["graph"] = graph_result

        # Phase 2: Event classification (depends on news)
        ctx2 = {**context, "articles": news_result.data.get("articles", [])}
        event_result = await self.event_agent.run(ctx2)
        results["events"] = event_result

        # Phase 3: Risk + Portfolio in parallel (depends on graph + events)
        ctx3 = {
            **context,
            "high_severity_events": event_result.data.get("high_severity_events", []),
        }
        risk_result, portfolio_result, research_result = await asyncio.gather(
            self.risk_agent.run(ctx3),
            self.portfolio_agent.run(ctx3),
            self.research_agent.run(ctx3),
        )
        results["risk"] = risk_result
        results["portfolio"] = portfolio_result
        results["research"] = research_result

        logger.info("Agent pipeline complete. Agents run: {}", list(results.keys()))
        return results
