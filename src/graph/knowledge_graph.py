"""
FEPSIP Financial Knowledge Graph
Dynamic knowledge graph of companies, sectors, commodities and their relationships.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import networkx as nx
import numpy as np
import pandas as pd
from src.utils import logger


class NodeType(str, Enum):
    COMPANY = "Company"
    SECTOR = "Sector"
    INDUSTRY = "Industry"
    COMMODITY = "Commodity"
    CURRENCY = "Currency"
    COUNTRY = "Country"
    ETF = "ETF"
    ECONOMIC_INDICATOR = "EconomicIndicator"
    GOVERNMENT_AGENCY = "GovernmentAgency"


class EdgeType(str, Enum):
    SUPPLIES_TO = "SUPPLIES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    COMPETES_WITH = "COMPETES_WITH"
    OWNS = "OWNS"
    BELONGS_TO = "BELONGS_TO"
    EXPOSED_TO = "EXPOSED_TO"
    AFFECTED_BY = "AFFECTED_BY"
    LOCATED_IN = "LOCATED_IN"
    TRADES_WITH = "TRADES_WITH"
    CORRELATED_WITH = "CORRELATED_WITH"


@dataclass
class NodeAttrs:
    node_type: NodeType
    name: str
    ticker: Optional[str] = None
    price: Optional[float] = None
    volatility: Optional[float] = None
    sentiment_score: Optional[float] = None
    risk_score: float = 0.0
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        d["node_type"] = self.node_type.value
        d["last_updated"] = self.last_updated.isoformat()
        return d


@dataclass
class EdgeAttrs:
    edge_type: EdgeType
    weight: float = 1.0
    correlation: Optional[float] = None
    impact_magnitude: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        d["edge_type"] = self.edge_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


class FinancialKnowledgeGraph:
    """Dynamic financial knowledge graph built on NetworkX."""

    SECTOR_MAP: dict[str, list[str]] = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD"],
        "Finance": ["JPM", "GS", "BAC", "WFC", "MS"],
        "Energy": ["XOM", "CVX", "COP", "SLB"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABT"],
        "Consumer": ["AMZN", "TSLA", "WMT", "HD"],
        "Industrial": ["BA", "GE", "CAT", "MMM"],
        "Materials": ["FCX", "NEM", "AA"],
        "Utilities": ["NEE", "DUK"],
        "Telecom": ["VZ", "T", "TMUS"],
    }

    SUPPLY_CHAIN_EDGES: list[tuple[str, str, EdgeType]] = [
        ("AAPL", "NVDA", EdgeType.DEPENDS_ON),
        ("TSLA", "XOM", EdgeType.DEPENDS_ON),
        ("BA", "GE", EdgeType.DEPENDS_ON),
        ("JPM", "GS", EdgeType.COMPETES_WITH),
        ("AAPL", "MSFT", EdgeType.COMPETES_WITH),
        ("GOOGL", "MSFT", EdgeType.COMPETES_WITH),
    ]

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._build_base_graph()
        logger.info("FinancialKnowledgeGraph initialized: {} nodes, {} edges",
                    self.G.number_of_nodes(), self.G.number_of_edges())

    def _build_base_graph(self) -> None:
        for sector, tickers in self.SECTOR_MAP.items():
            self.add_node(sector, NodeAttrs(node_type=NodeType.SECTOR, name=sector))
            for ticker in tickers:
                self.add_node(ticker, NodeAttrs(node_type=NodeType.COMPANY, name=ticker,
                                                 ticker=ticker, sector=sector))
                self.add_edge(ticker, sector, EdgeAttrs(edge_type=EdgeType.BELONGS_TO))

        for name, ntype in [
            ("US_RATES", NodeType.ECONOMIC_INDICATOR),
            ("INFLATION", NodeType.ECONOMIC_INDICATOR),
            ("CRUDE_OIL", NodeType.COMMODITY),
            ("GOLD", NodeType.COMMODITY),
            ("USD_INDEX", NodeType.CURRENCY),
        ]:
            self.add_node(name, NodeAttrs(node_type=ntype, name=name))

        for ticker in self.SECTOR_MAP.get("Energy", []):
            if self.G.has_node(ticker):
                self.add_edge(ticker, "CRUDE_OIL", EdgeAttrs(edge_type=EdgeType.EXPOSED_TO, weight=0.9))

        for src, dst, etype in self.SUPPLY_CHAIN_EDGES:
            if self.G.has_node(src) and self.G.has_node(dst):
                self.add_edge(src, dst, EdgeAttrs(edge_type=etype, weight=0.8))

    def add_node(self, node_id: str, attrs: NodeAttrs) -> None:
        self.G.add_node(node_id, **attrs.to_dict())

    def add_edge(self, src: str, dst: str, attrs: EdgeAttrs) -> None:
        self.G.add_edge(src, dst, **attrs.to_dict())

    def update_node_prices(self, price_data: dict[str, float]) -> None:
        for ticker, price in price_data.items():
            if self.G.has_node(ticker):
                self.G.nodes[ticker]["price"] = price
                self.G.nodes[ticker]["last_updated"] = datetime.now().isoformat()

    def update_correlations(self, returns_matrix: pd.DataFrame, threshold: float = 0.6) -> None:
        corr = returns_matrix.corr()
        added = 0
        tickers = list(corr.columns)
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i >= j:
                    continue
                c = float(corr.iloc[i, j])
                if abs(c) >= threshold and self.G.has_node(t1) and self.G.has_node(t2):
                    self.add_edge(t1, t2, EdgeAttrs(edge_type=EdgeType.CORRELATED_WITH,
                                                     weight=abs(c), correlation=c))
                    added += 1
        logger.info("Added {} correlation edges (threshold={})", added, threshold)

    def get_contagion_path(self, source: str, depth: int = 3) -> dict[str, list]:
        visited: dict[str, int] = {source: 0}
        queue = [source]
        paths: dict[str, list] = {source: [source]}
        while queue:
            node = queue.pop(0)
            if visited[node] >= depth:
                continue
            for neighbor in self.G.successors(node):
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    paths[neighbor] = paths[node] + [neighbor]
                    queue.append(neighbor)
        return paths

    def compute_centrality_scores(self) -> dict[str, dict[str, float]]:
        ug = self.G.to_undirected()
        return {
            "betweenness": nx.betweenness_centrality(ug),
            "degree": dict(nx.degree_centrality(ug)),
            "pagerank": nx.pagerank(self.G, alpha=0.85),
        }

    def get_systemic_risk_score(self, ticker: str) -> float:
        centrality = nx.degree_centrality(self.G.to_undirected()).get(ticker, 0.0)
        node_data = self.G.nodes.get(ticker, {})
        volatility = min(node_data.get("volatility", 0.2) or 0.2, 1.0)
        risk = node_data.get("risk_score", 0.0) / 100
        raw = (0.4 * centrality + 0.3 * volatility + 0.3 * risk) * 100
        return round(min(max(raw, 0), 100), 2)

    def update_risk_scores(self, risk_scores: dict[str, float]) -> None:
        for node, score in risk_scores.items():
            if self.G.has_node(node):
                self.G.nodes[node]["risk_score"] = score

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n, **dict(self.G.nodes[n])} for n in self.G.nodes],
            "edges": [{"source": u, "target": v, **dict(d)} for u, v, d in self.G.edges(data=True)],
            "stats": {"num_nodes": self.G.number_of_nodes(), "num_edges": self.G.number_of_edges()},
        }
