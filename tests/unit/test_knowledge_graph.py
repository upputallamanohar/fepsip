"""Unit tests for Financial Knowledge Graph."""
import pytest
import pandas as pd
import numpy as np


def test_graph_initialization():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    assert graph.G.number_of_nodes() > 0
    assert graph.G.number_of_edges() > 0


def test_node_types():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph, NodeType
    graph = FinancialKnowledgeGraph()
    node_types = {graph.G.nodes[n].get("node_type") for n in graph.G.nodes}
    assert "Company" in node_types
    assert "Sector" in node_types


def test_add_node():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph, NodeAttrs, NodeType
    graph = FinancialKnowledgeGraph()
    graph.add_node("TEST", NodeAttrs(node_type=NodeType.COMPANY, name="Test Co", ticker="TEST"))
    assert graph.G.has_node("TEST")
    assert graph.G.nodes["TEST"]["ticker"] == "TEST"


def test_update_prices():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    graph.update_node_prices({"AAPL": 189.5, "MSFT": 415.2})
    assert graph.G.nodes["AAPL"]["price"] == 189.5
    assert graph.G.nodes["MSFT"]["price"] == 415.2


def test_correlation_edges():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    initial_edges = graph.G.number_of_edges()

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "AAPL": np.random.randn(100) * 0.02,
        "MSFT": np.random.randn(100) * 0.02,
    }, index=dates)
    graph.update_correlations(returns, threshold=0.0)  # force add edges
    assert graph.G.number_of_edges() >= initial_edges


def test_contagion_path():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    paths = graph.get_contagion_path("AAPL", depth=2)
    assert "AAPL" in paths
    assert paths["AAPL"] == ["AAPL"]
    # Should reach Technology sector
    assert any("Technology" in path for path in paths.values())


def test_centrality_scores():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    centrality = graph.compute_centrality_scores()
    assert "betweenness" in centrality
    assert "degree" in centrality
    assert "pagerank" in centrality
    assert len(centrality["pagerank"]) > 0


def test_systemic_risk_score():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    score = graph.get_systemic_risk_score("AAPL")
    assert 0 <= score <= 100


def test_to_dict():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    d = graph.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert "stats" in d
    assert d["stats"]["num_nodes"] > 0


def test_risk_score_update():
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    graph = FinancialKnowledgeGraph()
    graph.update_risk_scores({"AAPL": 85.0, "MSFT": 42.0})
    assert graph.G.nodes["AAPL"]["risk_score"] == 85.0
