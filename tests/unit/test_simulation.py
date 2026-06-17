"""Unit tests for Scenario Simulator."""
import pytest


def test_simulator_basic():
    from src.simulation.scenario_simulator import ScenarioSimulator, SCENARIO_TEMPLATES
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    from src.prediction.predictor import PredictionEngine

    graph = FinancialKnowledgeGraph()
    engine = PredictionEngine()
    sim = ScenarioSimulator(graph, engine, steps=3)
    result = sim.simulate("factory_shutdown", "TSLA", magnitude=1.0)
    assert result.scenario_name == "factory_shutdown"
    assert result.source == "TSLA"
    assert 0.0 <= result.severity <= 1.0
    assert isinstance(result.steps, list)


def test_simulator_all_templates():
    from src.simulation.scenario_simulator import ScenarioSimulator, SCENARIO_TEMPLATES
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    from src.prediction.predictor import PredictionEngine

    graph = FinancialKnowledgeGraph()
    engine = PredictionEngine()
    sim = ScenarioSimulator(graph, engine, steps=2)

    for scenario_name in SCENARIO_TEMPLATES:
        result = sim.simulate(scenario_name, "AAPL")
        assert result is not None
        assert result.scenario_name == scenario_name


def test_simulate_from_text():
    from src.simulation.scenario_simulator import ScenarioSimulator
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    from src.prediction.predictor import PredictionEngine

    graph = FinancialKnowledgeGraph()
    engine = PredictionEngine()
    sim = ScenarioSimulator(graph, engine)
    result = sim.simulate_from_text("Tesla factory halt in Germany", "TSLA")
    assert result is not None


def test_simulation_result_to_dict():
    from src.simulation.scenario_simulator import ScenarioSimulator
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    from src.prediction.predictor import PredictionEngine

    graph = FinancialKnowledgeGraph()
    engine = PredictionEngine()
    sim = ScenarioSimulator(graph, engine, steps=2)
    result = sim.simulate("oil_price_shock", "XOM", magnitude=5.0)
    d = result.to_dict()
    assert "scenario_name" in d
    assert "severity" in d
    assert "steps" in d
    assert "portfolio_impact_pct" in d


def test_high_magnitude_increases_severity():
    from src.simulation.scenario_simulator import ScenarioSimulator
    from src.graph.knowledge_graph import FinancialKnowledgeGraph
    from src.prediction.predictor import PredictionEngine

    graph = FinancialKnowledgeGraph()
    engine = PredictionEngine()
    sim = ScenarioSimulator(graph, engine, steps=2)
    r_low = sim.simulate("oil_price_shock", "XOM", magnitude=1.0)
    r_high = sim.simulate("oil_price_shock", "XOM", magnitude=10.0)
    assert r_high.severity >= r_low.severity
