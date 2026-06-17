"""
FEPSIP Scenario Simulation Engine
Allows custom event injection and simulates propagation effects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
from src.utils import logger

SCENARIO_TEMPLATES = {
    "factory_shutdown": {
        "description": "{company} factory shutdown",
        "severity": 0.75,
        "affected_sectors": ["Industrial", "Consumer"],
        "macro_impact": {"supply_chain_stress": +0.3},
    },
    "chip_shortage": {
        "description": "{company} chip shortage",
        "severity": 0.80,
        "affected_sectors": ["Technology", "Automotive", "Consumer"],
        "macro_impact": {"supply_chain_stress": +0.5},
    },
    "oil_price_shock": {
        "description": "Oil price rises {magnitude}%",
        "severity": 0.85,
        "affected_sectors": ["Energy", "Airlines", "Consumer", "Industrial"],
        "macro_impact": {"inflation": +0.2, "energy_costs": +0.4},
    },
    "fed_rate_hike": {
        "description": "Fed raises rates by {magnitude} bps",
        "severity": 0.70,
        "affected_sectors": ["Finance", "RealEstate", "Utilities"],
        "macro_impact": {"interest_rates": +0.3, "credit_costs": +0.2},
    },
    "bank_failure": {
        "description": "{company} bank failure",
        "severity": 0.95,
        "affected_sectors": ["Finance", "RealEstate"],
        "macro_impact": {"credit_stress": +0.8, "systemic_risk": +0.6},
    },
    "geopolitical_crisis": {
        "description": "Geopolitical crisis in {region}",
        "severity": 0.85,
        "affected_sectors": ["Energy", "Defense", "Finance"],
        "macro_impact": {"risk_aversion": +0.5, "commodity_prices": +0.3},
    },
    "pandemic_outbreak": {
        "description": "Pandemic outbreak in {region}",
        "severity": 0.95,
        "affected_sectors": ["Healthcare", "Consumer", "Travel", "Finance"],
        "macro_impact": {"supply_chain_stress": +0.7, "consumer_demand": -0.4},
    },
}

SECTOR_IMPACT_MATRIX = {
    # (event_sector) -> {target_sector: multiplier}
    "Energy": {"Airlines": -0.8, "Industrial": -0.4, "Consumer": -0.3, "Finance": -0.2},
    "Finance": {"RealEstate": -0.6, "Consumer": -0.4, "Industrial": -0.3},
    "Technology": {"Consumer": 0.3, "Industrial": 0.2, "Finance": 0.1},
    "Industrial": {"Consumer": -0.3, "Technology": -0.2},
    "Consumer": {"Technology": -0.2, "Industrial": -0.1},
}


@dataclass
class ScenarioStep:
    step: int
    timestamp: str
    affected_nodes: list[dict]
    risk_delta: float
    description: str


@dataclass
class SimulationResult:
    scenario_name: str
    source: str
    severity: float
    steps: list[ScenarioStep]
    total_affected_tickers: list[str]
    portfolio_impact_pct: float
    peak_systemic_risk: float
    propagation_summary: dict
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "source": self.source,
            "severity": round(self.severity, 3),
            "steps": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "description": s.description,
                    "risk_delta": round(s.risk_delta, 2),
                    "affected_nodes": s.affected_nodes,
                }
                for s in self.steps
            ],
            "total_affected_tickers": self.total_affected_tickers,
            "portfolio_impact_pct": round(self.portfolio_impact_pct, 2),
            "peak_systemic_risk": round(self.peak_systemic_risk, 2),
            "propagation_summary": self.propagation_summary,
            "timestamp": self.timestamp.isoformat(),
        }


class ScenarioSimulator:
    """
    Simulates event propagation through the financial knowledge graph.
    Returns step-by-step ripple effects and risk evolution.
    """

    def __init__(self, graph, prediction_engine, steps: int = 5) -> None:
        self.graph = graph
        self.prediction_engine = prediction_engine
        self.steps = steps

    def simulate(
        self,
        scenario_type: str,
        source_ticker: str,
        magnitude: float = 1.0,
        company: Optional[str] = None,
        region: Optional[str] = None,
    ) -> SimulationResult:
        template = SCENARIO_TEMPLATES.get(scenario_type, SCENARIO_TEMPLATES["factory_shutdown"])
        description = template["description"].format(
            company=company or source_ticker,
            magnitude=f"{magnitude:.0f}",
            region=region or "Global",
        )
        severity = min(template["severity"] * magnitude / 10.0 if magnitude > 10 else template["severity"], 1.0)

        logger.info("Running simulation: {} on {} (severity={:.2f})", scenario_type, source_ticker, severity)

        # Get contagion paths
        paths = self.graph.get_contagion_path(source_ticker, depth=self.steps)
        all_steps = []
        total_risk_delta = 0.0
        affected_tickers = []

        for step_idx in range(1, self.steps + 1):
            step_nodes = [
                (node, path) for node, path in paths.items()
                if len(path) - 1 == step_idx
            ]
            if not step_nodes:
                break

            affected_nodes = []
            step_risk = 0.0

            for node_id, path in step_nodes:
                node_data = self.graph.G.nodes.get(node_id, {})
                decay = 0.6 ** step_idx
                impact = float(-severity * decay * np.random.uniform(0.5, 1.5))
                risk_increase = abs(impact) * 20

                affected_nodes.append({
                    "ticker": node_id,
                    "sector": node_data.get("sector", "Unknown"),
                    "node_type": node_data.get("node_type", "Unknown"),
                    "impact_pct": round(impact * 100, 2),
                    "risk_increase": round(risk_increase, 2),
                    "path": " → ".join(path),
                })
                if node_data.get("node_type") == "Company":
                    affected_tickers.append(node_id)
                step_risk += risk_increase

            all_steps.append(ScenarioStep(
                step=step_idx,
                timestamp=f"T+{step_idx}d",
                affected_nodes=affected_nodes,
                risk_delta=round(step_risk / max(len(affected_nodes), 1), 2),
                description=f"Wave {step_idx}: {len(affected_nodes)} entities affected",
            ))
            total_risk_delta += step_risk

        # Portfolio impact estimate
        portfolio_impact = -severity * 0.15 * 100  # rough -15% max for full severity

        return SimulationResult(
            scenario_name=scenario_type,
            source=source_ticker,
            severity=severity,
            steps=all_steps,
            total_affected_tickers=list(set(affected_tickers)),
            portfolio_impact_pct=round(portfolio_impact, 2),
            peak_systemic_risk=round(min(severity * 100, 100), 2),
            propagation_summary={
                "total_waves": len(all_steps),
                "total_affected": len(affected_tickers),
                "macro_impacts": template.get("macro_impact", {}),
                "affected_sectors": template.get("affected_sectors", []),
            },
        )

    def simulate_from_text(self, text: str, source_ticker: str) -> SimulationResult:
        """Auto-detect scenario type from text and simulate."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["shutdown", "halt", "factory"]):
            stype = "factory_shutdown"
        elif any(w in text_lower for w in ["chip", "shortage", "semiconductor"]):
            stype = "chip_shortage"
        elif any(w in text_lower for w in ["oil", "crude", "opec", "energy"]):
            stype = "oil_price_shock"
        elif any(w in text_lower for w in ["rate", "fed", "federal reserve"]):
            stype = "fed_rate_hike"
        elif any(w in text_lower for w in ["bank", "collapse", "failure"]):
            stype = "bank_failure"
        elif any(w in text_lower for w in ["war", "geopolit", "conflict", "sanction"]):
            stype = "geopolitical_crisis"
        else:
            stype = "factory_shutdown"

        return self.simulate(stype, source_ticker)
