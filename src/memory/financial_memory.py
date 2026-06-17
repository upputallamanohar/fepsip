"""
FEPSIP Financial Memory
RAG-based retrieval of historical financial events using Qdrant.
Falls back to in-memory store when Qdrant is unavailable.
"""
from __future__ import annotations
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
from src.utils import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed; using in-memory fallback")


@dataclass
class MemoryEntry:
    id: str
    event_type: str
    description: str
    outcome: str
    affected_sectors: list[str]
    impact_magnitude: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "description": self.description,
            "outcome": self.outcome,
            "affected_sectors": self.affected_sectors,
            "impact_magnitude": self.impact_magnitude,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# Pre-seeded historical crises
HISTORICAL_CRISES = [
    MemoryEntry(
        id="crisis_2008",
        event_type="bank_failure",
        description="2008 Global Financial Crisis - Lehman Brothers collapse, subprime mortgage crisis",
        outcome="S&P500 fell 56% peak to trough. Banks froze credit. Global recession followed.",
        affected_sectors=["Finance", "RealEstate", "Consumer", "Industrial"],
        impact_magnitude=0.95,
        timestamp=datetime(2008, 9, 15),
        metadata={"duration_months": 18, "peak_vix": 89.5},
    ),
    MemoryEntry(
        id="covid_2020",
        event_type="pandemic_outbreak",
        description="COVID-19 pandemic - global economic shutdown, supply chain collapse",
        outcome="S&P500 fell 34% in 33 days then recovered to all-time highs within 5 months.",
        affected_sectors=["Travel", "Consumer", "Healthcare", "Technology"],
        impact_magnitude=0.85,
        timestamp=datetime(2020, 2, 20),
        metadata={"duration_months": 2, "peak_vix": 82.7},
    ),
    MemoryEntry(
        id="dotcom_2000",
        event_type="asset_bubble",
        description="Dot-com bubble burst - overvalued tech stocks collapsed",
        outcome="NASDAQ fell 78% from peak. Tech sector took 15 years to recover previous highs.",
        affected_sectors=["Technology", "Finance"],
        impact_magnitude=0.80,
        timestamp=datetime(2000, 3, 10),
        metadata={"duration_months": 30},
    ),
    MemoryEntry(
        id="oil_shock_1973",
        event_type="commodity_shock",
        description="OPEC Oil Embargo 1973 - oil price quadrupled",
        outcome="Inflation surged, recession followed. Airlines and industrials severely impacted.",
        affected_sectors=["Energy", "Airlines", "Industrial", "Consumer"],
        impact_magnitude=0.75,
        timestamp=datetime(1973, 10, 17),
        metadata={"oil_price_change_pct": 300},
    ),
    MemoryEntry(
        id="svb_2023",
        event_type="bank_failure",
        description="Silicon Valley Bank collapse 2023 - bank run on tech-focused bank",
        outcome="Contagion spread to First Republic, Signature Bank. Fed emergency backstop deployed.",
        affected_sectors=["Finance", "Technology"],
        impact_magnitude=0.60,
        timestamp=datetime(2023, 3, 10),
        metadata={"assets_seized_bn": 209},
    ),
    MemoryEntry(
        id="ukraine_war_2022",
        event_type="geopolitical_event",
        description="Russia invades Ukraine 2022 - commodity shock, energy crisis in Europe",
        outcome="Oil and gas prices surged. European equities fell. Defense stocks rallied.",
        affected_sectors=["Energy", "Industrial", "Finance", "Consumer"],
        impact_magnitude=0.70,
        timestamp=datetime(2022, 2, 24),
        metadata={"oil_peak_usd": 130},
    ),
]


class InMemoryVectorStore:
    """Simple cosine-similarity in-memory vector store fallback."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, entry: MemoryEntry, embedding: np.ndarray) -> None:
        entry.embedding = embedding.tolist()
        self._entries.append(entry)
        self._embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-10))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        if not self._embeddings:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores = [float(np.dot(q, e)) for e in self._embeddings]
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._entries[i], s) for i, s in top]


class FinancialMemoryStore:
    """
    RAG-based financial memory using Qdrant (or in-memory fallback).
    Stores and retrieves historical events by semantic similarity.
    """

    COLLECTION = "financial_memory"
    VECTOR_DIM = 128  # Use simple random projections for demo

    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> None:
        self._qdrant: Optional[object] = None
        self._fallback = InMemoryVectorStore()
        self._use_qdrant = False

        if QDRANT_AVAILABLE:
            try:
                self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
                self._qdrant.get_collections()
                self._init_collection()
                self._use_qdrant = True
                logger.info("Qdrant connected at {}:{}", qdrant_host, qdrant_port)
            except Exception as e:
                logger.warning("Qdrant unavailable: {} — using in-memory store", e)

        self._seed_historical_data()

    def _init_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self.COLLECTION not in existing:
            self._qdrant.create_collection(
                self.COLLECTION,
                vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
            )

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Simple deterministic text embedding via hashing (production: use FinBERT)."""
        rng = np.random.RandomState(hash(text) % (2**32))
        return rng.randn(self.VECTOR_DIM).astype(np.float32)

    def _seed_historical_data(self) -> None:
        for crisis in HISTORICAL_CRISES:
            emb = self._text_to_embedding(crisis.description + " " + crisis.outcome)
            self.store(crisis, emb)
        logger.info("Seeded {} historical crises into memory store", len(HISTORICAL_CRISES))

    def store(self, entry: MemoryEntry, embedding: Optional[np.ndarray] = None) -> None:
        if embedding is None:
            embedding = self._text_to_embedding(entry.description)

        if self._use_qdrant:
            self._qdrant.upsert(
                collection_name=self.COLLECTION,
                points=[PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, entry.id)),
                    vector=embedding.tolist(),
                    payload=entry.to_dict(),
                )],
            )
        else:
            self._fallback.add(entry, embedding)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_emb = self._text_to_embedding(query)

        if self._use_qdrant:
            results = self._qdrant.search(
                collection_name=self.COLLECTION,
                query_vector=query_emb.tolist(),
                limit=top_k,
            )
            return [{"entry": r.payload, "score": r.score} for r in results]
        else:
            results = self._fallback.search(query_emb, top_k)
            return [{"entry": e.to_dict(), "score": s} for e, s in results]

    def query(self, question: str) -> str:
        """Answer a natural language query about historical events."""
        results = self.search(question, top_k=3)
        if not results:
            return "No relevant historical events found."

        answer_lines = [f"Based on historical financial events similar to your query: '{question}'\n"]
        for i, r in enumerate(results, 1):
            e = r["entry"]
            answer_lines.append(
                f"{i}. [{e.get('timestamp', 'Unknown date')}] {e.get('description', '')}\n"
                f"   Outcome: {e.get('outcome', '')}\n"
                f"   Affected: {', '.join(e.get('affected_sectors', []))}\n"
                f"   Relevance: {r['score']:.2%}\n"
            )
        return "\n".join(answer_lines)
