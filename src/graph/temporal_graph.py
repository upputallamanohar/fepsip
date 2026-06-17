"""FEPSIP Temporal Graph Store (torch optional)."""
from __future__ import annotations
from collections import deque
from datetime import datetime
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils import logger


class TemporalGraphStore:
    """Maintains a rolling window of graph snapshots."""

    def __init__(self, max_snapshots: int = 365) -> None:
        self.max_snapshots = max_snapshots
        self._snapshots: deque[tuple[datetime, dict]] = deque(maxlen=max_snapshots)

    def save_snapshot(self, graph) -> None:
        ts = datetime.now()
        self._snapshots.append((ts, graph.to_dict()))
        logger.debug("Graph snapshot saved at {}", ts.isoformat())

    def get_snapshots(self, n: int = 30) -> list[tuple[datetime, dict]]:
        snaps = list(self._snapshots)
        return snaps[-n:] if n < len(snaps) else snaps

    def get_node_feature_series(self, node_id: str, feature: str) -> list[float]:
        values = []
        for _, snapshot in self._snapshots:
            for node in snapshot.get("nodes", []):
                if node.get("id") == node_id:
                    values.append(node.get(feature, 0.0) or 0.0)
                    break
        return values

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)


if TORCH_AVAILABLE:
    class TemporalAttention(nn.Module):
        def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(embed_dim*2, embed_dim))
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            out, _ = self.attn(x, x, x)
            x = self.norm(x + out)
            return self.norm2(x + self.ff(x))

    class GraphSAGELayer(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim * 2, out_dim)
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, node_feats, adj):
            agg = torch.mm(adj, node_feats)
            combined = torch.cat([node_feats, agg], dim=-1)
            return F.relu(self.norm(self.linear(combined)))

    class TemporalGraphNetwork(nn.Module):
        def __init__(self, node_feature_dim=16, hidden_dim=64, output_dim=32,
                     num_gnn_layers=2, seq_len=30, num_attn_heads=4, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
            self.temporal_attn = TemporalAttention(hidden_dim, num_attn_heads, dropout)
            self.gnn_layers = nn.ModuleList([GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
            self.output_head = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))

        def forward(self, node_features_seq, adj_matrices):
            x = self.input_proj(node_features_seq)
            x = self.temporal_attn(x)
            x_last = x[:, -1, :]
            adj_last = adj_matrices[-1]
            for gnn in self.gnn_layers:
                x_last = gnn(x_last, adj_last)
            return self.output_head(x_last)

    class TemporalRiskPredictor(nn.Module):
        def __init__(self, embed_dim=32):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(embed_dim, 16), nn.ReLU(),
                                      nn.Linear(16, 1), nn.Sigmoid())
        def forward(self, embeddings):
            return self.head(embeddings).squeeze(-1) * 100

else:
    # Stub classes when torch not installed
    class TemporalGraphNetwork:
        """Stub: install torch to use TGN."""
        def __init__(self, **kwargs): pass

    class TemporalRiskPredictor:
        """Stub: install torch to use risk predictor."""
        def __init__(self, **kwargs): pass
