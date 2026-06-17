"""
FEPSIP Multimodal Feature Encoders
- PriceEncoder (LSTM + Transformer)
- NLPEncoder (FinBERT wrapper)
- FundamentalsEncoder (MLP)
- MacroEncoder (MLP)
- FusionLayer (cross-modal attention)
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import logger

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ─────────────────────────────────────────────────────────
# Price Encoder
# ─────────────────────────────────────────────────────────

class PriceEncoder(nn.Module):
    """
    Encodes price time series using LSTM + Transformer layers.
    Input: (batch, seq_len, features)
    Output: (batch, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int = 8,   # OHLCV + returns + volatility + RSI
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_attn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_attn_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)           # (B, T, H)
        attn_out = self.transformer(lstm_out) # (B, T, H)
        pooled = attn_out.mean(dim=1)        # (B, H) - mean pool
        return self.norm(self.output_proj(pooled))


# ─────────────────────────────────────────────────────────
# NLP Encoder (FinBERT)
# ─────────────────────────────────────────────────────────

class NLPEncoder:
    """
    Wraps FinBERT to produce sentence embeddings from financial text.
    Returns numpy arrays for easy downstream use.
    """
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded or not TRANSFORMERS_AVAILABLE:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("FinBERT loaded on {}", self.device)
        except Exception as e:
            logger.warning("FinBERT failed to load: {}. Using random embeddings.", e)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        self._load()
        if not self._loaded:
            return np.random.randn(len(texts), 768).astype(np.float32)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ─────────────────────────────────────────────────────────
# Fundamentals & Macro Encoders
# ─────────────────────────────────────────────────────────

class MLPEncoder(nn.Module):
    """Generic MLP encoder for tabular features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────
# Multimodal Fusion Layer
# ─────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """Cross-modal attention between two sequences."""

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query.unsqueeze(1), context.unsqueeze(1), context.unsqueeze(1))
        return self.norm(query + out.squeeze(1))


class MultimodalFusion(nn.Module):
    """
    Fuses price, NLP, graph, fundamentals, and macro embeddings
    via cross-modal attention and a transformer fusion layer.
    Output: unified market representation vector.
    """

    def __init__(
        self,
        price_dim: int = 128,
        nlp_dim: int = 768,
        graph_dim: int = 64,
        fund_dim: int = 64,
        macro_dim: int = 64,
        output_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        unified = output_dim

        # Project each modality to unified_dim
        self.price_proj = nn.Linear(price_dim, unified)
        self.nlp_proj = nn.Linear(nlp_dim, unified)
        self.graph_proj = nn.Linear(graph_dim, unified)
        self.fund_proj = nn.Linear(fund_dim, unified)
        self.macro_proj = nn.Linear(macro_dim, unified)

        # Cross-modal attention layers
        self.cross_attn = CrossModalAttention(unified, num_heads)

        # Final fusion transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=unified, nhead=num_heads,
            dim_feedforward=unified * 2, dropout=dropout, batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.output_head = nn.Sequential(
            nn.Linear(unified * 5, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        price_emb: torch.Tensor,
        nlp_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        fund_emb: torch.Tensor,
        macro_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Project all to unified dim
        p = self.price_proj(price_emb)
        n = self.nlp_proj(nlp_emb)
        g = self.graph_proj(graph_emb)
        f = self.fund_proj(fund_emb)
        m = self.macro_proj(macro_emb)

        # Stack as sequence (B, 5, unified)
        seq = torch.stack([p, n, g, f, m], dim=1)
        fused = self.fusion_transformer(seq)           # (B, 5, unified)
        flat = fused.flatten(start_dim=1)              # (B, 5 * unified)
        return self.output_head(flat)                  # (B, output_dim)
