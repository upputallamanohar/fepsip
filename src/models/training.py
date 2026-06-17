"""
FEPSIP Model Training Pipeline
Trains PriceEncoder, MultimodalFusion, and TGN models.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.encoders import PriceEncoder, MultimodalFusion
from src.utils import logger


class PriceDataset(Dataset):
    """Sliding-window dataset for price prediction."""

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        seq_len: int = 30,
        horizon: int = 5,
        feature_cols: Optional[list[str]] = None,
    ) -> None:
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_cols = feature_cols or ["open", "high", "low", "close", "volume",
                                              "returns", "volatility_20d", "rsi_14"]
        self.samples: list[tuple[np.ndarray, int]] = []
        self._build_samples(price_data)

    def _build_samples(self, price_data: dict) -> None:
        for ticker, df in price_data.items():
            df = df.dropna(subset=["returns"])
            available = [c for c in self.feature_cols if c in df.columns]
            if len(available) < 4 or len(df) < self.seq_len + self.horizon:
                continue

            feats = df[available].fillna(0).values
            # Normalize per ticker
            means = feats.mean(axis=0, keepdims=True)
            stds = feats.std(axis=0, keepdims=True) + 1e-8
            feats = (feats - means) / stds

            for i in range(len(feats) - self.seq_len - self.horizon):
                window = feats[i: i + self.seq_len]
                future_return = df["returns"].iloc[i + self.seq_len + self.horizon - 1]
                label = 0 if future_return > 0.01 else (2 if future_return < -0.01 else 1)
                # Pad features to fixed size 8
                if window.shape[1] < 8:
                    pad = np.zeros((self.seq_len, 8 - window.shape[1]))
                    window = np.concatenate([window, pad], axis=1)
                elif window.shape[1] > 8:
                    window = window[:, :8]
                self.samples.append((window.astype(np.float32), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)


class ModelTrainer:
    """Trains FEPSIP deep learning models."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ModelTrainer initialized on device={}", self.device)

    def train_price_encoder(
        self,
        price_data: dict[str, pd.DataFrame],
        hidden_dim: int = 128,
        num_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        seq_len: int = 30,
    ) -> PriceEncoder:
        dataset = PriceDataset(price_data, seq_len=seq_len)
        if len(dataset) == 0:
            logger.warning("Empty dataset — returning untrained model")
            return PriceEncoder(hidden_dim=hidden_dim)

        n_val = max(1, int(len(dataset) * 0.15))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

        model = PriceEncoder(input_dim=8, hidden_dim=hidden_dim).to(self.device)
        # Classification head for training
        head = nn.Linear(hidden_dim, 3).to(self.device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            model.train(); head.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                emb = model(x_batch)
                logits = head(emb)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # Validation
            model.eval(); head.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    emb = model(x_batch)
                    logits = head(emb)
                    val_loss += criterion(logits, y_batch).item()
                    correct += (logits.argmax(1) == y_batch).sum().item()
                    total += len(y_batch)

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            acc = correct / total if total > 0 else 0

            if (epoch + 1) % 5 == 0:
                logger.info("Epoch {}/{} | train_loss={:.4f} | val_loss={:.4f} | val_acc={:.3f}",
                            epoch + 1, num_epochs, avg_train, avg_val, acc)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                path = self.checkpoint_dir / "price_encoder_best.pt"
                torch.save(model.state_dict(), path)

        logger.info("PriceEncoder training complete. Best val_loss={:.4f}", best_val_loss)
        return model

    def load_price_encoder(self, hidden_dim: int = 128) -> Optional[PriceEncoder]:
        path = self.checkpoint_dir / "price_encoder_best.pt"
        if not path.exists():
            return None
        model = PriceEncoder(input_dim=8, hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        logger.info("PriceEncoder loaded from {}", path)
        return model

    def evaluate(
        self,
        model: PriceEncoder,
        price_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        dataset = PriceDataset(price_data)
        if len(dataset) == 0:
            return {"accuracy": 0.0, "samples": 0}

        head = nn.Linear(128, 3)
        head.eval()
        loader = DataLoader(dataset, batch_size=128)
        correct = total = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                emb = model(x_batch)
                preds = head(emb).argmax(1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return {"accuracy": correct / total if total > 0 else 0.0, "samples": total}
