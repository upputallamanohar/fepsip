"""
FEPSIP - Core Configuration & Settings
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).parent.parent.parent


class AppConfig(BaseModel):
    name: str = "FEPSIP"
    version: str = "1.0.0"
    env: str = "development"
    log_level: str = "INFO"


class MarketConfig(BaseModel):
    default_tickers: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    device: str = "cpu"


class RiskConfig(BaseModel):
    thresholds: dict[str, float] = {"high": 75, "critical": 90}
    contagion_depth: int = 3
    centrality_weight: float = 0.3
    sentiment_weight: float = 0.2
    volatility_weight: float = 0.3
    correlation_weight: float = 0.2


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True


class Settings(BaseSettings):
    """Main application settings loaded from YAML + env vars."""

    config_path: str = str(ROOT_DIR / "config" / "config.yaml")

    # Secrets (env only)
    polygon_api_key: str = ""
    finnhub_api_key: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    qdrant_host: str = "localhost"
    telegram_token: str = ""
    telegram_chat_id: str = ""

    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def load_yaml(self) -> dict[str, Any]:
        path = Path(self.config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @property
    def yaml_config(self) -> dict[str, Any]:
        return self.load_yaml()

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        d = self.yaml_config
        for p in parts:
            if not isinstance(d, dict):
                return default
            d = d.get(p, default)
        return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    import sys
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        ROOT_DIR / "logs" / "fepsip.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )
