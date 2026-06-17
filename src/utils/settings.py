"""FEPSIP Core Settings."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from pydantic_settings import BaseSettings

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _load_yaml() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        content = os.path.expandvars(f.read())
    return yaml.safe_load(content)


class Settings(BaseSettings):
    app_name: str = "FEPSIP"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_port: int = 8501

    finnhub_api_key: str = ""
    polygon_api_key: str = ""
    hf_token: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    qdrant_host: str = "localhost"
    telegram_token: str = ""
    telegram_chat_id: str = ""

    _yaml: dict[str, Any] = {}

    model_config = {"env_file": ".env", "extra": "allow",
                    "env_prefix": "", "populate_by_name": True}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, '_yaml', _load_yaml())

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val: Any = self._yaml
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
