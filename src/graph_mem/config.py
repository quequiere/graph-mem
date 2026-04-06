"""Configuration from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    graphiti_url: str = "http://127.0.0.1:8000"
    graphiti_api_key: str | None = None


def get_settings() -> Settings:
    return Settings(
        graphiti_url=os.environ.get("GRAPHITI_URL", "http://127.0.0.1:8000"),
        graphiti_api_key=os.environ.get("GRAPHITI_API_KEY"),
    )
