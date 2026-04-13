"""Configuration from environment variables and shared constants."""

import os
from dataclasses import dataclass

USER_PROFILE = "user_profile"


@dataclass
class Settings:
    graphiti_url: str = "http://127.0.0.1:8000"
    graphiti_api_key: str | None = None
    graphiti_timeout: float = 60.0


def get_settings() -> Settings:
    return Settings(
        graphiti_url=os.environ.get("GRAPHITI_URL", "http://127.0.0.1:8000"),
        graphiti_api_key=os.environ.get("GRAPHITI_API_KEY"),
        graphiti_timeout=float(os.environ.get("GRAPHITI_TIMEOUT", "60.0")),
    )
