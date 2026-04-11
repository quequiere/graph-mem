"""Configuration from environment variables and shared constants."""

import os
from dataclasses import dataclass

USER_PROFILE = "user_profile"


@dataclass
class Settings:
    graphiti_url: str = "http://127.0.0.1:8000"
    graphiti_api_key: str | None = None
    viewer_port: int = 8050
    neo4j_http_url: str = "http://127.0.0.1:7474"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphiti"


def get_settings() -> Settings:
    return Settings(
        graphiti_url=os.environ.get("GRAPHITI_URL", "http://127.0.0.1:8000"),
        graphiti_api_key=os.environ.get("GRAPHITI_API_KEY"),
        viewer_port=int(os.environ.get("GRAPH_MEM_VIEWER_PORT", "8050")),
        neo4j_http_url=os.environ.get("NEO4J_HTTP_URL", "http://127.0.0.1:7474"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "graphiti"),
    )
