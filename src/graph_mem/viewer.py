"""Embedded HTTP server for the graph viewer.

Starts a lightweight HTTP server in a daemon thread at MCP startup.
Serves a single-page app that visualizes the Neo4j knowledge graph.
"""

import logging
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from graph_mem.config import Settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class _ViewerHandler(SimpleHTTPRequestHandler):
    """Serves static files and injects Neo4j config."""

    def __init__(self, *args, settings: Settings, **kwargs):
        self._settings = settings
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_index()
        else:
            super().do_GET()

    def _serve_index(self):
        index_path = STATIC_DIR / "index.html"
        html = index_path.read_text(encoding="utf-8")
        html = html.replace("__NEO4J_HTTP_URL__", self._settings.neo4j_http_url)
        html = html.replace("__NEO4J_USER__", self._settings.neo4j_user)
        html = html.replace("__NEO4J_PASSWORD__", self._settings.neo4j_password)
        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Silence request logs — they'd go to stderr and pollute MCP stdio
        pass


def start_viewer(settings: Settings) -> None:
    """Start the viewer HTTP server in a daemon thread."""
    handler = partial(_ViewerHandler, settings=settings)
    try:
        server = HTTPServer(("127.0.0.1", settings.viewer_port), handler)
    except OSError as e:
        logger.warning("graph-mem viewer: port %d unavailable (%s), viewer disabled", settings.viewer_port, e)
        return

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("graph-mem viewer running at http://127.0.0.1:%d", settings.viewer_port)
