# Extends zepai/graphiti:latest with two fixes:
# 1. uvicorn called directly (uv is missing from the published image)
# 2. zep_graphiti.py patched to use OpenAIGenericClient + properly configured
#    embedder for Ollama (published image ignores OPENAI_BASE_URL for embedder)
FROM zepai/graphiti:latest

COPY zep_graphiti_patch.py /app/graph_service/zep_graphiti.py

CMD ["/app/.venv/bin/uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
