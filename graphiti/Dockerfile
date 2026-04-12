# Extends zepai/graphiti:latest with patched zep_graphiti.py that properly
# configures the embedder and cross-encoder from env vars (the stock image
# only configures the LLM client, leaving the embedder on OpenAI defaults).
FROM zepai/graphiti:latest

COPY zep_graphiti.py /app/graph_service/zep_graphiti.py
COPY ingest.py /app/graph_service/routers/ingest.py
COPY entities.py /app/graph_service/entities.py

CMD ["/app/.venv/bin/uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
