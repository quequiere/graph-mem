"""Async HTTP client for the Graphiti REST API."""

from typing import Any

import httpx


class GraphitiClient:
    """Wraps all Graphiti REST API calls."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=30.0)

    async def healthcheck(self) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get("/healthcheck")
            resp.raise_for_status()
            return resp.json()

    async def add_messages(
        self,
        group_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/messages",
                json={"group_id": group_id, "messages": messages},
            )
            resp.raise_for_status()
            return resp.json()

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_facts: int = 10,
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/search",
                json={"query": query, "group_ids": group_ids, "max_facts": max_facts},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_memory(
        self,
        group_id: str,
        messages: list[dict[str, Any]],
        max_facts: int = 10,
        center_node_uuid: str | None = None,
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/get-memory",
                json={
                    "group_id": group_id,
                    "max_facts": max_facts,
                    "center_node_uuid": center_node_uuid,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def add_entity_node(
        self,
        uuid: str,
        group_id: str,
        name: str,
        summary: str = "",
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/entity-node",
                json={"uuid": uuid, "group_id": group_id, "name": name, "summary": summary},
            )
            resp.raise_for_status()
            return resp.json()

    async def delete_group(self, group_id: str) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.delete(f"/group/{group_id}")
            resp.raise_for_status()
            return resp.json()

    async def get_episodes(
        self,
        group_id: str,
        last_n: int = 10,
    ) -> Any:
        async with self._client() as client:
            resp = await client.get(f"/episodes/{group_id}", params={"last_n": last_n})
            resp.raise_for_status()
            return resp.json()
