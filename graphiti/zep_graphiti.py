"""Patched zep_graphiti.py for OpenRouter / custom OpenAI-compatible backends.

Fixes two issues in the stock Graphiti server:
1. Embedder and cross-encoder are not configured from env vars (only LLM client is)
2. Many models (Gemma, Qwen, Llama, etc.) echo JSON Schema instead of returning values —
   ExampleLLMClient replaces schema instructions with concrete examples.
"""

import json
import logging
import os
import typing
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)

# Embedding dimension for Neo4j vector index. graphiti_core truncates client-side
# via [:embedding_dim], so this can be smaller than the model's native output.
EMBEDDING_DIM = 1024


def _schema_to_example(schema: dict, defs: dict | None = None) -> object:
    """Convert a JSON Schema to an example object with placeholder values."""
    if defs is None:
        defs = schema.get("$defs", {})

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        if ref_name in defs:
            return _schema_to_example(defs[ref_name], defs)
        return {}

    t = schema.get("type")
    if t == "string":
        words = (schema.get("description", "") or "text").split()[:3]
        return " ".join(words) + "..."
    elif t == "integer":
        return 0
    elif t == "number":
        return 0.0
    elif t == "boolean":
        return True
    elif t == "array":
        return [_schema_to_example(schema.get("items", {}), defs)]
    elif t == "object" or "properties" in schema:
        return {k: _schema_to_example(v, defs) for k, v in schema.get("properties", {}).items()}
    elif "anyOf" in schema or "oneOf" in schema:
        for opt in schema.get("anyOf", schema.get("oneOf", [{}])):
            if opt.get("type") != "null":
                return _schema_to_example(opt, defs)
        return None
    return "..."


class ExampleLLMClient(OpenAIGenericClient):
    """LLM client that converts JSON Schema prompts to example-based prompts.

    Many models (Gemma, Qwen, Llama, etc.) echo the JSON Schema structure as
    their output instead of returning actual values. This client appends a
    concrete example to the prompt, making the expected output unambiguous.
    """

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size=None,
    ) -> dict[str, typing.Any]:
        from graphiti_core.llm_client.client import MULTILINGUAL_EXTRACTION_RESPONSES
        from graphiti_core.llm_client.config import ModelSize
        from graphiti_core.llm_client.errors import RateLimitError, RefusalError
        import openai

        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model is not None:
            schema = response_model.model_json_schema()
            example = _schema_to_example(schema)
            example_str = json.dumps(example, indent=2)
            messages[-1].content += (
                f"\n\nIMPORTANT: Respond with a JSON object containing ACTUAL VALUES "
                f"(not the schema). Example format:\n\n{example_str}"
            )

        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        retry_count = 0
        last_error = None

        while retry_count <= self.MAX_RETRIES:
            try:
                return await self._generate_response(
                    messages, response_model, max_tokens=max_tokens,
                    model_size=model_size or ModelSize.medium,
                )
            except (RateLimitError, RefusalError):
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                raise
            except Exception as e:
                last_error = e
                if retry_count >= self.MAX_RETRIES:
                    raise
                retry_count += 1
                messages.append(Message(
                    role="user",
                    content=f"Invalid response: {e.__class__.__name__}: {e}. Please provide valid JSON with ACTUAL VALUES.",
                ))
                logger.warning(f"Retrying ({retry_count}/{self.MAX_RETRIES}): {e}")

        raise last_error or Exception("Max retries exceeded")


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        embedder=None,
        cross_encoder=None,
    ):
        super().__init__(uri, user, password, llm_client, embedder=embedder, cross_encoder=cross_encoder)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(name=name, uuid=uuid, group_id=group_id, summary=summary)
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            return await EntityEdge.get_by_uuid(self.driver, uuid)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])
        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)
        for node in nodes:
            await node.delete(self.driver)
        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


async def get_graphiti(settings: ZepEnvDep):
    # LLM config (from settings — stock Graphiti path)
    llm_api_key = settings.openai_api_key
    llm_base_url = settings.openai_base_url
    llm_model = settings.model_name

    # Embedder config (decoupled — read directly from env, no fallback).
    # KeyError here is intentional: missing EMBEDDING_* vars must fail loud
    # at startup rather than silently reusing the LLM provider.
    embed_api_key = os.environ["EMBEDDING_API_KEY"]
    embed_base_url = os.environ["EMBEDDING_BASE_URL"]
    embed_model = os.environ["EMBEDDING_MODEL_NAME"]

    llm_config = LLMConfig(
        api_key=llm_api_key,
        model=llm_model,
        small_model=llm_model,
        base_url=llm_base_url,
    )
    llm_client = ExampleLLMClient(config=llm_config)

    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=embed_api_key,
        embedding_model=embed_model,
        embedding_dim=EMBEDDING_DIM,
        base_url=embed_base_url,
    ))

    # Cross-encoder stays coupled to the LLM client: it reranks via text
    # generation, not embeddings, so it's consistent for it to follow the
    # LLM mode (remote or local).
    cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)

    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
