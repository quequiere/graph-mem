import json
import logging
import re
import typing
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.llm_client.config import LLMConfig  # type: ignore
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore
from graphiti_core.prompts.models import Message  # type: ignore
from pydantic import BaseModel

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)

# nomic-embed-text produces 768-dimensional vectors (not 1536 like text-embedding-3-small)
NOMIC_EMBED_DIM = 768


def _schema_to_example(schema: dict, defs: dict | None = None) -> object:
    """Convert a JSON Schema to an example object with placeholder values.

    This helps local models understand they should return actual values,
    not echo the schema structure.
    """
    if defs is None:
        defs = schema.get("$defs", {})

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        if ref_name in defs:
            return _schema_to_example(defs[ref_name], defs)
        return {}

    t = schema.get("type")
    if t == "string":
        desc = schema.get("description", "")
        # Return a short placeholder derived from the description
        words = desc.split()[:3] if desc else ["text"]
        return " ".join(words) + "..."
    elif t == "integer":
        return 0
    elif t == "number":
        return 0.0
    elif t == "boolean":
        return True
    elif t == "array":
        items = schema.get("items", {})
        return [_schema_to_example(items, defs)]
    elif t == "object" or "properties" in schema:
        result = {}
        for k, v in schema.get("properties", {}).items():
            result[k] = _schema_to_example(v, defs)
        return result
    elif "anyOf" in schema or "oneOf" in schema:
        options = schema.get("anyOf", schema.get("oneOf", [{}]))
        # Pick the first non-null option
        for opt in options:
            if opt.get("type") != "null":
                return _schema_to_example(opt, defs)
        return None
    else:
        return "..."


class OllamaLLMClient(OpenAIGenericClient):
    """OpenAIGenericClient subclass with improved prompts for local Ollama models.

    Local models (deepseek-r1, qwen2.5, llama3.1) confuse JSON Schema format
    instructions with the desired output format. This subclass replaces schema
    instructions with a concrete example, making the task unambiguous.
    """

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size=None,
    ) -> dict[str, typing.Any]:
        from graphiti_core.llm_client.client import MULTILINGUAL_EXTRACTION_RESPONSES
        from graphiti_core.llm_client.errors import RateLimitError, RefusalError
        from graphiti_core.llm_client.config import ModelSize
        import openai

        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model is not None:
            schema = response_model.model_json_schema()
            example = _schema_to_example(schema)
            example_str = json.dumps(example, indent=2)
            messages[-1].content += (
                f"\n\nIMPORTANT: Respond with a JSON object containing ACTUAL VALUES "
                f"(not the schema). Here is an example of the expected format with "
                f"placeholder values replaced by real data:\n\n{example_str}"
            )

        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        retry_count = 0
        last_error = None

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages,
                    response_model,
                    max_tokens=max_tokens,
                    model_size=model_size or ModelSize.medium,
                )
                return response
            except (RateLimitError, RefusalError):
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                raise
            except Exception as e:
                last_error = e
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f"Max retries exceeded. Last error: {e}")
                    raise
                retry_count += 1
                error_context = (
                    f"The previous response was invalid. Error: {e.__class__.__name__}: {e}. "
                    f"Please provide a valid JSON response with ACTUAL VALUES."
                )
                messages.append(Message(role="user", content=error_context))
                logger.warning(f"Retrying (attempt {retry_count}/{self.MAX_RETRIES}): {e}")

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
    base_url = settings.openai_base_url
    api_key = settings.openai_api_key or 'ollama'
    model = settings.model_name or 'deepseek-r1:7b'
    embedding_model = settings.embedding_model_name or 'nomic-embed-text'

    llm_config = LLMConfig(
        api_key=api_key,
        model=model,
        small_model=model,
        base_url=base_url,
    )
    # OllamaLLMClient: uses example-based prompts instead of raw JSON Schema,
    # which prevents local models from echoing the schema as output.
    llm_client = OllamaLLMClient(config=llm_config)

    # Build embedder with correct base_url at construction time (AsyncOpenAI client is
    # immutable after init, so patching config after the fact doesn't work)
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=api_key,
        embedding_model=embedding_model,
        embedding_dim=NOMIC_EMBED_DIM,
        base_url=base_url,
    ))

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
