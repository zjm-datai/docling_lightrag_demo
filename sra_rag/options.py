"""Runtime options for constructing SRA RAG components.

These objects describe values supplied by the host application. They are not
loaded from files or environment variables inside this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SraRagOptions:
    """Options required to create a reusable SRA RAG runtime."""

    working_dir: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    max_token_size: int = 8192
    default_llm_timeout: int = 180
    llm_model_max_async: int = 4
    entity_extract_max_gleaning: int = 1
    max_extract_input_tokens: int = 20480
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    timeout: float = 120.0
    query_max_entity_tokens: int = 2000
    query_max_relation_tokens: int = 2000
    query_max_total_tokens: int = 6000
    query_chunk_top_k: int = 8
    query_enable_rerank: bool = False


def normalize_rag_options(options: SraRagOptions | Mapping[str, Any]) -> SraRagOptions:
    """Normalize host-provided flat or nested options."""
    if isinstance(options, SraRagOptions):
        return options

    llm = _mapping_or_empty(options.get("llm"))
    embedding = _mapping_or_empty(options.get("embedding"))

    values = {
        "working_dir": options.get("working_dir"),
        "llm_base_url": options.get("llm_base_url", llm.get("base_url")),
        "llm_api_key": options.get("llm_api_key", llm.get("api_key")),
        "llm_model": options.get("llm_model", llm.get("model")),
        "embedding_model": options.get("embedding_model", embedding.get("model")),
        "embedding_dim": options.get("embedding_dim", embedding.get("dim")),
        "max_token_size": options.get(
            "max_token_size", embedding.get("max_token_size", 8192)
        ),
        "default_llm_timeout": options.get("default_llm_timeout", 180),
        "llm_model_max_async": options.get("llm_model_max_async", 4),
        "entity_extract_max_gleaning": options.get(
            "entity_extract_max_gleaning", 1
        ),
        "max_extract_input_tokens": options.get("max_extract_input_tokens", 20480),
        "chunk_token_size": options.get("chunk_token_size", 1200),
        "chunk_overlap_token_size": options.get("chunk_overlap_token_size", 100),
        "query_max_entity_tokens": options.get("query_max_entity_tokens", 2000),
        "query_max_relation_tokens": options.get("query_max_relation_tokens", 2000),
        "query_max_total_tokens": options.get("query_max_total_tokens", 6000),
        "query_chunk_top_k": options.get("query_chunk_top_k", 8),
        "query_enable_rerank": options.get("query_enable_rerank", False),
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Missing required SRA RAG option(s): {', '.join(missing)}")

    return SraRagOptions(
        working_dir=str(values["working_dir"]),
        llm_base_url=str(values["llm_base_url"]),
        llm_api_key=str(values["llm_api_key"]),
        llm_model=str(values["llm_model"]),
        embedding_model=str(values["embedding_model"]),
        embedding_dim=int(values["embedding_dim"]),
        max_token_size=int(values["max_token_size"]),
        default_llm_timeout=int(values["default_llm_timeout"]),
        llm_model_max_async=int(values["llm_model_max_async"]),
        entity_extract_max_gleaning=int(values["entity_extract_max_gleaning"]),
        max_extract_input_tokens=int(values["max_extract_input_tokens"]),
        chunk_token_size=int(values["chunk_token_size"]),
        chunk_overlap_token_size=int(values["chunk_overlap_token_size"]),
        query_max_entity_tokens=int(values["query_max_entity_tokens"]),
        query_max_relation_tokens=int(values["query_max_relation_tokens"]),
        query_max_total_tokens=int(values["query_max_total_tokens"]),
        query_chunk_top_k=int(values["query_chunk_top_k"]),
        query_enable_rerank=_to_bool(values["query_enable_rerank"]),
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Nested option groups must be mappings.")
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
