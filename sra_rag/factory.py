"""Factory functions for embedding SRA RAG into host applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sra_rag.indexer.lightrag_indexer import LightRAGIndexer
from sra_rag.options import SraRagOptions, normalize_rag_options
from sra_rag.parser.docling_parser import DoclingParser
from sra_rag.retrieval.lightrag_retrieval import LightRAGRetriever


@dataclass(frozen=True)
class SraRagRuntime:
    """Constructed SRA RAG components for a host application."""

    parser: DoclingParser
    indexer: LightRAGIndexer
    retriever: LightRAGRetriever


def create_sra_rag(options: SraRagOptions | Mapping[str, Any]) -> SraRagRuntime:
    """Create parser, indexer, and retriever from host-provided options."""
    normalized = normalize_rag_options(options)
    parser = DoclingParser()
    indexer = LightRAGIndexer(
        working_dir=normalized.working_dir,
        llm_base_url=normalized.llm_base_url,
        llm_api_key=normalized.llm_api_key,
        llm_model=normalized.llm_model,
        embedding_model=normalized.embedding_model,
        embedding_dim=normalized.embedding_dim,
        max_token_size=normalized.max_token_size,
        default_llm_timeout=normalized.default_llm_timeout,
        llm_model_max_async=normalized.llm_model_max_async,
        entity_extract_max_gleaning=normalized.entity_extract_max_gleaning,
        max_extract_input_tokens=normalized.max_extract_input_tokens,
        chunk_token_size=normalized.chunk_token_size,
        chunk_overlap_token_size=normalized.chunk_overlap_token_size,
    )
    retriever = LightRAGRetriever(
        indexer,
        default_query_options={
            "max_entity_tokens": normalized.query_max_entity_tokens,
            "max_relation_tokens": normalized.query_max_relation_tokens,
            "max_total_tokens": normalized.query_max_total_tokens,
            "chunk_top_k": normalized.query_chunk_top_k,
            "enable_rerank": normalized.query_enable_rerank,
        },
    )
    return SraRagRuntime(parser=parser, indexer=indexer, retriever=retriever)
