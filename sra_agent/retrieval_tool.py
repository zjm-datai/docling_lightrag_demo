"""Unified retrieval adapter used by report-generation agents."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Protocol

from sra_agent.state import RetrievalEvidence

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
MAX_RETRIEVAL_CONTENT_CHARS = 6000
MAX_RETRIEVAL_LOG_CHARS = 1200


class RetrieverLike(Protocol):
    """Minimal protocol implemented by LightRAGRetriever and test doubles."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        **kwargs: Any,
    ) -> list[Any]:
        """Retrieve relevant evidence for a query."""


class UnifiedRetrievalTool:
    """Small compatibility layer around LightRAG retrieval."""

    def __init__(
        self,
        retriever: RetrieverLike | None = None,
        default_mode: str | None = None,
    ):
        self.retriever = retriever
        self.default_mode = default_mode
        self._retrieval_lock = threading.Lock()

    def retrieve(
        self,
        query: str,
        *,
        purpose: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[RetrievalEvidence]:
        """Return evidence with stable metadata regardless of backend shape."""
        if self.retriever is None:
            logger.info(
                "No retriever configured for purpose=%s query=%s", purpose, query)
            return []

        effective_mode = self.default_mode or mode
        logger.info(
            "Unified retrieval start: purpose=%s requested_mode=%s effective_mode=%s "
            "top_k=%s query=%s",
            purpose,
            mode,
            effective_mode,
            top_k,
            query,
        )
        started_at = time.perf_counter()
        logger.info(
            "Unified retrieval waiting for backend lock: purpose=%s", purpose)
        with self._retrieval_lock:
            logger.info(
                "Unified retrieval acquired backend lock: purpose=%s", purpose)
            raw_results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                mode=effective_mode,
            )
            logger.info(
                "Unified retrieval released backend call: purpose=%s", purpose)
            logger.info(f"retrieval raw_results: {raw_results}")
        evidence: list[RetrievalEvidence] = []
        for item in raw_results:
            content = getattr(item, "content", None)
            metadata = getattr(item, "metadata", None)
            score = getattr(item, "score", 1.0)
            if isinstance(item, dict):
                content = item.get("content") or item.get(
                    "answer") or str(item)
                metadata = item.get("metadata", item)
                score = item.get("score", score)

            normalized_content = _sanitize_retrieval_content(
                str(content or ""))
            evidence.append(
                RetrievalEvidence(
                    query=query,
                    content=normalized_content,
                    source=str((metadata or {}).get("retriever", "lightrag")),
                    mode=str((metadata or {}).get("mode", effective_mode)),
                    score=float(score or 0.0),
                    metadata={"purpose": purpose, **dict(metadata or {})},
                )
            )

        for index, item in enumerate(evidence, 1):
            preview = item.content[:MAX_RETRIEVAL_LOG_CHARS]
            if len(item.content) > MAX_RETRIEVAL_LOG_CHARS:
                preview += "\n...[retrieval content truncated in logs]..."
            logger.info(
                "Unified retrieval evidence %s/%s: purpose=%s source=%s mode=%s "
                "score=%.3f chars=%s content=\n%s",
                index,
                len(evidence),
                purpose,
                item.source,
                item.mode,
                item.score,
                len(item.content),
                preview,
            )

        elapsed = time.perf_counter() - started_at
        total_chars = sum(len(item.content) for item in evidence)
        logger.info(
            "Unified retrieval done: purpose=%s elapsed=%.2fs raw_count=%s "
            "evidence_count=%s evidence_chars=%s",
            purpose,
            elapsed,
            len(raw_results),
            len(evidence),
            total_chars,
        )
        return evidence


def _sanitize_retrieval_content(content: str) -> str:
    text = _THINK_BLOCK_RE.sub("", content)
    text = _THINK_TAG_RE.sub("", text).strip()
    if len(text) <= MAX_RETRIEVAL_CONTENT_CHARS:
        return text
    return text[:MAX_RETRIEVAL_CONTENT_CHARS] + "\n\n[检索内容过长，已截断。]"
