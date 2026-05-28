"""SRA multi-agent report-generation package."""

from __future__ import annotations

from typing import Any
from .orchestrator import generate_research_report

__all__ = [
    "ResearchFormInput",
    "RetrievalEvidence",
    "SRAAgentResult",
    "SRAAgentState",
    "SRAOrchestrator",
    "SraLlmOptions",
    "SraLlmOptionsInput",
    "SraLlmOptionsLike",
    "build_sra_graph",
    "create_chat_openai",
    "create_sra_agent",
    "generate_research_report",
    "normalize_llm_options",
]


def __getattr__(name: str) -> Any:
    """Lazily expose public APIs without loading heavy graph/RAG deps on import."""
    if name in {"ResearchFormInput", "RetrievalEvidence", "SRAAgentState"}:
        from sra_agent import state

        return getattr(state, name)

    if name in {
        "SRAAgentResult",
        "SRAOrchestrator",
        "build_sra_graph",
        "create_chat_openai",
        "create_sra_agent",
        "generate_research_report",
    }:
        from sra_agent import orchestrator

        return getattr(orchestrator, name)

    if name in {
        "SraLlmOptions",
        "SraLlmOptionsInput",
        "SraLlmOptionsLike",
        "normalize_llm_options",
    }:
        from sra_agent import options

        return getattr(options, name)

    raise AttributeError(f"module 'sra_agent' has no attribute {name!r}")
