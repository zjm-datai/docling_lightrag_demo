"""LangGraph orchestrator for the SRA multi-agent report system."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from sra_agent.options import SraLlmOptionsInput, normalize_llm_options
from sra_agent.retrieval_tool import RetrieverLike, UnifiedRetrievalTool
from sra_agent.state import ResearchFormInput, SRAAgentState
from sra_agent.sub_agents.conclusion import create_conclusion_agent
from sra_agent.sub_agents.economics import create_economics_agent
from sra_agent.sub_agents.hotspot import create_hotspot_agent
from sra_agent.sub_agents.innovation import create_innovation_agent
from sra_agent.sub_agents.overview import create_overview_agent
from sra_agent.sub_agents.research_ideas import create_research_ideas_agent
from sra_agent.sub_agents.research_status import create_research_status_agent
from sra_agent.sub_agents.reviewer import create_reviewer_agent
from sra_agent.sub_agents.safety import create_safety_agent

logger = logging.getLogger(__name__)


@dataclass
class SRAAgentResult:
    """Structured result returned by the report orchestrator."""

    final_report: str
    sections: dict[str, str]
    retrieved_contexts: dict[str, list[dict[str, Any]]]
    review_notes: str = ""
    raw_state: dict[str, Any] | None = None


def create_chat_openai(
    options: SraLlmOptionsInput,
    **kwargs: Any,
) -> Any:
    """Create ChatOpenAI from host-provided OpenAI-compatible options."""
    normalized = normalize_llm_options(options)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required to create the default ChatOpenAI model. "
            "Run `uv sync` after installing project dependencies."
        ) from exc

    logger.info(
        "Creating ChatOpenAI: model=%s base_url=%s temperature=%s",
        normalized.model,
        normalized.base_url,
        kwargs.get("temperature", normalized.temperature),
    )
    return ChatOpenAI(
        model=normalized.model,
        base_url=normalized.base_url,
        api_key=normalized.api_key,
        temperature=kwargs.pop("temperature", normalized.temperature),
        timeout=kwargs.pop("timeout", normalized.timeout),
        **kwargs,
    )


class SRAOrchestrator:
    """Compile and run the SRA LangGraph multi-agent workflow."""

    def __init__(
        self,
        llm: Any | None = None,
        retriever: RetrieverLike | None = None,
        llm_options: SraLlmOptionsInput | None = None,
        retrieval_mode: str | None = "naive",
    ):
        if llm is None and llm_options is None:
            raise ValueError("Either llm or llm_options must be provided.")

        self.llm = llm or create_chat_openai(llm_options)
        self.retrieval_tool = UnifiedRetrievalTool(
            retriever,
            default_mode=retrieval_mode,
        )
        logger.info(
            "Initializing SRAOrchestrator: llm=%s retriever=%s retrieval_mode=%s",
            type(self.llm).__name__,
            type(retriever).__name__ if retriever else "None",
            retrieval_mode or "sub_agent_default",
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        logger.info("Building SRA LangGraph workflow")
        builder = StateGraph(SRAAgentState)

        builder.add_node("overview", create_overview_agent(self.llm))
        builder.add_node(
            "research_status",
            create_research_status_agent(self.llm, self.retrieval_tool),
        )
        builder.add_node(
            "innovation",
            create_innovation_agent(self.llm, self.retrieval_tool),
        )
        builder.add_node("research_ideas", create_research_ideas_agent(self.llm))
        builder.add_node("safety", create_safety_agent(self.llm, self.retrieval_tool))
        builder.add_node(
            "economics",
            create_economics_agent(self.llm, self.retrieval_tool),
        )
        builder.add_node("hotspot", create_hotspot_agent(self.llm, self.retrieval_tool))
        builder.add_node("conclusion", create_conclusion_agent(self.llm))
        builder.add_node("reviewer", create_reviewer_agent(self.llm))

        # Run nodes sequentially so RAG retrieval, LLM generation, and logs are
        # easy to follow in local/debug runs.
        builder.add_edge(START, "overview")
        builder.add_edge("overview", "research_status")
        builder.add_edge("research_status", "innovation")
        builder.add_edge("innovation", "research_ideas")
        builder.add_edge("research_ideas", "safety")
        builder.add_edge("safety", "economics")
        builder.add_edge("economics", "hotspot")
        builder.add_edge("hotspot", "conclusion")
        builder.add_edge("conclusion", "reviewer")
        builder.add_edge("reviewer", END)

        graph = builder.compile()
        logger.info("SRA LangGraph workflow compiled")
        return graph

    def run(self, user_input: dict[str, Any] | ResearchFormInput) -> SRAAgentResult:
        """Run the graph and return the final research innovation report."""
        normalized = ResearchFormInput.from_mapping(user_input)
        logger.info(
            "SRA report run start: goal=%s type=%s purpose=%s safety_economics=%s",
            normalized.research_goal,
            normalized.research_type,
            normalized.output_purpose,
            normalized.include_safety_economics,
        )
        initial_state: SRAAgentState = {
            "user_input": normalized.to_dict(),
            "report_sections": {},
            "retrieved_contexts": {},
        }
        started_at = time.perf_counter()
        state = self.graph.invoke(initial_state)
        elapsed = time.perf_counter() - started_at
        section_count = len(state.get("report_sections", {}))
        context_count = sum(
            len(items) for items in state.get("retrieved_contexts", {}).values()
        )
        final_report = state.get("final_report", "")
        logger.info(
            "SRA report run done: elapsed=%.2fs sections=%s contexts=%s "
            "final_report_chars=%s",
            elapsed,
            section_count,
            context_count,
            len(final_report),
        )
        return SRAAgentResult(
            final_report=final_report,
            sections=state.get("report_sections", {}),
            retrieved_contexts=state.get("retrieved_contexts", {}),
            review_notes=state.get("review_notes", ""),
            raw_state=dict(state),
        )


def build_sra_graph(
    llm: Any | None = None,
    retriever: RetrieverLike | None = None,
    llm_options: SraLlmOptionsInput | None = None,
    retrieval_mode: str | None = "naive",
) -> Any:
    """Build and return the compiled LangGraph app."""
    return SRAOrchestrator(
        llm=llm,
        retriever=retriever,
        llm_options=llm_options,
        retrieval_mode=retrieval_mode,
    ).graph


def create_sra_agent(
    *,
    llm: Any | None = None,
    retriever: RetrieverLike | None = None,
    llm_options: SraLlmOptionsInput | None = None,
    retrieval_mode: str | None = "naive",
) -> SRAOrchestrator:
    """Create the report orchestrator from injected host dependencies."""
    return SRAOrchestrator(
        llm=llm,
        retriever=retriever,
        llm_options=llm_options,
        retrieval_mode=retrieval_mode,
    )


def generate_research_report(
    user_input: dict[str, Any] | ResearchFormInput,
    *,
    llm: Any | None = None,
    retriever: RetrieverLike | None = None,
    llm_options: SraLlmOptionsInput | None = None,
    retrieval_mode: str | None = "naive",
) -> SRAAgentResult:
    """Convenience API for backend handlers."""
    orchestrator = SRAOrchestrator(
        llm=llm,
        retriever=retriever,
        llm_options=llm_options,
        retrieval_mode=retrieval_mode,
    )
    return orchestrator.run(user_input)
