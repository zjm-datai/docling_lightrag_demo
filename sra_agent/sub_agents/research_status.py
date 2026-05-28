"""Research-status sub-agent."""

from __future__ import annotations

import logging
from typing import Any, Callable

from sra_agent.retrieval_tool import UnifiedRetrievalTool
from sra_agent.state import SRAAgentState
from sra_agent.sub_agents.common import (
    evidence_text,
    form_summary,
    invoke_llm,
    log_agent_node,
    normalize_input,
    retrieve_for_section,
    section_update,
)

logger = logging.getLogger(__name__)


def create_research_status_agent(
    llm: Any,
    retrieval_tool: UnifiedRetrievalTool | None,
) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for research-status analysis."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        query_parts = [
            user_input.research_goal,
            user_input.research_direction,
            user_input.outcome_indicators,
            "研究现状 标题 摘要 年份 关键词 热点 分歧",
        ]
        query = " ".join(part for part in query_parts if part)
        evidence = retrieve_for_section(
            retrieval_tool,
            query,
            purpose="研究现状：检索文献元数据、摘要、关键词和趋势线索",
            top_k=8,
            mode="hybrid",
        )
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

检索证据：
{evidence_text(evidence)}

请生成“研究现状总结”，必须按以下小标题输出：
## 已被充分研究的内容
## 正在升温的热点内容
## 证据较少但有潜力的内容
## 不同研究之间的分歧点

要求：
- 优先利用题名、摘要、年份、关键词等元数据信息归纳。
- 每个小标题下给出 2-4 条要点。
- 对证据不足的判断单独说明，不要虚构确定性结论。
""".strip()
        output = invoke_llm(llm, "研究现状总结", prompt)
        logger.info(f"Research Status: {output}")
        return section_update(state, "research_status", output, evidence)

    return log_agent_node("research_status")(node)
