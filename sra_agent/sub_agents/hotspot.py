"""Research-hotspot sub-agent."""

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


def create_hotspot_agent(
    llm: Any,
    retrieval_tool: UnifiedRetrievalTool | None,
) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for hotspot recommendations."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        query = " ".join(
            part
            for part in [
                user_input.research_goal,
                user_input.research_direction,
                user_input.research_type,
                "2025 2026 研究热点 前沿 多组学 AI 真实世界研究",
            ]
            if part
        )
        evidence = retrieve_for_section(
            retrieval_tool,
            query,
            purpose="热点建议：检索系统热点库和前沿研究方向",
            top_k=6,
            mode="hybrid",
        )
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

检索证据：
{evidence_text(evidence)}

请生成“当前研究热点结合建议”。要求：
1. 给出 3-5 条能增强课题创新性的热点结合建议。
2. 可考虑多组学、AI/机器学习、真实世界研究、机制通路、临床转化等方向。
3. 每条建议说明“结合方式”和“能增强哪类创新性”。
4. 如果检索证据不足，请把建议标注为策略性建议而非已有文献结论。
""".strip()
        output = invoke_llm(llm, "当前研究热点结合建议", prompt)
        logger.info(f"Hotspot: {output}")
        return section_update(state, "hotspot", output, evidence)

    return log_agent_node("hotspot")(node)
