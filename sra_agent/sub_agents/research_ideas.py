"""Research-ideas sub-agent."""

from __future__ import annotations

import logging
from typing import Any, Callable

from sra_agent.state import SRAAgentState
from sra_agent.sub_agents.common import (
    form_summary,
    invoke_llm,
    log_agent_node,
    normalize_input,
    section_update,
)

logger = logging.getLogger(__name__)


def create_research_ideas_agent(llm: Any) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for specific research ideas."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        sections = state.get("report_sections") or {}
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

前序模块：研究现状总结
{sections.get("research_status", "尚无研究现状总结。")}

前序模块：创新性分析
{sections.get("innovation", "尚无创新性分析。")}

请生成“可研究点及科研思路”。要求：
1. 生成 3-5 个具体可研究点。
2. 每个可研究点必须包含“研究问题”“推荐思路”“关键指标/数据”“适配用途”。
3. 重点利用研究现状中的潜力内容、分歧点，以及创新性分析中的可行方向。
4. 按用户选择的研究类型（药理机制/方剂组合规律）调整实验或数据分析思路。
""".strip()
        output = invoke_llm(llm, "可研究点及科研思路", prompt)
        logger.info(f"Research Ideas: {output}")
        return section_update(state, "research_ideas", output, [])

    return log_agent_node("research_ideas")(node)
