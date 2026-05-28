"""Conclusion sub-agent."""

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


def create_conclusion_agent(llm: Any) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for conclusion and paper-title recommendation."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        sections = state.get("report_sections") or {}
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

创新性分析：
{sections.get("innovation", "尚无创新性分析。")}

可研究点及科研思路：
{sections.get("research_ideas", "尚无可研究点。")}

安全性研究：
{sections.get("safety", "尚无安全性分析。")}

经济性研究：
{sections.get("economics", "尚无经济性分析。")}

热点建议：
{sections.get("hotspot", "尚无热点建议。")}

请生成“结论与优先推荐方向”。要求：
1. 综合前序模块，给出 1-2 个最值得优先开展的研究方向。
2. 每个方向最终输出为论文标题形式。
3. 每个标题后用 2-3 句话说明推荐理由，包括创新性、可行性、安全/经济性注意点。
4. 不要再新增未被前序内容支撑的新方向。
""".strip()
        output = invoke_llm(llm, "结论与优先推荐方向", prompt)
        logger.info(f"Conclusion: {output}")
        return section_update(state, "conclusion", output, [])

    return log_agent_node("conclusion")(node)
