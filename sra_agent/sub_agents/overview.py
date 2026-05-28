"""Overview sub-agent: summarize user input without literature retrieval."""

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


def create_overview_agent(llm: Any) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for the research-content overview."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

请只基于用户输入生成“研究内容概述”。要求：
1. 用 1 段自然语言概括拟研究问题、对象、方向、指标、研究类型和输出用途。
2. 不检索、不引用文献。
3. 如果用户未提供方向、指标或创新点，请用谨慎表达，不要补充不存在的信息。
""".strip()
        output = invoke_llm(llm, "研究内容概述", prompt)
        logger.info(f"Overview: {output}")
        return section_update(state, "overview", output, [])

    return log_agent_node("overview")(node)
