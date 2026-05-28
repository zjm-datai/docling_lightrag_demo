"""Innovation-analysis sub-agent."""

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


def create_innovation_agent(
    llm: Any,
    retrieval_tool: UnifiedRetrievalTool | None,
) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for innovation analysis."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        innovation = "、".join(user_input.innovation_points) or "理论创新 方法创新 应用创新"
        query = " ".join(
            part
            for part in [
                user_input.research_goal,
                user_input.research_direction,
                user_input.focus_mechanism,
                user_input.focus_drug_or_formula,
                innovation,
                "创新点 机制 方法 应用 正文 摘要",
            ]
            if part
        )
        evidence = retrieve_for_section(
            retrieval_tool,
            query,
            purpose="创新性分析：检索创新点相关摘要和正文片段",
            top_k=8,
            mode="hybrid",
        )
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

检索证据：
{evidence_text(evidence)}

请生成“创新性分析”，必须包含以下三个小标题：
## 理论创新
## 方法创新
## 应用创新

每个小标题下按三项输出：
- 已有研究：
- 创新程度分析：
- 可行性分析：

要求：如果用户只勾选了部分创新点，也要简要评估未勾选类别是否不适合作为重点。
""".strip()
        output = invoke_llm(llm, "创新性分析", prompt)
        logger.info(f"Innovation: {output}")
        return section_update(state, "innovation", output, evidence)

    return log_agent_node("innovation")(node)
