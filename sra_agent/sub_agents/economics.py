"""Economics-analysis sub-agent."""

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


def create_economics_agent(
    llm: Any,
    retrieval_tool: UnifiedRetrievalTool | None,
) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for economics analysis."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        if not user_input.include_safety_economics:
            return section_update(
                state,
                "economics",
                "用户已关闭安全性/经济性分析，本模块未展开。",
                [],
            )

        query = " ".join(
            part
            for part in [
                user_input.research_goal,
                user_input.focus_drug_or_formula,
                "经济学评价 成本 疾病负担 成本效果 成本效益",
            ]
            if part
        )
        evidence = retrieve_for_section(
            retrieval_tool,
            query,
            purpose="经济性研究：检索成本、疾病负担和药物经济学评价证据",
            top_k=6,
            mode="hybrid",
        )
        logger.info("Economics retrieval complete; building prompt")
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

检索证据：
{evidence_text(evidence)}

请生成“经济性研究”。要求：
1. 分析疾病负担、干预成本、潜在经济价值。
2. 如果证据不足，明确建议后续开展药物经济学评价或真实世界成本研究。
3. 输出结构包含“已有经济学线索”“成本效益假设”“可加入课题设计的经济学指标”。
""".strip()
        output = invoke_llm(llm, "经济性研究", prompt)
        logger.info(f"Economics: {output}")
        return section_update(state, "economics", output, evidence)

    return log_agent_node("economics")(node)
