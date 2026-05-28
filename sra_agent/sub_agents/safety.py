"""Safety-analysis sub-agent."""

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


def create_safety_agent(
    llm: Any,
    retrieval_tool: UnifiedRetrievalTool | None,
) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for safety analysis."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        user_input = normalize_input(state)
        if not user_input.include_safety_economics:
            return section_update(
                state,
                "safety",
                "用户已关闭安全性/经济性分析，本模块未展开。",
                [],
            )

        target = user_input.focus_drug_or_formula or user_input.research_goal
        query = " ".join(
            part
            for part in [
                target,
                user_input.research_direction,
                "安全性 不良反应 毒性 禁忌 药物 方剂 疾病",
            ]
            if part
        )
        evidence = retrieve_for_section(
            retrieval_tool,
            query,
            purpose="安全性研究：检索药物/方剂/疾病安全性、不良反应、毒性和禁忌",
            top_k=6,
            mode="hybrid",
        )
        prompt = f"""
用户表单信息：
{form_summary(user_input)}

检索证据：
{evidence_text(evidence)}

请生成“安全性研究”。要求：
1. 如果用户填写了重点药物/方剂，优先围绕该对象分析。
2. 如果未填写药物/方剂，围绕研究疾病或研究方向中可能涉及的干预措施提示安全性关注点。
3. 输出“已知风险”“证据不足处”“后续研究中建议监测的安全性指标”。
4. 不要给出临床用药建议，只做科研设计层面的风险提示。
""".strip()
        output = invoke_llm(llm, "安全性研究", prompt)
        logger.info(f"Safety: {output}")
        return section_update(state, "safety", output, evidence)

    return log_agent_node("safety")(node)
