"""Reviewer sub-agent and final report assembler."""

from __future__ import annotations

import logging
from typing import Any, Callable

from sra_agent.state import SRAAgentState
from sra_agent.sub_agents.common import invoke_llm, log_agent_node

logger = logging.getLogger(__name__)

REPORT_ORDER = [
    ("overview", "1. 研究内容概述"),
    ("research_status", "2. 研究现状总结"),
    ("innovation", "3. 创新性分析"),
    ("research_ideas", "4. 可研究点及科研思路"),
    ("safety", "5. 安全性研究"),
    ("economics", "6. 经济性研究"),
    ("hotspot", "7. 当前研究热点结合建议"),
    ("conclusion", "8. 结论与优先推荐方向"),
]


def assemble_report(sections: dict[str, str], review_notes: str = "") -> str:
    """Assemble the final report in the product-approved order."""
    logger.info(
        "Assembling report: sections=%s has_review=%s",
        sorted(sections.keys()),
        bool(review_notes.strip()),
    )
    parts = ["# 科研创新性分析报告"]
    for key, title in REPORT_ORDER:
        content = sections.get(key, "").strip() or "本模块暂无输出。"
        parts.append(f"## {title}\n{content}")
    if review_notes.strip():
        parts.append(f"## Reviewer Agent 事实校验、引用检查、缺口提示\n{review_notes.strip()}")
    report = "\n\n".join(parts)
    logger.info("Report assembled: chars=%s", len(report))
    return report


def create_reviewer_agent(llm: Any) -> Callable[[SRAAgentState], dict[str, Any]]:
    """Create the LangGraph node for fact/citation gap review."""

    def node(state: SRAAgentState) -> dict[str, Any]:
        sections = state.get("report_sections") or {}
        contexts = state.get("retrieved_contexts") or {}
        draft = assemble_report(sections)
        evidence_index = "\n".join(
            f"- {section}: {len(items)} 条检索上下文"
            for section, items in contexts.items()
        ) or "无检索上下文。"
        prompt = f"""
以下是待审核报告草稿：
{draft}

检索上下文概况：
{evidence_index}

请作为 Reviewer Agent 输出审核意见，聚焦：
1. 事实校验风险：哪些判断需要更多文献支撑。
2. 引用检查：哪些模块已有检索证据，哪些模块没有证据或证据不足。
3. 缺口提示：下一步应补充哪些文献或数据。

要求：只输出审核意见，不要重写全文。
""".strip()
        review_notes = invoke_llm(llm, "Reviewer Agent", prompt)
        return {
            "review_notes": review_notes,
            "final_report": assemble_report(sections, review_notes),
        }

    return log_agent_node("reviewer")(node)
