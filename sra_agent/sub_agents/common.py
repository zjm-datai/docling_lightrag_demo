"""Common helpers for report-generation sub-agents."""

from __future__ import annotations

import logging
import re
import time
from functools import wraps
from typing import Any

from sra_agent.retrieval_tool import UnifiedRetrievalTool
from sra_agent.state import ResearchFormInput, RetrievalEvidence, SRAAgentState

logger = logging.getLogger(__name__)

MAX_EVIDENCE_CHARS_PER_ITEM = 4000
MAX_PROMPT_LOG_CHARS = 1200


SYSTEM_PROMPT = (
    "你是科研创新性分析系统中的专业子智能体。"
    "请用中文输出，语气专业、谨慎，严格围绕给定用户输入和检索证据。"
    "如果证据不足，请明确指出不足，不要编造文献结论。"
    "不要输出<think>、思考过程或内部推理，只输出最终正文。"
    "不要写“参考文献13”“文献[51]”这类无法核验的编号；"
    "如需指代来源，只能使用检索证据中明确给出的文档名、章节或页码。"
)

_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*[-*_]{3,}\s*$")


def normalize_input(state: SRAAgentState) -> ResearchFormInput:
    """Normalize the input payload stored in LangGraph state."""
    return ResearchFormInput.from_mapping(state["user_input"])


def form_summary(user_input: ResearchFormInput) -> str:
    """Create a compact prompt-ready summary of form fields."""
    data = user_input.to_dict()
    labels = {
        "research_goal": "研究目标",
        "research_direction": "研究方向/热点",
        "outcome_indicators": "结局指标",
        "innovation_points": "创新点",
        "research_type": "研究类型",
        "output_purpose": "输出用途",
        "selected_literature": "系统推荐文献",
        "uploaded_literature": "用户上传文献",
        "manual_literature": "手动文献信息",
        "focus_mechanism": "重点关注机制/通路",
        "focus_drug_or_formula": "重点关注药物/方剂",
        "include_safety_economics": "是否包含安全性/经济性分析",
    }
    lines = []
    for key, label in labels.items():
        value = data.get(key)
        if value in ("", [], None):
            continue
        if isinstance(value, list):
            value = "、".join(str(item) for item in value)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def evidence_text(evidence: list[RetrievalEvidence]) -> str:
    """Format evidence snippets for prompts."""
    if not evidence:
        return "未配置检索器或未检索到可用证据。"

    blocks = []
    for index, item in enumerate(evidence, 1):
        metadata = item.metadata or {}
        source = (
            metadata.get("title")
            or metadata.get("source_file")
            or metadata.get("query")
            or item.source
        )
        content = strip_think_text(item.content)
        if len(content) > MAX_EVIDENCE_CHARS_PER_ITEM:
            content = (
                content[:MAX_EVIDENCE_CHARS_PER_ITEM]
                + "\n\n[证据内容过长，已截断用于生成。]"
            )
        blocks.append(
            f"[证据{index}] query={item.query}\n"
            f"source={source}; mode={item.mode}; score={item.score}\n"
            f"{content}"
        )
    return "\n\n".join(blocks)


def strip_think_text(content: Any) -> str:
    """Remove model thinking blocks from retrieved or generated text."""
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)

    text = _THINK_BLOCK_RE.sub("", str(content))
    return _THINK_TAG_RE.sub("", text).strip()


def section_update(
    state: SRAAgentState,
    section_key: str,
    section_text: str,
    evidence: list[RetrievalEvidence] | None = None,
) -> dict[str, Any]:
    """Return a LangGraph-compatible partial state update."""
    update: dict[str, Any] = {
        "report_sections": {section_key: section_text},
    }
    if evidence is not None:
        update["retrieved_contexts"] = {
            section_key: [item.to_dict() for item in evidence]
        }

    return update


def clean_llm_output(content: Any, title: str) -> str:
    """Remove model-private thinking text and normalize section Markdown."""
    text = strip_think_text(content)
    lines = text.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and _is_duplicate_heading(lines[0], title):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    cleaned_lines = [
        _normalize_heading_level(line)
        for line in lines
        if not _HORIZONTAL_RULE_RE.match(line)
    ]
    return _collapse_blank_lines(cleaned_lines).strip()


def _is_duplicate_heading(line: str, title: str) -> bool:
    match = _HEADING_RE.match(line.strip())
    if not match:
        return False

    heading = _normalize_heading_text(match.group(2))
    expected = _normalize_heading_text(title)
    return expected in heading or heading in expected


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"[\s#*_：:（）()\[\]【】\-—]+", "", text).lower()


def _normalize_heading_level(line: str) -> str:
    match = _HEADING_RE.match(line)
    if not match:
        return line

    hashes, heading = match.groups()
    level = max(3, len(hashes))
    return f"{'#' * level} {heading}"


def _collapse_blank_lines(lines: list[str]) -> str:
    collapsed: list[str] = []
    blank_seen = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and blank_seen:
            continue
        collapsed.append(line)
        blank_seen = is_blank
    return "\n".join(collapsed)


def invoke_llm(llm: Any, title: str, prompt: str) -> str:
    """Invoke a LangChain chat model or a lightweight test double."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt_length = len(prompt)
    logger.info("LLM invoke start: module=%s prompt_chars=%s",
                title, prompt_length)
    logger.info(f"prompt: {prompt}")
    prompt_preview = prompt[:MAX_PROMPT_LOG_CHARS]
    if len(prompt) > MAX_PROMPT_LOG_CHARS:
        prompt_preview += "\n...[prompt truncated in logs]..."
    logger.debug("Prompt preview: module=%s content=%s", title, prompt_preview)
    started_at = time.perf_counter()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请完成模块：{title}\n\n{prompt}"),
    ]

    if hasattr(llm, "invoke"):
        logger.info("LLM invoke calling backend: module=%s", title)
        response = llm.invoke(messages)
    elif callable(llm):
        logger.info("LLM invoke calling callable backend: module=%s", title)
        response = llm(messages)
    else:
        raise TypeError("llm must provide invoke(messages) or be callable")

    content = getattr(response, "content", response)
    output = clean_llm_output(content, title)
    if not output and str(content).strip():
        logger.warning(
            "LLM output became empty after cleaning: module=%s raw_chars=%s. "
            "Using fallback text to avoid blocking the graph.",
            title,
            len(str(content)),
        )
        output = (
            f"本模块未获得有效正文输出。建议检查模型是否只返回了思考过程，"
            f"或降低检索证据长度后重新生成“{title}”。"
        )

    elapsed = time.perf_counter() - started_at
    logger.info(
        "LLM invoke done: module=%s elapsed=%.2fs output_chars=%s",
        title,
        elapsed,
        len(output),
    )
    return output


def retrieve_for_section(
    retrieval_tool: UnifiedRetrievalTool | None,
    query: str,
    *,
    purpose: str,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[RetrievalEvidence]:
    """Run retrieval if a tool is configured."""
    if retrieval_tool is None:
        logger.info(
            "Retrieval skipped: purpose=%s query_chars=%s reason=no_tool",
            purpose,
            len(query),
        )
        return []
    logger.info(
        "Retrieval requested: purpose=%s mode=%s top_k=%s query=%s",
        purpose,
        mode,
        top_k,
        query,
    )
    evidence = retrieval_tool.retrieve(
        query, purpose=purpose, top_k=top_k, mode=mode)
    logger.info(
        "Retrieval returned: purpose=%s evidence_count=%s evidence_chars=%s",
        purpose,
        len(evidence),
        sum(len(item.content) for item in evidence),
    )

    logger.info(f"evidence: {evidence}")

    return evidence


def log_agent_node(agent_name: str):
    """Decorate a LangGraph node with start/end/error logs."""

    def decorator(func):
        @wraps(func)
        def wrapper(state: SRAAgentState) -> dict[str, Any]:
            sections_before = sorted(
                (state.get("report_sections") or {}).keys())
            logger.info(
                "Agent start: %s existing_sections=%s",
                agent_name,
                sections_before,
            )
            started_at = time.perf_counter()
            try:
                update = func(state)
            except Exception:
                elapsed = time.perf_counter() - started_at
                logger.exception(
                    "Agent failed: %s elapsed=%.2fs", agent_name, elapsed)
                raise

            elapsed = time.perf_counter() - started_at
            section_keys = sorted((update.get("report_sections") or {}).keys())
            context_keys = sorted(
                (update.get("retrieved_contexts") or {}).keys())
            logger.info(
                "Agent done: %s elapsed=%.2fs updated_sections=%s updated_contexts=%s",
                agent_name,
                elapsed,
                section_keys,
                context_keys,
            )
            return update

        return wrapper

    return decorator
