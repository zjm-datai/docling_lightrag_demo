"""Tests for SRA agent state update/merge behavior."""

import pytest

from sra_agent.state import RetrievalEvidence, _merge_dicts
from sra_agent.sub_agents.common import clean_llm_output, section_update


def test_section_update_returns_only_section_delta():
    state = {
        "user_input": {},
        "report_sections": {"overview": "existing"},
        "retrieved_contexts": {"overview": [{"content": "old"}]},
    }
    evidence = [RetrievalEvidence(query="q", content="c", metadata={"title": "t"})]

    update = section_update(state, "innovation", "new text", evidence)

    assert update["report_sections"] == {"innovation": "new text"}
    assert set(update["retrieved_contexts"]) == {"innovation"}
    assert update["retrieved_contexts"]["innovation"][0]["content"] == "c"


def test_merge_dicts_combines_distinct_parallel_updates():
    assert _merge_dicts({"overview": "a"}, {"innovation": "b"}) == {
        "overview": "a",
        "innovation": "b",
    }


def test_merge_dicts_allows_idempotent_same_key_update():
    assert _merge_dicts({"overview": "a"}, {"overview": "a"}) == {"overview": "a"}


def test_merge_dicts_rejects_conflicting_same_key_update():
    with pytest.raises(ValueError, match="overview"):
        _merge_dicts({"overview": "a"}, {"overview": "b"})


def test_clean_llm_output_removes_think_block_and_duplicate_heading():
    output = clean_llm_output(
        "<think>\ninternal reasoning\n</think>\n\n### 经济性研究\n\n#### 已有线索\n正文",
        "经济性研究",
    )

    assert "<think>" not in output
    assert "internal reasoning" not in output
    assert output == "#### 已有线索\n正文"


def test_clean_llm_output_normalizes_top_level_headings_inside_section():
    output = clean_llm_output("## 已被充分研究的内容\n正文\n### 二级内容", "研究现状总结")

    assert output == "### 已被充分研究的内容\n正文\n### 二级内容"


def test_clean_llm_output_removes_horizontal_rules():
    output = clean_llm_output("第一段\n\n---  \n\n第二段\n***\n第三段", "研究现状总结")

    assert output == "第一段\n\n第二段\n第三段"
