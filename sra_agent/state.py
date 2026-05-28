"""Shared state definitions for the SRA LangGraph workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, NotRequired, TypedDict


@dataclass
class ResearchFormInput:
    """Normalized user form input for the research analysis report."""

    research_goal: str
    research_type: str
    output_purpose: str
    research_direction: str = ""
    outcome_indicators: str = ""
    innovation_points: list[str] = field(default_factory=list)
    selected_literature: list[str] = field(default_factory=list)
    uploaded_literature: list[str] = field(default_factory=list)
    manual_literature: list[str] = field(default_factory=list)
    focus_mechanism: str = ""
    focus_drug_or_formula: str = ""
    include_safety_economics: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | "ResearchFormInput") -> "ResearchFormInput":
        """Create a normalized input object from API/form data."""
        if isinstance(data, cls):
            return data

        aliases = {
            "goal": "research_goal",
            "direction": "research_direction",
            "hotspot": "research_direction",
            "indicators": "outcome_indicators",
            "innovation": "innovation_points",
            "type": "research_type",
            "purpose": "output_purpose",
            "drug": "focus_drug_or_formula",
            "formula": "focus_drug_or_formula",
            "mechanism": "focus_mechanism",
            "include_safety": "include_safety_economics",
        }
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            normalized[aliases.get(key, key)] = value

        innovation_points = normalized.get("innovation_points") or []
        if isinstance(innovation_points, str):
            innovation_points = [
                item.strip()
                for item in innovation_points.replace("，", ",").split(",")
                if item.strip()
            ]
        normalized["innovation_points"] = list(innovation_points)

        for list_key in ("selected_literature", "uploaded_literature", "manual_literature"):
            value = normalized.get(list_key) or []
            if isinstance(value, str):
                value = [value]
            normalized[list_key] = list(value)

        required = ("research_goal", "research_type", "output_purpose")
        missing = [key for key in required if not str(normalized.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Missing required research form fields: {', '.join(missing)}")

        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in normalized.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary safe for prompts and JSON APIs."""
        return asdict(self)


@dataclass
class RetrievalEvidence:
    """Evidence returned by the unified retrieval tool."""

    query: str
    content: str
    source: str = "lightrag"
    mode: str = "hybrid"
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable evidence payload."""
        return asdict(self)


class SRAAgentState(TypedDict):
    """LangGraph state shared by all report-generation agents."""

    user_input: dict[str, Any]
    report_sections: Annotated[dict[str, str], _merge_dicts]
    retrieved_contexts: Annotated[dict[str, list[dict[str, Any]]], _merge_dicts]
    final_report: NotRequired[str]
    review_notes: NotRequired[str]
    errors: NotRequired[Annotated[list[str], _merge_lists]]


def _merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge partial LangGraph dictionary updates."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key in merged and merged[key] != value:
            raise ValueError(f"Conflicting concurrent state update for key: {key}")
        merged[key] = value
    return merged


def _merge_lists(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Merge partial LangGraph list updates."""
    return [*(left or []), *(right or [])]
