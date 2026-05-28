"""Runtime options for embedding the SRA agent in a host application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable


@dataclass(frozen=True)
class SraLlmOptions:
    """OpenAI-compatible chat model options supplied by the host app."""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout: float = 120.0


@runtime_checkable
class SraLlmOptionsLike(Protocol):
    """Protocol for host option objects carrying OpenAI-compatible LLM fields."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str


SraLlmOptionsInput: TypeAlias = (
    SraLlmOptions | SraLlmOptionsLike | Mapping[str, Any]
)


def normalize_llm_options(options: SraLlmOptionsInput) -> SraLlmOptions:
    """Normalize flat or nested host-provided LLM options."""
    if isinstance(options, SraLlmOptions):
        return options

    llm = _mapping_or_empty(_option_value(options, "llm"))
    values = {
        "base_url": _option_value(
            options,
            "base_url",
            _option_value(options, "llm_base_url", llm.get("base_url")),
        ),
        "api_key": _option_value(
            options,
            "api_key",
            _option_value(options, "llm_api_key", llm.get("api_key")),
        ),
        "model": _option_value(
            options,
            "model",
            _option_value(options, "llm_model", llm.get("model")),
        ),
        "temperature": _option_value(
            options,
            "temperature",
            llm.get("temperature", 0.2),
        ),
        "timeout": _option_value(
            options,
            "timeout",
            _option_value(options, "llm_timeout", llm.get("timeout", 120.0)),
        ),
    }
    missing = [
        key for key in ("base_url", "api_key", "model") if values[key] is None
    ]
    if missing:
        raise ValueError(f"Missing required SRA agent LLM option(s): {', '.join(missing)}")

    return SraLlmOptions(
        base_url=str(values["base_url"]),
        api_key=str(values["api_key"]),
        model=str(values["model"]),
        temperature=float(values["temperature"]),
        timeout=float(values["timeout"]),
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Nested option groups must be mappings.")
    return value


def _option_value(options: Any, key: str, default: Any = None) -> Any:
    if isinstance(options, Mapping):
        return options.get(key, default)
    return getattr(options, key, default)
