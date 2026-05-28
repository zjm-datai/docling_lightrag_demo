"""Run the real SRA LangGraph report workflow with ChatOpenAI.

This script uses the complete supported input fields and writes the generated
research innovation report to a Markdown file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from sra_agent import generate_research_report
from sra_rag import SraRagOptions, create_sra_rag


DEFAULT_INPUT: dict[str, Any] = {
    "research_goal": "研究特应性皮炎儿童人群的炎症通路调控",
    "research_direction": "肠皮轴、Th2 免疫调控",
    "outcome_indicators": "EASI、瘙痒评分、IL-4、IgE",
    "innovation_points": ["理论创新", "方法创新", "应用创新"],
    "research_type": "药理机制",
    "output_purpose": "基金申报",
    "selected_literature": [
        "Interleukin-4 and Atopic Dermatitis: Why Does it Matter? A Narrative Review"
    ],
    "uploaded_literature": ["001_test.pdf"],
    "manual_literature": [
        "可在这里填写用户手动输入的标题、DOI 或 PMID"
    ],
    "focus_mechanism": "IL-4/IL-13 相关炎症通路",
    "focus_drug_or_formula": "",
    "include_safety_economics": True,
}


def load_user_input(input_json: Path | None) -> dict[str, Any]:
    """Load user input from JSON, or use the complete built-in example."""
    if input_json is None:
        return DEFAULT_INPUT

    with input_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a real SRA research innovation report."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional JSON file containing the full user input payload.",
    )
    parser.add_argument(
        "--working-dir",
        default="./sra_rag_data_test",
        help="LightRAG working directory. Default: ./sra_rag_data_test",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("SRA_LLM_BASE_URL"),
        help="OpenAI-compatible LLM base URL. Can also use SRA_LLM_BASE_URL.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("SRA_LLM_API_KEY"),
        help="OpenAI-compatible API key. Can also use SRA_LLM_API_KEY.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("SRA_LLM_MODEL"),
        help="Chat model name. Can also use SRA_LLM_MODEL.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("SRA_EMBEDDING_MODEL"),
        help="Embedding model name. Can also use SRA_EMBEDDING_MODEL.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=(
            int(os.environ["SRA_EMBEDDING_DIM"])
            if os.environ.get("SRA_EMBEDDING_DIM")
            else None
        ),
        help="Embedding vector dimension. Can also use SRA_EMBEDDING_DIM.",
    )
    parser.add_argument(
        "--max-token-size",
        type=int,
        default=int(os.environ.get("SRA_MAX_TOKEN_SIZE", "8192")),
        help="Embedding max token size. Default: 8192",
    )
    parser.add_argument(
        "--agent-llm-timeout",
        type=float,
        default=float(os.environ.get("SRA_AGENT_LLM_TIMEOUT", "120")),
        help="ChatOpenAI timeout for report-generation agents. Default: 120",
    )
    parser.add_argument(
        "--query-max-total-tokens",
        type=int,
        default=int(os.environ.get("SRA_QUERY_MAX_TOTAL_TOKENS", "6000")),
        help="LightRAG max total context tokens per query. Default: 6000",
    )
    parser.add_argument(
        "--rag-retrieval-mode",
        choices=["naive", "local", "global", "hybrid"],
        default=os.environ.get("SRA_RAG_RETRIEVAL_MODE", "naive"),
        help=(
            "Retrieval mode used by report sub-agents. Default: naive. "
            "Use hybrid only when the LLM keyword extraction path is stable."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/outputs/real_agent_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Use the real ChatOpenAI model but skip LightRAG retrieval.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Default: INFO",
    )
    parser.add_argument(
        "--retrieval-log-chars",
        type=int,
        default=int(os.environ.get("SRA_RETRIEVAL_LOG_CHARS", "1200")),
        help="Max characters printed per retrieved evidence item. Default: 1200",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("examples/outputs/real_agent_report.log"),
        help="Optional log file path.",
    )
    return parser.parse_args()


def configure_logging(level: str, log_file: Path | None) -> None:
    """Configure console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    args = parse_args()
    from sra_agent import retrieval_tool

    retrieval_tool.MAX_RETRIEVAL_LOG_CHARS = args.retrieval_log_chars
    configure_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Real agent report script started")
    logger.info("Arguments: %s", args)

    user_input = load_user_input(args.input_json)
    logger.info("Loaded user input with keys: %s", sorted(user_input.keys()))

    missing = [
        name
        for name in (
            "llm_base_url",
            "llm_api_key",
            "llm_model",
            "embedding_model",
            "embedding_dim",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Missing required runtime option(s): "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )

    options = SraRagOptions(
        working_dir=args.working_dir,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        max_token_size=args.max_token_size,
        timeout=args.agent_llm_timeout,
        query_max_total_tokens=args.query_max_total_tokens,
    )
    logger.info(
        "RAG config: working_dir=%s model=%s base_url=%s embedding=%s",
        options.working_dir,
        options.llm_model,
        options.llm_base_url,
        options.embedding_model,
    )
    rag = None if args.no_rag else create_sra_rag(options)
    retriever = None if rag is None else rag.retriever
    logger.info("Retriever configured: %s", type(retriever).__name__ if retriever else "None")

    result = generate_research_report(
        user_input=user_input,
        retriever=retriever,
        llm_options=options,
        retrieval_mode=args.rag_retrieval_mode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.final_report, encoding="utf-8")

    logger.info("Report written: %s", args.output.resolve())
    logger.info("Real agent report script finished")
    print(f"Report written to: {args.output.resolve()}")
    print(f"Sections generated: {', '.join(sorted(result.sections))}")
    if args.no_rag:
        print("RAG retrieval: skipped")
    else:
        print(f"RAG contexts: {sum(len(v) for v in result.retrieved_contexts.values())}")


if __name__ == "__main__":
    main()
