"""Example script for parsing and embedding documents with SRA RAG.

This script shows the host-application style:
1. Build a typed SraRagOptions object from runtime configuration.
2. Choose an embedding workflow mode with --embed-mode.
3. Parse documents and optionally index them into LightRAG.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from sra_rag import DoclingParser, SraRagOptions, create_sra_rag


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse documents and embed them into a LightRAG working dir."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Document files or directories to process.",
    )
    parser.add_argument(
        "--embed-mode",
        choices=["parse-only", "index", "index-and-query"],
        default="index",
        help=(
            "Embedding workflow mode. parse-only does not call LLM/embedding APIs; "
            "index parses and embeds documents; index-and-query also runs a query."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directory inputs.",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern for directory inputs. Default: *",
    )
    parser.add_argument(
        "--working-dir",
        default=os.environ.get("SRA_RAG_WORKING_DIR", "./sra_rag_data"),
        help="LightRAG working directory. Can also use SRA_RAG_WORKING_DIR.",
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
        "--llm-timeout",
        type=int,
        default=int(os.environ.get("SRA_LLM_TIMEOUT", "180")),
        help="LightRAG LLM call timeout in seconds. Default: 180",
    )
    parser.add_argument(
        "--llm-max-async",
        type=int,
        default=int(os.environ.get("SRA_LLM_MAX_ASYNC", "4")),
        help="LightRAG LLM concurrency during entity extraction. Default: 4",
    )
    parser.add_argument(
        "--entity-extract-max-gleaning",
        type=int,
        default=int(os.environ.get("SRA_ENTITY_EXTRACT_MAX_GLEANING", "1")),
        help="Extra entity/relation extraction passes per chunk. Default: 1",
    )
    parser.add_argument(
        "--max-extract-input-tokens",
        type=int,
        default=int(os.environ.get("SRA_MAX_EXTRACT_INPUT_TOKENS", "20480")),
        help="Max input tokens for entity/relation extraction. Default: 20480",
    )
    parser.add_argument(
        "--chunk-token-size",
        type=int,
        default=int(os.environ.get("SRA_CHUNK_TOKEN_SIZE", "1200")),
        help="LightRAG chunk token size. Lower values reduce LLM timeout risk.",
    )
    parser.add_argument(
        "--chunk-overlap-token-size",
        type=int,
        default=int(os.environ.get("SRA_CHUNK_OVERLAP_TOKEN_SIZE", "100")),
        help="LightRAG chunk overlap token size. Default: 100",
    )
    parser.add_argument(
        "--query",
        default="请总结这些文献的主要研究内容",
        help="Query used by --embed-mode index-and-query.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["naive", "local", "global", "hybrid"],
        default="hybrid",
        help="LightRAG retrieval mode for index-and-query. Default: hybrid.",
    )
    parser.add_argument(
        "--export-parsed-dir",
        type=Path,
        default=None,
        help="Optional directory for exported parsed markdown/metadata/chunks.",
    )
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> SraRagOptions:
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
            "Missing required option(s) for indexing: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )

    return SraRagOptions(
        working_dir=args.working_dir,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        max_token_size=args.max_token_size,
        default_llm_timeout=args.llm_timeout,
        llm_model_max_async=args.llm_max_async,
        entity_extract_max_gleaning=args.entity_extract_max_gleaning,
        max_extract_input_tokens=args.max_extract_input_tokens,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
    )


def iter_documents(
    inputs: Iterable[Path],
    *,
    recursive: bool,
    pattern: str,
) -> list[Path]:
    docs: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            docs.append(input_path)
            continue

        if input_path.is_dir():
            iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
            docs.extend(path for path in iterator if path.is_file())
            continue

        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    by_resolved_path: dict[Path, Path] = {}
    for path in docs:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        resolved = path.resolve()
        by_resolved_path.setdefault(resolved, path)
    return [by_resolved_path[key] for key in sorted(by_resolved_path)]


def parse_only(args: argparse.Namespace, docs: list[Path]) -> None:
    parser = DoclingParser()
    for doc_path in docs:
        parsed_doc = parser.parse(doc_path)
        export_dir = None
        if args.export_parsed_dir is not None:
            export_dir = args.export_parsed_dir / doc_path.stem
            parser.export_parsed_document(parsed_doc, export_dir)

        print(
            f"parsed title={parsed_doc.title!r} chunks={len(parsed_doc.chunks)} "
            f"chars={len(parsed_doc.content)}"
        )
        if export_dir is not None:
            print(f"exported parsed files: {export_dir.resolve()}")


def index_documents(args: argparse.Namespace, docs: list[Path]) -> None:
    options = build_options(args)
    rag = create_sra_rag(options)

    for doc_path in docs:
        parsed_doc = rag.parser.parse(doc_path)
        if args.export_parsed_dir is not None:
            rag.parser.export_parsed_document(
                parsed_doc,
                args.export_parsed_dir / doc_path.stem,
            )

        doc_id = rag.indexer.index_document(parsed_doc)
        print(
            f"indexed file={doc_path} doc_id={doc_id!r} "
            f"chunks={len(parsed_doc.chunks)}"
        )

    if args.embed_mode == "index-and-query":
        results = rag.retriever.retrieve(args.query, mode=args.retrieval_mode)
        print("\nquery:", args.query)
        print("retrieval_mode:", args.retrieval_mode)
        print("result_count:", len(results))
        if results:
            print("answer:")
            print(results[0].content)


def main() -> None:
    args = parse_args()
    docs = iter_documents(args.inputs, recursive=args.recursive, pattern=args.glob)
    if not docs:
        raise ValueError("No supported documents found.")

    print(f"embed_mode: {args.embed_mode}")
    print(f"document_count: {len(docs)}")
    if args.embed_mode != "parse-only":
        print(f"llm_timeout: {args.llm_timeout}s")
        print(f"llm_max_async: {args.llm_max_async}")
        print(f"chunk_token_size: {args.chunk_token_size}")
        print(f"entity_extract_max_gleaning: {args.entity_extract_max_gleaning}")
    for doc_path in docs:
        print(f"- {doc_path}")

    if args.embed_mode == "parse-only":
        parse_only(args, docs)
    else:
        index_documents(args, docs)


if __name__ == "__main__":
    main()
