"""Docling chunking 示例合集。

展示不同 chunking 模式的输出差异：
- NoChunker（近似演示）
- HierarchicalChunker
- HybridChunker

并支持将每种模式完整导出为独立 JSONL。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from docling.chunking import HierarchicalChunker, HybridChunker
from docling.document_converter import DocumentConverter


def _preview_text(text: str, max_chars: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[:max_chars]}..."


def _metadata_to_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if hasattr(meta, "model_dump"):
        return meta.model_dump()
    if hasattr(meta, "dict"):
        return meta.dict()
    if isinstance(meta, dict):
        return meta
    return {"raw": str(meta)}


def _chunk_to_record(chunk: Any, index: int, mode: str, source_file: Path) -> dict[str, Any]:
    text = getattr(chunk, "text", "") or ""
    return {
        "index": index,
        "mode": mode,
        "source_file": str(source_file),
        "text": text,
        "text_length": len(text),
        "metadata": _metadata_to_dict(getattr(chunk, "meta", None)),
    }


def _print_chunk_result(mode: str, chunks: list[Any], sample_limit: int) -> None:
    print("\n" + "=" * 80)
    print(f"Chunk 模式: {mode}")
    print("=" * 80)
    print(f"总 chunk 数量: {len(chunks)}")

    if not chunks:
        print("(无 chunk 输出)")
        return

    print(f"展示前 {min(sample_limit, len(chunks))} 个 chunk:")
    for index, chunk in enumerate(chunks[:sample_limit], 1):
        text = getattr(chunk, "text", "") or ""
        metadata = _metadata_to_dict(getattr(chunk, "meta", None))
        print(f"\n[{index}] 文本预览: {_preview_text(text)}")
        print(f"[{index}] 文本长度: {len(text)}")
        print(
            f"[{index}] 元数据: "
            f"{json.dumps(metadata, ensure_ascii=False, default=str)}"
        )


def _docling_to_chunks(doc_path: Path) -> tuple[Any, list[Any], list[Any], list[Any]]:
    converter = DocumentConverter()
    result = converter.convert(str(doc_path))
    dl_doc = result.document

    markdown_text = dl_doc.export_to_markdown()
    no_chunks = []
    for block in markdown_text.split("\n\n"):
        if block.strip():
            no_chunks.append(
                type("PlainChunk", (), {"text": block, "meta": {"mode": "no_chunk"}})()
            )

    hierarchical_chunks = list(HierarchicalChunker().chunk(dl_doc))
    hybrid_chunks = list(HybridChunker().chunk(dl_doc))
    return dl_doc, no_chunks, hierarchical_chunks, hybrid_chunks


def _export_jsonl(mode: str, chunks: list[Any], source_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_obj:
        for idx, chunk in enumerate(chunks, 1):
            record = _chunk_to_record(chunk, idx, mode, source_file)
            file_obj.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_examples(doc_path: Path, sample_limit: int = 5, export_dir: Path | None = None) -> None:
    if not doc_path.exists():
        raise FileNotFoundError(f"文件不存在: {doc_path}")

    print("=" * 80)
    print("Docling Chunking 示例合集")
    print("=" * 80)
    print(f"输入文档: {doc_path.resolve()}")

    dl_doc, no_chunks, hierarchical_chunks, hybrid_chunks = _docling_to_chunks(doc_path)

    print(f"\n文档导出 markdown 长度: {len(dl_doc.export_to_markdown())}")
    _print_chunk_result("NoChunker(近似演示)", no_chunks, sample_limit)
    _print_chunk_result("HierarchicalChunker", hierarchical_chunks, sample_limit)
    _print_chunk_result("HybridChunker", hybrid_chunks, sample_limit)

    if export_dir is not None:
        stem = doc_path.stem
        no_path = export_dir / f"{stem}.no_chunk.jsonl"
        hierarchical_path = export_dir / f"{stem}.hierarchical.jsonl"
        hybrid_path = export_dir / f"{stem}.hybrid.jsonl"

        _export_jsonl("no_chunk", no_chunks, doc_path, no_path)
        _export_jsonl("hierarchical", hierarchical_chunks, doc_path, hierarchical_path)
        _export_jsonl("hybrid", hybrid_chunks, doc_path, hybrid_path)

        print("\n" + "=" * 80)
        print("JSONL 导出完成")
        print("=" * 80)
        print(f"NoChunker: {no_path.resolve()}")
        print(f"HierarchicalChunker: {hierarchical_path.resolve()}")
        print(f"HybridChunker: {hybrid_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Docling chunking 示例合集")
    parser.add_argument("doc_path", type=Path, help="待解析文档路径（如 PDF）")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="每种模式展示前 N 个 chunk（默认 5）",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("examples") / "outputs" / "docling_chunks",
        help="每种模式 JSONL 导出目录",
    )
    args = parser.parse_args()
    run_examples(args.doc_path, args.sample_limit, args.export_dir)


if __name__ == "__main__":
    main()
