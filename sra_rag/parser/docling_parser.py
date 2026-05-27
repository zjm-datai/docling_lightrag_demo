"""Docling 文档解析器模块。

使用 Docling 解析多格式文档（PDF/DOCX/XLSX 等），并使用 HybridChunker 分块。
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from sra_rag.parser.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """基于 Docling 的文档解析器。

    支持 PDF、DOCX、XLSX、PPTX 等多种格式的文档解析，
    并能按文档结构进行智能分块，保留丰富的元数据信息。
    """

    def __init__(
        self,
        allowed_formats: Optional[list[InputFormat]] = None,
        format_options: Optional[dict[InputFormat, Any]] = None,
    ):
        """初始化 Docling 解析器。

        Args:
            allowed_formats: 允许的输入格式列表，默认为 None（支持所有格式）
            format_options: 格式特定选项配置，默认为 None
        """
        self.converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options=format_options,
        )

    def parse(self, file_path: Path) -> ParsedDocument:
        """解析文档文件。

        Args:
            file_path: 文档文件路径

        Returns:
            ParsedDocument: 包含解析结果的结构化对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"开始解析文档: {file_path}")

        # 执行文档转换
        result = self.converter.convert(file_path)

        if result.document is None:
            raise ValueError(f"无法解析文档: {file_path}")

        doc = result.document

        # 提取元数据
        metadata = self._extract_metadata(doc, file_path)

        # 导出为 Markdown
        content = doc.export_to_markdown()

        # Hybrid chunking (Docling built-in)
        chunks = self._chunk_by_hybrid(doc)

        logger.info(
            f"文档解析完成: {file_path.stem}, "
            f"文本块数量: {len(chunks)}, "
            f"内容长度: {len(content)} 字符"
        )

        return ParsedDocument(
            title=metadata.get("title", file_path.stem),
            content=content,
            metadata=metadata,
            chunks=chunks,
        )

    def export_parsed_document(
        self,
        parsed_doc: ParsedDocument,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        """保存 Docling 识别和切分结果。"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        markdown_path = output_path / "docling_content.md"
        metadata_path = output_path / "docling_metadata.json"
        chunks_path = output_path / "docling_chunks.json"
        summary_path = output_path / "docling_parsed_document.json"

        markdown_path.write_text(parsed_doc.content, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(parsed_doc.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        chunks_path.write_text(
            json.dumps(parsed_doc.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(
                {
                    "title": parsed_doc.title,
                    "content_file": markdown_path.name,
                    "metadata_file": metadata_path.name,
                    "chunks_file": chunks_path.name,
                    "content_length": len(parsed_doc.content),
                    "chunk_count": len(parsed_doc.chunks),
                    "metadata": parsed_doc.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "content": markdown_path,
            "metadata": metadata_path,
            "chunks": chunks_path,
            "summary": summary_path,
        }

    def _extract_metadata(
        self, doc: DoclingDocument, file_path: Path
    ) -> dict[str, Any]:
        """从 DoclingDocument 中提取元数据。

        Args:
            doc: Docling 文档对象
            file_path: 原始文件路径

        Returns:
            dict: 文档元数据字典
        """
        metadata = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "file_suffix": file_path.suffix,
        }

        # 尝试从文档中提取标题
        if hasattr(doc, "name") and doc.name:
            metadata["title"] = doc.name

        # 尝试提取文档描述（摘要）
        if hasattr(doc, "description") and doc.description:
            metadata["description"] = doc.description

        # 统计信息
        metadata["num_texts"] = len(getattr(doc, "texts", []))
        metadata["num_tables"] = len(getattr(doc, "tables", []))
        metadata["num_pictures"] = len(getattr(doc, "pictures", []))

        return metadata

    def _chunk_by_hybrid(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        """Use Docling HybridChunker and map chunks to the project schema."""
        chunks: list[dict[str, Any]] = []

        hybrid_chunks = list(HybridChunker().chunk(doc))
        for chunk in hybrid_chunks:
            text = str(getattr(chunk, "text", "") or "").strip()
            if not text:
                continue

            meta = getattr(chunk, "meta", None)
            section = ""
            block_type = "hybrid"
            page_num: Optional[int] = None
            section_path = ""

            if meta is not None:
                headings = getattr(meta, "headings", None)
                if headings:
                    section = str(headings[-1])
                    section_path = " > ".join(str(h) for h in headings if h)

                doc_items = getattr(meta, "doc_items", None) or []
                if doc_items:
                    first_item = doc_items[0]
                    block_type = str(
                        getattr(first_item, "label", block_type) or block_type
                    )
                    prov = getattr(first_item, "prov", None)
                    if prov and len(prov) > 0:
                        page_num = getattr(prov[0], "page_no", None)

            chunks.append(
                {
                    "text": text,
                    "section": section,
                    "page": page_num,
                    "block_type": block_type,
                    "metadata": {
                        "title": getattr(doc, "name", ""),
                        "section_path": section_path,
                    },
                }
            )

        logger.debug(f"Hybrid ?????? {len(chunks)} ????")
        return chunks

    def _get_page_number(self, item: Any) -> Optional[int]:
        """从元素中提取页面编号。

        Args:
            item: Docling 文档元素

        Returns:
            Optional[int]: 页面编号，如果无法提取则返回 None
        """
        try:
            # 尝试从 prov 属性中获取页面信息
            if hasattr(item, "prov") and item.prov:
                # prov 通常是一个列表，取第一个元素的 page 属性
                if isinstance(item.prov, list) and len(item.prov) > 0:
                    prov_item = item.prov[0]
                    if hasattr(prov_item, "page_no"):
                        return prov_item.page_no
                    elif hasattr(prov_item, "page"):
                        return prov_item.page
        except Exception as e:
            logger.debug(f"无法提取页面信息: {e}")

        return None

    def _get_section_path(self, item: Any) -> str:
        """获取元素的章节路径。

        Args:
            item: Docling 文档元素

        Returns:
            str: 章节路径字符串
        """
        # 简化实现：返回当前 section
        # 完整实现需要遍历文档树结构
        return getattr(item, "label", "")
