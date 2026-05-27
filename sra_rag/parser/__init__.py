"""文档解析器模块。

提供多种文档解析器实现，支持多格式文档的结构化解析。
"""

from sra_rag.parser.base import BaseParser, ParsedDocument
from sra_rag.parser.docling_parser import DoclingParser

__all__ = ["BaseParser", "ParsedDocument", "DoclingParser"]
