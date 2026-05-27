"""SRA RAG 模块。

文献解析与索引入库系统，基于 Docling 和 LightRAG 实现。

主要功能：
- 多格式文档解析（PDF/DOCX/XLSX 等）
- Layout-aware 智能分块
- LightRAG 双级检索（向量 + 知识图谱）
"""

from sra_rag.config import RAGConfig, default_config
from sra_rag.indexer.lightrag_indexer import LightRAGIndexer
from sra_rag.parser.docling_parser import DoclingParser
from sra_rag.retrieval.lightrag_retrieval import LightRAGRetriever

__all__ = [
    "DoclingParser",
    "LightRAGIndexer",
    "LightRAGRetriever",
    "RAGConfig",
    "default_config",
]
