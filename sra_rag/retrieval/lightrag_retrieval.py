"""LightRAG 检索器模块。

封装 LightRAG 的多种检索模式，提供统一的检索接口。
"""

import logging
from typing import Any, Optional

from lightrag import QueryParam

from sra_rag.indexer.lightrag_indexer import LightRAGIndexer
from sra_rag.retrieval.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class LightRAGRetriever(BaseRetriever):
    """基于 LightRAG 的文档检索器。

    支持多种检索模式：
    - naive: 简单向量检索，适用于简单问答
    - local: 局部知识图谱检索，适用于上下文相关问题
    - global: 全局知识图谱检索，适用于需要全局知识的问题
    - hybrid: 混合检索，结合所有优势（推荐）
    """

    def __init__(
        self,
        indexer: LightRAGIndexer,
    ):
        """初始化 LightRAG 检索器。

        Args:
            indexer: LightRAG 索引器实例
        """
        self.indexer = indexer
        self.rag = indexer.rag

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """执行检索。

        Args:
            query: 查询字符串
            top_k: 返回结果数量（LightRAG 当前版本可能不支持此参数）
            mode: 检索模式 (naive/local/global/hybrid)
            **kwargs: 其他检索参数

        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        logger.info(f"执行 LightRAG 检索: query='{query}', mode={mode}")

        try:
            param = QueryParam(mode=mode, top_k=top_k, **kwargs)
            result = self.rag.query(query, param=param)

            # LightRAG 返回的是完整答案，包装为 RetrievalResult
            retrieval_result = RetrievalResult(
                content=result,
                metadata={
                    "mode": mode,
                    "query": query,
                    "retriever": "lightrag",
                },
                score=1.0,
            )

            logger.info(f"检索完成，答案长度: {len(result)} 字符")
            return [retrieval_result]

        except Exception as e:
            logger.error(f"LightRAG 检索失败: {e}")
            raise

    def retrieve_with_context(
        self,
        query: str,
        mode: str = "hybrid",
        include_context: bool = True,
    ) -> dict[str, Any]:
        """执行检索并返回上下文信息。

        Args:
            query: 查询字符串
            mode: 检索模式
            include_context: 是否包含检索到的上下文

        Returns:
            dict: 包含答案和上下文的字典
        """
        logger.info(f"执行带上下文的检索: query='{query}', mode={mode}")

        result = self.retrieve(query, mode=mode)

        return {
            "answer": result[0].content if result else "",
            "contexts": result,
            "metadata": result[0].metadata if result else {},
        }

    def get_supported_modes(self) -> list[str]:
        """获取支持的检索模式列表。

        Returns:
            list[str]: 模式列表
        """
        return ["naive", "local", "global", "hybrid"]

    def get_mode_description(self, mode: str) -> str:
        """获取检索模式的描述。

        Args:
            mode: 检索模式

        Returns:
            str: 模式描述
        """
        descriptions = {
            "naive": "简单向量检索，适用于简单问答",
            "local": "局部知识图谱检索，适用于上下文相关问题",
            "global": "全局知识图谱检索，适用于需要全局知识的问题",
            "hybrid": "混合检索，结合所有优势（推荐）",
        }
        return descriptions.get(mode, "未知模式")
