"""索引器抽象基类模块。

定义文档索引器的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Any

from sra_rag.parser.base import ParsedDocument


class BaseIndexer(ABC):
    """文档索引器抽象基类。

    所有具体的索引器都应继承此类并实现 index 和 search 方法。
    """

    @abstractmethod
    def index(self, document: ParsedDocument) -> str:
        """索引文档。

        Args:
            document: 解析后的文档对象

        Returns:
            str: 文档 ID 或标识符
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索相关文档。

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            list[dict]: 检索结果列表
        """
        pass
