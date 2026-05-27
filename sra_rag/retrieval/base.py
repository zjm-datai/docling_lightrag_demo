"""检索器抽象基类模块。

定义文档检索器的通用接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """检索结果。

    Attributes:
        content: 检索到的内容
        metadata: 相关元数据（来源、分数等）
        score: 相关性分数
    """

    content: str
    metadata: dict[str, Any] = None
    score: float = 1.0

    def __post_init__(self):
        """初始化后处理。"""
        if self.metadata is None:
            self.metadata = {}


class BaseRetriever(ABC):
    """文档检索器抽象基类。

    所有具体的检索器都应继承此类并实现 retrieve 方法。
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """检索相关文档。

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            **kwargs: 其他检索参数

        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        pass
