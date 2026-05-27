"""解析器抽象基类模块。

定义文档解析器的通用接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedDocument:
    """解析后的文档表示。

    Attributes:
        title: 文档标题
        content: 结构化文本内容（Markdown 格式）
        metadata: 文档元数据（标题、摘要、年份、关键词等）
        chunks: 分块列表，每个分块包含 text 和元数据（section, page, block_type 等）
    """

    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)


class BaseParser(ABC):
    """文档解析器抽象基类。

    所有具体的文档解析器都应继承此类并实现 parse 方法。
    """

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """解析文档文件。

        Args:
            file_path: 文档文件路径

        Returns:
            ParsedDocument: 解析后的文档对象
        """
        pass
