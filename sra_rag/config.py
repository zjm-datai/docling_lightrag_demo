"""SRA RAG 配置模块。

集中管理 RAG 系统的所有配置参数。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGConfig:
    """RAG 系统配置。

    Attributes:
        working_dir: LightRAG 工作目录
        llm_base_url: LLM API 基础 URL
        llm_api_key: LLM API 密钥
        llm_model: LLM 模型名称
        embedding_model: Embedding 模型名称
        embedding_dim: Embedding 向量维度
        max_token_size: 最大 token 大小
        allowed_formats: 允许的文档格式
        chunk_metadata: 分块元数据配置
    """

    # LightRAG 工作目录
    working_dir: str = "./sra_rag_data"

    # LLM 配置
    llm_base_url: str = "http://211.90.240.240:30001/v1"
    llm_api_key: str = "gpustack_ba7863cefc4126b2_4ecbb5f42c239594a33e67ef49bdce9d"
    llm_model: str = "Qwen3-30B-A3B-GPTQ-Int4"

    # Embedding 配置
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    max_token_size: int = 8192

    # 文档解析配置
    allowed_formats: list[str] = field(
        default_factory=lambda: ["pdf", "docx", "xlsx", "pptx", "html", "md"]
    )

    # 其他元数据配置
    chunk_metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "include_section": True,
            "include_page": True,
            "include_block_type": True,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """将配置转换为字典。

        Returns:
            dict: 配置字典
        """
        return {
            "working_dir": self.working_dir,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RAGConfig":
        """从字典创建配置。

        Args:
            config_dict: 配置字典

        Returns:
            RAGConfig: 配置对象
        """
        return cls(**config_dict)


# 默认配置实例
default_config = RAGConfig()
