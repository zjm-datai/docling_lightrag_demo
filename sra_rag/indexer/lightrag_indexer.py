"""LightRAG 索引器模块。

集成 LightRAG 实现文档索引和检索功能，支持 Qwen3 LLM 和 bge-m3 嵌入模型。
"""

import asyncio
import hashlib
import logging
import os
from typing import Any

import aiohttp
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.operate import chunking_by_token_size
from lightrag.utils import EmbeddingFunc, sanitize_text_for_encoding

from sra_rag.parser.base import ParsedDocument

logger = logging.getLogger(__name__)


class _LLMCompleteAdapter:
    """LightRAG-safe LLM callable that avoids deepcopying the indexer."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def __deepcopy__(self, memo: dict[int, Any]) -> "_LLMCompleteAdapter":
        return self

    async def __call__(self, prompt: str, **kwargs: Any) -> str:
        kwargs.pop("model", None)
        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        return await openai_complete_if_cache(
            self.model,
            prompt,
            base_url=self.base_url,
            api_key=self.api_key,
            **kwargs,
        )


class _DoclingChunker:
    """LightRAG-safe chunking callable that uses registered Docling chunks."""

    def __init__(self):
        self.chunks_by_content: dict[str, list[dict[str, Any]]] = {}

    def __deepcopy__(self, memo: dict[int, Any]) -> "_DoclingChunker":
        return self

    def __call__(
        self,
        tokenizer: Any,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
    ) -> list[dict[str, Any]]:
        docling_chunks = self.chunks_by_content.get(content)
        if docling_chunks is None:
            return chunking_by_token_size(
                tokenizer,
                content,
                split_by_character,
                split_by_character_only,
                chunk_overlap_token_size,
                chunk_token_size,
            )

        chunks: list[dict[str, Any]] = []
        for index, chunk in enumerate(docling_chunks):
            chunk_content = str(chunk.get("text") or "").strip()
            if not chunk_content:
                continue

            token_count = len(tokenizer.encode(chunk_content))
            if token_count > chunk_token_size:
                logger.warning(
                    "Docling chunk %s 超过 LightRAG chunk_token_size: %s > %s",
                    index,
                    token_count,
                    chunk_token_size,
                )

            chunks.append(
                {
                    "tokens": token_count,
                    "content": chunk_content,
                    "chunk_order_index": index,
                    "metadata": chunk.get("metadata", {}),
                }
            )

        return chunks


class LightRAGIndexer:
    """基于 LightRAG 的文档索引器。

    使用 LightRAG 的双级检索系统（向量 + 知识图谱）进行文档索引和检索。
    支持配置自定义的 LLM 和 Embedding 模型。
    """

    def __init__(
        self,
        *,
        working_dir: str,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        embedding_model: str,
        embedding_dim: int,
        max_token_size: int = 8192,
        default_llm_timeout: int = 180,
        llm_model_max_async: int = 4,
        entity_extract_max_gleaning: int = 1,
        max_extract_input_tokens: int = 20480,
        chunk_token_size: int = 1200,
        chunk_overlap_token_size: int = 100,
    ):
        """初始化 LightRAG 索引器。

        Args:
            working_dir: LightRAG 工作目录，用于存储索引数据
            llm_base_url: LLM API 基础 URL
            llm_api_key: LLM API 密钥
            llm_model: LLM 模型名称
            embedding_model: Embedding 模型名称
            embedding_dim: Embedding 向量维度
            max_token_size: 最大 token 大小
            default_llm_timeout: LightRAG 单次 LLM 调用超时时间（秒）
            llm_model_max_async: LightRAG LLM 并发数
            entity_extract_max_gleaning: 实体关系补充抽取轮数
            max_extract_input_tokens: 实体关系抽取输入 token 上限
            chunk_token_size: LightRAG chunk token 大小
            chunk_overlap_token_size: LightRAG chunk overlap token 大小
        """
        self.working_dir = working_dir
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._docling_chunker = _DoclingChunker()

        # 创建工作目录
        os.makedirs(working_dir, exist_ok=True)
        logger.info(f"LightRAG 工作目录: {working_dir}")

        # 配置 Embedding 函数
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=lambda texts: self._embed_texts(texts),
        )

        # 初始化 LightRAG
        logger.info("正在初始化 LightRAG...")
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=_LLMCompleteAdapter(
                model=llm_model,
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            llm_model_name=llm_model,
            embedding_func=embedding_func,
            chunking_func=self._docling_chunker,
            default_llm_timeout=default_llm_timeout,
            llm_model_max_async=llm_model_max_async,
            entity_extract_max_gleaning=entity_extract_max_gleaning,
            max_extract_input_tokens=max_extract_input_tokens,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
        )

        # LightRAG 1.4+ 需要手动初始化存储
        # 这里必须确保初始化完成后再允许 insert/query，避免竞态导致
        # JsonDocStatusStorage not initialized。
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 某些上下文下 get_event_loop 可能报错（例如没有当前 loop），
            # 回退到 asyncio.run 创建并运行临时事件循环。
            asyncio.run(self.rag.initialize_storages())
        else:
            if loop.is_running():
                raise RuntimeError(
                    "Detected running event loop during LightRAGIndexer init. "
                    "Please use an async initialization path that awaits "
                    "rag.initialize_storages() before indexing."
                )
            loop.run_until_complete(self.rag.initialize_storages())
        logger.info("LightRAG 初始化完成")

    def index_document(self, doc: ParsedDocument) -> str:
        """索引解析后的文档。

        Args:
            doc: 解析后的文档对象

        Returns:
            str: 文档标题/ID
        """
        logger.info(f"开始索引文档: {doc.title}")

        docling_chunks = self._build_docling_chunks(doc)

        if not docling_chunks:
            logger.warning(f"文档 {doc.title} 没有有效的文本块")
            return doc.title

        logger.info(f"将插入 1 篇文档，使用 {len(docling_chunks)} 个 Docling 文本块")

        file_path = str(
            doc.metadata.get("source_file")
            or doc.metadata.get("file_name")
            or doc.title
        )
        doc_content = doc.content.strip() or "\n\n".join(
            chunk["text"] for chunk in docling_chunks
        )
        sanitized_doc_content = sanitize_text_for_encoding(doc_content)
        doc_id = self._make_doc_id(doc.title, file_path, doc_content)
        self._docling_chunker.chunks_by_content[doc_content] = docling_chunks
        self._docling_chunker.chunks_by_content[sanitized_doc_content] = docling_chunks

        try:
            self.rag.insert(
                doc_content,
                ids=doc_id,
                file_paths=file_path,
            )
        finally:
            self._docling_chunker.chunks_by_content.pop(doc_content, None)
            self._docling_chunker.chunks_by_content.pop(sanitized_doc_content, None)

        logger.info(f"文档索引完成: {doc.title}")
        return doc.title

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """查询检索。

        Args:
            query: 查询字符串
            mode: 检索模式 (naive/local/global/hybrid)
            top_k: 返回结果数量

        Returns:
            dict: 包含检索结果的字典
        """
        logger.info(f"执行查询: {query}, 模式: {mode}")

        param = QueryParam(mode=mode, top_k=top_k, **kwargs)
        result = self.rag.query(query, param=param)

        return {
            "answer": result,
            "mode": mode,
            "query": query,
        }

    def index(self, document: ParsedDocument) -> str:
        """兼容 BaseIndexer 的索引入口。"""
        return self.index_document(document)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """兼容 BaseIndexer 的检索入口。"""
        return [self.query(query, top_k=top_k)]

    def _build_docling_chunks(self, doc: ParsedDocument) -> list[dict[str, Any]]:
        """构造带文档上下文的 Docling chunk。"""
        chunk_texts: list[dict[str, Any]] = []
        source_file = doc.metadata.get("source_file") or doc.metadata.get("file_name")

        for chunk in doc.chunks:
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue

            context_parts = [f"Document: {doc.title}"]
            if source_file:
                context_parts.append(f"Source: {source_file}")
            if chunk.get("section"):
                context_parts.append(f"Section: {chunk['section']}")
            if chunk.get("page"):
                context_parts.append(f"Page: {chunk['page']}")
            if chunk.get("block_type"):
                context_parts.append(f"Block type: {chunk['block_type']}")

            chunk_texts.append(
                {
                    "text": "\n".join([*context_parts, "", text]),
                    "metadata": {
                        "title": doc.title,
                        "source_file": source_file,
                        "section": chunk.get("section"),
                        "page": chunk.get("page"),
                        "block_type": chunk.get("block_type"),
                    },
                }
            )

        return chunk_texts

    def _make_doc_id(self, doc_title: str, file_path: str, content: str) -> str:
        """生成稳定的 LightRAG 文档 ID。"""
        digest = hashlib.md5(
            f"{doc_title}:{file_path}:{content}".encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
        return f"doc-{digest}"

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """调用 bge-m3 嵌入模型生成文本向量。

        使用 OpenAI 兼容接口调用 Embedding 模型。

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.llm_base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}"
                    },
                    json={
                        "model": self.embedding_model,
                        "input": texts,
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Embedding API 错误: {response.status} - {error_text}"
                        )
                        raise ValueError(
                            f"Embedding API 返回错误: {response.status}"
                        )

                    data = await response.json()
                    embeddings = [
                        item["embedding"] for item in data["data"]
                    ]
                    embeddings_array = np.asarray(embeddings, dtype=np.float32)
                    if embeddings_array.ndim != 2:
                        raise ValueError(
                            "Embedding API 返回格式错误: 期望二维向量数组"
                        )
                    if embeddings_array.shape[1] != self.embedding_dim:
                        raise ValueError(
                            "Embedding API 返回维度不匹配: "
                            f"期望 {self.embedding_dim}, "
                            f"实际 {embeddings_array.shape[1]}"
                        )
                    logger.debug(
                        f"成功生成 {len(embeddings_array)} 个嵌入向量"
                    )
                    return embeddings_array

        except Exception as e:
            logger.error(f"Embedding 生成失败: {e}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """删除已索引的文档。

        Args:
            doc_id: 文档 ID

        Returns:
            bool: 是否成功删除
        """
        try:
            # LightRAG 可能不支持直接按文档 ID 删除
            # 需要根据实际 API 调整
            logger.warning("LightRAG 当前版本可能不支持文档删除功能")
            return False
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
