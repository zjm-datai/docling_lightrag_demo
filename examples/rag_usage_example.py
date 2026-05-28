"""SRA RAG 模块使用示例。

展示如何使用 sra_rag 模块进行文档解析、索引和检索。
"""

from pathlib import Path

from sra_rag import (
    DoclingParser,
    LightRAGIndexer,
    LightRAGRetriever,
    SraRagOptions,
    create_sra_rag,
)


EXAMPLE_OPTIONS = SraRagOptions(
    working_dir="./sra_rag_data",
    llm_base_url="http://your-api/v1",
    llm_api_key="your-api-key",
    llm_model="your-chat-model",
    embedding_model="your-embedding-model",
    embedding_dim=1024,
    max_token_size=8192,
)


def example_basic_usage():
    """基本使用示例。"""
    print("=" * 60)
    print("SRA RAG 基本使用示例")
    print("=" * 60)

    # 1. 解析文档
    print("\n步骤 1: 解析文档")
    parser = DoclingParser()

    # 示例文件路径（需要替换为实际文件）
    file_path = Path("example.pdf")

    if file_path.exists():
        parsed_doc = parser.parse(file_path)
        print(f"  文档标题: {parsed_doc.title}")
        print(f"  文本块数量: {len(parsed_doc.chunks)}")
        print(f"  内容长度: {len(parsed_doc.content)} 字符")
        print(f"  元数据: {parsed_doc.metadata}")
    else:
        print(f"  跳过: 文件 {file_path} 不存在")
        return

    # 2. 索引文档
    print("\n步骤 2: 索引文档")
    indexer = LightRAGIndexer(
        working_dir=EXAMPLE_OPTIONS.working_dir,
        llm_base_url=EXAMPLE_OPTIONS.llm_base_url,
        llm_api_key=EXAMPLE_OPTIONS.llm_api_key,
        llm_model=EXAMPLE_OPTIONS.llm_model,
        embedding_model=EXAMPLE_OPTIONS.embedding_model,
        embedding_dim=EXAMPLE_OPTIONS.embedding_dim,
        max_token_size=EXAMPLE_OPTIONS.max_token_size,
    )
    doc_id = indexer.index_document(parsed_doc)
    print(f"  文档 ID: {doc_id}")

    # 3. 检索
    print("\n步骤 3: 执行检索")
    retriever = LightRAGRetriever(indexer)

    query = "请解释主要概念"
    print(f"  查询: {query}")

    results = retriever.retrieve(query, mode="hybrid")
    print(f"  检索结果数量: {len(results)}")
    print(f"  答案长度: {len(results[0].content)} 字符")
    print(f"  检索模式: {results[0].metadata.get('mode')}")


def example_different_modes():
    """不同检索模式示例。"""
    print("\n" + "=" * 60)
    print("不同检索模式对比")
    print("=" * 60)

    # 假设已经初始化了 indexer
    # rag = create_sra_rag(EXAMPLE_OPTIONS)
    # retriever = rag.retriever

    modes = ["naive", "local", "global", "hybrid"]
    query = "什么是图神经网络？"

    for mode in modes:
        print(f"\n模式: {mode}")
        print(f"  描述: {LightRAGRetriever.get_mode_description(None, mode)}")
        # result = retriever.retrieve(query, mode=mode)
        # print(f"  答案: {result[0].content[:100]}...")


def example_factory():
    """统一入口示例。"""
    print("\n" + "=" * 60)
    print("统一入口示例")
    print("=" * 60)

    print("外部项目负责读取配置，然后传入 create_sra_rag。")
    print(f"  工作目录: {EXAMPLE_OPTIONS.working_dir}")
    print(f"  LLM 模型: {EXAMPLE_OPTIONS.llm_model}")
    print(f"  Embedding 模型: {EXAMPLE_OPTIONS.embedding_model}")
    print(f"  嵌入维度: {EXAMPLE_OPTIONS.embedding_dim}")


def example_batch_processing():
    """批量处理示例。"""
    print("\n" + "=" * 60)
    print("批量处理示例")
    print("=" * 60)

    parser = DoclingParser()
    rag = create_sra_rag(EXAMPLE_OPTIONS)
    indexer = rag.indexer

    # 批量解析和索引
    doc_dir = Path("./documents")
    if doc_dir.exists():
        for file_path in doc_dir.glob("*.pdf"):
            print(f"\n处理: {file_path.name}")
            try:
                parsed_doc = parser.parse(file_path)
                indexer.index_document(parsed_doc)
                print(f"  ✓ 完成")
            except Exception as e:
                print(f"  ✗ 失败: {e}")
    else:
        print(f"目录 {doc_dir} 不存在")


if __name__ == "__main__":
    # 运行示例
    example_basic_usage()
    example_different_modes()
    example_factory()
    example_batch_processing()
