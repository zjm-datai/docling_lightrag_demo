"""测试 PDF 文档处理流程。

处理指定的 PDF 文件，展示完整的解析、索引和检索流程，
并显示数据存储位置。
"""

import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """打印分隔线。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_pdf_file():
    """检查 PDF 文件是否存在。"""
    print_section("步骤 0: 检查 PDF 文件")

    pdf_path = Path(
        "c:/Desktop/project/science_research/sra_agent/resources/"
        "001_test.pdf"
    )

    if not pdf_path.exists():
        print(f"✗ 文件不存在: {pdf_path}")
        return None

    file_size = pdf_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✓ 文件存在: {pdf_path.name}")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  文件路径: {pdf_path.absolute()}")

    return pdf_path


def test_parsing(pdf_path: Path, output_dir: str = "./sra_rag_data_test"):
    """测试文档解析。"""
    print_section("步骤 1: 文档解析 (Docling Parser)")

    from sra_rag import DoclingParser

    parser = DoclingParser()

    print("\n开始解析文档...")
    print("这可能需要几分钟时间，取决于文档大小和复杂度...")

    try:
        parsed_doc = parser.parse(pdf_path)
        exported_files = parser.export_parsed_document(parsed_doc, output_dir)

        print(f"\n✓ 解析成功！")
        print(f"\n文档信息:")
        print(f"  标题: {parsed_doc.title}")
        print(f"  内容长度: {len(parsed_doc.content):,} 字符")
        print(f"  文本块数量: {len(parsed_doc.chunks)}")
        print(f"  表格数量: {parsed_doc.metadata.get('num_tables', 0)}")
        print(f"  图片数量: {parsed_doc.metadata.get('num_pictures', 0)}")

        print(f"\n元数据:")
        for key, value in parsed_doc.metadata.items():
            print(f"  {key}: {value}")

        print(f"\nDocling 解析结果已保存:")
        for name, path in exported_files.items():
            print(f"  {name}: {path.absolute()}")

        # 显示前 5 个 chunk 的信息
        print(f"\n前 5 个文本块示例:")
        for i, chunk in enumerate(parsed_doc.chunks[:5], 1):
            print(f"\n  Chunk {i}:")
            print(f"    文本长度: {len(chunk.get('text', ''))} 字符")
            print(f"    章节: {chunk.get('section', 'N/A')}")
            print(f"    页面: {chunk.get('page', 'N/A')}")
            print(f"    类型: {chunk.get('block_type', 'N/A')}")
            print(f"    文本预览: {chunk.get('text', '')[:100]}...")

        return parsed_doc

    except Exception as e:
        print(f"\n✗ 解析失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_indexing(parsed_doc, working_dir: str = "./sra_rag_data_test"):
    """测试文档索引。"""
    print_section("步骤 2: 文档索引 (LightRAG Indexer)")

    from sra_rag import LightRAGIndexer

    print(f"\n工作目录: {working_dir}")
    print("开始初始化 LightRAG...")
    print("这将创建索引文件并构建知识图谱...")
    print("首次索引可能需要较长时间（取决于文档大小和 LLM 响应速度）...")

    try:
        indexer = LightRAGIndexer(
            working_dir=working_dir,
            llm_base_url="http://211.90.240.240:30001/v1",
            llm_api_key="replace-with-api-key",
            llm_model="Qwen3-30B-A3B-GPTQ-Int4",
            embedding_model="bge-m3",
            embedding_dim=1024,
        )

        print("\n开始索引文档...")
        doc_id = indexer.index_document(parsed_doc)

        print(f"\n✓ 索引成功！")
        print(f"  文档 ID: {doc_id}")

        return indexer

    except Exception as e:
        print(f"\n✗ 索引失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def show_storage_info(working_dir: str):
    """显示存储信息。"""
    print_section("数据存储位置说明")

    working_path = Path(working_dir)

    print(f"\n📁 主工作目录: {working_path.absolute()}")
    print(f"   所有索引数据都存储在这个目录中")

    if working_path.exists():
        print(f"\n📂 已生成的文件:")

        # 列出所有文件
        for file_path in working_path.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                size_str = (
                    f"{file_size / (1024*1024):.2f} MB"
                    if file_size > 1024 * 1024
                    else f"{file_size / 1024:.2f} KB"
                    if file_size > 1024
                    else f"{file_size} B"
                )
                relative_path = file_path.relative_to(working_path)
                print(f"   📄 {relative_path} ({size_str})")

        print(f"\n📊 文件类型说明:")
        print(f"   • kv_store_full_docs.json - 完整文档存储")
        print(f"   • kv_store_llm_response_cache.json - LLM 响应缓存")
        print(f"   • kv_store_text_chunks.json - 文本块存储")
        print(f"   • kv_store_entities.json - 实体存储")
        print(f"   • kv_store_entity_chunks.json - 实体块存储")
        print(f"   • kv_store_relationships.json - 关系存储")
        print(f"   • graph_chunk_entity_relation.graphml - 知识图谱（GraphML 格式）")
        print(f"   • lightrag.log - 运行日志")

    print(f"\n💡 重要说明:")
    print(f"   1. 这些文件已在 .gitignore 中配置，不会被提交到 Git")
    print(f"   2. 可以安全删除整个目录重新索引")
    print(f"   3. 保留这些文件可以避免重复索引（节省时间和 API 调用）")
    print(f"   4. 如果要更换 Embedding 模型，需要删除目录重新索引")


def test_querying(indexer):
    """测试检索功能。"""
    print_section("步骤 3: 文档检索 (LightRAG Retriever)")

    from sra_rag import LightRAGRetriever

    if indexer is None:
        print("✗ 索引器未初始化，跳过检索测试")
        return

    retriever = LightRAGRetriever(indexer)

    # 测试不同的查询
    test_queries = [
        "What is the role of Interleukin-4 in Atopic Dermatitis?",
        "Why does IL-4 matter in dermatitis?",
        "What are the clinical implications?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 70)

        try:
            # 使用 hybrid 模式检索
            result = retriever.retrieve(query, mode="hybrid")

            print(f"✓ 检索成功")
            print(f"  答案长度: {len(result[0].content)} 字符")
            print(f"  检索模式: {result[0].metadata.get('mode')}")
            print(f"\n  答案预览:")
            print(f"  {result[0].content[:200]}...")

        except Exception as e:
            print(f"✗ 检索失败: {e}")


def compare_modes(indexer):
    """比较不同检索模式。"""
    print_section("步骤 4: 检索模式对比")

    from sra_rag import LightRAGRetriever

    if indexer is None:
        print("✗ 索引器未初始化，跳过模式对比")
        return

    retriever = LightRAGRetriever(indexer)

    query = "What is the relationship between IL-4 and skin inflammation?"
    modes = ["naive", "local", "global", "hybrid"]

    print(f"\n查询: {query}\n")

    for mode in modes:
        print(f"模式: {mode}")
        print(f"  描述: {retriever.get_mode_description(mode)}")

        try:
            result = retriever.retrieve(query, mode=mode)
            print(f"  ✓ 答案长度: {len(result[0].content)} 字符")
        except Exception as e:
            print(f"  ✗ 检索失败: {e}")

        print()


def main():
    """主函数。"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SRA RAG 模块 - PDF 处理测试".center(68) + "█")
    print("█" + "  测试完整的解析、索引、检索流程".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # 配置
    working_dir = "./sra_rag_data_test"

    # 步骤 0: 检查文件
    pdf_path = check_pdf_file()
    if pdf_path is None:
        return

    # 步骤 1: 解析文档
    parsed_doc = test_parsing(pdf_path, working_dir)
    if parsed_doc is None:
        return

    # 步骤 2: 索引文档
    indexer = test_indexing(parsed_doc, working_dir)

    # 显示存储信息
    show_storage_info(working_dir)

    # 步骤 3: 测试检索
    if indexer:
        test_querying(indexer)
        compare_modes(indexer)

    # 总结
    print_section("测试总结")
    print(f"\n✓ 测试完成！")
    print(f"\n数据存储位置:")
    print(f"  工作目录: {Path(working_dir).absolute()}")
    print(f"  包含文件: 向量索引、知识图谱、缓存等")
    print(f"\n下次使用:")
    print(f"  - 保留该目录可以避免重复索引")
    print(f"  - 删除该目录可以重新索引")
    print(f"  - 目录已在 .gitignore 中排除")
    print()


if __name__ == "__main__":
    main()
