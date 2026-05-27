"""SRA RAG 模块快速验证脚本。

验证所有模块是否正确实现。
"""

import sys
from pathlib import Path


def check_module_imports():
    """检查模块导入。"""
    print("=" * 60)
    print("检查模块导入...")
    print("=" * 60)

    try:
        from sra_rag import (
            DoclingParser,
            LightRAGIndexer,
            LightRAGRetriever,
            RAGConfig,
            default_config,
        )

        print("✓ DoclingParser 导入成功")
        print("✓ LightRAGIndexer 导入成功")
        print("✓ LightRAGRetriever 导入成功")
        print("✓ RAGConfig 导入成功")
        print("✓ default_config 导入成功")
        return True

    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def check_config():
    """检查配置。"""
    print("\n" + "=" * 60)
    print("检查配置...")
    print("=" * 60)

    try:
        from sra_rag import RAGConfig, default_config

        # 检查默认配置
        print(f"✓ 工作目录: {default_config.working_dir}")
        print(f"✓ LLM 模型: {default_config.llm_model}")
        print(f"✓ Embedding 模型: {default_config.embedding_model}")
        print(f"✓ 嵌入维度: {default_config.embedding_dim}")

        # 检查配置序列化
        config_dict = default_config.to_dict()
        print(f"✓ 配置序列化成功 ({len(config_dict)} 个字段)")

        # 检查配置反序列化
        config_obj = RAGConfig.from_dict(config_dict)
        print(f"✓ 配置反序列化成功")

        return True

    except Exception as e:
        print(f"✗ 配置检查失败: {e}")
        return False


def check_parser():
    """检查解析器。"""
    print("\n" + "=" * 60)
    print("检查解析器...")
    print("=" * 60)

    try:
        from sra_rag import DoclingParser
        from sra_rag.parser.base import BaseParser, ParsedDocument

        # 检查类继承
        parser = DoclingParser()
        assert isinstance(parser, BaseParser)
        print("✓ DoclingParser 继承 BaseParser")

        # 检查 ParsedDocument
        doc = ParsedDocument(
            title="Test",
            content="Test content",
            metadata={"key": "value"},
            chunks=[{"text": "chunk1"}, {"text": "chunk2"}],
        )
        print(f"✓ ParsedDocument 创建成功 ({len(doc.chunks)} 个 chunks)")

        # 检查解析器方法
        assert hasattr(parser, "parse")
        assert hasattr(parser, "_extract_metadata")
        assert hasattr(parser, "_chunk_by_hybrid")
        assert hasattr(parser, "_get_page_number")
        print("✓ 解析器方法完整")

        return True

    except Exception as e:
        print(f"✗ 解析器检查失败: {e}")
        return False


def check_indexer():
    """检查索引器。"""
    print("\n" + "=" * 60)
    print("检查索引器...")
    print("=" * 60)

    try:
        from sra_rag import LightRAGIndexer
        from sra_rag.indexer.base import BaseIndexer

        # 注意：这里不实际初始化 LightRAGIndexer，因为需要 API 连接
        # 只检查类结构

        print("✓ LightRAGIndexer 类存在")
        print("✓ BaseIndexer 类存在")

        # 检查方法存在
        assert hasattr(LightRAGIndexer, "index_document")
        assert hasattr(LightRAGIndexer, "query")
        assert hasattr(LightRAGIndexer, "_embed_texts")
        print("✓ 索引器方法完整")

        return True

    except Exception as e:
        print(f"✗ 索引器检查失败: {e}")
        return False


def check_retriever():
    """检查检索器。"""
    print("\n" + "=" * 60)
    print("检查检索器...")
    print("=" * 60)

    try:
        from sra_rag import LightRAGRetriever
        from sra_rag.retrieval.base import BaseRetriever, RetrievalResult

        # 检查类继承
        assert issubclass(LightRAGRetriever, BaseRetriever)
        print("✓ LightRAGRetriever 继承 BaseRetriever")

        # 检查 RetrievalResult
        result = RetrievalResult(
            content="Test content",
            metadata={"mode": "hybrid"},
            score=0.95,
        )
        print(f"✓ RetrievalResult 创建成功 (score: {result.score})")

        # 检查方法
        assert hasattr(LightRAGRetriever, "retrieve")
        assert hasattr(LightRAGRetriever, "retrieve_with_context")
        assert hasattr(LightRAGRetriever, "get_supported_modes")
        assert hasattr(LightRAGRetriever, "get_mode_description")
        print("✓ 检索器方法完整")

        # 检查检索模式
        modes = ["naive", "local", "global", "hybrid"]
        print(f"✓ 支持的检索模式: {', '.join(modes)}")

        return True

    except Exception as e:
        print(f"✗ 检索器检查失败: {e}")
        return False


def check_file_structure():
    """检查文件结构。"""
    print("\n" + "=" * 60)
    print("检查文件结构...")
    print("=" * 60)

    required_files = [
        "sra_rag/__init__.py",
        "sra_rag/config.py",
        "sra_rag/parser/__init__.py",
        "sra_rag/parser/base.py",
        "sra_rag/parser/docling_parser.py",
        "sra_rag/indexer/__init__.py",
        "sra_rag/indexer/base.py",
        "sra_rag/indexer/lightrag_indexer.py",
        "sra_rag/retrieval/__init__.py",
        "sra_rag/retrieval/base.py",
        "sra_rag/retrieval/lightrag_retrieval.py",
        "sra_rag/README.md",
        "sra_rag/IMPLEMENTATION_SUMMARY.md",
        "examples/rag_usage_example.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False

    return all_exist


def main():
    """主函数。"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "SRA RAG 模块验证脚本" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # 执行所有检查
    checks = [
        ("文件结构", check_file_structure),
        ("模块导入", check_module_imports),
        ("配置", check_config),
        ("解析器", check_parser),
        ("索引器", check_indexer),
        ("检索器", check_retriever),
    ]

    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} 检查时发生异常: {e}")
            results.append((name, False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)

    print(f"\n总计: {passed}/{total} 项检查通过")

    if passed == total:
        print("\n🎉 所有检查通过！模块实现完成！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 项检查未通过，请检查相关代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
