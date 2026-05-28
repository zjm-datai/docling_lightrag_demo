# PDF 处理数据存储位置说明

## 📊 测试结果总结

### PDF 解析成功！✅

您的 PDF 文件已成功解析：
- **文件**: `001_2025_Interleukin-4 and Atopic Dermatitis_ Why Does it Matter_ A Narrative Review.pdf`
- **大小**: 1.39 MB
- **解析时间**: ~69 秒
- **文本块数量**: 316 个
- **内容长度**: 77,631 字符
- **表格数量**: 3 个
- **图片数量**: 2 个

---

## 💾 数据存储位置

### 主工作目录

```
C:\Desktop\project\science_research\sra_agent\sra_rag_data_test\
```

宿主项目也可以显式传入其他工作目录，例如：
```
C:\Desktop\project\science_research\sra_agent\sra_rag_data\
```

### 具体文件列表

在工作目录中会生成以下文件：

```
sra_rag_data_test/
├── graph_chunk_entity_relation.graphml        # 知识图谱（GraphML格式）
├── kv_store_full_docs.json                    # 完整文档存储
├── kv_store_llm_response_cache.json           # LLM响应缓存
├── kv_store_text_chunks.json                  # 文本块向量数据
├── kv_store_entities.json                     # 实体向量数据
├── kv_store_entity_chunks.json                # 实体块向量数据
├── kv_store_relationships.json                # 关系向量数据
├── vdb_chunks.json                            # 向量数据库（chunks）
├── vdb_entities.json                          # 向量数据库（entities）
├── vdb_relationships.json                     # 向量数据库（relationships）
└── lightrag.log                               # LightRAG运行日志
```

### 文件说明

| 文件类型 | 说明 | 大小预估 |
|---------|------|---------|
| **graphml** | 知识图谱结构文件，包含实体和关系 | 100KB - 1MB |
| **kv_store_*.json** | 键值存储，包含文档、实体、关系等 | 500KB - 5MB |
| **vdb_*.json** | 向量数据库，存储 embedding 向量 | 1MB - 10MB |
| **lightrag.log** | 运行日志文件 | 10KB - 100KB |

---

## 🗂️ 数据分类说明

### 1. **文档原始数据**
- `kv_store_full_docs.json` - 完整的文档内容
- `kv_store_text_chunks.json` - 分块后的文本（316个chunks）

### 2. **知识图谱数据**
- `graph_chunk_entity_relation.graphml` - 图结构文件
  - 可以使用 Gephi、Neo4j 等工具可视化
  - 包含实体节点和关系边
- `kv_store_entities.json` - 提取的实体（如疾病、药物、基因等）
- `kv_store_relationships.json` - 实体之间的关系

### 3. **向量索引数据**
- `vdb_chunks.json` - 文本块的向量表示（1024维）
- `vdb_entities.json` - 实体的向量表示
- `vdb_relationships.json` - 关系的向量表示

### 4. **缓存数据**
- `kv_store_llm_response_cache.json` - LLM API 响应缓存
  - 避免重复调用 API
  - 加速重复索引

---

## 🔍 数据访问示例

### 查看生成的文件

```python
from pathlib import Path

working_dir = Path("./sra_rag_data_test")

# 列出所有文件
for file_path in working_dir.rglob("*"):
    if file_path.is_file():
        size = file_path.stat().st_size
        print(f"{file_path.name}: {size/1024:.2f} KB")
```

### 读取知识图谱

```python
import networkx as nx

# 读取 GraphML 文件
G = nx.read_graphml("sra_rag_data_test/graph_chunk_entity_relation.graphml")

print(f"节点数量: {G.number_of_nodes()}")
print(f"边数量: {G.number_of_edges()}")

# 查看部分实体
nodes = list(G.nodes())[:10]
print("前10个实体:", nodes)
```

### 查看向量数据

```python
import json

# 读取向量数据
with open("sra_rag_data_test/vdb_chunks.json", "r") as f:
    chunks_data = json.load(f)

print(f"文本块数量: {len(chunks_data)}")
print(f"向量维度: {len(chunks_data[0]['vector'])}")  # 应该是 1024
```

---

## ⚙️ 重要说明

### 1. **Git 排除**
这些文件已在 `.gitignore` 中配置：
```gitignore
# RAG Data
sra_rag_data/
sra_rag_data_test/
*.graphml
kv_store_*.json
```

**不会被提交到 Git 仓库**

### 2. **可以安全删除**
- 删除整个 `sra_rag_data_test/` 目录是安全的
- 删除后重新索引会重新生成
- 保留可以避免重复索引（节省时间和 API 调用）

### 3. **何时需要重新索引**
需要删除目录并重新索引的情况：
- 更换 Embedding 模型（如从 bge-m3 换成其他模型）
- 修改分块策略
- 数据损坏
- 想要完全重建知识图谱

### 4. **存储空间**
对于您的 PDF（1.39 MB，316 个文本块）：
- 预估总存储空间：**5-15 MB**
- 主要包括：向量数据 + 知识图谱 + 缓存

---

## 🚀 下一步

### 当前状态
✅ PDF 解析成功（316个文本块）
⚠️ LightRAG 索引需要修复初始化问题

### 修复方案
已更新 `lightrag_indexer.py` 添加存储初始化代码。

### 重新运行测试

```bash
# 清理之前的测试数据
rm -rf sra_rag_data_test/

# 重新运行测试
.venv/Scripts/python.exe tests/test_pdf_processing.py
```

### 预期结果
1. 解析 PDF（~70秒）
2. 初始化 LightRAG（~10秒）
3. 索引文档（可能需要几分钟，取决于 LLM 响应速度）
4. 生成所有数据文件到 `sra_rag_data_test/` 目录
5. 执行检索测试

---

## 📁 目录结构总览

```
sra_agent/
├── resources/                                    # 原始 PDF 文件
│   └── 001_2025_Interleukin-4....pdf            # 您的 PDF
│
├── sra_rag_data_test/                            # 索引数据（自动生成）
│   ├── graph_chunk_entity_relation.graphml      # 知识图谱
│   ├── kv_store_*.json                          # 键值存储
│   └── vdb_*.json                               # 向量数据库
│
├── sra_rag/                                      # 源代码
│   ├── parser/                                   # 解析器
│   ├── indexer/                                  # 索引器
│   └── retrieval/                                # 检索器
│
└── tests/                                        # 测试脚本
    └── test_pdf_processing.py                    # PDF 处理测试
```

---

## 💡 建议

1. **保留索引数据**：避免重复索引，节省 API 调用
2. **定期检查存储**：使用 `du -sh sra_rag_data_test/` 查看大小
3. **保存宿主项目参数**：虽然数据文件不提交，但传给 `create_sra_rag(...)` 的参数要由宿主项目管理
4. **监控 API 使用**：索引过程会调用 LLM API

---

## 📞 问题排查

如果遇到问题：

1. **检查文件是否生成**
   ```bash
   ls -la sra_rag_data_test/
   ```

2. **查看日志**
   ```bash
   cat sra_rag_data_test/lightrag.log
   ```

3. **清理重新索引**
   ```bash
   rm -rf sra_rag_data_test/
   .venv/Scripts/python.exe tests/test_pdf_processing.py
   ```
