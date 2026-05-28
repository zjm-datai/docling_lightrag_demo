
```bash
uv run python examples/embed_documents_example.py resources/001_test.pdf \
  --embed-mode index \
  --working-dir ./sra_rag_data_test \
  --llm-base-url http://211.90.240.240:30001/v1 \
  --llm-api-key gpustack_ba7863cefc4126b2_4ecbb5f42c239594a33e67ef49bdce9d \
  --llm-model Qwen3-30B-A3B-GPTQ-Int4 \
  --embedding-model bge-m3 \
  --embedding-dim 1024
```




