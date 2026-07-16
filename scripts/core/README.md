# Core Scripts

这里放核心本地流水线脚本。

## 常用命令

```powershell
py scripts\core\run_local_pipeline.py
```

这个命令会执行：

```text
1. scripts/WorldCloud_Analysis/generate_review_tag_evidence.py
2. scripts/core/import_to_sqlite.py
3. scripts/core/build_vector_store.py
```

最终让 Agent 可以查询：

```text
data/runtime/pet_products.sqlite3
data/runtime/chroma
```

## 文件说明

`run_local_pipeline.py`

核心本地流水线入口。适合每次重新抓取或修改 raw/processed 数据后运行。

`import_to_sqlite.py`

把 `data/raw` 和 `data/processed` 导入 SQLite。

`build_vector_store.py`

把商品摘要和 `data/processed/review_tag_evidence.csv` 中的代表性评论证据写入 ChromaDB。
默认每个 `product_id + tag_name + evidence_type` 保留 1 条代表证据，避免把 2 万多条原始评论全部塞入向量库。

`download_reranker_model.py`

把 BGE reranker 下载到 `models/rerank/bge-reranker-base`，供证据精排从本地加载。

核心 Agent 运行依赖本地 SQLite 与 ChromaDB。
