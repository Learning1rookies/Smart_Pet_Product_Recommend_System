# Local Models

这里放本地模型文件。

## BGE reranker

默认路径：

```text
models/rerank/bge-reranker-base
```

下载命令：

```powershell
py scripts\core\download_reranker_model.py
```

下载完成后，`config/rerank.yml` 会从本地目录加载模型，不再访问 HuggingFace。

模型权重体积较大，不随公开 GitHub 仓库分发。克隆项目后使用上述脚本下载，`models/rerank/` 已由 `.gitignore` 排除。
