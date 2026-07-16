from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pet_agent.utils.config_loader import load_rerank_config
from pet_agent.utils.path_tool import get_abs_path


DEFAULT_REPO_ID = "BAAI/bge-reranker-base"
DEFAULT_LOCAL_DIR = "models/rerank/bge-reranker-base"


def main() -> None:
    config = load_rerank_config()
    repo_id = str(config.get("download_repo_id") or DEFAULT_REPO_ID)
    endpoint = str(config.get("download_endpoint") or "").strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    local_dir = get_abs_path(str(config.get("model_name") or DEFAULT_LOCAL_DIR))
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download reranker models.") from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print(f"repo_id: {repo_id}")
    print(f"endpoint: {endpoint or 'https://huggingface.co'}")
    print(f"local_dir: {local_dir}")
    print("download_status: done")


if __name__ == "__main__":
    main()
