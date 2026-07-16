"""
MD5文件处理
"""


from __future__ import annotations

import hashlib
import json
from pathlib import Path


# 文本转成MD5十六进制内容
def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# 文件转成MD5十六进制内容
def md5_file(path: Path, chunk_size: int = 4096) -> str:
    md5_obj = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            md5_obj.update(chunk)
    return md5_obj.hexdigest()

# MD5文本内容转换成json内容
def md5_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return md5_text(payload)

# 哈希存储的异常处理
try:
    from pet_agent.storage.hash_store import ContentHashStore
except ImportError:
    ContentHashStore = None  # type: ignore
