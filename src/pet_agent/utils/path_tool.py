"""
项目根目录定位工具：
    统一处理项目路径，避免因为运行位置不同导致找不到文件
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]   # 项目根目录: parents从0开始


# 获取项目根目录
def get_project_root() -> Path:
    return PROJECT_ROOT

# 获取项目绝对路径
def get_abs_path(path: str | Path) -> Path:
    candidate = Path(path)   # 将输入变量变成Path对象
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate

