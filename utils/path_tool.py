# 文件用途：确定项目根目录并把配置中的相对路径安全转换成绝对路径。
# 调用关系：配置、日志、模型、RAG、Skill 和存储模块都调用本文件；本文件只依赖 pathlib。
# 修改易踩坑：项目根目录计算错误会让配置、数据库和知识库全部指向错误位置。
"""项目路径工具。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> str:
    """返回项目根目录的绝对路径。"""
    return str(PROJECT_ROOT)


def get_abs_path(path: str | Path) -> str:
    """将项目内相对路径转换为绝对路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())
