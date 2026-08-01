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
