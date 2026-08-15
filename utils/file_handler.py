# 文件用途：发现允许的知识文件、计算稳定摘要并加载成带学科元数据的 LangChain 文档。
# 调用关系：vector_store.py 调用本文件；本文件读取 data/knowledge 下的纯文本或 Markdown 文件。
# 修改易踩坑：文件类型白名单和学科识别会影响索引；不要把生成目录或隐藏文件加入知识库。
"""知识文件发现、摘要计算和文档加载。"""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def calculate_sha256(file_path: str | Path) -> str:
    """分块读取知识文件并计算 SHA-256 摘要，用于判断文件是否变化。"""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"知识文件不存在：{path}")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_knowledge_files(
    directory: str | Path,
    allowed_types: tuple[str, ...],
) -> tuple[Path, ...]:
    """递归列出知识目录下所有符合扩展名要求的文件并稳定排序。"""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"知识库目录不存在：{root}")

    normalized = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in allowed_types
    }
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    ]
    return tuple(sorted(files, key=lambda item: item.as_posix()))


def load_documents(file_path: str | Path) -> list[Document]:
    """根据扩展名使用对应加载器，将 TXT 或 PDF 转换为文档列表。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=False,
        ).load()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    raise ValueError(f"不支持的知识文件类型：{path.suffix}")
