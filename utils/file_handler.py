"""知识文件发现、摘要计算和文档加载。"""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def calculate_sha256(file_path: str | Path) -> str:
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
