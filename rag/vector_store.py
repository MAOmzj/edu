"""教育知识库的增量索引和检索。

这一层负责“知识文件”，vector_backends.py 负责“向量数据库”。主要流程：

    data/knowledge 中的 TXT/PDF
      -> 计算 SHA-256 判断是否变化
      -> 切成小片段并补充学科/来源元数据
      -> Embedding 向量化后写入 SQLite 或 Chroma
      -> manifest JSON 记录每个文件对应哪些 chunk_id

manifest 只是增量同步清单，不保存正文和向量；真正数据在所选向量后端中。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import create_embedding_model
from rag.vector_backends import (
    ChromaVectorBackend,
    SQLiteVectorBackend,
    VectorBackend,
)
from utils.config_handler import app_conf, knowledge_conf
from utils.file_handler import calculate_sha256, list_knowledge_files, load_documents
from utils.logger_handler import logger
from utils.path_tool import PROJECT_ROOT, get_abs_path


# V4 表示默认向量已切换为 text-embedding-v4。
# 版本提升后，旧的本地哈希向量不会冒充在线语义向量。
MANIFEST_VERSION = 4
SUPPORTED_VECTOR_BACKENDS = {"sqlite", "chroma"}


@dataclass
class IndexSyncReport:
    """一次同步的统计结果，方便命令行用 JSON 告诉用户处理了多少内容。"""

    added_files: int = 0
    updated_files: int = 0
    removed_files: int = 0
    unchanged_files: int = 0
    indexed_chunks: int = 0


class EducationVectorStore:
    def __init__(
        self,
        embedding_model: Embeddings | None = None,
        backend_name: str | None = None,
    ):
        """读取配置，并准备可切换的 SQLite/Chroma 后端和切分器。"""
        configured_backend = (
            backend_name
            or os.getenv("EDUCATION_VECTOR_BACKEND")
            or knowledge_conf["vector_backend"]
        )
        self.backend_name = str(configured_backend).strip().lower()
        if self.backend_name not in SUPPORTED_VECTOR_BACKENDS:
            choices = "、".join(sorted(SUPPORTED_VECTOR_BACKENDS))
            raise ValueError(f"vector_backend 只能是：{choices}")

        self._embedding_model = embedding_model
        self._vector_store: VectorBackend | None = None
        self._sync_lock = threading.RLock()
        self.data_path = Path(get_abs_path(knowledge_conf["data_path"]))
        self.sqlite_database_path = Path(
            get_abs_path(knowledge_conf["sqlite_database_path"])
        )
        self.chroma_persist_directory = Path(
            get_abs_path(knowledge_conf["chroma_persist_directory"])
        )
        manifest_key = f"{self.backend_name}_manifest_path"
        self.manifest_path = Path(get_abs_path(knowledge_conf[manifest_key]))
        self.persist_directory = (
            self.sqlite_database_path
            if self.backend_name == "sqlite"
            else self.chroma_persist_directory
        )
        self.allowed_types = tuple(knowledge_conf["allowed_file_types"])
        self.k = int(knowledge_conf["k"])
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(knowledge_conf["chunk_size"]),
            chunk_overlap=int(knowledge_conf["chunk_overlap"]),
            separators=list(knowledge_conf["separators"]),
            length_function=len,
        )

    def _get_store(self) -> VectorBackend:
        """按照配置按需创建并复用 SQLite 或 Chroma 向量后端。"""
        if self._vector_store is None:
            # 延迟创建很重要：`main.py --status` 只看清单，不应调用在线 Embedding，
            # 也不应因为没有 DASHSCOPE_API_KEY 而无法查看状态。
            embedding_model = self._embedding_model or create_embedding_model()
            if self.backend_name == "sqlite":
                self._vector_store = SQLiteVectorBackend(
                    database_path=self.sqlite_database_path,
                    collection_name=knowledge_conf["collection_name"],
                    embedding_model=embedding_model,
                    embedding_batch_size=int(
                        knowledge_conf["embedding_batch_size"]
                    ),
                    vector_weight=float(
                        knowledge_conf["hybrid_vector_weight"]
                    ),
                    bm25_weight=float(
                        knowledge_conf["hybrid_bm25_weight"]
                    ),
                    bm25_k1=float(knowledge_conf["bm25_k1"]),
                    bm25_b=float(knowledge_conf["bm25_b"]),
                )
            else:
                self._vector_store = ChromaVectorBackend(
                    persist_directory=self.chroma_persist_directory,
                    collection_name=knowledge_conf["collection_name"],
                    embedding_model=embedding_model,
                    vector_weight=float(
                        knowledge_conf["hybrid_vector_weight"]
                    ),
                    bm25_weight=float(
                        knowledge_conf["hybrid_bm25_weight"]
                    ),
                    bm25_k1=float(knowledge_conf["bm25_k1"]),
                    bm25_b=float(knowledge_conf["bm25_b"]),
                )
        return self._vector_store

    def _load_manifest(self) -> dict[str, Any]:
        """读取索引清单；文件不存在或版本过期时返回空清单。"""
        empty_manifest = {
            "version": MANIFEST_VERSION,
            "backend": self.backend_name,
            "files": {},
        }
        if not self.manifest_path.is_file():
            return empty_manifest
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"索引清单损坏：{self.manifest_path}") from exc
        if (
            manifest.get("version") != MANIFEST_VERSION
            or manifest.get("backend") != self.backend_name
        ):
            return empty_manifest
        manifest.setdefault("files", {})
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """先写入临时文件再原子替换，安全保存最新索引清单。"""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    def _relative_source(self, path: Path) -> str:
        """将知识文件绝对路径转换成适合写入文档元数据的相对路径。"""
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative = path.resolve().relative_to(self.data_path.resolve())
            return (Path("data/knowledge") / relative).as_posix()

    @staticmethod
    def _subject_for(path: Path) -> str:
        """根据知识文件名开头判断学科，无法识别时返回“综合”。"""
        for subject in app_conf["supported_subjects"]:
            if path.stem.startswith(subject):
                return subject
        return "综合"

    #生成ids：relative_source file_hash index组成sha256
    @staticmethod
    def _chunk_id(relative_source: str, file_hash: str, index: int) -> str:
        """根据来源、文件摘要和分片序号生成稳定的向量分片编号。"""
        raw = f"{relative_source}:{file_hash}:{index}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _split_file(self, path: Path, file_hash: str) -> tuple[list[Document], list[str]]:
        """加载并切分一个知识文件，同时补充元数据和生成分片编号。"""
        relative_source = self._relative_source(path)
        subject = self._subject_for(path)
        source_documents = load_documents(path)
        for document in source_documents:
            document.metadata = {
                **document.metadata,
                "source": relative_source,
                "subject": subject,
                "file_hash": file_hash,
            }

        chunks = self.splitter.split_documents(source_documents)
        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            ids.append(self._chunk_id(relative_source, file_hash, index))
        return chunks, ids

    def sync(self) -> IndexSyncReport:
        """比较文件摘要与索引清单，增量添加、更新或删除向量分片。"""
        with self._sync_lock:
            return self._sync_locked()

    #开始遍历
    def _sync_locked(self) -> IndexSyncReport:
        """在已经取得同步锁的情况下完成一次增量索引更新。"""
        report = IndexSyncReport()
        files = list_knowledge_files(self.data_path, self.allowed_types)
        current = {
            self._relative_source(path): {
                "path": path,
                "sha256": calculate_sha256(path),
            }
            for path in files
        }
        manifest = self._load_manifest()
        previous: dict[str, Any] = manifest["files"]
        store = self._get_store()

        # 第一步：清理“清单里有、磁盘上已经没有”的旧文件分片。
        for removed_source in sorted(set(previous) - set(current)):
            old_ids = previous[removed_source].get("ids", [])
            if old_ids:
                store.delete(ids=old_ids)
            report.removed_files += 1
            logger.info("已从知识库移除文件：%s", removed_source)

        # 第二步：摘要未变化就跳过；新增或变化的文件重新切片并写向量。
        next_files: dict[str, Any] = {}
        for source, item in current.items():
            previous_item = previous.get(source)
            if previous_item and previous_item.get("sha256") == item["sha256"]:
                next_files[source] = previous_item
                report.unchanged_files += 1
                continue

            if previous_item:
                old_ids = previous_item.get("ids", [])
                if old_ids:
                    store.delete(ids=old_ids)

            chunks, ids = self._split_file(item["path"], item["sha256"])
            if chunks:
                store.add_documents(chunks, ids=ids)
            next_files[source] = {"sha256": item["sha256"], "ids": ids}
            report.indexed_chunks += len(chunks)
            if previous_item:
                report.updated_files += 1
                logger.info("已更新知识文件：%s（%d 个分片）", source, len(chunks))
            else:
                report.added_files += 1
                logger.info("已添加知识文件：%s（%d 个分片）", source, len(chunks))

        # 最后才替换清单。这样如果前面向量化失败，不会把未完成状态写进清单。
        self._save_manifest(
            {
                "version": MANIFEST_VERSION,
                "backend": self.backend_name,
                "files": next_files,
            }
        )
        return report

    def rebuild(self) -> IndexSyncReport:
        """清空现有向量集合和索引清单，再从知识文件完整重建索引。"""
        with self._sync_lock:
            store = self._get_store()
            store.reset_collection()
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            return self._sync_locked()

    def search(
        self,
        query: str,
        subject: str | None = None,
        k: int | None = None,
    ) -> list[Document]:
        """同步最新知识文件后，按查询文本和可选学科执行相似度检索。"""
        # 每次查询前做增量同步；未变化文件只比较摘要，不会重复请求 Embedding。
        self.sync()
        filter_value = {"subject": subject} if subject and subject != "综合" else None
        search_options = {"filter": filter_value} if filter_value else {}
        return self._get_store().similarity_search(
            query,
            k=k or self.k,
            **search_options,
        )

    def status(self) -> dict[str, Any]:
        """返回集合名称、知识文件数、分片数和本地存储路径。"""
        manifest = self._load_manifest()
        chunk_count = sum(
            len(item.get("ids", [])) for item in manifest["files"].values()
        )
        return {
            "backend": self.backend_name,
            "retrieval_mode": "bm25_cosine_hybrid",
            "collection": knowledge_conf["collection_name"],
            "knowledge_files": len(manifest["files"]),
            "indexed_chunks": chunk_count,
            "data_path": str(self.data_path),
            "storage_path": str(self.persist_directory),
            # 保留旧字段，避免已有前端或脚本升级后读取失败。
            "persist_directory": str(self.persist_directory),
            "manifest_path": str(self.manifest_path),
            "hybrid_weights": {
                "cosine": float(knowledge_conf["hybrid_vector_weight"]),
                "bm25": float(knowledge_conf["hybrid_bm25_weight"]),
            },
        }



#python main.py --sync-index