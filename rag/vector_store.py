# 文件用途：发现知识文件、切分文档、增量建索引、维护清单并向上层提供检索。
# 调用关系：qa_service.py 和 CLI 调用本文件；本文件调用 file_handler、model.factory 和 vector_backends。
# 修改易踩坑：文件哈希清单必须与所选后端分开，切分参数或 Embedding 改变后必须重建索引。
"""教育知识库的增量索引和检索。

这一层负责“知识文件”，vector_backends.py 负责“向量数据库”。主要流程：

    data/knowledge 中的 TXT/PDF
      -> 计算 SHA-256 判断是否变化
      -> 切成小片段并补充学科/来源元数据
      -> Embedding 向量化后写入 SQLite、Chroma 或 Qdrant
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
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import create_embedding_model
from rag.vector_backends import (
    ChromaVectorBackend,
    QdrantVectorBackend,
    SQLiteVectorBackend,
    VectorBackend,
)
from utils.config_handler import app_conf, knowledge_conf, model_conf
from utils.file_handler import calculate_sha256, list_knowledge_files, load_documents
from utils.logger_handler import logger
from utils.path_tool import PROJECT_ROOT, get_abs_path


# V5 记录实际存储身份，避免从嵌入式 Chroma 切换到远程 Server 后误用旧清单。
MANIFEST_VERSION = 5
SUPPORTED_VECTOR_BACKENDS = {"sqlite", "chroma", "qdrant"}


def _environment_bool(name: str, default: bool) -> bool:
    """读取常见布尔环境变量写法，非法值立即报错。"""

    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false 或 1/0")


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
        """读取配置，并准备可切换的 SQLite/Chroma/Qdrant 后端和切分器。"""
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
        self.sync_on_search = _environment_bool(
            "EDUCATION_SYNC_ON_SEARCH",
            bool(knowledge_conf.get("sync_on_search", True)),
        )
        self.data_path = Path(get_abs_path(knowledge_conf["data_path"]))
        self.sqlite_database_path = Path(
            get_abs_path(knowledge_conf["sqlite_database_path"])
        )
        self.chroma_persist_directory = Path(
            get_abs_path(knowledge_conf["chroma_persist_directory"])
        )
        chroma_conf = knowledge_conf.get("chroma_server") or {}
        self.chroma_mode = str(
            os.getenv("CHROMA_MODE") or chroma_conf.get("mode", "http")
        ).strip().lower()
        self.chroma_host = str(
            os.getenv("CHROMA_HOST") or chroma_conf.get("host", "127.0.0.1")
        ).strip()
        self.chroma_port = int(
            os.getenv("CHROMA_PORT") or chroma_conf.get("port", 8000)
        )
        self.chroma_ssl = _environment_bool(
            "CHROMA_SSL",
            bool(chroma_conf.get("ssl", False)),
        )
        self.chroma_tenant = str(
            os.getenv("CHROMA_TENANT")
            or chroma_conf.get("tenant", "default_tenant")
        ).strip()
        self.chroma_database = str(
            os.getenv("CHROMA_DATABASE")
            or chroma_conf.get("database", "default_database")
        ).strip()
        self.chroma_auth_token = os.getenv("CHROMA_AUTH_TOKEN", "").strip()
        self.chroma_native_candidate_multiplier = int(
            chroma_conf.get("native_candidate_multiplier", 4)
        )
        self.chroma_native_max_candidates = int(
            chroma_conf.get("native_max_candidates", 100)
        )
        scheme = "https" if self.chroma_ssl else "http"
        self.chroma_endpoint = f"{scheme}://{self.chroma_host}:{self.chroma_port}"

        qdrant_conf = knowledge_conf.get("qdrant_server") or {}
        self.qdrant_url = str(
            os.getenv("QDRANT_URL")
            or qdrant_conf.get("url", "http://127.0.0.1:6333")
        ).strip().rstrip("/")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        self.qdrant_grpc_port = int(
            os.getenv("QDRANT_GRPC_PORT")
            or qdrant_conf.get("grpc_port", 6334)
        )
        self.qdrant_prefer_grpc = _environment_bool(
            "QDRANT_PREFER_GRPC",
            bool(qdrant_conf.get("prefer_grpc", False)),
        )
        self.qdrant_timeout = int(
            os.getenv("QDRANT_TIMEOUT") or qdrant_conf.get("timeout", 10)
        )
        self.qdrant_vector_size = int(qdrant_conf.get("vector_size", 1024))
        self.qdrant_distance = str(
            qdrant_conf.get("distance", "cosine")
        ).strip().lower()
        self.qdrant_upsert_batch_size = int(
            qdrant_conf.get("upsert_batch_size", 64)
        )
        self.qdrant_delete_batch_size = int(
            qdrant_conf.get("delete_batch_size", 256)
        )
        self.qdrant_shard_number = int(qdrant_conf.get("shard_number", 1))
        self.qdrant_replication_factor = int(
            qdrant_conf.get("replication_factor", 1)
        )
        self.qdrant_write_consistency_factor = int(
            qdrant_conf.get("write_consistency_factor", 1)
        )
        self.qdrant_write_ordering = str(
            qdrant_conf.get("write_ordering", "medium")
        ).strip().lower()
        self.qdrant_read_consistency = str(
            qdrant_conf.get("read_consistency", "majority")
        ).strip().lower()
        self.qdrant_on_disk_payload = bool(
            qdrant_conf.get("on_disk_payload", True)
        )
        self.qdrant_on_disk_vectors = bool(
            qdrant_conf.get("on_disk_vectors", True)
        )
        self.qdrant_hnsw_m = int(qdrant_conf.get("hnsw_m", 16))
        self.qdrant_hnsw_ef_construct = int(
            qdrant_conf.get("hnsw_ef_construct", 100)
        )
        self.qdrant_hnsw_full_scan_threshold = int(
            qdrant_conf.get("hnsw_full_scan_threshold", 10_000)
        )
        self.qdrant_search_hnsw_ef = int(
            qdrant_conf.get("search_hnsw_ef", 128)
        )
        self.qdrant_native_candidate_multiplier = int(
            qdrant_conf.get("native_candidate_multiplier", 4)
        )
        self.qdrant_native_max_candidates = int(
            qdrant_conf.get("native_max_candidates", 100)
        )
        qdrant_schema = {
            "schema_version": 1,
            "embedding_model": str(model_conf["embedding_model_name"]),
            "vector_size": self.qdrant_vector_size,
            "distance": self.qdrant_distance,
            "chunk_size": int(knowledge_conf["chunk_size"]),
            "chunk_overlap": int(knowledge_conf["chunk_overlap"]),
            "separators": list(knowledge_conf["separators"]),
            "shard_number": self.qdrant_shard_number,
            "replication_factor": self.qdrant_replication_factor,
            "write_consistency_factor": (
                self.qdrant_write_consistency_factor
            ),
            "on_disk_payload": self.qdrant_on_disk_payload,
            "on_disk_vectors": self.qdrant_on_disk_vectors,
            "hnsw_m": self.qdrant_hnsw_m,
            "hnsw_ef_construct": self.qdrant_hnsw_ef_construct,
            "hnsw_full_scan_threshold": (
                self.qdrant_hnsw_full_scan_threshold
            ),
        }
        self.qdrant_schema_fingerprint = hashlib.sha256(
            json.dumps(
                qdrant_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        manifest_key = f"{self.backend_name}_manifest_path"
        self.manifest_path = Path(get_abs_path(knowledge_conf[manifest_key]))
        if self.backend_name == "sqlite":
            self.persist_directory = self.sqlite_database_path
        elif self.backend_name == "chroma":
            self.persist_directory = (
                self.chroma_endpoint
                if self.chroma_mode == "http"
                else self.chroma_persist_directory
            )
        else:
            self.persist_directory = self.qdrant_url
        self.allowed_types = tuple(knowledge_conf["allowed_file_types"])
        self.k = int(knowledge_conf["k"])
        rerank_conf = knowledge_conf.get("rerank") or {}
        self.rerank_enabled = bool(rerank_conf.get("enabled", False))
        self.rerank_model = str(
            os.getenv("DASHSCOPE_RERANK_MODEL") or rerank_conf.get("model", "gte-rerank")
        ).strip()
        self.rerank_candidate_k = max(1, int(rerank_conf.get("candidate_k", 20)))
        self.rerank_timeout = float(rerank_conf.get("timeout", 10))
        self._reranker: DashScopeReranker | None = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(knowledge_conf["chunk_size"]),
            chunk_overlap=int(knowledge_conf["chunk_overlap"]),
            separators=list(knowledge_conf["separators"]),
            length_function=len,
        )

    def _get_store(self) -> VectorBackend:
        """按照配置按需创建并复用 SQLite、Chroma 或 Qdrant 向量后端。"""
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
            elif self.backend_name == "chroma":
                self._vector_store = ChromaVectorBackend(
                    collection_name=knowledge_conf["collection_name"],
                    embedding_model=embedding_model,
                    deployment_mode=self.chroma_mode,
                    persist_directory=self.chroma_persist_directory,
                    host=self.chroma_host,
                    port=self.chroma_port,
                    ssl=self.chroma_ssl,
                    tenant=self.chroma_tenant,
                    database=self.chroma_database,
                    auth_token=self.chroma_auth_token,
                    native_candidate_multiplier=(
                        self.chroma_native_candidate_multiplier
                    ),
                    native_max_candidates=self.chroma_native_max_candidates,
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
                self._vector_store = QdrantVectorBackend(
                    collection_name=knowledge_conf["collection_name"],
                    embedding_model=embedding_model,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    grpc_port=self.qdrant_grpc_port,
                    prefer_grpc=self.qdrant_prefer_grpc,
                    timeout=self.qdrant_timeout,
                    vector_size=self.qdrant_vector_size,
                    distance=self.qdrant_distance,
                    embedding_batch_size=int(
                        knowledge_conf["embedding_batch_size"]
                    ),
                    upsert_batch_size=self.qdrant_upsert_batch_size,
                    delete_batch_size=self.qdrant_delete_batch_size,
                    shard_number=self.qdrant_shard_number,
                    replication_factor=self.qdrant_replication_factor,
                    write_consistency_factor=(
                        self.qdrant_write_consistency_factor
                    ),
                    write_ordering=self.qdrant_write_ordering,
                    read_consistency=self.qdrant_read_consistency,
                    on_disk_payload=self.qdrant_on_disk_payload,
                    on_disk_vectors=self.qdrant_on_disk_vectors,
                    hnsw_m=self.qdrant_hnsw_m,
                    hnsw_ef_construct=self.qdrant_hnsw_ef_construct,
                    hnsw_full_scan_threshold=(
                        self.qdrant_hnsw_full_scan_threshold
                    ),
                    search_hnsw_ef=self.qdrant_search_hnsw_ef,
                    native_candidate_multiplier=(
                        self.qdrant_native_candidate_multiplier
                    ),
                    native_max_candidates=self.qdrant_native_max_candidates,
                    vector_weight=float(
                        knowledge_conf["hybrid_vector_weight"]
                    ),
                    bm25_weight=float(
                        knowledge_conf["hybrid_bm25_weight"]
                    ),
                    bm25_k1=float(knowledge_conf["bm25_k1"]),
                    bm25_b=float(knowledge_conf["bm25_b"]),
                    schema_fingerprint=self.qdrant_schema_fingerprint,
                )
        return self._vector_store

    def _storage_identity(self) -> str:
        """返回清单绑定的真实存储目标，切换目标后自动要求重新建索引。"""

        if self.backend_name == "sqlite":
            return f"sqlite:{self.sqlite_database_path.resolve()}"
        if self.backend_name == "qdrant":
            return (
                f"qdrant:{self.qdrant_url}:"
                f"{knowledge_conf['collection_name']}:"
                f"schema={self.qdrant_schema_fingerprint}"
            )
        if self.chroma_mode == "http":
            return (
                f"chroma:{self.chroma_endpoint}:"
                f"{self.chroma_tenant}:{self.chroma_database}:"
                f"{knowledge_conf['collection_name']}"
            )
        return (
            f"chroma-persistent:{self.chroma_persist_directory.resolve()}:"
            f"{self.chroma_tenant}:{self.chroma_database}:"
            f"{knowledge_conf['collection_name']}"
        )

    def _get_reranker(self) -> DashScopeReranker:
        """按需创建阿里云百炼 Rerank 客户端，避免状态查询时加载或调用它。"""
        if self._reranker is None:
            from rag.reranker import DashScopeReranker

            self._reranker = DashScopeReranker(
                model=self.rerank_model,
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                timeout=self.rerank_timeout,
            )
        return self._reranker

    def _rerank_documents(
        self,
        query: str,
        documents: list[Document],
        *,
        top_n: int,
    ) -> list[Document]:
        """对混合召回候选做 Rerank；失败时保留原混合排序结果，不让检索中断。"""
        if not documents or len(documents) <= top_n:
            return documents[:top_n]
        try:
            return self._get_reranker().rerank(query, documents, top_n=top_n)
        except Exception:
            # Rerank 属于精排增强。接口失败或 Key 缺失时回退到向量 + BM25 顺序，
            # 保证主 Agent 仍能拿到资料，而不是因为可选增强功能让本轮提问失败。
            logger.exception("阿里云 Rerank 执行失败，回退到混合检索排序结果")
            return documents[:top_n]

    def _load_manifest(self) -> dict[str, Any]:
        """读取索引清单；文件不存在或版本过期时返回空清单。"""
        empty_manifest = {
            "version": MANIFEST_VERSION,
            "backend": self.backend_name,
            "storage_identity": self._storage_identity(),
            "files": {},
            "_manifest_valid": False,
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
            or manifest.get("storage_identity") != self._storage_identity()
        ):
            return empty_manifest
        manifest.setdefault("files", {})
        manifest["_manifest_valid"] = True
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """先写入临时文件再原子替换，安全保存最新索引清单。"""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{os.getpid()}."
            f"{threading.get_ident()}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

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
            chunk_id = self._chunk_id(relative_source, file_hash, index)
            chunk.metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)
        return chunks, ids

    def sync(self) -> IndexSyncReport:
        """比较文件摘要与索引清单，增量添加、更新或删除向量分片。"""
        with self._sync_lock:
            return self._sync_locked()

    #开始遍历
    def _sync_locked(
        self,
        *,
        backend_already_reset: bool = False,
    ) -> IndexSyncReport:
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
        manifest_valid = bool(manifest.pop("_manifest_valid", False))
        previous: dict[str, Any] = manifest["files"]
        store = self._get_store()

        if self.backend_name == "qdrant":
            if not isinstance(store, QdrantVectorBackend):
                raise RuntimeError("Qdrant 后端类型与配置不一致")
            if not manifest_valid:
                # manifest 是远程集合的所有权清单。首次建立、版本变化或目标变化时，
                # 先清空应用专属集合，避免无主旧点与新索引混合。
                if not backend_already_reset:
                    store.reset_collection()
                previous = {}
            else:
                expected_points = sum(
                    len(item.get("ids", [])) for item in previous.values()
                )
                actual_points = store.count_documents()
                if actual_points != expected_points:
                    logger.warning(
                        "Qdrant 点数与 manifest 不一致（实际 %d，期望 %d），"
                        "自动执行全量重建",
                        actual_points,
                        expected_points,
                    )
                    store.reset_collection()
                    previous = {}

        # 第一步：清理“清单里有、磁盘上已经没有”的旧文件分片。
        for removed_source in sorted(set(previous) - set(current)):
            old_ids = previous[removed_source].get("ids", [])
            if old_ids:
                store.delete(ids=old_ids)
            report.removed_files += 1
            logger.info("已从知识库移除文件：%s", removed_source)

        # 第二步：摘要未变化就跳过；新增或变化的文件重新切片并写向量。
        next_files: dict[str, Any] = {}

        def save_progress_checkpoint() -> None:
            """保存已完成文件，同时保留尚未处理文件的旧清单以支持断点恢复。"""
            checkpoint_files = {
                source: item
                for source, item in previous.items()
                if source in current
            }
            checkpoint_files.update(next_files)
            self._save_manifest(
                {
                    "version": MANIFEST_VERSION,
                    "backend": self.backend_name,
                    "storage_identity": self._storage_identity(),
                    "files": checkpoint_files,
                }
            )

        for source, item in current.items():
            previous_item = previous.get(source)
            if previous_item and previous_item.get("sha256") == item["sha256"]:
                next_files[source] = previous_item
                report.unchanged_files += 1
                continue

            old_ids = previous_item.get("ids", []) if previous_item else []

            chunks, ids = self._split_file(item["path"], item["sha256"])
            if chunks:
                store.add_documents(chunks, ids=ids)

            # 新文件使用基于新哈希生成的 ids，可以先安全写入。必须等新向量全部
            # 成功后再删除旧 ids，避免在线 Embedding 中断时把仍可检索的旧版本丢掉。
            if old_ids:
                store.delete(ids=old_ids)
            next_files[source] = {"sha256": item["sha256"], "ids": ids}
            report.indexed_chunks += len(chunks)
            if previous_item:
                report.updated_files += 1
                logger.info("已更新知识文件：%s（%d 个分片）", source, len(chunks))
            else:
                report.added_files += 1
                logger.info("已添加知识文件：%s（%d 个分片）", source, len(chunks))

            # 在线 Embedding 可能因瞬时网络错误中断。每个文件成功后保存一个安全
            # 检查点，使下一次同步跳过已完成文件，同时不丢失未处理文件的旧 ids。
            save_progress_checkpoint()

        # 全部文件完成后写入最终清单；此时 next_files 已覆盖所有当前知识文件。
        self._save_manifest(
            {
                "version": MANIFEST_VERSION,
                "backend": self.backend_name,
                "storage_identity": self._storage_identity(),
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
            return self._sync_locked(backend_already_reset=True)

    def search(
        self,
        query: str,
        subject: str | None = None,
        k: int | None = None,
    ) -> list[Document]:
        """同步最新知识文件后，按查询文本和可选学科执行相似度检索。"""
        # 每次查询前做增量同步；未变化文件只比较摘要，不会重复请求 Embedding。
        if self.sync_on_search:
            self.sync()
        filter_value = {"subject": subject} if subject and subject != "综合" else None
        search_options = {"filter": filter_value} if filter_value else {}
        top_k = int(k or self.k)
        candidate_k = (
            max(top_k, self.rerank_candidate_k)
            if self.rerank_enabled
            else top_k
        )
        documents = self._get_store().similarity_search(
            query,
            k=candidate_k,
            **search_options,
        )
        if not self.rerank_enabled:
            return documents[:top_k]
        return self._rerank_documents(query, documents, top_n=top_k)

    def status(self) -> dict[str, Any]:
        """返回集合名称、知识文件数、分片数和当前存储目标。"""
        manifest = self._load_manifest()
        chunk_count = sum(
            len(item.get("ids", [])) for item in manifest["files"].values()
        )
        return {
            "backend": self.backend_name,
            "retrieval_mode": (
                "bm25_cosine_hybrid_aliyun_rerank"
                if self.rerank_enabled
                else "bm25_cosine_hybrid"
            ),
            "collection": knowledge_conf["collection_name"],
            "knowledge_files": len(manifest["files"]),
            "indexed_chunks": chunk_count,
            "data_path": str(self.data_path),
            "storage_path": str(self.persist_directory),
            # 保留旧字段，避免已有前端或脚本升级后读取失败。
            "persist_directory": str(self.persist_directory),
            "manifest_path": str(self.manifest_path),
            "storage_identity": self._storage_identity(),
            "sync_on_search": self.sync_on_search,
            "chroma_server": (
                {
                    "mode": self.chroma_mode,
                    "endpoint": self.chroma_endpoint,
                    "tenant": self.chroma_tenant,
                    "database": self.chroma_database,
                    "native_candidate_multiplier": (
                        self.chroma_native_candidate_multiplier
                    ),
                    "native_max_candidates": self.chroma_native_max_candidates,
                }
                if self.backend_name == "chroma"
                else None
            ),
            "qdrant_server": (
                {
                    "url": self.qdrant_url,
                    "grpc_port": self.qdrant_grpc_port,
                    "prefer_grpc": self.qdrant_prefer_grpc,
                    "timeout": self.qdrant_timeout,
                    "vector_size": self.qdrant_vector_size,
                    "distance": self.qdrant_distance,
                    "upsert_batch_size": self.qdrant_upsert_batch_size,
                    "delete_batch_size": self.qdrant_delete_batch_size,
                    "shard_number": self.qdrant_shard_number,
                    "replication_factor": self.qdrant_replication_factor,
                    "write_consistency_factor": (
                        self.qdrant_write_consistency_factor
                    ),
                    "write_ordering": self.qdrant_write_ordering,
                    "read_consistency": self.qdrant_read_consistency,
                    "on_disk_payload": self.qdrant_on_disk_payload,
                    "on_disk_vectors": self.qdrant_on_disk_vectors,
                    "hnsw_m": self.qdrant_hnsw_m,
                    "hnsw_ef_construct": self.qdrant_hnsw_ef_construct,
                    "hnsw_full_scan_threshold": (
                        self.qdrant_hnsw_full_scan_threshold
                    ),
                    "search_hnsw_ef": self.qdrant_search_hnsw_ef,
                    "native_candidate_multiplier": (
                        self.qdrant_native_candidate_multiplier
                    ),
                    "native_max_candidates": (
                        self.qdrant_native_max_candidates
                    ),
                    "schema_fingerprint": self.qdrant_schema_fingerprint,
                }
                if self.backend_name == "qdrant"
                else None
            ),
            "hybrid_weights": {
                "cosine": float(knowledge_conf["hybrid_vector_weight"]),
                "bm25": float(knowledge_conf["hybrid_bm25_weight"]),
            },
            "rerank": {
                "enabled": self.rerank_enabled,
                "model": self.rerank_model,
                "candidate_k": self.rerank_candidate_k,
                "top_k": self.k,
            },
        }



#python main.py --sync-index
