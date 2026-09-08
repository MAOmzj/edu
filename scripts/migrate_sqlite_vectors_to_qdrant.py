"""把现有 SQLite 向量原样迁移到 Qdrant，不重复调用 Embedding API。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vector_backends import QdrantVectorBackend
from rag.vector_store import MANIFEST_VERSION, EducationVectorStore
from utils.file_handler import calculate_sha256
from utils.path_tool import get_abs_path


class _MigrationOnlyEmbeddings(Embeddings):
    """阻止迁移过程意外访问在线 Embedding 服务。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise RuntimeError("SQLite 向量迁移不应重新生成文档向量")

    def embed_query(self, text: str) -> list[float]:
        del text
        raise RuntimeError("SQLite 向量迁移不应生成查询向量")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将完整 SQLite 索引中的预计算向量迁移到当前 Qdrant"
    )
    parser.add_argument(
        "--sqlite-database",
        default="storage/knowledge_vectors.sqlite3",
        help="SQLite 向量数据库路径",
    )
    parser.add_argument(
        "--sqlite-manifest",
        default="storage/index_manifest_sqlite.json",
        help="SQLite 文件清单路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每批迁移的向量数量",
    )
    parser.add_argument(
        "--confirm-current-schema",
        action="store_true",
        help=(
            "确认旧 SQLite 使用当前配置中的 Embedding 模型、chunk_size、"
            "chunk_overlap 和 separators；旧 v4 清单本身不记录这些字段"
        ),
    )
    return parser.parse_args()


def _load_source_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"SQLite 索引清单不存在：{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SQLite 索引清单损坏：{path}") from exc
    if manifest.get("version") != 4:
        raise RuntimeError("源 SQLite 清单必须是受支持的 v4 格式")
    if manifest.get("backend") != "sqlite":
        raise RuntimeError("源索引清单不是 SQLite 后端")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("SQLite 索引清单没有任何知识文件")
    return manifest


def _validate_source_files(
    files: dict[str, object],
) -> dict[str, tuple[str, str]]:
    expected_by_id: dict[str, tuple[str, str]] = {}
    for source, raw_item in files.items():
        if not isinstance(raw_item, dict):
            raise RuntimeError(f"SQLite 清单条目格式错误：{source}")
        file_path = PROJECT_ROOT / source
        if not file_path.is_file():
            raise RuntimeError(f"知识文件不存在，拒绝迁移旧向量：{source}")
        expected_hash = str(raw_item.get("sha256", ""))
        if not expected_hash or calculate_sha256(file_path) != expected_hash:
            raise RuntimeError(f"知识文件已经变化，请先更新 SQLite 索引：{source}")
        raw_ids = raw_item.get("ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise RuntimeError(f"SQLite 清单没有分片 ID：{source}")
        for raw_id in raw_ids:
            chunk_id = str(raw_id)
            if chunk_id in expected_by_id:
                raise RuntimeError(f"SQLite 清单存在重复分片 ID：{chunk_id}")
            expected_by_id[chunk_id] = (source, expected_hash)
    return expected_by_id


def _validate_sqlite_rows(
    connection: sqlite3.Connection,
    *,
    collection_name: str,
    expected_by_id: dict[str, tuple[str, str]],
    vector_size: int,
) -> None:
    rows = connection.execute(
        """
        SELECT chunk_id, metadata_json, subject, source, file_hash,
               embedding, embedding_dimension
        FROM education_vector_chunks
        WHERE collection_name = ?
        """,
        (collection_name,),
    )
    stored_ids: set[str] = set()
    for (
        raw_chunk_id,
        metadata_json,
        subject_column,
        source_column,
        hash_column,
        blob,
        dimension,
    ) in rows:
        chunk_id = str(raw_chunk_id)
        stored_ids.add(chunk_id)
        expected = expected_by_id.get(chunk_id)
        if expected is None:
            continue
        expected_source, expected_hash = expected
        if int(dimension) != vector_size or len(blob) != vector_size * 4:
            raise RuntimeError(f"SQLite 向量维度或字节长度不正确：{chunk_id}")
        vector = _decode_vector(blob, int(dimension))
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"SQLite 向量包含 NaN 或无穷大：{chunk_id}")
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SQLite 分片元数据 JSON 损坏：{chunk_id}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError(f"SQLite 分片元数据不是对象：{chunk_id}")
        if (
            str(source_column) != expected_source
            or str(hash_column) != expected_hash
            or str(metadata.get("source", "")) != expected_source
            or str(metadata.get("file_hash", "")) != expected_hash
            or str(metadata.get("subject", "")) != str(subject_column)
        ):
            raise RuntimeError(f"SQLite 分片来源或冗余列不一致：{chunk_id}")
        metadata_chunk_id = metadata.get("chunk_id")
        if metadata_chunk_id is not None and str(metadata_chunk_id) != chunk_id:
            raise RuntimeError(f"SQLite metadata.chunk_id 不一致：{chunk_id}")
        chunk_index = metadata.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
            raise RuntimeError(f"SQLite chunk_index 格式错误：{chunk_id}")
        if (
            EducationVectorStore._chunk_id(
                expected_source,
                expected_hash,
                chunk_index,
            )
            != chunk_id
        ):
            raise RuntimeError(f"SQLite 分片 ID 公式校验失败：{chunk_id}")

    expected_ids = set(expected_by_id)
    if stored_ids != expected_ids:
        missing = len(expected_ids - stored_ids)
        extra = len(stored_ids - expected_ids)
        raise RuntimeError(
            "SQLite 数据与清单不一致，拒绝迁移："
            f"缺少 {missing} 条，多出 {extra} 条"
        )


def _decode_vector(blob: bytes, dimension: int) -> Sequence[float]:
    return struct.unpack(f"<{dimension}f", blob)


def _validate_qdrant_points(
    backend: QdrantVectorBackend,
    expected_ids: set[str],
) -> int:
    """逐页核对原始 chunk_id 与确定性 UUID，不能只比较点数。"""

    actual_ids: set[str] = set()
    offset = None
    while True:
        points, offset = backend._client.scroll(
            collection_name=backend.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            consistency=backend.read_consistency,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            if not chunk_id:
                raise RuntimeError(f"Qdrant 点缺少 chunk_id：{point.id}")
            if str(point.id) != backend._point_id(chunk_id):
                raise RuntimeError(f"Qdrant 点 UUID 与 chunk_id 不匹配：{chunk_id}")
            actual_ids.add(chunk_id)
        if offset is None:
            break
    if actual_ids != expected_ids:
        missing = len(expected_ids - actual_ids)
        extra = len(actual_ids - expected_ids)
        raise RuntimeError(
            "Qdrant 点 ID 与 SQLite 清单不一致，暂不覆盖目标 manifest："
            f"缺少 {missing} 条，多出 {extra} 条"
        )
    info = backend._client.get_collection(
        collection_name=backend.collection_name
    )
    status = getattr(getattr(info, "status", None), "value", info.status)
    if str(status).lower() != "green":
        raise RuntimeError(f"Qdrant 集合尚未就绪，当前状态：{status}")
    return len(actual_ids)


def migrate(
    *,
    sqlite_database: Path,
    sqlite_manifest: Path,
    batch_size: int,
    confirm_current_schema: bool,
) -> dict[str, int]:
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")
    if not confirm_current_schema:
        raise RuntimeError(
            "旧 v4 清单没有 Embedding/切分指纹；确认源索引确由当前模型和"
            "切分配置生成后，请显式传入 --confirm-current-schema"
        )
    source_manifest_digest = hashlib.sha256(
        sqlite_manifest.read_bytes()
    ).hexdigest()
    source_manifest = _load_source_manifest(sqlite_manifest)
    source_files = source_manifest["files"]
    if not isinstance(source_files, dict):
        raise RuntimeError("SQLite 索引清单 files 格式错误")
    expected_by_id = _validate_source_files(source_files)
    expected_ids = set(expected_by_id)

    target_store = EducationVectorStore(
        embedding_model=_MigrationOnlyEmbeddings(),
        backend_name="qdrant",
    )
    target_backend = target_store._get_store()
    if not isinstance(target_backend, QdrantVectorBackend):
        raise RuntimeError("当前目标不是 Qdrant 后端")
    if target_store.sync_on_search:
        raise RuntimeError(
            "迁移期间必须设置 EDUCATION_SYNC_ON_SEARCH=false，禁止在线请求并发同步"
        )
    if batch_size > target_backend.upsert_batch_size:
        raise RuntimeError(
            "迁移 batch_size 不能超过 Qdrant upsert_batch_size："
            f"{target_backend.upsert_batch_size}"
        )

    collection_name = target_backend.collection_name
    if not sqlite_database.is_file():
        raise RuntimeError(f"SQLite 向量数据库不存在：{sqlite_database}")
    connection = sqlite3.connect(f"file:{sqlite_database.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        _validate_sqlite_rows(
            connection,
            collection_name=collection_name,
            expected_by_id=expected_by_id,
            vector_size=target_backend.vector_size,
        )
        cursor = connection.execute(
            """
            SELECT chunk_id, document, metadata_json,
                   embedding, embedding_dimension
            FROM education_vector_chunks
            WHERE collection_name = ?
            ORDER BY chunk_id
            """,
            (collection_name,),
        )
        migrated = 0
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            documents: list[Document] = []
            ids: list[str] = []
            embeddings: list[Sequence[float]] = []
            for chunk_id, text, metadata_json, blob, dimension in rows:
                metadata = json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"分片元数据不是对象：{chunk_id}")
                documents.append(
                    Document(page_content=str(text), metadata=metadata)
                )
                ids.append(str(chunk_id))
                embeddings.append(_decode_vector(blob, int(dimension)))
            target_backend.upsert_precomputed_documents(
                documents,
                ids=ids,
                embeddings=embeddings,
            )
            migrated += len(rows)
            if migrated % 1024 == 0 or migrated == len(expected_ids):
                print(f"已迁移 {migrated}/{len(expected_ids)} 个分片", flush=True)
    finally:
        connection.close()

    if hashlib.sha256(sqlite_manifest.read_bytes()).hexdigest() != source_manifest_digest:
        raise RuntimeError("迁移期间 SQLite manifest 发生变化，暂不提交目标 manifest")
    if set(_validate_source_files(source_files)) != expected_ids:
        raise RuntimeError("迁移期间知识文件发生变化，暂不提交目标 manifest")
    actual_count = _validate_qdrant_points(target_backend, expected_ids)
    target_store._save_manifest(
        {
            "version": MANIFEST_VERSION,
            "backend": "qdrant",
            "storage_identity": target_store._storage_identity(),
            "files": source_files,
            "migration": {
                "source_backend": "sqlite",
                "source_manifest_version": 4,
                "current_schema_confirmed": True,
            },
        }
    )
    return {
        "knowledge_files": len(source_files),
        "migrated_chunks": len(expected_ids),
        "qdrant_points": actual_count,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = migrate(
            sqlite_database=Path(get_abs_path(args.sqlite_database)),
            sqlite_manifest=Path(get_abs_path(args.sqlite_manifest)),
            batch_size=args.batch_size,
            confirm_current_schema=args.confirm_current_schema,
        )
    except Exception as exc:
        print(f"迁移失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
