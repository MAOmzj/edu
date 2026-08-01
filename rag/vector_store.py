"""教育知识库的增量索引和检索。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import create_embedding_model
from utils.config_handler import app_conf, knowledge_conf
from utils.file_handler import calculate_sha256, list_knowledge_files, load_documents
from utils.logger_handler import logger
from utils.path_tool import PROJECT_ROOT, get_abs_path


MANIFEST_VERSION = 1


@dataclass
class IndexSyncReport:
    added_files: int = 0
    updated_files: int = 0
    removed_files: int = 0
    unchanged_files: int = 0
    indexed_chunks: int = 0


class EducationVectorStore:
    def __init__(self, embedding_model: Embeddings | None = None):
        self._embedding_model = embedding_model
        self._vector_store: Chroma | None = None
        self.data_path = Path(get_abs_path(knowledge_conf["data_path"]))
        self.persist_directory = Path(
            get_abs_path(knowledge_conf["persist_directory"])
        )
        self.manifest_path = Path(get_abs_path(knowledge_conf["manifest_path"]))
        self.allowed_types = tuple(knowledge_conf["allowed_file_types"])
        self.k = int(knowledge_conf["k"])
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(knowledge_conf["chunk_size"]),
            chunk_overlap=int(knowledge_conf["chunk_overlap"]),
            separators=list(knowledge_conf["separators"]),
            length_function=len,
        )

    def _get_store(self) -> Chroma:
        if self._vector_store is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            embedding_model = self._embedding_model or create_embedding_model()
            self._vector_store = Chroma(
                collection_name=knowledge_conf["collection_name"],
                embedding_function=embedding_model,
                persist_directory=str(self.persist_directory),
            )
        return self._vector_store

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"version": MANIFEST_VERSION, "files": {}}
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"索引清单损坏：{self.manifest_path}") from exc
        if manifest.get("version") != MANIFEST_VERSION:
            return {"version": MANIFEST_VERSION, "files": {}}
        manifest.setdefault("files", {})
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    def _relative_source(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative = path.resolve().relative_to(self.data_path.resolve())
            return (Path("data/knowledge") / relative).as_posix()

    @staticmethod
    def _subject_for(path: Path) -> str:
        for subject in app_conf["supported_subjects"]:
            if path.stem.startswith(subject):
                return subject
        return "综合"

    @staticmethod
    def _chunk_id(relative_source: str, file_hash: str, index: int) -> str:
        raw = f"{relative_source}:{file_hash}:{index}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _split_file(self, path: Path, file_hash: str) -> tuple[list[Document], list[str]]:
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

        for removed_source in sorted(set(previous) - set(current)):
            old_ids = previous[removed_source].get("ids", [])
            if old_ids:
                store.delete(ids=old_ids)
            report.removed_files += 1
            logger.info("已从知识库移除文件：%s", removed_source)

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

        self._save_manifest(
            {"version": MANIFEST_VERSION, "files": next_files}
        )
        return report

    def rebuild(self) -> IndexSyncReport:
        store = self._get_store()
        store.reset_collection()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        return self.sync()

    def search(
        self,
        query: str,
        subject: str | None = None,
        k: int | None = None,
    ) -> list[Document]:
        self.sync()
        filter_value = {"subject": subject} if subject and subject != "综合" else None
        search_options = {"filter": filter_value} if filter_value else {}
        return self._get_store().similarity_search(
            query,
            k=k or self.k,
            **search_options,
        )

    def status(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        chunk_count = sum(
            len(item.get("ids", [])) for item in manifest["files"].values()
        )
        return {
            "collection": knowledge_conf["collection_name"],
            "knowledge_files": len(manifest["files"]),
            "indexed_chunks": chunk_count,
            "data_path": str(self.data_path),
            "persist_directory": str(self.persist_directory),
        }
