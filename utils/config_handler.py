# 文件用途：读取并在启动阶段校验 model/knowledge/prompts/app 四份 YAML 配置。
# 调用关系：几乎所有后端模块导入本文件的配置字典；本文件调用 path_tool 和 PyYAML。
# 修改易踩坑：这里在导入时执行校验，新增必填项必须同步配置文件和测试，否则整个应用无法导入。
"""集中加载并校验 YAML 配置。"""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from utils.path_tool import get_abs_path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """读取指定 YAML 配置文件，并保证顶层内容是字典对象。"""
    config_path = Path(get_abs_path(path))
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是对象：{config_path}")
    return config


def require_keys(config: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    """检查配置字典是否包含全部必需字段，缺失时立即报告。"""
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{name} 缺少配置项：{', '.join(missing)}")


model_conf = load_yaml_config("config/model.yml")
knowledge_conf = load_yaml_config("config/knowledge.yml")
prompts_conf = load_yaml_config("config/prompts.yml")
app_conf = load_yaml_config("config/app.yml")

require_keys(
    model_conf,
    (
        "chat_model_name",
        "embedding_model_name",
        "local_embedding_dimension",
        "thinking_enabled",
    ),
    "model.yml",
)
require_keys(
    knowledge_conf,
    (
        "collection_name",
        "vector_backend",
        "sqlite_database_path",
        "sqlite_manifest_path",
        "chroma_persist_directory",
        "chroma_manifest_path",
        "qdrant_manifest_path",
        "data_path",
        "allowed_file_types",
        "hybrid_vector_weight",
        "hybrid_bm25_weight",
        "bm25_k1",
        "bm25_b",
        "chunk_size",
        "chunk_overlap",
        "embedding_batch_size",
        "k",
        "sync_on_search",
    ),
    "knowledge.yml",
)

if "rerank" in knowledge_conf:
    require_keys(
        knowledge_conf["rerank"],
        ("enabled", "model", "candidate_k", "timeout"),
        "knowledge.yml:rerank",
    )
if "chroma_server" in knowledge_conf:
    require_keys(
        knowledge_conf["chroma_server"],
        (
            "mode",
            "host",
            "port",
            "ssl",
            "tenant",
            "database",
            "native_candidate_multiplier",
            "native_max_candidates",
        ),
        "knowledge.yml:chroma_server",
    )
if knowledge_conf["vector_backend"] not in {"sqlite", "chroma", "qdrant"}:
    raise ValueError(
        "knowledge.yml:vector_backend 只能是 sqlite、chroma 或 qdrant"
    )

require_keys(
    knowledge_conf.get("qdrant_server") or {},
    (
        "url",
        "grpc_port",
        "prefer_grpc",
        "timeout",
        "vector_size",
        "distance",
        "upsert_batch_size",
        "delete_batch_size",
        "shard_number",
        "replication_factor",
        "write_consistency_factor",
        "write_ordering",
        "read_consistency",
        "on_disk_payload",
        "on_disk_vectors",
        "hnsw_m",
        "hnsw_ef_construct",
        "hnsw_full_scan_threshold",
        "search_hnsw_ef",
        "native_candidate_multiplier",
        "native_max_candidates",
    ),
    "knowledge.yml:qdrant_server",
)
qdrant_conf = knowledge_conf["qdrant_server"]
qdrant_url = urlparse(str(qdrant_conf["url"]).strip())
if qdrant_url.scheme not in {"http", "https"} or not qdrant_url.netloc:
    raise ValueError("knowledge.yml:qdrant_server.url 必须是有效的 HTTP(S) URL")
if qdrant_url.username is not None or qdrant_url.password is not None:
    raise ValueError(
        "knowledge.yml:qdrant_server.url 不能包含凭据，请使用 QDRANT_API_KEY"
    )
try:
    qdrant_http_port = qdrant_url.port
except ValueError as exc:
    raise ValueError(
        "knowledge.yml:qdrant_server.url 端口必须在 1 到 65535 之间"
    ) from exc
if qdrant_http_port is not None and not 1 <= qdrant_http_port <= 65_535:
    raise ValueError(
        "knowledge.yml:qdrant_server.url 端口必须在 1 到 65535 之间"
    )
if str(qdrant_conf["distance"]).strip().lower() != "cosine":
    raise ValueError("knowledge.yml:qdrant_server.distance 当前只支持 cosine")
for key in (
    "grpc_port",
    "timeout",
    "vector_size",
    "upsert_batch_size",
    "delete_batch_size",
    "shard_number",
    "replication_factor",
    "write_consistency_factor",
    "hnsw_m",
    "hnsw_ef_construct",
    "hnsw_full_scan_threshold",
    "search_hnsw_ef",
    "native_candidate_multiplier",
    "native_max_candidates",
):
    value = qdrant_conf[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"knowledge.yml:qdrant_server.{key} 必须是大于 0 的整数"
        )
if qdrant_conf["write_consistency_factor"] > qdrant_conf["replication_factor"]:
    raise ValueError(
        "knowledge.yml:qdrant_server.write_consistency_factor "
        "不能大于 replication_factor"
    )
if str(qdrant_conf["write_ordering"]).strip().lower() not in {
    "weak",
    "medium",
    "strong",
}:
    raise ValueError(
        "knowledge.yml:qdrant_server.write_ordering "
        "只能是 weak、medium 或 strong"
    )
if str(qdrant_conf["read_consistency"]).strip().lower() not in {
    "majority",
    "quorum",
    "all",
}:
    raise ValueError(
        "knowledge.yml:qdrant_server.read_consistency "
        "只能是 majority、quorum 或 all"
    )
for key in ("prefer_grpc", "on_disk_payload", "on_disk_vectors"):
    if not isinstance(qdrant_conf[key], bool):
        raise ValueError(
            f"knowledge.yml:qdrant_server.{key} 必须是 true 或 false"
        )
if qdrant_conf["grpc_port"] > 65_535:
    raise ValueError(
        "knowledge.yml:qdrant_server.grpc_port 必须在 1 到 65535 之间"
    )
if not isinstance(knowledge_conf["sync_on_search"], bool):
    raise ValueError("knowledge.yml:sync_on_search 必须是 true 或 false")
require_keys(
    prompts_conf,
    ("system_prompt_path", "rag_answer_prompt_path"),
    "prompts.yml",
)
require_keys(
    app_conf,
    ("supported_subjects", "agent", "memory", "context"),
    "app.yml",
)
if "skills" in app_conf:
    require_keys(
        app_conf["skills"],
        ("enabled", "max_instruction_chars", "subject_paths"),
        "app.yml:skills",
    )
require_keys(
    app_conf["context"],
    (
        "recent_message_limit",
        "max_conversation_chars",
        "max_long_term_chars",
        "max_review_feedback_chars",
        "summary_enabled",
        "summary_trigger_messages",
        "summary_keep_recent_messages",
        "summary_trigger_chars",
        "max_summary_chars",
    ),
    "app.yml:context",
)
require_keys(
    app_conf["memory"],
    ("database_path", "max_long_term_memories", "retrieval_limit"),
    "app.yml:memory",
)
