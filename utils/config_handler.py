# 文件用途：读取并在启动阶段校验 model/knowledge/prompts/app 四份 YAML 配置。
# 调用关系：几乎所有后端模块导入本文件的配置字典；本文件调用 path_tool 和 PyYAML。
# 修改易踩坑：这里在导入时执行校验，新增必填项必须同步配置文件和测试，否则整个应用无法导入。
"""集中加载并校验 YAML 配置。"""

from pathlib import Path
from typing import Any

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
    ),
    "knowledge.yml",
)
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
