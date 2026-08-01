"""集中加载并校验 YAML 配置。"""

from pathlib import Path
from typing import Any

import yaml

from utils.path_tool import get_abs_path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(get_abs_path(path))
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是对象：{config_path}")
    return config


def require_keys(config: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{name} 缺少配置项：{', '.join(missing)}")


model_conf = load_yaml_config("config/model.yml")
knowledge_conf = load_yaml_config("config/knowledge.yml")
prompts_conf = load_yaml_config("config/prompts.yml")
app_conf = load_yaml_config("config/app.yml")

require_keys(
    model_conf,
    ("chat_model_name", "embedding_model_name"),
    "model.yml",
)
require_keys(
    knowledge_conf,
    (
        "collection_name",
        "persist_directory",
        "data_path",
        "manifest_path",
        "allowed_file_types",
        "chunk_size",
        "chunk_overlap",
        "k",
    ),
    "knowledge.yml",
)
require_keys(
    prompts_conf,
    ("system_prompt_path", "rag_answer_prompt_path"),
    "prompts.yml",
)
require_keys(app_conf, ("supported_subjects", "agent", "memory"), "app.yml")
require_keys(
    app_conf["memory"],
    ("database_path", "max_long_term_memories", "retrieval_limit"),
    "app.yml:memory",
)
