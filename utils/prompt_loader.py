"""读取教育问答提示词。"""

from pathlib import Path

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path


def _load_prompt(config_key: str) -> str:
    """根据配置字段读取对应提示词文件并返回去除首尾空白的内容。"""
    try:
        prompt_path = Path(get_abs_path(prompts_conf[config_key]))
    except KeyError as exc:
        raise ValueError(f"prompts.yml 缺少配置项：{config_key}") from exc

    if not prompt_path.is_file():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_system_prompt() -> str:
    """读取教育 Agent 使用的系统提示词。"""
    return _load_prompt("system_prompt_path")


def load_rag_answer_prompt() -> str:
    """读取直接 RAG 问答链使用的回答提示词模板。"""
    return _load_prompt("rag_answer_prompt_path")
