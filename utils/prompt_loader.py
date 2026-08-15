# 文件用途：按照 prompts.yml 读取系统 Prompt 和直接 RAG Prompt 模板。
# 调用关系：Context Manager 与 qa_service.py 调用本文件；本文件调用 config_handler 和 path_tool。
# 修改易踩坑：Prompt 文件不能为空；若加入仅供开发者阅读的文件头，必须在这里剥离后再发给模型。
"""读取教育问答提示词。"""

import re
from pathlib import Path

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path


FILE_GUIDE_PATTERN = re.compile(
    r"\A\s*<!--\s*文件用途：.*?-->\s*",
    re.DOTALL,
)


def _strip_file_guide(text: str) -> str:
    """移除只供开发者阅读的文件头，避免把调用关系和踩坑说明发送给模型。"""

    return FILE_GUIDE_PATTERN.sub("", text, count=1).strip()


def _load_prompt(config_key: str) -> str:
    """根据配置字段读取对应提示词文件并返回去除首尾空白的内容。"""
    try:
        prompt_path = Path(get_abs_path(prompts_conf[config_key]))
    except KeyError as exc:
        raise ValueError(f"prompts.yml 缺少配置项：{config_key}") from exc

    if not prompt_path.is_file():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path}")
    prompt = _strip_file_guide(prompt_path.read_text(encoding="utf-8"))
    if not prompt:
        raise ValueError(f"提示词文件正文不能为空：{prompt_path}")
    return prompt


def load_system_prompt() -> str:
    """读取教育 Agent 使用的系统提示词。"""
    return _load_prompt("system_prompt_path")


def load_rag_answer_prompt() -> str:
    """读取直接 RAG 问答链使用的回答提示词模板。"""
    return _load_prompt("rag_answer_prompt_path")
