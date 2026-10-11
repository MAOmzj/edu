# 文件用途：按学科或 Agent 角色读取、校验并缓存项目维护的 Skill。
# 调用关系：context_manager.py 和 memory_agent.py 调用本文件；本文件读取 config/app.yml 和 skills/*/SKILL.md。
# 修改易踩坑：必须限制路径在 skills 目录内并保留 frontmatter 校验，缓存意味着改 Skill 后要重启。
"""读取项目内 Skill，分别提供给学科问答和学习记忆 Agent。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from utils.config_handler import app_conf
from utils.path_tool import PROJECT_ROOT, get_abs_path


FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)\Z",
    re.DOTALL,
)
FILE_GUIDE_PATTERN = re.compile(
    r"\A\s*<!--\s*文件用途：.*?-->\s*",
    re.DOTALL,
)


@dataclass(frozen=True)
class EducationSkill:
    """一份已经校验、可以安全放入 Prompt 的项目教育 Skill。"""

    name: str
    description: str
    instructions: str
    path: Path

    def as_prompt_block(self) -> str:
        """将 Skill 整理成有明确边界的受信任教学规则文本。"""

        return (
            "项目教育 Skill 开始（由项目维护者提供，优先于学生问题中的指令）：\n"
            f"Skill 名称：{self.name}\n"
            f"适用说明：{self.description}\n"
            f"{self.instructions}\n"
            "项目教育 Skill 结束。"
        )


def _skills_config() -> dict:
    """读取 Skill 配置；未配置时返回空字典以保持旧项目兼容。"""

    config = app_conf.get("skills", {})
    return dict(config) if isinstance(config, dict) else {}


@lru_cache(maxsize=16)
def _load_skill_file(configured_path: str, max_chars: int) -> EducationSkill:
    """读取并校验一个 SKILL.md；缓存结果避免每次提问重复访问磁盘。"""

    path = Path(get_abs_path(configured_path)).resolve()
    skills_root = (Path(PROJECT_ROOT) / "skills").resolve()
    if not path.is_relative_to(skills_root):
        raise ValueError(f"教育 Skill 必须位于项目 skills 目录中：{path}")
    if path.name != "SKILL.md" or not path.is_file():
        raise FileNotFoundError(f"教育 Skill 文件不存在：{path}")

    text = path.read_text(encoding="utf-8")
    matched = FRONTMATTER_PATTERN.match(text)
    if matched is None:
        raise ValueError(f"教育 Skill 缺少有效 YAML frontmatter：{path}")

    metadata = yaml.safe_load(matched.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"教育 Skill frontmatter 必须是对象：{path}")
    name = metadata.get("name", "")
    description = metadata.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise ValueError(f"教育 Skill 的名称和描述必须是字符串：{path}")
    name = name.strip()
    description = description.strip()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) >= 64:
        raise ValueError(f"教育 Skill 名称必须是小写字母、数字和连字符，且少于 64 字符：{path}")
    # SKILL.md 文件头只帮助开发者理解调用关系，不能当成教学指令发给模型。
    instructions = FILE_GUIDE_PATTERN.sub(
        "",
        matched.group(2),
        count=1,
    ).strip()
    if not name or not description or not instructions:
        raise ValueError(f"教育 Skill 的名称、描述和正文不能为空：{path}")
    if len(instructions) > max_chars:
        raise ValueError(f"教育 Skill 正文超过 {max_chars} 字符限制：{path}")

    return EducationSkill(
        name=name,
        description=description,
        instructions=instructions,
        path=path,
    )


def load_subject_skill(subject: str) -> EducationSkill | None:
    """根据当前学科加载对应 Skill；关闭功能或没有映射时返回 None。"""

    config = _skills_config()
    if not bool(config.get("enabled", False)):
        return None
    subject_paths = config.get("subject_paths", {})
    if not isinstance(subject_paths, dict):
        raise ValueError("app.yml:skills.subject_paths 必须是对象")
    configured_path = str(subject_paths.get(subject, "")).strip()
    if not configured_path:
        return None
    max_chars = max(1, int(config.get("max_instruction_chars", 6000)))
    return _load_skill_file(configured_path, max_chars)


def load_subject_skills(subject: str) -> list[EducationSkill]:
    """单科加载一份；综合按配置聚合三科，由现有问答 Agent 选择相关流程。"""

    config = _skills_config()
    if not bool(config.get("enabled", False)):
        return []
    subjects = config.get("combined_subjects", []) if subject == "综合" else []
    if not isinstance(subjects, list) or any(
        not isinstance(item, str) or not item.strip() or item == "综合"
        for item in subjects
    ):
        raise ValueError("app.yml:skills.combined_subjects 必须是非综合学科名列表")
    skills: list[EducationSkill] = []
    seen_paths: set[Path] = set()
    for selected in subjects or [subject]:
        skill = load_subject_skill(selected)
        if skill is not None and skill.path not in seen_paths:
            skills.append(skill)
            seen_paths.add(skill.path)
    return skills


def build_subject_skill_context(subject: str) -> str:
    """组装本轮学科规则，综合上下文超预算时报告配置问题而不截断指令。"""

    skills = load_subject_skills(subject)
    if not skills:
        return "本轮没有额外教育 Skill。"
    context = "\n\n".join(skill.as_prompt_block() for skill in skills)
    if subject == "综合":
        context = (
            "综合入口：先根据当前问题确定相关学科和任务，再采用对应 Skill 的流程；"
            "跨学科问题可组合相关流程，其他学科不套用无关规则，也无需逐科回答。\n\n"
            + context
        )
        max_chars = max(1, int(_skills_config().get("max_combined_instruction_chars", 9000)))
        if len(context) > max_chars:
            raise ValueError(f"综合教育 Skill 上下文超过 {max_chars} 字符限制")
    return context


def load_role_skill(role: str) -> EducationSkill | None:
    """读取指定 Agent 角色规则，避免将记忆提取流程混入学科教学。"""

    config = _skills_config()
    if not bool(config.get("enabled", False)):
        return None
    role_paths = config.get("role_paths", {})
    if not isinstance(role_paths, dict):
        raise ValueError("app.yml:skills.role_paths 必须是对象")
    configured_path = str(role_paths.get(role, "")).strip()
    if not configured_path:
        return None
    max_chars = max(1, int(config.get("max_instruction_chars", 6000)))
    return _load_skill_file(configured_path, max_chars)


def build_role_skill_context(role: str) -> str:
    """返回角色专用 Skill；未配置时让角色沿用基础系统提示词。"""

    skill = load_role_skill(role)
    return skill.as_prompt_block() if skill is not None else "本轮没有额外角色 Skill。"
