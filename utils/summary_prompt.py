# 文件用途：定义线上压缩与离线微调共用的摘要守则、消息格式和字符裁剪。
# 调用关系：context_manager.py 和 summary_training.py 调用；本模块仅依赖标准库。
# 修改易踩坑：修改守则会同时改变线上与训练输入；裁剪必须优先保留最新的完整消息。
"""摘要输入的共同契约，不加载模型、配置或学生数据库。"""

SUMMARY_SYSTEM_PROMPT = (
    "你负责压缩小学生问答会话。只保留已确认的学习主题、"
    "正确结论、学生明确表达的困难或偏好、尚未解决的问题。"
    "保留旧摘要中仍有效的信息，如实记录讲解内容和本轮实际表现；"
    "不从助手讲解或学生单次答对推断已掌握或熟练。"
    "后续纠正应替代旧结论，已解决的问题应更新状态，题目人物的经历不能归为学生经历。"
    "不要编造信息，不保存姓名、学校、联系方式等隐私，"
    "不要记录内部工具调用、草稿或审核过程。只输出简短摘要。"
)


def truncate_prefix(value: str, max_chars: int) -> str:
    """保留开头，超限时以省略号结束。"""
    cleaned = value.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars == 1:
        return "…"
    return cleaned[: max_chars - 1].rstrip() + "…"


def format_summary_conversation(messages: list[dict], max_chars: int) -> str:
    """在字符预算内保留最近对话，与在线上下文格式一致。"""
    newest_first = []
    used_chars = 0
    for message in reversed(messages):
        text = message["content"].strip()
        if not text:
            continue
        role = "小助手" if message["role"] == "assistant" else "学生"
        line = f"{role}：{text}"
        separator_chars = 1 if newest_first else 0
        remaining = max_chars - used_chars - separator_chars
        if remaining < 1:
            break
        if len(line) > remaining:
            if not newest_first:
                newest_first.append(truncate_prefix(line, remaining))
            break
        newest_first.append(line)
        used_chars += len(line) + separator_chars
    return "\n".join(reversed(newest_first)) or "无"


def build_summary_messages(previous_summary: str, older_context: str) -> list[dict]:
    """构建共同的 system/user 输入；目标摘要由训练代码单独追加。"""
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"已有较早会话摘要：\n{previous_summary}\n\n"
                f"本次需要并入摘要的旧消息：\n{older_context}"
            ),
        },
    ]
