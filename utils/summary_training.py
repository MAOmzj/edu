# 文件用途：校验摘要训练数据、检查分割泄漏并生成只监督目标摘要的 token labels。
# 调用关系：train_summary.py 调用；复用 summary_prompt.py；无需 torch 或线上服务。
# 修改易踩坑：padding labels 必须为 -100，目标 EOS 仍参与损失；超长样本不能静默截断。
"""摘要 SFT 的数据与监督边界。"""

from __future__ import annotations

import json
from pathlib import Path

from utils.summary_prompt import build_summary_messages, format_summary_conversation, truncate_prefix


def load_summary_records(path: str | Path, max_summary_chars: int = 1200) -> list[dict]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location} 不是有效 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{location} 需要 JSON 对象")
        for key in ("conversation_id", "target_summary"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{location} 的 {key} 需要非空文本")
        previous = record.get("previous_summary", "")
        if not isinstance(previous, str):
            raise ValueError(f"{location} 的 previous_summary 需要文本")
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{location} 的 messages 需要非空列表")
        normalized = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError(f"{location} 的消息仅支持 user/assistant 与非空 content")
            normalized.append({"role": message["role"], "content": message["content"].strip()})
        target = record["target_summary"].strip()
        if len(target) > max_summary_chars:
            raise ValueError(f"{location} 的目标摘要超过 {max_summary_chars} 字符，请人工修订")
        records.append({
            "conversation_id": record["conversation_id"].strip(),
            "previous_summary": previous.strip(),
            "messages": normalized,
            "target_summary": target,
        })
    if not records:
        raise ValueError(f"{path} 训练数据不能为空")
    return records


def validate_split(
    train: list[dict], validation: list[dict],
    *, max_summary_chars: int = 1200, summary_input_chars: int = 8000,
) -> None:
    """禁止同一会话或相同输入同时出现在训练与验证中。"""
    overlap = {r["conversation_id"] for r in train} & {r["conversation_id"] for r in validation}
    if overlap:
        raise ValueError("训练与验证集包含相同 conversation_id，请按完整会话分割")
    def inputs(records):
        return {
            json.dumps(example_messages(r, max_summary_chars, summary_input_chars), ensure_ascii=False, sort_keys=True)
            for r in records
        }
    if inputs(train) & inputs(validation):
        raise ValueError("训练与验证集包含重复输入（按实际摘要预算裁剪后），即使 conversation_id 不同也会泄漏")


def example_messages(record: dict, max_summary_chars: int, summary_input_chars: int) -> list[dict]:
    return build_summary_messages(
        truncate_prefix(record["previous_summary"] or "无", max_summary_chars),
        format_summary_conversation(record["messages"], summary_input_chars),
    )


def encode_example(
    record: dict, tokenizer, max_seq_length: int,
    *, max_summary_chars: int = 1200, summary_input_chars: int = 8000,
) -> dict:
    """使用推理时的 prompt token 前缀，单独编码摘要和 EOS，屏蔽全部输入。"""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer 缺少 eos_token_id")
    texts = [record["previous_summary"], record["target_summary"]]
    texts.extend(m["content"] for m in record["messages"])
    for special in tokenizer.all_special_tokens:
        if special and any(special in text for text in texts):
            raise ValueError("样本含模型保留控制 token，请改为普通文本后再训练")
    prompt_ids = tokenizer.apply_chat_template(
        example_messages(record, max_summary_chars, summary_input_chars),
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    target_ids = tokenizer.encode(record["target_summary"], add_special_tokens=False)
    if not prompt_ids or not target_ids:
        raise ValueError("Prompt 或目标摘要编码为空")
    target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"样本 {record['conversation_id']} 有 {len(input_ids)} tokens，"
            f"超过 {max_seq_length}；请按线上预算修订，不能截断目标摘要"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def pad_features(features: list[dict], pad_token_id: int) -> dict:
    """按位置右侧 padding；即使 pad==EOS 也保留真实摘要 EOS 的标签。"""
    if not features:
        raise ValueError("Batch 不能为空")
    length = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        padding = length - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        batch["labels"].append(feature["labels"] + [-100] * padding)
    return batch
