# 文件用途：离线验证摘要训练数据、分割泄漏、目标监督与线上 Prompt 一致性。
# 调用关系：unittest 调用本文件；测试调用 summary_training.py 与真实 ContextManager。
# 修改易踩坑：使用假 Tokenizer 和临时 JSONL，不下载模型；真实 EOS 不能因与 padding 同 ID 被屏蔽。
"""摘要 LoRA 训练的数据与 token 监督边界测试。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent.context_manager import EducationContextManager
from utils.summary_training import (
    encode_example,
    example_messages,
    load_summary_records,
    pad_features,
    validate_split,
)


def sample_record(conversation_id="conversation-a"):
    return {
        "conversation_id": conversation_id,
        "previous_summary": "此前学习了面积。",
        "messages": [
            {"role": "user", "content": "现在问周长。"},
            {"role": "assistant", "content": "旧助手答案，只作为输入背景。"},
        ],
        "target_summary": "人工核验目标：面积已学，正在学习周长。",
    }


class CharTokenizer:
    """字符到 token 一一映射，显式记录模板消息与推理前缀。"""

    eos_token_id = 1_000_000
    all_special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]

    def __init__(self):
        self.template_calls = []
        self.prompt_ids = []

    @staticmethod
    def text_ids(text):
        return [ord(character) + 1 for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict):
        self.template_calls.append({
            "messages": deepcopy(messages),
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
            "return_dict": return_dict,
        })
        # 包含非思考推理 prefill；它和全部历史同样应在监督目标外。
        prompt = "\n".join(f'{message["role"]}: {message["content"]}' for message in messages)
        prompt += "\nassistant: <think>\n\n</think>\n\n"
        self.prompt_ids = self.text_ids(prompt)
        return list(self.prompt_ids)

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("目标摘要必须单独编码，不添加第二套模板控制 token")
        return self.text_ids(text)


class RecordingSummaryModel:
    """捕获真实在线摘要入口交给模型的消息，不访问模型服务。"""

    def __init__(self):
        self.requests = []

    def invoke(self, messages):
        self.requests.append(deepcopy(messages))
        return AIMessage(content="已完成本轮学习主题摘要。")


class SummaryTrainingDataTests(unittest.TestCase):
    def _load_text(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(text, encoding="utf-8")
            return load_summary_records(path, **kwargs)

    def _load_record(self, record, **kwargs):
        return self._load_text(json.dumps(record, ensure_ascii=False) + "\n", **kwargs)

    def test_loader_normalizes_text_and_allows_blank_lines_and_optional_previous_summary(self):
        record = sample_record("  conversation-a  ")
        record.pop("previous_summary")
        record["target_summary"] = "  已确认周长。  "
        record["messages"] = [{"role": "user", "content": "  请解释周长。  "}]
        loaded = self._load_text("\n" + json.dumps(record, ensure_ascii=False) + "\n\n")
        self.assertEqual(loaded, [{
            "conversation_id": "conversation-a",
            "previous_summary": "",
            "messages": [{"role": "user", "content": "请解释周长。"}],
            "target_summary": "已确认周长。",
        }])

    def test_loader_rejects_malformed_nonobject_and_empty_jsonl(self):
        for text in ("{", "[]", "null", "", " \n\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self._load_text(text)

    def test_loader_rejects_invalid_record_fields(self):
        invalid_fields = (
            ("conversation_id", None), ("conversation_id", 123), ("conversation_id", " "),
            ("target_summary", None), ("target_summary", False), ("target_summary", " "),
            ("previous_summary", None), ("previous_summary", []),
            ("messages", None), ("messages", {}), ("messages", []),
        )
        for field, value in invalid_fields:
            record = sample_record()
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self._load_record(record)

    def test_loader_rejects_invalid_roles_message_shapes_and_content_types(self):
        invalid_messages = [None, 123, "not-a-message", {}]
        invalid_messages.extend(
            {"role": role, "content": "普通会话文本"}
            for role in (None, True, 7, [], {}, "system", "tool", "developer")
        )
        invalid_messages.extend(
            {"role": "user", "content": content}
            for content in (None, True, 123, [], {}, "", " \t ")
        )
        for message in invalid_messages:
            record = sample_record()
            record["messages"] = [message]
            with self.subTest(message=message), self.assertRaises(ValueError):
                self._load_record(record)

    def test_loader_rejects_overlong_target_without_silent_truncation(self):
        record = sample_record()
        record["target_summary"] = "12345"
        self.assertEqual(self._load_record(record, max_summary_chars=5)[0]["target_summary"], "12345")
        record["target_summary"] = "123456"
        with self.assertRaises(ValueError):
            self._load_record(record, max_summary_chars=5)

    def test_split_rejects_shared_conversation_even_for_different_windows(self):
        training = sample_record("shared-conversation")
        validation = sample_record("shared-conversation")
        validation["messages"] = [{"role": "user", "content": "同一会话较晚的一轮。"}]
        with self.assertRaisesRegex(ValueError, "conversation_id"):
            validate_split([training], [validation])

    def test_split_rejects_duplicate_input_with_different_ids_and_targets(self):
        training = sample_record("train-conversation")
        validation = sample_record("validation-conversation")
        validation["target_summary"] = "措辞不同的人工摘要。"
        validation["messages"] = [
            {"content": message["content"], "role": message["role"]}
            for message in training["messages"]
        ]
        with self.assertRaisesRegex(ValueError, "重复输入"):
            validate_split([training], [validation])

    def test_split_rejects_inputs_that_become_identical_after_context_budget(self):
        training = sample_record("train-conversation")
        validation = sample_record("validation-conversation")
        recent_messages = [
            {"role": "assistant", "content": "相同的最近答案。"},
            {"role": "user", "content": "相同的最近问题。"},
        ]
        training["messages"] = [
            {"role": "user", "content": "训练集中不同的较旧消息" * 10},
            *deepcopy(recent_messages),
        ]
        validation["messages"] = [
            {"role": "user", "content": "验证集中另一条较旧消息" * 10},
            *deepcopy(recent_messages),
        ]
        self.assertNotEqual(training["messages"], validation["messages"])
        self.assertEqual(example_messages(training, 20, 60), example_messages(validation, 20, 60))
        with self.assertRaises(ValueError):
            validate_split([training], [validation], max_summary_chars=20, summary_input_chars=60)

    def test_split_allows_distinct_inputs_even_when_targets_match(self):
        training = sample_record("train-conversation")
        validation = sample_record("validation-conversation")
        validation["messages"] = [{"role": "user", "content": "另一个会话的不同输入。"}]
        validate_split([training], [validation])


class SummaryTrainingTokenTests(unittest.TestCase):
    def test_only_final_summary_and_eos_are_supervised_with_inference_prefix_unchanged(self):
        record = sample_record()
        tokenizer = CharTokenizer()
        encoded = encode_example(record, tokenizer, max_seq_length=10_000)
        prompt_length = len(tokenizer.prompt_ids)
        expected_target = tokenizer.text_ids(record["target_summary"]) + [tokenizer.eos_token_id]
        self.assertEqual(encoded["input_ids"][:prompt_length], tokenizer.prompt_ids)
        self.assertEqual(encoded["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(encoded["labels"][prompt_length:], expected_target)
        self.assertEqual(encoded["input_ids"][prompt_length:], expected_target)
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))
        self.assertEqual([token for token in encoded["labels"] if token != -100], expected_target)
        call = tokenizer.template_calls[0]
        self.assertIs(call["tokenize"], True)
        self.assertIs(call["add_generation_prompt"], True)
        self.assertIs(call["enable_thinking"], False)
        self.assertIs(call["return_dict"], False)
        self.assertEqual([message["role"] for message in call["messages"]], ["system", "user"])
        self.assertIn(record["messages"][1]["content"], call["messages"][1]["content"])
        self.assertNotIn(record["target_summary"], call["messages"][1]["content"])

    def test_sequence_boundary_includes_eos_and_overlong_sample_is_rejected(self):
        record = sample_record()
        original = deepcopy(record)
        encoded = encode_example(record, CharTokenizer(), max_seq_length=10_000)
        length = len(encoded["input_ids"])
        at_limit = encode_example(record, CharTokenizer(), max_seq_length=length)
        self.assertEqual(at_limit, encoded)
        with self.assertRaisesRegex(ValueError, "不能截断"):
            encode_example(record, CharTokenizer(), max_seq_length=length - 1)
        self.assertEqual(record, original)

    def test_reserved_control_tokens_in_history_previous_summary_or_target_are_rejected(self):
        for field in ("previous_summary", "target_summary", "history_user", "history_assistant"):
            for token in CharTokenizer.all_special_tokens:
                record = sample_record()
                if field.startswith("history_"):
                    index = 0 if field == "history_user" else 1
                    record["messages"][index]["content"] += token
                else:
                    record[field] += token
                with self.subTest(field=field, token=token), self.assertRaisesRegex(ValueError, "控制 token"):
                    encode_example(record, CharTokenizer(), max_seq_length=10_000)

    def test_missing_eos_and_empty_target_encoding_are_rejected(self):
        tokenizer = CharTokenizer()
        tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            encode_example(sample_record(), tokenizer, max_seq_length=10_000)
        empty_target = sample_record()
        empty_target["target_summary"] = ""
        with self.assertRaisesRegex(ValueError, "编码为空"):
            encode_example(empty_target, CharTokenizer(), max_seq_length=10_000)

    def test_padding_masks_only_added_positions_when_pad_and_eos_share_id(self):
        shorter = sample_record("shorter")
        shorter["target_summary"] = "短摘要。"
        longer = sample_record("longer")
        tokenizer = CharTokenizer()
        features = [encode_example(record, tokenizer, 10_000) for record in (shorter, longer)]
        originals = deepcopy(features)
        batch = pad_features(features, pad_token_id=tokenizer.eos_token_id)
        total_length = len(features[1]["input_ids"])
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            padding = total_length - length
            with self.subTest(row=index):
                self.assertEqual(len(batch["input_ids"][index]), total_length)
                self.assertEqual(batch["labels"][index][:length], feature["labels"])
                self.assertEqual(batch["labels"][index][length - 1], tokenizer.eos_token_id)
                self.assertEqual(batch["attention_mask"][index][length - 1], 1)
                self.assertEqual(batch["input_ids"][index][length:], [tokenizer.eos_token_id] * padding)
                self.assertEqual(batch["attention_mask"][index][length:], [0] * padding)
                self.assertEqual(batch["labels"][index][length:], [-100] * padding)
        self.assertEqual(features, originals)
        with self.assertRaises(ValueError):
            pad_features([], tokenizer.eos_token_id)

    def test_training_prompt_matches_actual_context_manager_invoke_with_same_budgets(self):
        record = sample_record()
        record["previous_summary"] = "较早主题" * 20
        record["messages"] = [
            {"role": "user", "content": "会被预算移除的较旧消息" * 10},
            {"role": "assistant", "content": "最近完整答案。"},
            {"role": "user", "content": "最近完整问题。"},
        ]
        model = RecordingSummaryModel()
        manager = EducationContextManager(
            None, summary_model=model,
            max_conversation_chars=100, max_summary_chars=20, summary_trigger_chars=30,
        )
        older_messages = [
            (AIMessage if message["role"] == "assistant" else HumanMessage)(content=message["content"])
            for message in record["messages"]
        ]
        manager.summarize_conversation(
            previous_summary=record["previous_summary"], older_messages=older_messages,
        )
        tokenizer = CharTokenizer()
        encode_example(record, tokenizer, 10_000, max_summary_chars=20, summary_input_chars=60)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(tokenizer.template_calls[0]["messages"], model.requests[0])
        self.assertEqual(example_messages(record, 20, 60), model.requests[0])
        user_input = model.requests[0][1]["content"]
        self.assertNotIn("会被预算移除", user_input)
        self.assertIn("小助手：最近完整答案。", user_input)
        self.assertIn("学生：最近完整问题。", user_input)
        self.assertIn("…", user_input)


if __name__ == "__main__":
    unittest.main()
