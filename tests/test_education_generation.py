# 文件用途：验证教学数据生成的本地调用、真实目标接受条件、分割与来源记录。
# 调用关系：unittest调用generate_education_data，不调用网络、模型、GPU或真实输出目录。
# 修改易踩坑：截断/内部控制标记不能变成训练标签，参考元数据不得进入模型输入。
import copy
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import generate_education_data as generator


class Tokenizer:
    eos_token_id = 9
    all_special_tokens = ["<|im_end|>"]
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [1, 2, 3]
    def encode(self, text, **kwargs):
        return list(range(10, 10 + len(text)))


def seed():
    return {"id": "math-add-01", "group_id": "math-add", "subject": "数学", "grade": 1,
            "skill": "加法", "split": "train", "source": "synthetic_authored",
            "messages": [{"role": "user", "content": "2颗红珠子和3颗蓝珠子一共几颗？"}],
            "quality_checks": {"answer": 5}}


def response(content="把两组珠子合起来：2+3=5，所以一共5颗。", finish="stop", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
        message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        model="qwen3.8-27b", id="local-response-1",
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}))


class EducationGenerationTests(unittest.TestCase):
    def incremental_seeds(self):
        rows = []
        for subject in ["数学", "语文", "英语", "科学"]:
            item = seed()
            item.update(id="extra-"+subject, group_id="extra-"+subject, subject=subject,
                        messages=[{"role": "user", "content": subject+"独立教学问题？"}])
            rows.append(item)
        return rows

    def test_incremental_training_batch_can_have_no_new_held_out_data(self):
        rows = self.incremental_seeds()
        generator.validate_seeds(rows, "指令", {"train": 4, "validation": 0, "test": 0})
        for expected in [{"train": 0, "validation": 0, "test": 0},
                         {"train": 3, "validation": 0, "test": 0}]:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                generator.validate_seeds(rows, "指令", expected)

    def test_teacher_reasoning_is_recorded_as_flag_and_not_training_text(self):
        actual = response()
        actual.choices[0].message.reasoning_content = "只属于教师响应的测试字段"
        original = seed()
        expected = generator.generation_messages(original, "指令")
        metadata = {"teacher_model": "qwen3.8-27b", "teacher_endpoint": "http://127.0.0.1:8000/v1",
                    "input_seed_sha256": generator.digest(original), "prompt_sha256": generator.digest(expected),
                    "enable_thinking": True}
        row = generator.accepted_record(original, actual, Tokenizer(), "指令", 2048, metadata)
        self.assertTrue(row["generation"]["reasoning_content_present"])
        self.assertNotIn("只属于教师响应的测试字段", json.dumps(row, ensure_ascii=False))
        generator.validate_candidate(row, original, Tokenizer(), {"instruction":"指令", "max_seq_length":2048},
                                     {"model":"qwen3.8-27b", "base_url":"http://127.0.0.1:8000/v1", "enable_thinking":True})

    def test_incremental_candidates_are_reviewable_before_resumable_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.incremental_seeds()
            args = SimpleNamespace(output_dir=root, model="qwen3.8-27b", base_url="http://127.0.0.1:8000/v1",
                temperature=.3, teacher_thinking=True, max_tokens=2048, concurrency=2,
                train_count=4, validation_count=0, test_count=0, resume=False, defer_publication=True)
            create = AsyncMock(side_effect=[response() for _ in rows])
            client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(
                data=[SimpleNamespace(id=args.model)]))), chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                close=AsyncMock())
            modules = {"openai":SimpleNamespace(AsyncOpenAI=Mock(return_value=client)),
                       "httpx":SimpleNamespace(AsyncClient=Mock(return_value=object()))}
            with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                asyncio.run(generator.generate(args, rows, {"instruction":"指令", "max_seq_length":2048}, Tokenizer()))
                original = (root / "model_candidates.jsonl").read_bytes()
                self.assertFalse((root / "train.jsonl").exists())
                self.assertEqual(json.loads((root / "generation_progress.json").read_text())["status"], "candidates_ready")
                args.resume, args.defer_publication = True, False
                asyncio.run(generator.generate(args, rows, {"instruction":"指令", "max_seq_length":2048}, Tokenizer()))
            self.assertEqual(create.await_count, 4)
            self.assertEqual((root / "model_candidates.jsonl").read_bytes(), original)
            self.assertEqual(len((root / "train.jsonl").read_text().splitlines()), 4)
            self.assertFalse((root / "validation.jsonl").exists())
            self.assertFalse((root / "test.jsonl").exists())
            self.assertTrue(all(call.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
                                for call in create.await_args_list))
    def test_only_local_server_without_credentials_allowed(self):
        self.assertEqual(generator.local_url("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")
        for url in ["https://example.com/v1", "http://user:secret@127.0.0.1:8000/v1",
                    "http://127.0.0.1/v1", "http://127.0.0.1:8000/v1?token=secret"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                generator.local_url(url)

    def test_reference_metadata_is_not_sent_to_teacher(self):
        messages = generator.generation_messages(seed(), "小学学习助手")
        self.assertEqual(messages, [{"role": "system", "content": "小学学习助手"}] + seed()["messages"])
        self.assertNotIn("quality_checks", str(messages))

    def test_only_actual_content_is_target_and_remains_pending(self):
        original = seed()
        tokenizer = Tokenizer()
        row = generator.accepted_record(original, response(), tokenizer, "小学学习助手", 2048,
                                         {"teacher_model": "qwen3.8-27b"})
        self.assertEqual(row["source"], "local_model")
        self.assertEqual(row["review_status"], "pending")
        self.assertEqual(row["messages"][-1]["content"], response().choices[0].message.content)
        self.assertNotIn("quality_checks", row)
        self.assertEqual(original["messages"][-1]["role"], "user")
        self.assertEqual(row["generation"]["finish_reason"], "stop")

    def test_truncation_empty_tools_and_control_labels_rejected(self):
        invalid = [response(finish="length"), response(content=""), response(tool_calls=[{}]),
                   response(content="<think>内部过程</think>然后回答"),
                   response(content="<tool_call>假工具调用</tool_call>")]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator.accepted_record(seed(), value, Tokenizer(), "指令", 2048, {})

    def test_overlong_answer_rejected_without_cutting(self):
        with self.assertRaisesRegex(ValueError, "不能截断"):
            generator.accepted_record(seed(), response(), Tokenizer(), "指令", 10, {})

    def test_short_correct_teaching_answer_is_not_rejected_by_character_count(self):
        row = generator.accepted_record(seed(), response(content="互相垂直。"), Tokenizer(), "指令", 2048, {})
        self.assertEqual(row["messages"][-1]["content"], "互相垂直。")

    def test_resume_rejects_changed_content_or_original_prompt(self):
        original = seed()
        metadata = {"teacher_model": "qwen3.8-27b", "teacher_endpoint": "http://127.0.0.1:8000/v1",
                    "input_seed_sha256": generator.digest(original),
                    "prompt_sha256": generator.digest(generator.generation_messages(original, "指令")),
                    "enable_thinking": False}
        row = generator.accepted_record(original, response(), Tokenizer(), "指令", 2048, metadata)
        config = {"instruction": "指令", "max_seq_length": 2048}
        parameters = {"model": "qwen3.8-27b", "base_url": "http://127.0.0.1:8000/v1"}
        generator.validate_candidate(row, original, Tokenizer(), config, parameters)
        for change in ["group_id", "subject", "split", "source", "review_status", "messages"]:
            altered = copy.deepcopy(row)
            if change == "messages": altered[change][-1]["content"] = "改成了另外一个未经记录的答案。"
            else: altered[change] = "changed"
            with self.subTest(change=change), self.assertRaises(ValueError):
                generator.validate_candidate(altered, original, Tokenizer(), config, parameters)

    @staticmethod
    def revision_candidate():
        original = seed()
        config = {"instruction": "指令", "max_seq_length": 2048}
        parameters = {"model": "qwen3.8-27b", "base_url": "http://127.0.0.1:8000/v1"}
        expected = generator.generation_messages(original, config["instruction"])
        metadata = {"teacher_model": parameters["model"], "teacher_endpoint": parameters["base_url"],
                    "input_seed_sha256": generator.digest(original), "prompt_sha256": generator.digest(expected),
                    "enable_thinking": False}
        first = generator.accepted_record(original, response(content="两组共有6颗珠子。"), Tokenizer(),
                                          config["instruction"], 2048, metadata)
        feedback = "请核对加法结果，并重新给出学生可见的教学回答。"
        request = expected + [copy.deepcopy(first["messages"][-1]), {"role": "user", "content": feedback}]
        revised_metadata = {**metadata, "kind": "model_revision_after_review",
                            "original_prompt_sha256": generator.digest(expected),
                            "original_candidate_sha256": first["candidate_sha256"],
                            "revision_request_messages": request, "review_feedback": feedback,
                            "prompt_sha256": generator.digest(request)}
        revised = generator.accepted_record(original, response(), Tokenizer(), config["instruction"],
                                            2048, revised_metadata)
        return revised, original, config, parameters

    @staticmethod
    def resign_candidate(row):
        row["candidate_sha256"] = generator.digest({key: value for key, value in row.items()
                                                    if key != "candidate_sha256"})

    def test_revision_validates_actual_teacher_prompt_and_keeps_feedback_out_of_sft(self):
        row, original, config, parameters = self.revision_candidate()
        generator.validate_candidate(row, original, Tokenizer(), config, parameters)
        expected = generator.generation_messages(original, config["instruction"])
        self.assertEqual(row["messages"][:-1], expected)
        self.assertEqual(len(row["messages"]), len(expected) + 1)
        self.assertNotEqual(row["generation"]["prompt_sha256"], row["generation"]["original_prompt_sha256"])
        self.assertEqual(row["generation"]["prompt_sha256"],
                         generator.digest(row["generation"]["revision_request_messages"]))

    def test_revision_can_enable_teacher_thinking_with_original_setting_recorded(self):
        row, original, config, parameters = self.revision_candidate()
        row["generation"]["enable_thinking"] = True
        self.resign_candidate(row)
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            generator.validate_candidate(row, original, Tokenizer(), config, parameters)
        row["generation"]["original_teacher_enable_thinking"] = False
        self.resign_candidate(row)
        generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_regeneration_keeps_reference_feedback_in_teacher_system_only(self):
        row, original, config, parameters = self.revision_candidate()
        metadata = row["generation"]
        metadata["kind"] = "model_regeneration_after_review"
        metadata["revision_request_messages"] = generator.regeneration_messages(
            original, config["instruction"], metadata["review_feedback"])
        metadata["prompt_sha256"] = generator.digest(metadata["revision_request_messages"])
        self.resign_candidate(row)
        generator.validate_candidate(row, original, Tokenizer(), config, parameters)
        self.assertEqual(row["messages"][:-1], generator.generation_messages(original, config["instruction"]))
        self.assertNotIn(metadata["review_feedback"], str(row["messages"]))
        self.assertEqual(len(metadata["revision_request_messages"]), len(row["messages"])-1)
        metadata["review_feedback"] = "没有记录进实际请求的另一条反馈"
        self.resign_candidate(row)
        with self.assertRaisesRegex(ValueError, "修订请求"):
            generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_revision_rejects_original_prompt_hash_claimed_as_actual_request(self):
        row, original, config, parameters = self.revision_candidate()
        row["generation"]["prompt_sha256"] = row["generation"]["original_prompt_sha256"]
        self.resign_candidate(row)
        with self.assertRaisesRegex(ValueError, "prompt_sha256"):
            generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_revision_rejects_tampered_request_prefix_even_when_hash_recomputed(self):
        row, original, config, parameters = self.revision_candidate()
        row["generation"]["revision_request_messages"][0]["content"] = "更换原始系统指令"
        row["generation"]["prompt_sha256"] = generator.digest(row["generation"]["revision_request_messages"])
        self.resign_candidate(row)
        with self.assertRaisesRegex(ValueError, "修订请求"):
            generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_revision_rejects_original_hash_and_candidate_identity_tampering(self):
        for key, value in [("original_prompt_sha256", "0" * 64),
                           ("original_candidate_sha256", "A" * 64),
                           ("original_candidate_sha256", "f" * 63),
                           ("original_candidate_sha256", "g" * 64),
                           ("original_candidate_sha256", None)]:
            row, original, config, parameters = self.revision_candidate()
            row["generation"][key] = value
            self.resign_candidate(row)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_revision_requires_exact_tail_roles_nonempty_content_and_matching_feedback(self):
        changes = ["feedback_mismatch", "empty_feedback", "empty_original", "wrong_role", "extra_tail_field"]
        for change in changes:
            row, original, config, parameters = self.revision_candidate()
            metadata = row["generation"]
            request = metadata["revision_request_messages"]
            if change == "feedback_mismatch": request[-1]["content"] = "换一条反馈"
            elif change == "empty_feedback": metadata["review_feedback"] = request[-1]["content"] = " "
            elif change == "empty_original": request[-2]["content"] = " "
            elif change == "wrong_role": request[-2]["role"] = "user"
            else: request[-1]["extra"] = "不属于严格请求结构的字段"
            metadata["prompt_sha256"] = generator.digest(request)
            self.resign_candidate(row)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "修订请求"):
                generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_revision_feedback_leakage_in_training_messages_is_rejected(self):
        row, original, config, parameters = self.revision_candidate()
        row["messages"] = copy.deepcopy(row["generation"]["revision_request_messages"]) + [row["messages"][-1]]
        self.resign_candidate(row)
        with self.assertRaisesRegex(ValueError, "原始题面"):
            generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_unknown_revision_kind_is_rejected(self):
        for kind in ["human_edit", "unknown_revision", None]:
            row, original, config, parameters = self.revision_candidate()
            row["generation"]["kind"] = kind
            self.resign_candidate(row)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "未知kind"):
                generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_revision_still_checks_common_model_provenance_and_checksum(self):
        fields = ["teacher_model", "response_model", "teacher_endpoint", "finish_reason",
                  "enable_thinking", "input_seed_sha256"]
        for key in fields:
            row, original, config, parameters = self.revision_candidate()
            row["generation"][key] = "changed"
            self.resign_candidate(row)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                generator.validate_candidate(row, original, Tokenizer(), config, parameters)
        row, original, config, parameters = self.revision_candidate()
        row["messages"][-1]["content"] = "未重新记录的回答"
        with self.assertRaisesRegex(ValueError, "校验值"):
            generator.validate_candidate(row, original, Tokenizer(), config, parameters)

    def test_resume_isolates_only_incomplete_last_json_line(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candidates.jsonl"
            path.write_bytes(b'{"id":"one"}\n{"id":"two"')
            self.assertEqual(generator.read_candidates(path), [{"id": "one"}])
            self.assertEqual(path.read_bytes(), b'{"id":"one"}\n')
            self.assertTrue(path.with_suffix(".incomplete-last-line.bin").exists())
            path.write_bytes(b'broken\n{"id":"two"}\n')
            with self.assertRaisesRegex(ValueError, "损坏"):
                generator.read_candidates(path)

    def test_interrupted_publication_resumes_only_owned_files(self):
        row = seed()
        row["messages"].append({"role": "assistant", "content": "一共5颗珠子。"})
        splits = {"train": [row], "validation": [{**row, "id": "validation", "group_id": "validation"}],
                  "test": [{**row, "id": "test", "group_id": "test"}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generator.publish_splits(root, splits)
            (root / "validation.jsonl").unlink()
            (root / "test.jsonl").unlink()
            original_train = (root / "train.jsonl").read_bytes()
            generator.publish_splits(root, splits)
            self.assertEqual((root / "train.jsonl").read_bytes(), original_train)
            self.assertTrue((root / "validation.jsonl").exists())
            (root / "test.jsonl").write_text("external changed data")
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                generator.publish_splits(root, splits)

    def test_seed_splits_block_relabelled_groups_and_duplicates(self):
        rows = []
        for split, count in [("train", 250), ("validation", 25), ("test", 25)]:
            for subject in ["数学", "语文", "英语", "科学"]:
                for index in range(count):
                    row = seed()
                    identity = f"{split}-{subject}-{index}"
                    row.update(id=identity, group_id=identity, split=split, subject=subject,
                               messages=[{"role": "user", "content": f"独立问题{identity}"}])
                    rows.append(row)
        generator.validate_seeds(rows, "指令")
        bad = copy.deepcopy(rows)
        bad[-1]["group_id"] = bad[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "跨集合"):
            generator.validate_seeds(bad, "指令")
        bad = copy.deepcopy(rows)
        bad[-1]["messages"] = bad[0]["messages"]
        with self.assertRaisesRegex(ValueError, "实际输入重复"):
            generator.validate_seeds(bad, "指令")


if __name__ == "__main__":
    unittest.main()
