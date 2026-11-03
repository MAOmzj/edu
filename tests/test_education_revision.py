# 文件用途：回归验证教学候选修订的请求重试、版本保留与训练输入隔离。
# 调用关系：unittest使用临时目录、假分词器和本地API替身调用revise，不访问网络或模型。
# 修改易踩坑：测试素材只能留在临时目录；失败不能替换原候选，反馈不能进入训练消息。
import asyncio
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import generate_education_data as generator
from scripts import revise_education_data as revision
from tests.test_education_generation import Tokenizer, response, seed


class EducationRevisionTests(unittest.TestCase):
    def prepare(self, root, revise_both=False):
        config = {"instruction": "小学教学测试指令", "max_seq_length": 2048}
        parameters = {"model": "qwen3.8-27b", "base_url": "http://127.0.0.1:8000/v1"}
        first = seed()
        second = copy.deepcopy(first)
        second.update(id="math-add-02", group_id="math-add-two",
                      messages=[{"role": "user", "content": "4颗红珠子和3颗蓝珠子一共几颗？"}])
        seeds = [first, second]
        rows = []
        for item in seeds:
            original_response = response(content="两组珠子共有6颗。")
            original_response.id = "original-" + item["id"]
            expected = generator.generation_messages(item, config["instruction"])
            metadata = {"teacher_model": parameters["model"],
                        "teacher_endpoint": parameters["base_url"],
                        "input_seed_sha256": generator.digest(item),
                        "prompt_sha256": generator.digest(expected), "enable_thinking": False}
            rows.append(generator.accepted_record(item, original_response, Tokenizer(),
                        config["instruction"], config["max_seq_length"], metadata))
        feedback = {first["id"]: "测试复核反馈标记：核对2+3，并仅给原问题的学生可见回答。"}
        if revise_both:
            feedback[second["id"]] = "第二条测试反馈：核对4+3，并仅给原问题的学生可见回答。"
        generator.save_lines(root / "inputs.jsonl", seeds)
        generator.save_lines(root / "model_candidates.jsonl", rows)
        (root / "generation_manifest.json").write_text(json.dumps({
            "inputs_sha256": generator.digest(seeds),
            "instruction_sha256": generator.digest(config["instruction"]),
            "parameters": parameters}, ensure_ascii=False), encoding="utf-8")
        feedback_path = root / "test_feedback.json"
        feedback_path.write_text(json.dumps(feedback, ensure_ascii=False), encoding="utf-8")
        args = SimpleNamespace(output_dir=root, feedback=feedback_path)
        return args, config, seeds, rows, feedback

    def run_revision(self, args, config, outcomes):
        create = AsyncMock(side_effect=outcomes)
        client = SimpleNamespace(
            models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(
                data=[SimpleNamespace(id="qwen3.8-27b")]))),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock())
        openai_factory = Mock(return_value=client)
        http_factory = Mock(return_value=object())
        sleep = AsyncMock()
        fake_modules = {"openai": SimpleNamespace(AsyncOpenAI=openai_factory),
                        "httpx": SimpleNamespace(AsyncClient=http_factory)}
        with patch.dict(sys.modules, fake_modules), patch.object(revision.asyncio, "sleep", sleep), \
                redirect_stdout(io.StringIO()):
            try:
                asyncio.run(revision.revise(args, Tokenizer(), config))
            except Exception as error:
                return client, create, sleep, error
        openai_factory.assert_called_once()
        http_factory.assert_called_once_with(trust_env=False)
        return client, create, sleep, None

    def assert_unpublished(self, root):
        for name in ["train.jsonl", "validation.jsonl", "test.jsonl", "publication_manifest.json"]:
            self.assertFalse((root / name).exists(), name)

    def assert_backup(self, root, original_bytes):
        backups = list(root.glob("model_candidates.before_revision.*.jsonl"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original_bytes)

    def test_api_failure_retries_and_keeps_feedback_out_of_training_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config, seeds, original_rows, feedback = self.prepare(root)
            original_bytes = (root / "model_candidates.jsonl").read_bytes()
            actual_response = response()
            actual_response.id = "mock-revision-success-2"
            client, create, sleep, error = self.run_revision(args, config, [
                TimeoutError("模拟本地请求超时"), actual_response])
            self.assertIsNone(error)
            self.assertEqual(create.await_count, 2)
            sleep.assert_awaited_once_with(1)
            client.close.assert_awaited_once()
            ledger = [json.loads(line) for line in
                      (root / "model_revision_responses.jsonl").read_text().splitlines()]
            self.assertEqual([item["accepted"] for item in ledger], [False, True])
            self.assertEqual(ledger[0]["attempt"], 1)
            self.assertEqual(ledger[0]["error_type"], "TimeoutError")
            self.assertIsNone(ledger[0]["response"])
            updated = [json.loads(line) for line in
                       (root / "model_candidates.jsonl").read_text().splitlines()]
            revised = updated[0]
            expected = generator.generation_messages(seeds[0], config["instruction"])
            note = feedback[seeds[0]["id"]]
            teacher_request = expected + [original_rows[0]["messages"][-1],
                                          {"role": "user", "content": note}]
            for call in create.await_args_list:
                self.assertEqual(call.kwargs["messages"], teacher_request)
            self.assertEqual(revised["messages"][:-1], original_rows[0]["messages"][:-1])
            self.assertEqual(revised["messages"][-1]["content"], actual_response.choices[0].message.content)
            self.assertNotIn(note, "\n".join(message["content"] for message in revised["messages"]))
            self.assertEqual(revised["generation"]["review_feedback"], note)
            self.assertEqual(revised["generation"]["revision_request_messages"], teacher_request)
            self.assertEqual(revised["generation"]["response_id"], actual_response.id)
            self.assertEqual(revised["generation"]["attempt"], 2)
            self.assertEqual(revised["generation"]["original_candidate_sha256"],
                             original_rows[0]["candidate_sha256"])
            self.assertEqual(ledger[1]["revised_candidate"], revised)
            self.assertEqual(revised["review_status"], "pending")
            self.assertEqual(updated[1], original_rows[1])
            self.assert_backup(root, original_bytes)
            self.assert_unpublished(root)

    def test_three_api_failures_preserve_whole_batch_even_after_another_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config, _, _, _ = self.prepare(root, revise_both=True)
            original_bytes = (root / "model_candidates.jsonl").read_bytes()
            client, create, sleep, error = self.run_revision(args, config, [response(),
                ConnectionError("模拟失败1"), ConnectionError("模拟失败2"), ConnectionError("模拟失败3")])
            self.assertIsInstance(error, ConnectionError)
            self.assertEqual(create.await_count, 4)
            self.assertEqual([call.args for call in sleep.await_args_list], [(1,), (2,)])
            client.close.assert_awaited_once()
            self.assertEqual((root / "model_candidates.jsonl").read_bytes(), original_bytes)
            self.assert_backup(root, original_bytes)
            ledger = [json.loads(line) for line in
                      (root / "model_revision_responses.jsonl").read_text().splitlines()]
            failed = [item for item in ledger if not item["accepted"]]
            self.assertEqual([item["attempt"] for item in failed], [1, 2, 3])
            self.assertTrue(all(item["response"] is None for item in failed))
            self.assertTrue(all(item["error_type"] == "ConnectionError" for item in failed))
            self.assertEqual(sum(item["accepted"] for item in ledger), 1)
            self.assertFalse(list(root.glob("revision_report.*.json")))
            self.assert_unpublished(root)

    def test_stale_old_temporary_file_does_not_block_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config, _, _, feedback = self.prepare(root)
            original_bytes = (root / "model_candidates.jsonl").read_bytes()
            original_sha = hashlib.sha256(original_bytes).hexdigest()
            request_sha = generator.digest({"candidates_sha256": original_sha, "feedback": feedback})
            stale = root / ("model_candidates.revision." + request_sha[:12] + ".tmp")
            stale.write_bytes(b"interrupted old write")
            client, create, _, error = self.run_revision(args, config, [response()])
            self.assertIsNone(error)
            self.assertEqual(create.await_count, 1)
            client.close.assert_awaited_once()
            self.assertNotEqual((root / "model_candidates.jsonl").read_bytes(), original_bytes)
            self.assertEqual(stale.read_bytes(), b"interrupted old write")
            self.assertFalse(list(root.glob(".education-revision-*")))
            self.assert_backup(root, original_bytes)
            self.assert_unpublished(root)

    def test_rejected_teacher_response_does_not_store_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config, _, _, _ = self.prepare(root)
            args.concurrency = 12
            rejected = response(finish="length")
            rejected.choices[0].message.reasoning_content = "private-teacher-reasoning-marker"
            _, create, _, error = self.run_revision(args, config, [rejected, response()])
            self.assertIsNone(error)
            self.assertEqual(create.await_count, 2)
            ledger_text = (root / "model_revision_responses.jsonl").read_text()
            self.assertNotIn("private-teacher-reasoning-marker", ledger_text)
            first = json.loads(ledger_text.splitlines()[0])
            self.assertFalse(first["accepted"])
            self.assertEqual(first["response"]["finish_reasons"], ["length"])
            self.assertEqual(first["metadata"]["revision_concurrency"], 12)


if __name__ == "__main__":
    unittest.main()
