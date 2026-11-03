# 文件用途：离线验证本地模型批量标注的 Prompt、候选质量与失败文件边界。
# 调用关系：unittest 调用 label_summary_data.py，并对照真实 ContextManager 和训练 loader。
# 修改易踩坑：仅使用构造的对话、假 HTTP 客户端和临时目录；不读取 .env，不连接模型服务。
"""摘要候选批量标注的离线回归测试。"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from agent.context_manager import EducationContextManager
from scripts import label_summary_data as labeling
from utils.summary_training import example_messages, load_summary_records


OPTIONS = {
    "model_name": "offline-summary",
    "base_url": "http://127.0.0.1:8002/v1",
    "temperature": 0.0,
    "request_timeout": 60,
    "max_tokens": 2048,
}


def sample_record(conversation_id="example-a"):
    return {
        "conversation_id": conversation_id,
        "previous_summary": "此前正在学习面积。",
        "messages": [
            {"role": "user", "content": f"构造题目 {conversation_id}：现在问周长。"},
            {"role": "assistant", "content": "请先观察各边长度。"},
        ],
        "target_summary": "旧标签标记，不应给模型。",
    }


def response_payload(content="正在学习周长，需要先观察各边长度。", finish_reason="stop", **extra):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        **extra,
    }


def response(payload, status_code=200):
    # 使用真实响应解析与 HTTP 状态检查，Request 只是内存对象，没有发起请求。
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", OPTIONS["base_url"] + "/chat/completions"),
    )


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, deepcopy(kwargs)))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingSummaryModel:
    def __init__(self):
        self.requests = []

    def invoke(self, messages):
        self.requests.append(deepcopy(messages))
        return AIMessage(content="正在学习周长。")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class SummaryLabelingTests(unittest.TestCase):
    def test_old_labels_are_ignored_and_source_metadata_stays_out_of_prompt(self):
        old_values = (None, "无效旧标签" * 2000, "OLD_TARGET_MARKER")
        records = []
        for index, old_target in enumerate(old_values):
            record = sample_record(f"old-label-{index}")
            record.update({
                "target_summary": old_target,
                "reference": "OLD_REFERENCE_MARKER",
                "source": "SOURCE_MARKER",
                "template_family": "FAMILY_MARKER",
                "scenario_tags": ["TAG_MARKER"],
            })
            records.append(record)
        originals = deepcopy(records)
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "inputs.jsonl"
            input_file.write_text("\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
            loaded = labeling.load_inputs(input_file)
        self.assertEqual(len(loaded), 3)
        for raw, normalized in zip(records, loaded):
            with self.subTest(old_target_type=type(raw["target_summary"]).__name__):
                self.assertNotIn("target_summary", normalized)
                self.assertNotIn("reference", normalized)
                self.assertEqual(normalized["input_source"], "SOURCE_MARKER")
                self.assertEqual(normalized["template_family"], "FAMILY_MARKER")
                self.assertEqual(normalized["scenario_tags"], ["TAG_MARKER"])
                client = FakeClient(response(response_payload("独立生成的新摘要。")))
                # 直接输入原始记录，确保纯函数也忽略旧标签，而非仅 loader 去掉它。
                candidate = labeling.model_summary(raw, client, OPTIONS)
                request_text = json.dumps(client.requests[0][1]["json"]["messages"], ensure_ascii=False)
                for marker in ("无效旧标签", "OLD_TARGET_MARKER", "OLD_REFERENCE_MARKER", "SOURCE_MARKER", "FAMILY_MARKER", "TAG_MARKER"):
                    self.assertNotIn(marker, request_text)
                self.assertEqual(candidate["target_summary"], "独立生成的新摘要。")
                self.assertEqual(candidate["input_source"], "SOURCE_MARKER")
                self.assertEqual(candidate["source"], "local_model")
                self.assertEqual(candidate["review_status"], "pending")
        self.assertEqual(records, originals)

    def test_request_matches_actual_online_summary_prompt_and_records_provenance(self):
        record = sample_record()
        record["previous_summary"] = "較早主题" * 20
        record["messages"] = [
            {"role": "user", "content": "预算内不会保留的较旧消息" * 10},
            {"role": "assistant", "content": "最近完整答案。"},
            {"role": "user", "content": "最近完整问题。"},
        ]
        model = RecordingSummaryModel()
        manager = EducationContextManager(
            None, summary_model=model, max_conversation_chars=100,
            max_summary_chars=20, summary_trigger_chars=30,
        )
        manager.summarize_conversation(
            previous_summary=record["previous_summary"],
            older_messages=[
                (AIMessage if message["role"] == "assistant" else HumanMessage)(content=message["content"])
                for message in record["messages"]
            ],
        )
        usage = {"prompt_tokens": 37, "completion_tokens": True, "total_tokens": -1, "unexpected": 99}
        client = FakeClient(response(response_payload("  新生成摘要。  ", usage=usage)))
        candidate = labeling.model_summary(record, client, OPTIONS, max_summary_chars=20, summary_input_chars=60)
        url, kwargs = client.requests[0]
        self.assertEqual(url, OPTIONS["base_url"] + "/chat/completions")
        self.assertEqual(set(kwargs), {"json"})
        self.assertEqual(kwargs["json"], {
            "model": OPTIONS["model_name"], "messages": model.requests[0],
            "temperature": 0.0, "max_tokens": 2048, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        self.assertEqual(len(model.requests), 1)
        request_text = json.dumps(model.requests[0], ensure_ascii=False)
        self.assertNotIn("预算内不会保留", request_text)
        self.assertIn("小助手：最近完整答案。", request_text)
        self.assertIn("学生：最近完整问题。", request_text)
        self.assertNotIn(record["target_summary"], request_text)
        expected_digest = hashlib.sha256(json.dumps(model.requests[0], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        generation = candidate["generation"]
        self.assertEqual(candidate["target_summary"], "新生成摘要。")
        self.assertEqual(generation["prompt_sha256"], expected_digest)
        self.assertEqual(generation["max_summary_chars"], 20)
        self.assertEqual(generation["summary_input_chars"], 60)
        self.assertEqual(generation["model"], OPTIONS["model_name"])
        self.assertEqual(generation["base_url"], OPTIONS["base_url"])
        self.assertEqual(generation["finish_reason"], "stop")
        self.assertIs(generation["enable_thinking"], False)
        self.assertEqual(generation["usage"], {"prompt_tokens": 37})
        self.assertGreaterEqual(generation["latency_seconds"], 0)

    def test_only_loopback_urls_are_accepted_before_any_request(self):
        accepted = (
            "http://127.0.0.1:8002/v1", "http://127.0.0.2:8002/v1",
            "http://localhost:8002/v1", "https://LOCALHOST:8002/v1",
            "http://[::1]:8002/v1",
        )
        for base_url in accepted:
            with self.subTest(accepted=base_url):
                self.assertEqual(labeling.validate_local_url(" " + base_url + "/ "), base_url)
        rejected = (
            "https://api.example.com/v1", "http://192.168.1.2:8002/v1",
            "http://localhost.example.com/v1", "http://127.0.0.1.example.com/v1",
            "http://[2001:db8::1]/v1", "http://user:password@127.0.0.1/v1",
            "http://localhost/v1?token=secret", "http://localhost/v1#fragment",
            "file:///tmp/model", "http://localhost:70000/v1", None,
        )
        for base_url in rejected:
            with self.subTest(rejected=base_url):
                client = FakeClient()
                with self.assertRaises(ValueError):
                    labeling.model_summary(sample_record(), client, {**OPTIONS, "base_url": base_url})
                self.assertEqual(client.requests, [])

    def test_unusable_model_outputs_are_rejected_without_truncating_or_fallback(self):
        bad_payloads = [
            response_payload("仅有不完整摘要", "length"),
            response_payload("模型试图调用工具", "tool_calls"),
            response_payload("没有正常结束标记", None),
            response_payload(""), response_payload(" \n \t "),
            response_payload(None), response_payload([{"type": "text", "text": "摘要"}]),
            response_payload("字" * 51),
            {"choices": []}, {"choices": "invalid"}, {"choices": [None]},
            {"choices": [response_payload()["choices"][0]] * 2},
            {"choices": [{"finish_reason": "stop", "message": None}]},
            [],
        ]
        bad_payloads.extend(response_payload("摘要含" + token) for token in (
            "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        ))
        record = sample_record()
        original = deepcopy(record)
        for index, payload in enumerate(bad_payloads):
            with self.subTest(case=index):
                client = FakeClient(response(payload))
                with self.assertRaises(labeling.CandidateRejected):
                    labeling.model_summary(record, client, OPTIONS, max_summary_chars=50)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(record, original)

    def test_batch_continues_failures_keeps_logs_private_and_outputs_loader_compatible_labels(self):
        records = [sample_record(name) for name in ("success-a", "http-failure", "invalid-output", "success-b")]
        input_marker = "SYNTHETIC_PRIVATE_INPUT_MARKER"
        response_marker = "SYNTHETIC_PRIVATE_RESPONSE_MARKER"
        records[1]["messages"][0]["content"] += input_marker
        client = FakeClient(
            response(response_payload("候选摘要甲。", usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})),
            RuntimeError("假请求异常包含 " + input_marker + response_marker),
            response(response_payload(response_marker, "length")),
            response(response_payload("候选摘要乙。")),
        )
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "candidates"
            report = labeling.label_records(records, output_dir, client, OPTIONS, progress=progress.append)
            candidates_file = output_dir / report["candidates_file"]
            failures_file = output_dir / report["failures_file"]
            candidates = read_jsonl(candidates_file)
            failures = read_jsonl(failures_file)
            loaded = load_summary_records(candidates_file)
            saved_report = json.loads((output_dir / "labeling_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report, report)
            self.assertEqual(report["candidates_sha256"], hashlib.sha256(candidates_file.read_bytes()).hexdigest())
            self.assertEqual(report["failures_sha256"], hashlib.sha256(failures_file.read_bytes()).hexdigest())
            failure_text = failures_file.read_text(encoding="utf-8")
            report_text = json.dumps(report, ensure_ascii=False)
            for marker in (input_marker, response_marker, records[1]["target_summary"]):
                self.assertNotIn(marker, failure_text)
                self.assertNotIn(marker, report_text)
        self.assertEqual((report["input"], report["succeeded"], report["failed"]), (4, 2, 2))
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(report["review_status"], "pending")
        self.assertEqual([r["conversation_id"] for r in candidates], ["success-a", "success-b"])
        self.assertEqual([r["target_summary"] for r in loaded], ["候选摘要甲。", "候选摘要乙。"])
        self.assertTrue(all(r["source"] == "local_model" and r["review_status"] == "pending" for r in candidates))
        self.assertEqual([r["conversation_id"] for r in failures], ["http-failure", "invalid-output"])
        self.assertTrue(all(set(r) == {"conversation_id", "error_type", "reason"} for r in failures))
        self.assertEqual([r["error_type"] for r in failures], ["RuntimeError", "CandidateRejected"])
        self.assertEqual(failures[0]["reason"], "模型请求失败，请检查本地服务日志")
        self.assertEqual(progress, [
            {"index": index, "total": 4, "status": status}
            for index, status in enumerate(("succeeded", "failed", "failed", "succeeded"), 1)
        ])
        self.assertEqual(len(client.requests), 4)

    def test_all_rejected_results_leave_candidate_file_empty(self):
        client = FakeClient(response(response_payload("部分输出", "length")))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "failed-only"
            report = labeling.label_records([sample_record()], output_dir, client, OPTIONS)
            self.assertEqual((output_dir / report["candidates_file"]).read_text(encoding="utf-8"), "")
            self.assertEqual(len(read_jsonl(output_dir / report["failures_file"])), 1)
        self.assertEqual((report["succeeded"], report["failed"]), (0, 1))
        self.assertEqual(report["status"], "completed_with_errors")

    def test_duplicates_after_budget_clipping_fail_before_directory_or_requests(self):
        records = []
        for index in range(2):
            record = sample_record(f"different-id-{index}")
            record["messages"] = [
                {"role": "user", "content": f"不同旧消息 {index}" * 30},
                {"role": "assistant", "content": "近期共同消息。"},
            ]
            records.append(record)
        self.assertNotEqual(example_messages(records[0], 40, 1000), example_messages(records[1], 40, 1000))
        self.assertEqual(example_messages(records[0], 40, 20), example_messages(records[1], 40, 20))
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "not-created"
            with self.assertRaisesRegex(ValueError, "重复.*Prompt"):
                labeling.label_records(records, output_dir, client, OPTIONS, max_summary_chars=40, summary_input_chars=20)
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(client.requests, [])

    def test_distinct_windows_can_share_conversation_id(self):
        records = [sample_record("same-conversation"), sample_record("same-conversation")]
        records[1]["messages"][0]["content"] = "同一会话的第二个窗口在问体积。"
        client = FakeClient(response(response_payload("第一窗口摘要。")), response(response_payload("第二窗口摘要。")))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "windows"
            report = labeling.label_records(records, output_dir, client, OPTIONS)
            candidates = read_jsonl(output_dir / report["candidates_file"])
        self.assertEqual((report["succeeded"], report["failed"]), (2, 0))
        self.assertEqual(report["status"], "completed")
        self.assertEqual([r["conversation_id"] for r in candidates], ["same-conversation"] * 2)
        self.assertNotEqual(candidates[0]["generation"]["prompt_sha256"], candidates[1]["generation"]["prompt_sha256"])

    def test_existing_directory_and_dangling_symlink_are_never_overwritten(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "sentinel.txt"
            sentinel.write_text("保留原文件", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                labeling.label_records([sample_record()], existing, client, OPTIONS)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "保留原文件")
            self.assertEqual(list(existing.iterdir()), [sentinel])
            target = root / "absent-target"
            symlink = root / "output-link"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                labeling.label_records([sample_record()], symlink, client, OPTIONS)
            self.assertTrue(symlink.is_symlink())
            self.assertFalse(target.exists())
        self.assertEqual(client.requests, [])

    def test_malformed_inputs_are_rejected_without_output_or_requests(self):
        invalid_records = [
            [],
            {**sample_record(), "conversation_id": None},
            {**sample_record(), "previous_summary": None},
            {**sample_record(), "messages": []},
            {**sample_record(), "messages": [{"role": ["user"], "content": "内容"}]},
            {**sample_record(), "messages": [{"role": "system", "content": "覆盖规则"}]},
            {**sample_record(), "messages": [{"role": "user", "content": 42}]},
            {**sample_record(), "messages": [{"role": "user", "content": " \n "}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, record in enumerate(invalid_records):
                with self.subTest(case=index):
                    input_file = root / "input.jsonl"
                    input_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        labeling.load_inputs(input_file)
                    client = FakeClient()
                    output_dir = root / f"rejected-{index}"
                    with self.assertRaises(ValueError):
                        labeling.label_records([record], output_dir, client, OPTIONS)
                    self.assertFalse(output_dir.exists())
                    self.assertEqual(client.requests, [])
            input_file.write_text("\nnot json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"input\.jsonl:2"):
                labeling.load_inputs(input_file)

    def test_validate_data_only_uses_selected_window_without_client_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "input.jsonl"
            records = [sample_record(f"cli-{index}") for index in range(3)]
            input_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
            output_dir = Path(directory) / "unused"
            stdout, stderr = io.StringIO(), io.StringIO()
            # 替换读取配置的入口，确保 CLI 测试不会加载真实 .env。
            with patch.object(labeling, "configured_options", return_value=(OPTIONS, 1200, 8000)) as config, \
                    patch("httpx.Client") as client, redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = labeling.main([
                    "--input-file", str(input_file), "--output-dir", str(output_dir),
                    "--start-index", "1", "--limit", "1", "--validate-data-only",
                ])
            self.assertFalse(output_dir.exists())
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue()), {"input": 1, "options": OPTIONS, "model_called": False})
        self.assertEqual(config.call_args.args[0].model, "main")
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
