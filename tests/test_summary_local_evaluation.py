# 文件用途：离线验证本地摘要评测的输入、参考答案隔离与输出保护。
# 调用关系：unittest 调用 evaluate_summary_local.py；模型依赖使用 mock。
# 修改易踩坑：测试必须在生成前停止，不加载模型、不访问网络或 CUDA。
"""本地摘要评测的轻量输入边界测试。"""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_summary_local as evaluator


def sample_case(identifier="case-a"):
    return {
        "id": identifier,
        "previous_summary": "此前讨论了长方形。",
        "messages": [
            {"role": "user", "content": "请解释周长。"},
            {"role": "assistant", "content": "周长是边长的总和。"},
        ],
        "reference": "仅供评分的参考答案标记：REFERENCE_ONLY_7461。",
    }


class SummaryLocalEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cases_path = self.root / "cases.jsonl"
        self.output_path = self.root / "report.json"

    def write_cases(self, records, path=None):
        path = path or self.cases_path
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def arguments(self, *extra):
        return [
            "--model-path", str(self.root / "local-base"),
            "--cases", str(self.cases_path),
            "--output", str(self.output_path),
            *extra,
        ]

    def test_loader_allows_blank_lines_and_missing_previous_summary(self):
        record = sample_case()
        record.pop("previous_summary")
        self.cases_path.write_text(
            "\n" + json.dumps(record, ensure_ascii=False) + "\n\n", encoding="utf-8",
        )
        cases = evaluator.load_cases([self.cases_path])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["previous_summary"], "")
        self.assertEqual(cases[0]["reference"], record["reference"])
        self.assertEqual(cases[0]["case_source"], str(self.cases_path.resolve()))

    def test_loader_rejects_empty_malformed_and_nonobject_jsonl(self):
        for text in ("", " \n\n", "{", "[]\n", "null\n", "123\n"):
            with self.subTest(text=text):
                self.cases_path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluator.load_cases([self.cases_path])
        with self.assertRaisesRegex(ValueError, "不能为空"):
            evaluator.load_cases([])

    def test_loader_rejects_invalid_ids_previous_summary_and_reference(self):
        invalid = (
            ("id", None), ("id", 123), ("id", ""), ("id", " \t"),
            ("previous_summary", None), ("previous_summary", []),
            ("reference", None), ("reference", False), ("reference", ""), ("reference", " \t"),
        )
        for field, value in invalid:
            record = sample_case()
            record[field] = value
            with self.subTest(field=field, value=value):
                self.write_cases([record])
                with self.assertRaises(ValueError):
                    evaluator.load_cases([self.cases_path])

    def test_loader_rejects_empty_message_sets_and_bad_message_shapes(self):
        invalid = (
            None, {}, [], [None], ["text"], [{}],
            [{"role": "system", "content": "不能作为对话角色。"}],
            [{"role": "tool", "content": "不能作为对话角色。"}],
            [{"role": [], "content": "普通文本。"}],
            [{"role": "user", "content": None}],
            [{"role": "assistant", "content": {"text": "普通文本。"}}],
            [{"role": "user", "content": " \t"}],
        )
        for messages in invalid:
            record = sample_case()
            record["messages"] = messages
            with self.subTest(messages=messages):
                self.write_cases([record])
                with self.assertRaises(ValueError):
                    evaluator.load_cases([self.cases_path])

    def test_duplicate_ids_are_rejected_within_and_across_case_files(self):
        original = sample_case()
        changed = sample_case()
        changed["messages"][0]["content"] = "另一段不同的输入。"
        self.write_cases([original, changed])
        with self.assertRaisesRegex(ValueError, "id 缺失或重复"):
            evaluator.load_cases([self.cases_path])
        self.write_cases([original])
        other_path = self.write_cases([changed], self.root / "other.jsonl")
        with self.assertRaisesRegex(ValueError, "id 缺失或重复"):
            evaluator.load_cases([self.cases_path, other_path])

    def test_multiple_files_keep_case_order_for_per_case_sampling_seed(self):
        first = self.write_cases([sample_case("first"), sample_case("second")])
        other = self.write_cases([sample_case("third")], self.root / "other.jsonl")
        self.assertEqual(
            [case["id"] for case in evaluator.load_cases([first, other])],
            ["first", "second", "third"],
        )

    def test_changing_reference_does_not_change_input_prompt(self):
        first = sample_case()
        changed = deepcopy(first)
        changed["reference"] = "另一份仅供评分的答案 SECOND_REFERENCE_ONLY。"
        self.write_cases([first])
        loaded_first = evaluator.load_cases([self.cases_path])[0]
        self.write_cases([changed])
        loaded_changed = evaluator.load_cases([self.cases_path])[0]
        self.assertEqual(
            evaluator.example_messages(loaded_first, 1200, 8000),
            evaluator.example_messages(loaded_changed, 1200, 8000),
        )

    def test_actual_cli_tokenizer_prompt_excludes_reference_and_label_metadata(self):
        record = sample_case()
        record.update(target_summary="LABEL_ONLY_5821", provenance={"answer": "METADATA_ONLY_3975"})
        self.write_cases([record])
        fake_torch = ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.cuda = Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.is_bf16_supported.return_value = True
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = Mock()
        fake_transformers.AutoModelForCausalLM = Mock()
        fake_transformers.set_seed = Mock()
        tokenizer = fake_transformers.AutoTokenizer.from_pretrained.return_value

        class StopBeforeTokenization(RuntimeError):
            pass

        tokenizer.apply_chat_template.side_effect = StopBeforeTokenization
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}), \
                patch.object(evaluator, "summary_budgets", return_value=(1200, 8000)):
            with self.assertRaises(StopBeforeTokenization):
                evaluator.main(self.arguments())
        prompt = tokenizer.apply_chat_template.call_args.args[0]
        prompt_text = json.dumps(prompt, ensure_ascii=False)
        self.assertEqual([message["role"] for message in prompt], ["system", "user"])
        self.assertIn(record["previous_summary"], prompt_text)
        for message in record["messages"]:
            self.assertIn(message["content"], prompt_text)
        for forbidden in (record["reference"], record["target_summary"], "METADATA_ONLY_3975"):
            self.assertNotIn(forbidden, prompt_text)
        self.assertIs(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"], False)
        fake_transformers.AutoModelForCausalLM.from_pretrained.return_value.generate.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_cli_rejects_existing_output_and_dangling_symlink_before_reading_cases(self):
        for kind in ("file", "directory", "dangling_symlink"):
            with self.subTest(kind=kind):
                if kind == "file":
                    self.output_path.write_text("keep this report", encoding="utf-8")
                elif kind == "directory":
                    self.output_path.mkdir()
                else:
                    self.output_path.symlink_to(self.root / "missing-report")
                with patch.object(evaluator, "load_cases") as loader, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        evaluator.main(self.arguments())
                self.assertEqual(error.exception.code, 2)
                loader.assert_not_called()
                if kind == "file":
                    self.assertEqual(self.output_path.read_text(), "keep this report")
                if kind == "directory":
                    self.output_path.rmdir()
                else:
                    self.output_path.unlink()

    def test_cli_rejects_nonpositive_token_budget_before_reading_cases(self):
        for budget in ("0", "-1"):
            with self.subTest(budget=budget), patch.object(evaluator, "load_cases") as loader, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    evaluator.main(self.arguments("--max-new-tokens", budget))
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()
            self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
