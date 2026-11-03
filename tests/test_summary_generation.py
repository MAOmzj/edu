# 文件用途：验证离线摘要候选数据生成、按模板族分割、泄漏拦截与输出保护。
# 调用关系：unittest 调用本文件；调用 generate_summary_data.py 和真实训练数据 loader。
# 修改易踩坑：所有输出放临时目录，不下载或调用模型；pending 合成目标不能冒充已审核金标。
"""虚构摘要数据生成流程测试，不使用学生数据库或模型服务。"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from scripts.generate_summary_data import generate_records, split_records, write_dataset
from utils.summary_training import example_messages, load_summary_records, validate_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_record(conversation_id, family, content):
    return {
        "conversation_id": conversation_id,
        "previous_summary": "此前正在学习面积。",
        "messages": [{"role": "user", "content": content}],
        "target_summary": "学生继续学习面积，当前问题尚未解决。",
        "source": "synthetic_template",
        "review_status": "pending",
        "template_family": family,
        "scenario_tags": ["unresolved_question"],
    }


class SummaryGenerationTests(unittest.TestCase):
    def test_generation_is_seeded_and_preserves_candidate_review_metadata(self):
        records = generate_records(count=200, seed=42)
        self.assertEqual(records, generate_records(count=200, seed=42))
        self.assertNotEqual(records, generate_records(count=200, seed=43))
        self.assertEqual(len(records), 200)
        self.assertEqual(len({record["conversation_id"] for record in records}), 200)
        for record in records:
            with self.subTest(conversation_id=record["conversation_id"]):
                self.assertEqual(record["source"], "synthetic_template")
                self.assertEqual(record["review_status"], "pending")
                self.assertIsInstance(record["template_family"], str)
                self.assertTrue(record["template_family"])
                self.assertIsInstance(record["scenario_tags"], list)
                self.assertTrue(record["scenario_tags"])
                self.assertTrue(all(isinstance(tag, str) and tag for tag in record["scenario_tags"]))
                self.assertTrue(0 < len(record["target_summary"]) <= 1200)

    def test_revised_chinese_writing_length_is_distinct_and_replaces_old_requirement(self):
        records = [record for record in generate_records(200, 42)
                   if record["template_family"] == "chinese_writing_requirement"]
        self.assertEqual(len(records), 10)
        self.assertEqual(sum(record["conversation_id"].endswith("-00004") for record in records), 1)
        for record in records:
            with self.subTest(conversation_id=record["conversation_id"]):
                old_match = re.search(r"需要写(\d+)字", record["previous_summary"])
                new_match = re.search(r"约(\d+)字", record["messages"][0]["content"])
                self.assertIsNotNone(old_match)
                self.assertIsNotNone(new_match)
                old_limit = int(old_match.group(1))
                new_limit = int(new_match.group(1))
                self.assertNotEqual(old_limit, new_limit)
                self.assertIn(f"不是{old_limit}字", record["messages"][0]["content"])
                self.assertIn(f"约{new_limit}字", record["target_summary"])
                self.assertRegex(record["target_summary"], rf"旧[^；。]*{old_limit}字[^；。]*被替代")
                if record["conversation_id"].endswith("-00004"):
                    # 此变体曾把新旧要求都写成 300 字，形成直接矛盾。
                    self.assertEqual(new_limit, 300)
                    self.assertEqual(old_limit, 400)

    def test_split_is_deterministic_preserves_records_and_isolates_template_families(self):
        records = generate_records(count=200, seed=42)
        original = deepcopy(records)
        splits = split_records(records, seed=17)
        self.assertEqual(splits, split_records(records, seed=17))
        self.assertEqual(records, original)
        self.assertEqual(set(splits), {"train", "validation", "test"})
        ids = []
        families = {}
        for name, split in splits.items():
            self.assertTrue(split)
            ids.extend(record["conversation_id"] for record in split)
            families[name] = {record["template_family"] for record in split}
        self.assertCountEqual(ids, [record["conversation_id"] for record in records])
        self.assertEqual(len(ids), len(set(ids)))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            with self.subTest(left=left, right=right):
                self.assertFalse(families[left] & families[right])
                validate_split(splits[left], splits[right])

    def test_split_rejects_different_raw_inputs_with_the_same_budgeted_prompt(self):
        training = synthetic_record("record-a", "family-a", "较旧训练内容" * 2000)
        validation = synthetic_record("record-b", "family-b", "较旧验证内容" * 2000)
        recent = {"role": "user", "content": "相同的近期面积问题。"}
        training["messages"].append(deepcopy(recent))
        validation["messages"].append(deepcopy(recent))
        third = synthetic_record("record-c", "family-c", "另一条不同输入。")
        self.assertNotEqual(training["messages"], validation["messages"])
        self.assertEqual(example_messages(training, 1200, 8000), example_messages(validation, 1200, 8000))
        with self.assertRaisesRegex(ValueError, "重复输入"):
            split_records([training, validation, third], seed=42)

    def test_split_uses_nondefault_input_and_summary_budgets_for_leakage_checks(self):
        shared_recent_content = "三组共同的近期会话内容。" * 800
        self.assertGreater(len(shared_recent_content), 8000)
        records = []
        for index in range(3):
            record = synthetic_record(f"record-{index}", f"family-{index}", f"第{index}组不同的较早问题。")
            record["messages"].append({"role": "assistant", "content": shared_recent_content})
            records.append(record)
        default_prompts = [example_messages(record, 1200, 8000) for record in records]
        self.assertTrue(all(prompt == default_prompts[0] for prompt in default_prompts))
        with self.assertRaisesRegex(ValueError, "重复输入"):
            split_records(records, seed=42)
        larger_prompts = [example_messages(record, 1400, 20_000) for record in records]
        self.assertNotEqual(larger_prompts[0], larger_prompts[1])
        splits = split_records(records, seed=42, max_summary_chars=1400, summary_input_chars=20_000)
        self.assertEqual({name: len(items) for name, items in splits.items()}, {"train": 1, "validation": 1, "test": 1})
        # 同一个调用也必须传递目标摘要预算，而不只传递输入预算。
        with self.assertRaisesRegex(ValueError, "目标摘要"):
            split_records(records, seed=42, max_summary_chars=5, summary_input_chars=20_000)

    def test_invalid_split_ratios_are_rejected(self):
        records = generate_records(count=40, seed=42)
        for validation_ratio, test_ratio in ((0, 0.1), (-0.1, 0.1), (float("nan"), 0.1), (0.5, 0.5)):
            with self.subTest(validation_ratio=validation_ratio, test_ratio=test_ratio), self.assertRaises(ValueError):
                split_records(records, validation_ratio=validation_ratio, test_ratio=test_ratio)

    def test_writer_outputs_loadable_jsonl_reports_hashes_and_reuses_test_cases_for_evaluation(self):
        splits = split_records(generate_records(count=200, seed=42), seed=42)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "candidates"
            report = write_dataset(output_dir, splits, seed=42)
            self.assertEqual(report["source"], "synthetic_template")
            self.assertEqual(report["review_status"], "pending")
            self.assertEqual(report["seed"], 42)
            self.assertEqual(json.loads((output_dir / "dataset_report.json").read_text(encoding="utf-8")), report)
            for name, records in splits.items():
                info = report["splits"][name]
                path = output_dir / info["file"]
                loaded = load_summary_records(path)
                with self.subTest(split=name):
                    self.assertEqual(path.name, f"{name}.candidates.jsonl")
                    self.assertEqual(info["examples"], len(records))
                    self.assertEqual(len(loaded), len(records))
                    self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertEqual(set(info["template_families"]), {record["template_family"] for record in records})
                    self.assertEqual(sum(info["template_families"].values()), len(records))
                    self.assertEqual(
                        loaded,
                        [{key: record[key] for key in ("conversation_id", "previous_summary", "messages", "target_summary")}
                         for record in records],
                    )
            eval_path = output_dir / report["test_eval_file"]
            evaluation = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(evaluation), len(splits["test"]))
            self.assertEqual(report["test_eval_sha256"], hashlib.sha256(eval_path.read_bytes()).hexdigest())
            source_by_id = {record["conversation_id"]: record for record in splits["test"]}
            self.assertEqual({record["id"] for record in evaluation}, set(source_by_id))
            for record in evaluation:
                source = source_by_id[record["id"]]
                self.assertEqual(record["reference"], source["target_summary"])
                self.assertEqual(record["messages"], source["messages"])
                self.assertEqual(record["previous_summary"], source["previous_summary"])
                self.assertEqual(record["source"], "synthetic_template")
                self.assertEqual(record["review_status"], "pending")

    def test_writer_never_overwrites_existing_directory_or_dangling_symlink(self):
        splits = split_records(generate_records(count=40, seed=42), seed=42)
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("existing data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_dataset(existing, splits, seed=42)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing data")
            self.assertEqual({path.name for path in existing.iterdir()}, {"keep.txt"})
            missing = Path(directory) / "missing-target"
            link = Path(directory) / "linked-output"
            link.symlink_to(missing, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                write_dataset(link, splits, seed=42)
            self.assertTrue(link.is_symlink())
            self.assertFalse(missing.exists())

    def test_writer_rejects_cross_split_duplicate_prompts_before_creating_output(self):
        first = synthetic_record("record-a", "family-a", "相同输入。")
        duplicate = synthetic_record("record-b", "family-b", "相同输入。")
        third = synthetic_record("record-c", "family-c", "另一个不同输入。")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected"
            with self.assertRaisesRegex(ValueError, "重复输入"):
                write_dataset(output, {"train": [first], "validation": [duplicate], "test": [third]}, seed=42)
            self.assertFalse(output.exists())

    def test_writer_rejects_nontraining_message_roles_before_creating_output(self):
        first = synthetic_record("record-a", "family-a", "不同输入一。")
        first["messages"][0]["role"] = "system"
        second = synthetic_record("record-b", "family-b", "不同输入二。")
        third = synthetic_record("record-c", "family-c", "不同输入三。")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected"
            with self.assertRaises(ValueError):
                write_dataset(output, {"train": [first], "validation": [second], "test": [third]}, seed=42)
            self.assertFalse(output.exists())

    def test_cli_generates_default_dataset_without_optional_model_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cli-candidates"
            # -S 禁用 site-packages；生成入口只依赖标准库和项目的纯数据模块。
            result = subprocess.run(
                [sys.executable, "-S", str(PROJECT_ROOT / "scripts/generate_summary_data.py"), "--output-dir", str(output)],
                cwd=PROJECT_ROOT,
                env={"PATH": os.defpath, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"},
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(result.stdout)
            self.assertEqual(status["review_status"], "pending")
            self.assertEqual(sum(status["counts"].values()), 200)
            self.assertEqual(Path(status["output_dir"]), output)
            report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["seed"], 42)
            for name in ("train", "validation", "test"):
                self.assertEqual(len(load_summary_records(output / report["splits"][name]["file"])), status["counts"][name])

    def test_cli_reports_expanded_home_path_without_changing_parent_home(self):
        original_home = os.environ.get("HOME")
        with tempfile.TemporaryDirectory() as directory:
            child_home = Path(directory) / "child-home"
            child_home.mkdir()
            working_dir = Path(directory) / "working"
            working_dir.mkdir()
            expected_output = child_home / "summary-candidates"
            result = subprocess.run(
                [sys.executable, "-S", str(PROJECT_ROOT / "scripts/generate_summary_data.py"),
                 "--output-dir", "~/summary-candidates", "--count", "40", "--seed", "23"],
                cwd=working_dir,
                env={"HOME": str(child_home), "PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(result.stdout)
            self.assertEqual(Path(status["output_dir"]), expected_output.resolve())
            self.assertTrue((expected_output / "dataset_report.json").is_file())
            self.assertFalse((working_dir / "~").exists())
        self.assertEqual(os.environ.get("HOME"), original_home)


if __name__ == "__main__":
    unittest.main()
