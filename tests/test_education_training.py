# 文件用途：验证教育SFT的数据边界、语言层目标、准备状态与无GPU检查入口。
# 调用关系：unittest调用education_training与train_education；不启动真实训练或模型服务。
# 修改易踩坑：用假tokenizer检查最后回答监督；示例测试不能变成正式教学训练数据。
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_education_finetune as launcher
from scripts import train_education
from utils import education_training as training


class FakeTokenizer:
    eos_token_id = 9
    # 真实Qwen tokenizer的think/tool标记不属于all_special_tokens。
    all_special_tokens = ["<|im_end|>"]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        self.target = text
        return [4, 5]


def record(identity="one", group="group-one", question="1米有多少厘米？"):
    return {"id": identity, "group_id": group, "messages": [
        {"role": "user", "content": question}, {"role": "assistant", "content": "1米有100厘米。"},
        {"role": "user", "content": "那2米呢？"}, {"role": "assistant", "content": "2米是200厘米。"},
    ]}


class EducationTrainingTests(unittest.TestCase):
    def write_records(self, root, name, rows):
        path = root / name
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_final_answer_only_and_non_thinking_prompt(self):
        tokenizer = FakeTokenizer()
        sample = record()
        feature = training.encode_record(sample, tokenizer, "小学学习助手", 10)
        self.assertEqual(feature["labels"], [-100, -100, -100, 4, 5, 9])
        self.assertEqual(feature["input_ids"], [1, 2, 3, 4, 5, 9])
        self.assertEqual(tokenizer.target, sample["messages"][-1]["content"])
        self.assertEqual(tokenizer.messages[1:], sample["messages"][:-1])
        self.assertEqual(tokenizer.messages[0]["role"], "system")
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_custom_system_is_preserved(self):
        sample = record()
        sample["messages"].insert(0, {"role": "system", "content": "只教英语"})
        messages = training.prompt_messages(sample, "默认指令")
        self.assertEqual(messages[0]["content"], "只教英语")
        self.assertEqual(len([m for m in messages if m["role"] == "system"]), 1)

    def test_padding_does_not_mask_real_eos(self):
        feature = training.encode_record(record(), FakeTokenizer(), "指令", 10)
        short = {key: value[:-1] for key, value in feature.items()}
        batch = training.pad_batch([feature, short], 9)
        self.assertEqual(batch["labels"][0][-1], 9)
        self.assertEqual(batch["labels"][1][-1], -100)
        self.assertEqual(batch["attention_mask"][1][-1], 0)

    def test_overlong_target_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "不能截断"):
            training.encode_record(record(), FakeTokenizer(), "指令", 5)

    def test_control_tokens_rejected_in_history_and_answer(self):
        for token in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_end|>"]:
            for index in [0, 1, 3]:
                sample = record()
                sample["messages"][index]["content"] += token
                with self.subTest(token=token, index=index), self.assertRaisesRegex(ValueError, "控制token"):
                    training.encode_record(sample, FakeTokenizer(), "指令", 10)

    def test_interrupt_during_spawn_cleans_up_new_child(self):
        handlers = {}
        child = Mock(pid=987654)
        child.wait.return_value = 0
        def register(sig, handler):
            handlers[sig] = handler
        def spawn(*_args, **_kwargs):
            # 模拟子进程已创建、Popen尚未将句柄返回时收到中断。
            handlers[launcher.signal.SIGTERM](launcher.signal.SIGTERM, None)
            return child
        with patch.object(train_education, "preparation_report", return_value={"issues": []}), \
             patch.object(launcher.signal, "getsignal", return_value=launcher.signal.SIG_DFL), \
             patch.object(launcher.signal, "signal", side_effect=register), \
             patch.object(launcher.subprocess, "Popen", side_effect=spawn) as popen, \
             patch.object(launcher.os, "killpg") as killpg, \
             contextlib.redirect_stdout(io.StringIO()):
            code = launcher.main([])
        self.assertEqual(code, 130)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(child.pid, launcher.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=15)

    def test_loader_rejects_invalid_dialogue_or_missing_target(self):
        for messages in [[], [{"role": "user", "content": "问题"}],
                         [{"role": "assistant", "content": "回答"}, {"role": "user", "content": "问题"}],
                         [{"role": "user", "content": "问题"}, {"role": "assistant", "content": ""}],
                         [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答", "tool_calls": [{}]}]]:
            sample = record()
            sample["messages"] = messages
            with self.subTest(messages=messages), tempfile.TemporaryDirectory() as temp:
                path = self.write_records(Path(temp), "train.jsonl", [sample])
                with self.assertRaises(ValueError):
                    training.load_records(path)

    def test_pending_review_is_recorded_without_approval_gate(self):
        sample = record()
        sample["review_status"] = "pending"
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_records(Path(temp), "train.jsonl", [sample])
            self.assertEqual(training.load_records(path)[0]["review_status"], "pending")
        training.reject_examples(["train.jsonl"], [sample])

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_records(Path(temp), "train.jsonl", [record(), record()])
            with self.assertRaisesRegex(ValueError, "重复id"):
                training.load_records(path)

    def test_split_blocks_group_and_effective_prompt_leakage(self):
        train = record()
        validation = record("two", "group-one", "不同题目")
        with self.assertRaisesRegex(ValueError, "group_id"):
            training.validate_split([train], [validation], "指令")
        validation = record("two", "group-two")
        with self.assertRaisesRegex(ValueError, "相同实际输入"):
            training.validate_split([train], [validation], "指令")

    def test_examples_rejected_by_filename_or_metadata(self):
        with self.assertRaisesRegex(ValueError, "演示示例"):
            training.reject_examples(["data.example.jsonl"], [record()])
        sample = record()
        sample["dataset_kind"] = "example"
        with self.assertRaisesRegex(ValueError, "演示示例"):
            training.reject_examples(["train.jsonl"], [sample])

    def create_base(self, root, *, quantized=False):
        config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
                  "text_config": {"num_hidden_layers": 64, "hidden_size": 5120}}
        if quantized:
            config["quantization_config"] = {"quant_method": "compressed-tensors"}
        (root / "config.json").write_text(json.dumps(config))
        for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
            (root / name).write_text("test placeholder")

    def test_quantized_base_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_base(root, quantized=True)
            with self.assertRaisesRegex(ValueError, "未量化"):
                training.validate_base_model(root)

    def test_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_base(root)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
            with self.assertRaisesRegex(FileNotFoundError, "分片缺失"):
                training.validate_base_model(root)

    def test_language_targets_cover_hybrid_layers_and_exclude_vision_mtp(self):
        modules = []
        for layer in range(64):
            attention = "self_attn.q_proj" if layer % 4 == 3 else "linear_attn.in_proj_qkv"
            modules.extend([(f"model.language_model.layers.{layer}.{attention}", object()),
                            (f"model.language_model.layers.{layer}.mlp.up_proj", object())])
        modules.extend([("model.visual.blocks.0.mlp.up_proj", object()), ("mtp.layers.0.mlp.up_proj", object())])
        model = Mock()
        model.named_modules.return_value = modules
        targets = train_education.discover_targets(model)
        self.assertEqual(len(targets), 128)
        self.assertFalse(any("visual" in name or "mtp" in name for name in targets))
        model.named_modules.return_value = modules[:2]
        with self.assertRaisesRegex(ValueError, "64"):
            train_education.discover_targets(model)

    def test_check_reports_all_missing_inputs_without_training_imports(self):
        config = training.load_config(train_education.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            for key in ["model_path", "training_python", "train_file", "validation_file", "output_dir"]:
                config[key] = str(Path(temp) / key)
            report = train_education.preparation_report(config)
        self.assertEqual(len(report["issues"]), 4)
        self.assertIs(report["gpu_used"], False)
        self.assertIs(report["model_loaded"], False)

    def test_demo_data_check_needs_no_model_and_not_trainable(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            code = train_education.main(["--validate-data-only",
                "--train-file", str(training.PROJECT_ROOT / "data/education_train.example.jsonl"),
                "--validation-file", str(training.PROJECT_ROOT / "data/education_validation.example.jsonl")])
        report = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["train_examples"], 4)
        self.assertEqual(report["validation_examples"], 2)
        self.assertFalse(report["usable_for_training"])
        self.assertFalse(report["gpu_used"])

    def test_gpu_owner_guard_does_not_stop_other_processes(self):
        with patch.object(train_education.subprocess, "run", return_value=Mock(stdout="123\n")):
            with self.assertRaisesRegex(RuntimeError, "123"):
                train_education.ensure_gpu_idle()

    def test_default_education_config_is_separate_from_summary(self):
        config = training.load_config(train_education.DEFAULT_CONFIG)
        self.assertEqual(config["method"], "qlora")
        self.assertEqual(config["model_path"], "/root/MAOMI/models/Qwen3.8-27B")
        self.assertEqual(config["training_python"], "/root/MAOMI/envs/education-training/bin/python")
        self.assertEqual(config["output_dir"], "/root/MAOMI/training/education-lora-v1")
        self.assertIn("小学", config["instruction"])
        self.assertNotIn("summary_dataset", config["train_file"])


if __name__ == "__main__":
    unittest.main()
