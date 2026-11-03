# 文件用途：验证Python摘要训练启动器的预检、暂停、训练与恢复控制流。
# 调用关系：unittest调用run_summary_finetune.py；模型、进程与服务操作全部mock。
# 修改易踩坑：测试不能运行真实训练或启停服务，异常清理必须早于模型恢复。
"""Exercise the training launcher without models, GPU commands, or live services."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import run_summary_finetune as launcher
import scripts


class SummaryFinetuneLauncherTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.default_command = launcher.training_command()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.output = self.root / "new-adapter"
        self.train = self.root / "train.jsonl"
        self.validation = self.root / "validation.jsonl"
        self.python = self.root / "training-env" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.touch()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "max_position_embeddings": 40960,
        }) + "\n")
        (self.model / "model.safetensors").touch()
        records = [
            {
                "conversation_id": f"unit-{index}",
                "previous_summary": "",
                "messages": [{"role": "user", "content": f"练习{index}"}],
                "target_summary": "测试摘要",
                "review_status": "assistant_reviewed" if index < 160 else "pending",
            }
            for index in range(1000)
        ]
        self.train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        self.validation.write_text(json.dumps(records[0], ensure_ascii=False) + "\n")
        self.original_train_bytes = self.train.read_bytes()

        # A synthetic module avoids importing project configuration or reading
        # credentials. Every process/service operation below is also patched.
        self.serve = ModuleType("scripts.serve")
        self.serve.LOG_DIR = self.logs
        self.serve._process_group_alive = mock.Mock(return_value=False)
        self.stack.enter_context(mock.patch.dict(sys.modules, {"scripts.serve": self.serve}))
        self.stack.enter_context(mock.patch.object(scripts, "serve", self.serve, create=True))
        for name, value in {
            "PROJECT_ROOT": self.root,
            "TRAINING_PYTHON": self.python,
            "BASE_MODEL": self.model,
            "TRAIN_FILE": self.train,
            "VALIDATION_FILE": self.validation,
            "OUTPUT_DIR": self.output,
        }.items():
            self.stack.enter_context(mock.patch.object(launcher, name, value))
        self.stack.enter_context(mock.patch.object(
            launcher.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)
        ))
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(redirect_stderr(self.stderr))
        # launch() shields cleanup from further interrupts. Do not modify the
        # test runner's actual signal handlers while testing that control flow.
        self.stack.enter_context(mock.patch.object(launcher.signal, "getsignal", return_value="previous-handler"))
        self.stack.enter_context(mock.patch.object(launcher.signal, "signal"))
        # Fail closed if a later implementation adds an unmocked operation.
        # The child-cleanup test overrides Popen explicitly with fake children.
        for name in ("Popen", "run"):
            self.stack.enter_context(mock.patch.object(
                launcher.subprocess, name,
                side_effect=AssertionError("unit test must not launch a real subprocess"),
            ))
        for name in ("kill", "killpg", "execv"):
            self.stack.enter_context(mock.patch.object(
                launcher.os, name,
                side_effect=AssertionError("unit test must not signal or replace a real process"),
            ))
        self.events = []
        self.snapshots = [
            launcher.ServiceSnapshot(
                service=SimpleNamespace(key=key, title=f"unit {key}"),
                pid=900001 + index,
                identity=12345 + index,
                argv=("/fake/service", "--token", "unit-argv-secret"),
                cwd=self.root,
                environment={"UNIT_API_TOKEN": "unit-environment-secret"},
            )
            for index, key in enumerate(("chat", "web"))
        ]
        self.real_run_child = launcher.run_child
        self.check_paths = self.stack.enter_context(mock.patch.object(
            launcher, "check_paths", wraps=launcher.check_paths
        ))
        self.run_child = self.stack.enter_context(mock.patch.object(
            launcher, "run_child", side_effect=self.child_succeeds
        ))
        self.capture_services = self.stack.enter_context(mock.patch.object(
            launcher, "capture_services", side_effect=self.capture
        ))
        self.verify_gpu = self.stack.enter_context(mock.patch.object(
            launcher, "verify_gpu_ownership", side_effect=self.verify
        ))
        self.pause_services = self.stack.enter_context(mock.patch.object(
            launcher, "pause_services", side_effect=self.pause
        ))
        self.restore_services = self.stack.enter_context(mock.patch.object(
            launcher, "restore_services", side_effect=self.restore
        ))

    def child_succeeds(self, command):
        if "--dry-run" in command:
            self.events.append("preflight")
        elif "-c" in command:
            self.events.append("dependencies")
        else:
            self.events.append("training")
        return 0

    def capture(self, manager):
        self.assertIs(manager, self.serve)
        self.events.append("capture")
        return self.snapshots

    def verify(self, snapshots):
        self.events.append("gpu_before" if snapshots else "gpu_after")

    def pause(self, manager, snapshots):
        self.assertIs(manager, self.serve)
        self.assertIs(snapshots, self.snapshots)
        self.events.append("pause")

    def restore(self, manager, snapshots):
        self.assertIs(manager, self.serve)
        self.assertIs(snapshots, self.snapshots)
        self.events.append("restore")
        return []

    def report(self):
        paths = list(self.logs.glob("summary-finetune-launch-*.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def assert_services_untouched(self):
        self.pause_services.assert_not_called()
        self.restore_services.assert_not_called()

    def test_training_command_uses_absolute_paths_and_configured_epochs(self):
        for command in (self.default_command, launcher.training_command()):
            for index in (0, 1):
                self.assertTrue(Path(command[index]).is_absolute())
            for option in ("--model-path", "--train-file", "--validation-file", "--output-dir"):
                self.assertTrue(Path(command[command.index(option) + 1]).is_absolute(), option)
            self.assertEqual(command[command.index("--epochs") + 1], str(launcher.EPOCHS))
        command = launcher.training_command()
        self.assertTrue(Path(command[0]).is_absolute())
        self.assertTrue(Path(command[1]).is_absolute())
        self.assertEqual(Path(command[1]), self.root / "scripts" / "train_summary.py")
        for option, expected in {
            "--model-path": self.model,
            "--train-file": self.train,
            "--validation-file": self.validation,
            "--output-dir": self.output,
        }.items():
            value = Path(command[command.index(option) + 1])
            self.assertTrue(value.is_absolute(), option)
            self.assertEqual(value, expected)
        self.assertEqual(command[command.index("--epochs") + 1], str(launcher.EPOCHS))

    def test_dry_run_only_runs_data_preflight(self):
        self.assertEqual(launcher.launch(dry_run=True), 0)
        self.run_child.assert_called_once_with(launcher.training_command() + ["--dry-run"])
        self.capture_services.assert_not_called()
        self.verify_gpu.assert_not_called()
        self.assert_services_untouched()

    def test_nonzero_data_preflight_does_not_pause_services(self):
        self.run_child.side_effect = None
        self.run_child.return_value = 7
        with self.assertRaises(RuntimeError):
            launcher.launch()
        self.assertEqual(self.run_child.call_count, 1)
        self.capture_services.assert_not_called()
        self.verify_gpu.assert_not_called()
        self.assert_services_untouched()

    def test_nonzero_dependency_preflight_does_not_pause_services(self):
        self.run_child.side_effect = [0, 9]
        with self.assertRaises(RuntimeError):
            launcher.launch()
        self.assertEqual(self.run_child.call_count, 2)
        self.capture_services.assert_not_called()
        self.assert_services_untouched()

    def test_existing_output_fails_before_child_or_service_operations(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            launcher.launch()
        self.run_child.assert_not_called()
        self.capture_services.assert_not_called()
        self.assert_services_untouched()

    def test_output_created_during_preflight_is_rechecked_before_pausing(self):
        def create_output_after_preflight(command):
            if "--dry-run" in command:
                self.output.mkdir()
            return 0
        self.run_child.side_effect = create_output_after_preflight
        with self.assertRaises(FileExistsError):
            launcher.launch()
        self.capture_services.assert_not_called()
        self.assert_services_untouched()

    def test_successful_training_restores_services_after_training(self):
        self.assertEqual(launcher.launch(), 0)
        self.pause_services.assert_called_once_with(self.serve, self.snapshots)
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        self.assertLess(self.events.index("preflight"), self.events.index("pause"))
        self.assertLess(self.events.index("gpu_before"), self.events.index("pause"))
        self.assertLess(self.events.index("pause"), self.events.index("training"))
        self.assertLess(self.events.index("training"), self.events.index("restore"))
        self.assertEqual(self.report()["status"], "completed")
        self.assertEqual(self.train.read_bytes(), self.original_train_bytes)
        self.assertEqual(self.report()["review_status_counts"], {"assistant_reviewed": 160, "pending": 840})

    def test_nonzero_training_exit_still_restores_services(self):
        self.run_child.side_effect = [0, 0, 23]
        self.assertEqual(launcher.launch(), 23)
        self.pause_services.assert_called_once()
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        self.assertEqual(self.report()["status"], "training_failed")
        self.assertNotIn("训练完成：", self.stdout.getvalue())

    def test_keyboard_interrupt_during_training_still_restores_services(self):
        self.run_child.side_effect = [0, 0, KeyboardInterrupt()]
        self.assertEqual(launcher.launch(), 130)
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        self.assertEqual(self.report()["status"], "interrupted")

    def test_failure_after_partial_pause_still_enters_restoration(self):
        def partial_pause(_manager, _snapshots):
            if self.pause_services.call_count == 1:
                self.events.append("partially_paused")
                self.serve._process_group_alive.return_value = True
                raise RuntimeError("unit partial pause failure")
            self.events.append("pause_cleanup")
            self.serve._process_group_alive.return_value = False
        self.pause_services.side_effect = partial_pause
        with self.assertRaisesRegex(RuntimeError, "partial pause"):
            launcher.launch()
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        self.assertEqual(self.pause_services.call_count, 2)
        self.assertLess(self.events.index("partially_paused"), self.events.index("restore"))
        self.assertLess(self.events.index("pause_cleanup"), self.events.index("restore"))
        self.assertEqual(self.report()["status"], "failed")
        self.assertEqual(self.run_child.call_count, 2)

    def test_interrupt_during_partial_pause_finishes_shutdown_before_restore(self):
        def interrupted_pause(_manager, _snapshots):
            if self.pause_services.call_count == 1:
                self.serve._process_group_alive.return_value = True
                self.events.append("pause_interrupted")
                raise KeyboardInterrupt()
            self.serve._process_group_alive.return_value = False
            self.events.append("pause_cleanup")
        self.pause_services.side_effect = interrupted_pause
        self.assertEqual(launcher.launch(), 130)
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        self.assertEqual(self.pause_services.call_count, 2)
        self.assertLess(self.events.index("pause_cleanup"), self.events.index("restore"))
        self.assertEqual(self.report()["status"], "interrupted")
        self.assertEqual(self.run_child.call_count, 2)

    def test_unknown_gpu_owner_fails_before_pause(self):
        self.verify_gpu.side_effect = RuntimeError("unit unknown GPU owner")
        with self.assertRaisesRegex(RuntimeError, "unknown GPU"):
            launcher.launch()
        self.verify_gpu.assert_called_once_with(self.snapshots)
        self.assert_services_untouched()
        self.assertEqual(self.run_child.call_count, 2)

    def test_gpu_check_failure_after_pause_still_restores_services(self):
        self.verify_gpu.side_effect = [None, RuntimeError("unit GPU still occupied"), None]
        with self.assertRaisesRegex(RuntimeError, "still occupied"):
            launcher.launch()
        self.pause_services.assert_called_once()
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        self.assertEqual(self.run_child.call_count, 2)

    def test_gpu_still_occupied_during_cleanup_only_restores_web(self):
        self.verify_gpu.side_effect = [None, None, RuntimeError("unit GPU cleanup failed")]
        def restore_web_only(manager, snapshots):
            self.assertIs(manager, self.serve)
            self.assertEqual([item.service.key for item in snapshots], ["web"])
            self.events.append("restore_web")
            return []
        self.restore_services.side_effect = restore_web_only
        self.assertNotEqual(launcher.launch(), 0)
        self.restore_services.assert_called_once_with(self.serve, [self.snapshots[1]])
        self.assertTrue(self.report()["restoration_errors"])
        self.assertNotIn("训练完成：", self.stdout.getvalue())

    def test_child_cleanup_failure_never_restores_models_even_with_free_gpu(self):
        self.run_child.side_effect = [
            0, 0, launcher.ChildCleanupError("unit training group did not exit"),
        ]
        # The GPU verifier succeeds: cleanup failure must independently block
        # restoring GPU services, including when nvidia-smi would look empty.
        def restore_web_only(manager, snapshots):
            self.assertIs(manager, self.serve)
            self.assertEqual([item.service.key for item in snapshots], ["web"])
            return []
        self.restore_services.side_effect = restore_web_only
        with self.assertRaises(launcher.ChildCleanupError):
            launcher.launch()
        self.restore_services.assert_called_once_with(self.serve, [self.snapshots[1]])
        self.assertEqual(self.report()["status"], "child_cleanup_failed")
        self.assertTrue(self.report()["restoration_errors"])
        self.assertNotIn("训练完成：", self.stdout.getvalue())

    def test_restoration_errors_make_successful_training_return_failure(self):
        self.restore_services.side_effect = None
        self.restore_services.return_value = ["chat: unit restoration failure"]
        self.assertNotEqual(launcher.launch(), 0)
        self.assertEqual(self.report()["restoration_errors"], ["chat: unit restoration failure"])
        self.assertNotIn("训练完成：", self.stdout.getvalue())

    def test_report_and_console_never_include_original_argv_or_environment(self):
        self.assertEqual(launcher.launch(), 0)
        report = self.report()
        serialized = json.dumps(report) + self.stdout.getvalue() + self.stderr.getvalue()
        for sensitive in ("unit-argv-secret", "unit-environment-secret", "UNIT_API_TOKEN"):
            self.assertNotIn(sensitive, serialized)
        self.assertEqual(report["original_services"], ["chat", "web"])
        self.assertNotIn("argv", report)
        self.assertNotIn("environment", report)

    def test_training_child_cleanup_happens_before_service_restoration(self):
        children = []
        for name in ("preflight", "dependencies", "training"):
            child = mock.Mock()
            child.name = name
            child.wait.side_effect = KeyboardInterrupt() if name == "training" else [0]
            children.append(child)
        with mock.patch.object(launcher.subprocess, "Popen", side_effect=children) as popen:
            with mock.patch.object(launcher, "stop_child", side_effect=lambda proc: self.events.append(f"stop_{proc.name}")) as stop:
                self.run_child.side_effect = self.real_run_child
                self.assertEqual(launcher.launch(), 130)
        self.assertEqual(popen.call_count, 3)
        self.assertEqual(stop.call_args_list, [mock.call(child) for child in children])
        self.assertLess(self.events.index("stop_training"), self.events.index("restore"))
        self.restore_services.assert_called_once_with(self.serve, self.snapshots)
        for call in popen.call_args_list:
            self.assertTrue(call.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
