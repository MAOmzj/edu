# 文件用途：验证摘要训练启动器的服务恢复、进程组清理和信号隔离。
# 调用关系：unittest 调用run_summary_finetune；全部进程、信号与GPU操作使用mock。
# 修改易踩坑：只使用临时日志目录，不启动模型、不读取真实进程或发送真实信号。
"""摘要启动器的低层服务生命周期测试。"""

import contextlib
import io
import itertools
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import scripts
import run_summary_finetune as launcher


class SummaryLauncherServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager = SimpleNamespace(
            LOG_DIR=self.root,
            _port_in_use=Mock(return_value=False),
            _process_group_alive=Mock(return_value=False),
            _wait_ready=Mock(return_value=(True, "")),
            _stop_started=Mock(return_value=True),
        )
        self.output = contextlib.ExitStack()
        self.addCleanup(self.output.close)
        self.output.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.output.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def snapshot(self, key, pid):
        service = SimpleNamespace(
            key=key, title=f"original {key}",
            port={"chat": 8000, "embedding": 8001, "summary": 8002, "web": 8766}[key],
            ready_timeout=15,
            pid_path=self.root / f"serve-{key}.pid",
            log_path=self.root / f"serve-{key}.log",
        )
        return launcher.ServiceSnapshot(
            service=service, pid=pid, identity=pid + 100,
            argv=("/original/python", "-m", f"original.{key}", "--original-option", "retained"),
            cwd=self.root / f"original-{key}-cwd",
            environment={"ORIGINAL_SETTING": key, "TEST_ONLY_SECRET": "sentinel-never-logged"},
        )

    @staticmethod
    def process(pid):
        return Mock(pid=pid)

    def test_restore_preserves_original_launch_parameters_and_restores_web_last(self):
        web, chat, embedding = [self.snapshot(key, pid) for key, pid in (
            ("web", 10), ("chat", 20), ("embedding", 30),
        )]
        snapshots = [web, chat, embedding]
        processes = [self.process(pid) for pid in (120, 130, 110)]
        with patch.object(launcher, "process_identity", return_value=None), \
                patch.object(launcher.subprocess, "Popen", side_effect=processes) as popen, \
                patch.object(launcher.os, "killpg") as killpg:
            errors = launcher.restore_services(self.manager, snapshots)
        self.assertEqual(errors, [])
        ordered = [chat, embedding, web]
        self.assertEqual(popen.call_count, 3)
        for invocation, item, proc in zip(popen.call_args_list, ordered, processes):
            self.assertEqual(invocation.args, (item.argv,))
            self.assertEqual(invocation.kwargs["cwd"], str(item.cwd))
            self.assertEqual(invocation.kwargs["env"], item.environment)
            self.assertIs(invocation.kwargs["start_new_session"], True)
            self.assertEqual(invocation.kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(invocation.kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(item.service.pid_path.read_text(), f"{proc.pid}\n")
            self.assertTrue(item.service.log_path.is_symlink())
        self.assertEqual(
            [invocation.args[0].key for invocation in self.manager._wait_ready.call_args_list],
            ["chat", "embedding", "web"],
        )
        # summary未出现在快照中，不能因为配置存在而自动启动。
        self.assertFalse((self.root / "serve-summary.pid").exists())
        self.manager._stop_started.assert_not_called()
        killpg.assert_not_called()

    def test_empty_snapshots_do_not_start_any_previously_inactive_service(self):
        with patch.object(launcher.subprocess, "Popen") as popen, \
                patch.object(launcher, "process_identity") as identity:
            self.assertEqual(launcher.restore_services(self.manager, []), [])
        popen.assert_not_called()
        identity.assert_not_called()

    def test_live_original_process_is_not_duplicated_or_replaced(self):
        item = self.snapshot("chat", 20)
        with patch.object(launcher, "process_identity", return_value=item.identity), \
                patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.restore_services(self.manager, [item]), [])
        popen.assert_not_called()
        self.manager._port_in_use.assert_not_called()
        self.manager._stop_started.assert_not_called()

    def test_port_conflict_is_reported_without_forcing_or_signaling_another_instance(self):
        item = self.snapshot("chat", 20)
        self.manager._port_in_use.return_value = True
        with patch.object(launcher, "process_identity", return_value=None), \
                patch.object(launcher.subprocess, "Popen") as popen, \
                patch.object(launcher.os, "killpg") as killpg:
            errors = launcher.restore_services(self.manager, [item])
        self.assertEqual(errors, ["chat: RuntimeError"])
        popen.assert_not_called()
        killpg.assert_not_called()
        self.manager._stop_started.assert_not_called()
        self.assertFalse(item.service.pid_path.exists())

    def test_original_group_must_be_gone_before_a_replacement_can_start(self):
        item = self.snapshot("chat", 20)
        self.manager._process_group_alive.return_value = True
        with patch.object(launcher, "process_identity", return_value=None), \
                patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.restore_services(self.manager, [item]), ["chat: RuntimeError"])
        popen.assert_not_called()

    def test_pid_or_log_registration_failure_cleans_only_the_process_just_started(self):
        for failure_at in ("pid", "log"):
            with self.subTest(failure_at=failure_at):
                item = self.snapshot("chat", 20)
                proc = self.process(120)
                if failure_at == "pid":
                    item.service.pid_path = Mock()
                    item.service.pid_path.write_text.side_effect = OSError("registration failed")
                else:
                    item.service.log_path = Mock()
                    item.service.log_path.symlink_to.side_effect = OSError("link failed")
                self.manager._stop_started.reset_mock()
                self.manager._wait_ready.reset_mock()
                with patch.object(launcher, "process_identity", return_value=None), \
                        patch.object(launcher.subprocess, "Popen", return_value=proc):
                    errors = launcher.restore_services(self.manager, [item])
                self.assertEqual(errors, ["chat: OSError"])
                self.manager._stop_started.assert_called_once_with(item.service, proc)
                self.manager._wait_ready.assert_not_called()

    def test_readiness_failure_is_cleaned_and_other_services_are_still_restored(self):
        chat, web = self.snapshot("chat", 20), self.snapshot("web", 10)
        chat_proc, web_proc = self.process(120), self.process(110)
        self.manager._wait_ready.side_effect = [(False, "health timeout"), (True, "")]
        with patch.object(launcher, "process_identity", return_value=None), \
                patch.object(launcher.subprocess, "Popen", side_effect=[chat_proc, web_proc]) as popen:
            errors = launcher.restore_services(self.manager, [web, chat])
        self.assertEqual(errors, ["chat: RuntimeError"])
        self.assertEqual(popen.call_count, 2)
        self.manager._stop_started.assert_called_once_with(chat.service, chat_proc)
        self.assertEqual(web.service.pid_path.read_text(), "110\n")
        self.assertEqual(self.manager._wait_ready.call_args_list[-1], call(web.service, web_proc, 15))

    def test_process_lookup_race_when_signaling_original_is_harmless(self):
        item = self.snapshot("chat", 20)
        for disappearance in ("getpgid", "killpg"):
            with self.subTest(disappearance=disappearance), \
                    patch.object(launcher, "process_identity", return_value=item.identity), \
                    patch.object(launcher.os, "getpgid", return_value=item.pid) as getpgid, \
                    patch.object(launcher.os, "killpg") as killpg:
                (getpgid if disappearance == "getpgid" else killpg).side_effect = ProcessLookupError
                launcher.signal_original(item, signal.SIGTERM)
                if disappearance == "getpgid":
                    killpg.assert_not_called()

    def test_reused_pid_or_changed_group_never_receives_original_service_signal(self):
        item = self.snapshot("chat", 20)
        with patch.object(launcher, "process_identity", return_value=item.identity + 1), \
                patch.object(launcher.os, "getpgid") as getpgid, \
                patch.object(launcher.os, "killpg") as killpg:
            launcher.signal_original(item, signal.SIGTERM)
        getpgid.assert_not_called()
        killpg.assert_not_called()
        with patch.object(launcher, "process_identity", return_value=item.identity), \
                patch.object(launcher.os, "getpgid", return_value=999), \
                patch.object(launcher.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "进程组"):
                launcher.signal_original(item, signal.SIGTERM)
        killpg.assert_not_called()

    def test_identity_is_checked_again_immediately_before_sending_signal(self):
        item = self.snapshot("chat", 20)
        with patch.object(launcher, "process_identity", side_effect=[item.identity, item.identity + 1]), \
                patch.object(launcher.os, "getpgid", return_value=item.pid), \
                patch.object(launcher.os, "killpg") as killpg:
            launcher.signal_original(item, signal.SIGTERM)
        killpg.assert_not_called()

    def test_run_child_masks_interrupts_before_cleanup_and_restores_handlers(self):
        proc = self.process(123)
        proc.wait.return_value = 7
        original_handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
        active_handlers = dict(original_handlers)
        events = []

        def set_handler(sig, handler):
            active_handlers[sig] = handler
            events.append((sig, handler))

        def cleanup(child):
            self.assertIs(child, proc)
            self.assertEqual(active_handlers, {signal.SIGINT: signal.SIG_IGN, signal.SIGTERM: signal.SIG_IGN})
            events.append("cleanup")

        with patch.object(launcher.subprocess, "Popen", return_value=proc) as popen, \
                patch.object(launcher.signal, "getsignal", side_effect=original_handlers.__getitem__), \
                patch.object(launcher.signal, "signal", side_effect=set_handler), \
                patch.object(launcher, "stop_child", side_effect=cleanup):
            self.assertEqual(launcher.run_child(["fake-training-python", "train.py"]), 7)
        popen.assert_called_once_with(
            ["fake-training-python", "train.py"], cwd=str(launcher.PROJECT_ROOT), start_new_session=True,
        )
        self.assertEqual(active_handlers, original_handlers)
        self.assertEqual(events[2], "cleanup")

    def test_wait_interruption_still_cleans_child_before_propagating(self):
        proc = self.process(123)
        proc.wait.side_effect = KeyboardInterrupt
        with patch.object(launcher.subprocess, "Popen", return_value=proc), \
                patch.object(launcher.signal, "getsignal", return_value=None), \
                patch.object(launcher.signal, "signal") as set_handler, \
                patch.object(launcher, "stop_child") as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                launcher.run_child(["fake-training-python"])
        cleanup.assert_called_once_with(proc)
        self.assertEqual(set_handler.call_args_list[:2], [call(signal.SIGINT, signal.SIG_IGN), call(signal.SIGTERM, signal.SIG_IGN)])
        self.assertEqual(set_handler.call_args_list[-2:], [call(signal.SIGINT, None), call(signal.SIGTERM, None)])

    def test_cleanup_failure_propagates_and_handlers_are_restored(self):
        proc = self.process(123)
        proc.wait.return_value = 0
        failure = RuntimeError("group still running")
        with patch.object(launcher.subprocess, "Popen", return_value=proc), \
                patch.object(launcher.signal, "getsignal", return_value=None), \
                patch.object(launcher.signal, "signal") as set_handler, \
                patch.object(launcher, "stop_child", side_effect=failure):
            with self.assertRaises(launcher.ChildCleanupError) as error:
                launcher.run_child(["fake-training-python"])
        self.assertIs(error.exception.__cause__, failure)
        self.assertEqual(set_handler.call_args_list[-2:], [call(signal.SIGINT, None), call(signal.SIGTERM, None)])

    def fake_serve(self, alive):
        module = ModuleType("scripts.serve")
        module._process_group_alive = alive
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict(sys.modules, {"scripts.serve": module}))
        stack.enter_context(patch.object(scripts, "serve", module, create=True))
        return stack

    def test_exited_training_parent_still_causes_orphan_group_cleanup(self):
        proc = self.process(123)
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        with self.fake_serve(Mock(return_value=False)), \
                patch.object(launcher.Path, "exists", return_value=False), \
                patch.object(launcher.os, "killpg") as killpg, \
                patch.object(launcher.time, "monotonic", return_value=0), \
                patch.object(launcher.time, "sleep"):
            launcher.stop_child(proc)
        killpg.assert_called_once_with(proc.pid, signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=30)

    def test_cleanup_escalates_when_training_parent_does_not_exit(self):
        proc = self.process(123)
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("fake-training", 30), 0]
        with self.fake_serve(Mock(return_value=False)), \
                patch.object(launcher.os, "killpg") as killpg, \
                patch.object(launcher.time, "monotonic", return_value=0), \
                patch.object(launcher.time, "sleep"):
            launcher.stop_child(proc)
        self.assertEqual(killpg.call_args_list, [call(proc.pid, signal.SIGTERM), call(proc.pid, signal.SIGKILL)])
        self.assertEqual(proc.wait.call_args_list, [call(timeout=30), call(timeout=10)])

    def test_cleanup_rejects_a_group_that_survives_sigkill(self):
        proc = self.process(123)
        proc.poll.return_value = None
        proc.wait.return_value = 0
        with self.fake_serve(Mock(return_value=True)), \
                patch.object(launcher.os, "killpg") as killpg, \
                patch.object(launcher.time, "monotonic", side_effect=itertools.count(0, 20).__next__), \
                patch.object(launcher.time, "sleep"):
            with self.assertRaisesRegex(launcher.ChildCleanupError, "不能恢复"):
                launcher.stop_child(proc)
        self.assertEqual(killpg.call_args_list, [call(proc.pid, signal.SIGTERM), call(proc.pid, signal.SIGKILL)])

    def test_reused_training_pid_is_not_signaled(self):
        proc = self.process(123)
        proc.poll.return_value = 0
        with patch.object(launcher.Path, "exists", return_value=True), \
                patch.object(launcher.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "复用"):
                launcher.stop_child(proc)
        killpg.assert_not_called()
        proc.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
