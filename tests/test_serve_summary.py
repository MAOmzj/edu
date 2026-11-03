# 文件用途：验证独立摘要服务配置、启动顺序与启动失败或中断后的清理。
# 调用关系：unittest 调用 scripts/serve.py；所有进程、信号和 HTTP 探活使用 mock。
# 修改易踩坑：测试不得启动真实模型、停止现有服务或访问外部网络。
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import serve
from utils.config_handler import load_yaml_config


class SummaryServiceTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.config = patch.dict(serve.model_conf, {
            "summary": {}, "chat_lora_path": "", "chat_quantization": "",
            "chat_tensor_parallel_size": 1,
        }, clear=False)
        self.config.start()
        for key in ("chat_gpu_memory_utilization", "embedding_gpu_memory_utilization",
                    "summary_gpu_memory_utilization", "embedding_tensor_parallel_size",
                    "summary_tensor_parallel_size"):
            serve.model_conf.pop(key, None)
        for key in ("chat_cuda_visible_devices", "embedding_cuda_visible_devices",
                    "summary_cuda_visible_devices"):
            serve.model_conf.pop(key, None)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.config.stop()
        self.environment.stop()

    @staticmethod
    def argument(service, name):
        return service.argv[service.argv.index(name) + 1]

    def test_summary_defaults_are_independent_and_local(self):
        table = serve._build_services()
        summary = table["summary"]
        self.assertEqual(summary.port, 8002)
        self.assertIn("/root/MAOMI/models/Qwen3-4B", summary.argv)
        self.assertEqual(self.argument(summary, "--served-model-name"), "education-summary")
        self.assertEqual(self.argument(summary, "--host"), "127.0.0.1")
        self.assertEqual(self.argument(summary, "--max-model-len"), "16384")
        self.assertEqual(self.argument(summary, "--max-num-seqs"), "1")
        self.assertEqual(self.argument(summary, "--max-num-batched-tokens"), "2048")
        self.assertEqual(self.argument(summary, "--gpu-memory-utilization"), "0.20")
        self.assertEqual(self.argument(summary, "--dtype"), "bfloat16")
        self.assertIn("--enforce-eager", summary.argv)
        self.assertEqual(summary.env["OMP_NUM_THREADS"], "1")
        self.assertFalse(serve.get_summary_model_config()["enabled"])
        self.assertEqual(self.argument(table["chat"], "--gpu-memory-utilization"), "0.80")

    def test_enabled_summary_adjusts_chat_default_but_respects_override(self):
        serve.model_conf["summary"] = {"enabled": True}
        table = serve._build_services()
        self.assertEqual(self.argument(table["chat"], "--gpu-memory-utilization"), "0.70")
        os.environ["QWEN_GPU_MEM_UTIL"] = "0.65"
        os.environ["SUMMARY_GPU_MEM_UTIL"] = "0.25"
        os.environ["SUMMARY_MAX_MODEL_LEN"] = "32768"
        table = serve._build_services()
        self.assertEqual(self.argument(table["chat"], "--gpu-memory-utilization"), "0.65")
        self.assertEqual(self.argument(table["summary"], "--gpu-memory-utilization"), "0.25")
        self.assertEqual(self.argument(table["summary"], "--max-model-len"), "32768")

    def test_environment_overrides_summary_identity_and_enabled(self):
        os.environ.update({
            "EDUCATION_SUMMARY_ENABLED": "true",
            "EDUCATION_SUMMARY_BASE_URL": "http://127.0.0.1:8102/v1",
            "EDUCATION_SUMMARY_MODEL_NAME": "custom-summary",
        })
        table = serve._build_services()
        self.assertEqual(table["summary"].port, 8102)
        self.assertEqual(self.argument(table["summary"], "--served-model-name"), "custom-summary")
        self.assertEqual(self.argument(table["chat"], "--gpu-memory-utilization"), "0.70")

    def test_project_all_deploys_merged_v2_with_matching_client_identity(self):
        serve.model_conf["summary"] = load_yaml_config("config/model.yml")["summary"]
        options = serve.get_summary_model_config()
        table = serve._build_services()
        summary = table["summary"]
        self.assertTrue(options["enabled"])
        self.assertEqual(
            options["model_path"], "/root/MAOMI/models/Qwen3-4B-summary-lora-v2"
        )
        self.assertIn(options["model_path"], summary.argv)
        self.assertEqual(self.argument(summary, "--served-model-name"), options["model_name"])
        self.assertEqual(summary.health_url, options["base_url"] + "/models")
        with patch.object(serve, "services", return_value=table), patch.object(
            serve, "start", return_value=0
        ) as start:
            self.assertEqual(serve.main(["all", "-d"]), 0)
        self.assertEqual(start.call_args.args[0], ["chat", "embedding", "summary", "web"])

    def test_invalid_enabled_environment_is_rejected(self):
        os.environ["EDUCATION_SUMMARY_ENABLED"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "EDUCATION_SUMMARY_ENABLED"):
            serve._build_services()

    def test_nonpositive_context_length_is_rejected(self):
        os.environ["SUMMARY_MAX_MODEL_LEN"] = "0"
        with self.assertRaisesRegex(ValueError, "SUMMARY_MAX_MODEL_LEN"):
            serve._build_services()

    def test_remote_summary_endpoint_cannot_be_started_locally(self):
        serve.model_conf["summary"] = {"base_url": "http://example.com:8002/v1"}
        with self.assertRaisesRegex(ValueError, "不在本机"):
            serve._build_services()

    def test_all_selects_summary_only_when_enabled_and_explicit_command_works(self):
        table = serve._build_services()
        with patch.object(serve, "services", return_value=table), patch.object(
            serve, "start", return_value=0
        ) as start:
            self.assertEqual(serve.main(["all", "-d"]), 0)
            self.assertEqual(start.call_args.args[0], ["chat", "embedding", "web"])
            serve.model_conf["summary"] = {"enabled": True}
            self.assertEqual(serve.main(["all", "-d"]), 0)
            self.assertEqual(start.call_args.args[0], ["chat", "embedding", "summary", "web"])
            serve.model_conf["summary"] = {"enabled": False}
            self.assertEqual(serve.main(["summary", "-d"]), 0)
            self.assertEqual(start.call_args.args[0], ["summary"])

    def test_stop_all_includes_explicitly_started_disabled_summary(self):
        table = serve._build_services()
        with patch.object(serve, "services", return_value=table), patch.object(
            serve, "stop", return_value=0
        ) as stop:
            self.assertEqual(serve.main(["stop", "all"]), 0)
            self.assertIn("summary", stop.call_args.args[0])
            self.assertEqual(serve.main(["stop", "summary"]), 0)
            self.assertEqual(stop.call_args.args[0], ["summary"])

    def start_with_mocks(self, keys, *, detach=False, wait_result=(True, ""), wait_effect=None):
        table = serve._build_services()
        events = []
        signals = {}

        def start(service, **kwargs):
            events.append(("start", service.key))
            return Mock(pid=100 + len(events))

        def wait(service, proc, timeout):
            events.append(("wait", service.key))
            if wait_effect is not None:
                return wait_effect(service, signals)
            return wait_result

        def stop(service, proc):
            events.append(("stop", service.key))
            return True

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(serve, "services", return_value=table))
            stack.enter_context(patch.object(serve, "_preflight", return_value=True))
            stack.enter_context(patch.object(serve, "_start_one", side_effect=start))
            stack.enter_context(patch.object(serve, "_wait_ready", side_effect=wait))
            stack.enter_context(patch.object(serve, "_stop_started", side_effect=stop))
            stack.enter_context(patch.object(serve, "_report_failure"))
            stack.enter_context(patch.object(serve, "_supervise", return_value=0))
            stack.enter_context(patch.object(serve.signal, "getsignal", return_value=None))
            stack.enter_context(patch.object(
                serve.signal, "signal", side_effect=lambda sig, handler: signals.update({sig: handler})
            ))
            result = serve.start(keys, detach=detach, force=False)
        return result, events, signals

    def test_summary_enabled_detached_all_waits_before_starting_web(self):
        result, events, signals = self.start_with_mocks(
            ["chat", "embedding", "summary", "web"], detach=True
        )
        self.assertEqual(result, 0)
        web_start = events.index(("start", "web"))
        for key in ("chat", "embedding", "summary"):
            self.assertLess(events.index(("wait", key)), web_start)
        self.assertLess(events.index(("wait", "chat")), events.index(("start", "embedding")))
        self.assertLess(events.index(("wait", "embedding")), events.index(("start", "summary")))
        self.assertIn(("wait", "web"), events)
        self.assertFalse(any(action == "stop" for action, key in events))
        self.assertTrue(all(handler is None for handler in signals.values()))

    def test_disabled_detached_all_also_waits_for_models_and_web(self):
        result, events, _ = self.start_with_mocks(["chat", "embedding", "web"], detach=True)
        self.assertEqual(result, 0)
        self.assertEqual(events, [
            ("start", "chat"), ("wait", "chat"),
            ("start", "embedding"), ("wait", "embedding"),
            ("start", "web"), ("wait", "web"),
        ])

    def test_single_detached_model_preserves_async_start(self):
        result, events, _ = self.start_with_mocks(["summary"], detach=True)
        self.assertEqual(result, 0)
        self.assertEqual(events, [("start", "summary")])

    def test_first_model_failure_skips_later_models_and_web(self):
        result, events, _ = self.start_with_mocks(
            ["chat", "embedding", "summary", "web"],
            detach=True, wait_result=(False, "timeout"),
        )
        self.assertEqual(result, 1)
        self.assertEqual(events, [("start", "chat"), ("wait", "chat"), ("stop", "chat")])

    def test_middle_model_failure_only_cleans_started_models(self):
        def fail_embedding(service, signals):
            return (False, "timeout") if service.key == "embedding" else (True, "")

        result, events, _ = self.start_with_mocks(
            ["chat", "embedding", "summary", "web"], detach=True,
            wait_effect=fail_embedding,
        )
        self.assertEqual(result, 1)
        self.assertNotIn(("start", "summary"), events)
        self.assertNotIn(("start", "web"), events)
        self.assertEqual(events[-2:], [("stop", "embedding"), ("stop", "chat")])

    def test_summary_readiness_timeout_cleans_all_spawned_models(self):
        def fail_summary(service, signals):
            return (False, "timeout") if service.key == "summary" else (True, "")

        result, events, _ = self.start_with_mocks(
            ["chat", "embedding", "summary", "web"], wait_effect=fail_summary
        )
        self.assertEqual(result, 1)
        self.assertNotIn(("start", "web"), events)
        self.assertEqual(events[-3:], [("stop", "summary"), ("stop", "embedding"), ("stop", "chat")])

    def test_web_readiness_timeout_cleans_web_too(self):
        result, events, _ = self.start_with_mocks(["web"], wait_result=(False, "timeout"))
        self.assertEqual(result, 1)
        self.assertEqual(events[-1], ("stop", "web"))

    def test_detached_all_web_failure_cleans_all_started_services(self):
        def fail_web(service, signals):
            return (False, "timeout") if service.key == "web" else (True, "")

        result, events, _ = self.start_with_mocks(
            ["chat", "embedding", "summary", "web"], detach=True, wait_effect=fail_web
        )
        self.assertEqual(result, 1)
        self.assertEqual(events[-4:], [
            ("stop", "web"), ("stop", "summary"),
            ("stop", "embedding"), ("stop", "chat"),
        ])

    def test_spawn_error_cleans_models_that_already_started(self):
        table = serve._build_services()
        chat_proc = Mock(pid=123)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(serve, "services", return_value=table))
            stack.enter_context(patch.object(serve, "_preflight", return_value=True))
            stack.enter_context(patch.object(
                serve, "_start_one", side_effect=[chat_proc, OSError("spawn failed")]
            ))
            stack.enter_context(patch.object(serve, "_wait_ready", return_value=(True, "")))
            stop = stack.enter_context(patch.object(serve, "_stop_started", return_value=True))
            stack.enter_context(patch.object(serve.signal, "getsignal", return_value=None))
            signal_handler = stack.enter_context(patch.object(serve.signal, "signal"))
            with self.assertRaisesRegex(OSError, "spawn failed"):
                serve.start(["chat", "summary"], detach=False, force=False)
        stop.assert_called_once_with(table["chat"], chat_proc)
        self.assertEqual(signal_handler.call_args_list[-2].args, (serve.signal.SIGINT, None))
        self.assertEqual(signal_handler.call_args_list[-1].args, (serve.signal.SIGTERM, None))

    def test_signal_during_summary_loading_cleans_spawned_process(self):
        def interrupt(service, signals):
            signals[serve.signal.SIGTERM](serve.signal.SIGTERM, None)

        result, events, signals = self.start_with_mocks(["summary"], wait_effect=interrupt)
        self.assertEqual(result, 0)
        self.assertEqual(events[-1], ("stop", "summary"))
        self.assertTrue(all(handler is None for handler in signals.values()))

    def test_pid_registration_failure_or_interrupt_cleans_created_group(self):
        service = serve._build_services()["summary"]
        for failure in (OSError("pid write failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temp:
                proc = Mock(pid=123)
                proc.poll.return_value = None
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(serve, "LOG_DIR", Path(temp)))
                    stack.enter_context(patch.object(serve.subprocess, "Popen", return_value=proc))
                    stack.enter_context(patch.object(serve, "_tee_output"))
                    stack.enter_context(patch.object(Path, "write_text", side_effect=failure))
                    stack.enter_context(patch.object(serve, "_process_group_alive", return_value=False))
                    kill_group = stack.enter_context(patch.object(serve.os, "killpg"))
                    with self.assertRaises(type(failure)):
                        serve._start_one(service, detach=False)
                kill_group.assert_called_once_with(proc.pid, serve.signal.SIGTERM)

    def test_exited_parent_with_live_children_still_stops_owned_group(self):
        service = serve._build_services()["summary"]
        proc = Mock(pid=123)
        proc.poll.return_value = 1
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(Path, "exists", return_value=False))
            stack.enter_context(patch.object(serve, "_process_group_alive", side_effect=[True, False]))
            stack.enter_context(patch.object(serve.time, "sleep"))
            stack.enter_context(patch.object(serve, "_read_pid", return_value=proc.pid))
            forget = stack.enter_context(patch.object(serve, "_forget"))
            kill_group = stack.enter_context(patch.object(serve.os, "killpg"))
            self.assertTrue(serve._stop_started(service, proc))
        kill_group.assert_called_once_with(proc.pid, serve.signal.SIGTERM)
        forget.assert_called_once_with(service)

    def test_owned_group_with_unresponsive_children_is_force_stopped(self):
        service = serve._build_services()["summary"]
        proc = Mock(pid=123)
        proc.poll.return_value = 1
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(Path, "exists", return_value=False))
            stack.enter_context(patch.object(serve, "_process_group_alive", return_value=True))
            stack.enter_context(patch.object(serve, "_read_pid", return_value=None))
            kill_group = stack.enter_context(patch.object(serve.os, "killpg"))
            self.assertTrue(serve._stop_started(service, proc, timeout=0))
        self.assertEqual(
            [call.args for call in kill_group.call_args_list],
            [(proc.pid, serve.signal.SIGTERM), (proc.pid, serve.signal.SIGKILL)],
        )

    def test_cleanup_preserves_pid_file_replaced_by_another_start(self):
        service = serve._build_services()["summary"]
        proc = Mock(pid=123)
        proc.poll.return_value = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(serve, "_process_group_alive", return_value=False))
            stack.enter_context(patch.object(serve, "_read_pid", return_value=999))
            forget = stack.enter_context(patch.object(serve, "_forget"))
            kill_group = stack.enter_context(patch.object(serve.os, "killpg"))
            self.assertTrue(serve._stop_started(service, proc))
        kill_group.assert_called_once_with(proc.pid, serve.signal.SIGTERM)
        forget.assert_not_called()

    def test_cleanup_does_not_signal_reused_parent_pid(self):
        service = serve._build_services()["summary"]
        proc = Mock(pid=123)
        proc.poll.return_value = 1
        with patch.object(Path, "exists", return_value=True), patch.object(
            serve.os, "killpg"
        ) as kill_group, patch.object(serve, "_forget") as forget:
            self.assertFalse(serve._stop_started(service, proc))
        kill_group.assert_not_called()
        forget.assert_not_called()

    def test_health_probe_bypasses_environment_proxy(self):
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=Mock(status=200))
        opener.open.return_value.__exit__ = Mock(return_value=False)
        with patch.object(serve.urllib.request, "ProxyHandler") as proxy, patch.object(
            serve.urllib.request, "build_opener", return_value=opener
        ):
            self.assertTrue(serve._is_ready("http://127.0.0.1:8002/v1/models"))
        proxy.assert_called_once_with({})
        opener.open.assert_called_once_with("http://127.0.0.1:8002/v1/models", timeout=2.0)


if __name__ == "__main__":
    unittest.main()
