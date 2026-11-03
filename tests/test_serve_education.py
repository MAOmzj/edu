# 文件用途：验证教育适配器部署的请求路由与基座匹配检查。
# 调用关系：unittest 调用 serve._build_services/_chat_model_options；不启动模型或访问网络。
# 修改易踩坑：临时权重仅作路径检查，不是可用于推理的真实适配器。
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import serve


class EducationDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.base = root / "base"
        self.adapter = root / "adapter"
        self.base.mkdir()
        self.adapter.mkdir()
        (self.base / "config.json").write_text("{}")
        self.info = {"peft_type": "LORA", "r": 16,
                     "base_model_name_or_path": str(self.base)}
        self.write_info()
        (self.adapter / "adapter_model.safetensors").touch()
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.dict(serve.model_conf, {
            "chat_model_path": str(self.base), "chat_lora_path": str(self.adapter),
            "chat_model_name": "education-v2", "summary": {"enabled": True},
            "chat_quantization": "", "chat_tensor_parallel_size": 1,
        }))
        for key in ("chat_gpu_memory_utilization", "embedding_gpu_memory_utilization",
                    "summary_gpu_memory_utilization", "embedding_tensor_parallel_size",
                    "summary_tensor_parallel_size", "chat_quantized_base_manifest"):
            serve.model_conf.pop(key, None)
        for key in ("chat_cuda_visible_devices", "embedding_cuda_visible_devices",
                    "summary_cuda_visible_devices"):
            serve.model_conf.pop(key, None)

    def write_info(self):
        # 下游路径校验函数读取此配置；本测试不加载其中任何模型权重。
        (self.adapter / "adapter_config.json").write_text(json.dumps(self.info))

    @staticmethod
    def argument(service, name):
        return service.argv[service.argv.index(name) + 1]

    def test_student_client_name_selects_adapter_instead_of_base(self):
        table = serve._build_services()
        chat = table["chat"]
        module = json.loads(self.argument(chat, "--lora-modules"))
        self.assertEqual(module["name"], serve.model_conf["chat_model_name"])
        self.assertEqual(module["path"], str(self.adapter))
        self.assertEqual(module["base_model_name"], self.argument(chat, "--served-model-name"))
        self.assertNotEqual(module["name"], module["base_model_name"])
        self.assertIn("--enable-lora", chat.argv)
        self.assertEqual(self.argument(chat, "--max-lora-rank"), "16")
        self.assertEqual(self.argument(chat, "--dtype"), "bfloat16")
        self.assertEqual(self.argument(chat, "--gpu-memory-utilization"), "0.75")
        self.assertEqual(self.argument(table["embedding"], "--gpu-memory-utilization"), "0.04")
        self.assertEqual(self.argument(table["summary"], "--gpu-memory-utilization"), "0.16")

    def test_wrong_base_is_rejected_before_starting(self):
        self.info["base_model_name_or_path"] = str(self.base.parent / "other-base")
        self.write_info()
        with self.assertRaisesRegex(ValueError, "训练基座一致"):
            serve._chat_model_options()

    def test_missing_adapter_never_silently_serves_base(self):
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            serve._chat_model_options()

    def test_blank_adapter_path_uses_regular_served_name(self):
        serve.model_conf["chat_lora_path"] = ""
        base, name, args = serve._chat_model_options()
        self.assertEqual(base, self.base)
        self.assertEqual(name, "education-v2")
        self.assertEqual(args, ())

    def test_nf4_keeps_original_base_and_adapter_routing(self):
        serve.model_conf["chat_quantization"] = "nf4"
        serve.model_conf["chat_gpu_memory_utilization"] = 0.35
        chat = serve._build_services()["chat"]
        self.assertIn(str(self.base), chat.argv)
        self.assertIn("--enable-lora", chat.argv)
        self.assertEqual(self.argument(chat, "--quantization"), "bitsandbytes")
        self.assertEqual(self.argument(chat, "--load-format"), "bitsandbytes")
        quant = json.loads(self.argument(chat, "--hf-overrides"))["quantization_config"]
        self.assertTrue(quant["load_in_4bit"])
        self.assertEqual(quant["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(quant["bnb_4bit_use_double_quant"])
        self.assertIn("lm_head", quant["llm_int8_skip_modules"])
        self.assertEqual(json.loads(self.argument(chat, "--lora-modules"))["name"], "education-v2")
        self.assertEqual(self.argument(chat, "--gpu-memory-utilization"), "0.35")

    def configure_int4(self):
        # 下游 serve 读取独立 INT4 目录；原适配器仍记录 self.base，不能修改训练来源。
        self.quantized_base = self.base.parent / "int4-base"
        self.quantized_base.mkdir()
        self.int4_config = {"quantization_config": {
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {"group_0": {
                "weights": {"type": "int", "num_bits": 4},
                "input_activations": None, "output_activations": None,
            }},
        }}
        self.write_int4_config()
        self.manifest = self.base.parent / "quantized-base-manifest.json"
        self.manifest.write_text("{}")
        serve.model_conf.update({
            "chat_model_path": str(self.quantized_base),
            "chat_quantization": "compressed-tensors",
            "chat_quantized_base_manifest": str(self.manifest),
        })

    def write_int4_config(self):
        (self.quantized_base / "config.json").write_text(json.dumps(self.int4_config))

    def test_int4_uses_bound_manifest_and_preserves_education_inference(self):
        self.configure_int4()
        # 下游真实来源校验有独立测试；这里验证 serve 必须传入实际四个绑定对象。
        with patch.object(serve, "validate_quantized_lora_base") as validate:
            chat = serve._build_services()["chat"]
        validate.assert_called_once_with(
            self.quantized_base, self.base.resolve(), self.adapter.resolve(), self.manifest,
        )
        self.assertIn(str(self.quantized_base), chat.argv)
        self.assertEqual(self.argument(chat, "--quantization"), "compressed-tensors")
        self.assertEqual(self.argument(chat, "--load-format"), "auto")
        self.assertNotIn("--hf-overrides", chat.argv)
        self.assertNotIn("--model-loader-extra-config", chat.argv)
        self.assertNotIn("bitsandbytes", chat.argv)
        self.assertIn("INT4 W4A16", chat.title)
        self.assertNotIn("NF4", chat.title)
        self.assertIn("教育微调", chat.title)
        module = json.loads(self.argument(chat, "--lora-modules"))
        self.assertEqual(module, {"name": "education-v2", "path": str(self.adapter.resolve()),
                                  "base_model_name": "education-v2-base"})
        self.assertEqual(self.argument(chat, "--served-model-name"), "education-v2-base")
        self.assertIn("--enable-lora", chat.argv)
        self.assertEqual(self.argument(chat, "--max-lora-rank"), "16")
        self.assertEqual(self.argument(chat, "--dtype"), "bfloat16")
        self.assertEqual(json.loads(self.argument(chat, "--limit-mm-per-prompt")),
                         {"image": 0, "video": 0})
        self.assertEqual(json.loads(self.argument(chat, "--default-chat-template-kwargs")),
                         {"enable_thinking": False})
        self.assertEqual(self.argument(chat, "--max-num-seqs"), "4")
        self.assertEqual(self.argument(chat, "--max-num-batched-tokens"), "2048")
        self.assertEqual(self.argument(chat, "--tensor-parallel-size"), "1")
        # 检查构建参数没有偷偷把适配器训练基座改成 INT4 路径。
        self.assertEqual(json.loads((self.adapter / "adapter_config.json").read_text()), self.info)

    def test_int4_rejects_missing_manifest_before_validation(self):
        self.configure_int4()
        for missing in (None, "", "   "):
            with self.subTest(manifest=missing):
                serve.model_conf["chat_quantized_base_manifest"] = missing
                with patch.object(serve, "validate_quantized_lora_base") as validate:
                    with self.assertRaisesRegex(ValueError, "chat_quantized_base_manifest"):
                        serve._chat_model_options()
                validate.assert_not_called()

    def test_int4_propagates_failed_provenance_check(self):
        self.configure_int4()
        with patch.object(serve, "validate_quantized_lora_base",
                          side_effect=ValueError("量化基座来源不匹配")) as validate:
            with self.assertRaisesRegex(ValueError, "量化基座来源不匹配"):
                serve._build_services()
        validate.assert_called_once()

    def test_int4_requires_exported_integer_weight_only_four_bits(self):
        self.configure_int4()
        valid = json.dumps(self.int4_config)
        invalid_fields = (
            ("quant_method", "bitsandbytes"), ("format", "dense"),
            ("quantization_status", "initialized"), ("config_groups", {}),
        )
        invalid_weights = (("type", "float"), ("num_bits", 8), ("num_bits", None))
        for key, value in invalid_fields:
            with self.subTest(field=key, value=value):
                self.int4_config = json.loads(valid)
                self.int4_config["quantization_config"][key] = value
                self.write_int4_config()
                with self.assertRaisesRegex(ValueError, "W4A16 INT4"):
                    serve._chat_quantization_options()
        for key, value in invalid_weights:
            with self.subTest(weight=key, value=value):
                self.int4_config = json.loads(valid)
                group = self.int4_config["quantization_config"]["config_groups"]["group_0"]
                group["weights"][key] = value
                self.write_int4_config()
                with self.assertRaisesRegex(ValueError, "W4A16 INT4"):
                    serve._chat_quantization_options()
        for key in ("input_activations", "output_activations"):
            with self.subTest(activation=key):
                self.int4_config = json.loads(valid)
                group = self.int4_config["quantization_config"]["config_groups"]["group_0"]
                group[key] = {"num_bits": 8, "type": "int"}
                self.write_int4_config()
                with self.assertRaisesRegex(ValueError, "W4A16 INT4"):
                    serve._chat_quantization_options()

    def test_nf4_does_not_use_int4_manifest(self):
        serve.model_conf["chat_quantization"] = "nf4"
        serve.model_conf["chat_quantized_base_manifest"] = "/missing/int4-manifest.json"
        with patch.object(serve, "validate_quantized_lora_base") as validate:
            chat = serve._build_services()["chat"]
        validate.assert_not_called()
        self.assertEqual(self.argument(chat, "--quantization"), "bitsandbytes")

    def test_int4_manifest_never_bypasses_other_modes_training_base_match(self):
        self.configure_int4()
        for mode in ("", "bf16", "nf4"):
            with self.subTest(quantization=mode):
                serve.model_conf["chat_quantization"] = mode
                with patch.object(serve, "validate_quantized_lora_base") as validate:
                    with self.assertRaisesRegex(ValueError, "训练基座一致"):
                        serve._chat_model_options()
                validate.assert_not_called()

    def test_unsupported_quantization_is_rejected(self):
        serve.model_conf["chat_quantization"] = "unknown"
        with self.assertRaisesRegex(ValueError, "chat_quantization"):
            serve._build_services()

    def test_gpu_budget_environment_override_takes_precedence(self):
        serve.model_conf["chat_gpu_memory_utilization"] = 0.35
        os.environ["QWEN_GPU_MEM_UTIL"] = "0.40"
        chat = serve._build_services()["chat"]
        self.assertEqual(self.argument(chat, "--gpu-memory-utilization"), "0.40")

    def test_invalid_gpu_budget_is_rejected(self):
        for value in [0, -0.1, 1.1, float("nan")]:
            serve.model_conf["chat_gpu_memory_utilization"] = value
            with self.assertRaisesRegex(ValueError, "chat_gpu_memory_utilization"):
                serve._build_services()

    def test_tp2_keeps_adapter_and_isolates_small_models(self):
        serve.model_conf.update({
            "chat_tensor_parallel_size": 2,
            "chat_cuda_visible_devices": "0,1",
            "embedding_cuda_visible_devices": "0",
            "summary_cuda_visible_devices": "0",
            "chat_gpu_memory_utilization": 0.25,
        })
        table = serve._build_services()
        chat = table["chat"]
        self.assertEqual(self.argument(chat, "--tensor-parallel-size"), "2")
        self.assertEqual(self.argument(chat, "--distributed-executor-backend"), "mp")
        self.assertIn("TP=2", chat.title)
        self.assertEqual(chat.env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(self.argument(chat, "--gpu-memory-utilization"), "0.25")
        self.assertIn(str(self.base), chat.argv)
        self.assertEqual(json.loads(self.argument(chat, "--lora-modules"))["name"], "education-v2")
        self.assertEqual(self.argument(chat, "--max-num-seqs"), "4")
        for key in ("embedding", "summary"):
            self.assertEqual(table[key].env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertNotIn("--tensor-parallel-size", table[key].argv)
            self.assertNotIn("--distributed-executor-backend", table[key].argv)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", table["web"].env)

    def test_all_models_tp2_have_independent_per_gpu_budgets(self):
        # 下游 _build_services 给三个独立 vLLM 实例各自创建两卡 worker，
        # 每实例预算按每张卡计算；测试不启动 GPU 进程。
        for prefix, budget in (("chat", 0.70), ("embedding", 0.03), ("summary", 0.12)):
            serve.model_conf.update({
                f"{prefix}_tensor_parallel_size": 2,
                f"{prefix}_cuda_visible_devices": "0,1",
                f"{prefix}_gpu_memory_utilization": budget,
            })
        table = serve._build_services()
        for prefix, budget in (("chat", "0.7"), ("embedding", "0.03"), ("summary", "0.12")):
            with self.subTest(service=prefix):
                service = table[prefix]
                self.assertEqual(self.argument(service, "--tensor-parallel-size"), "2")
                self.assertEqual(self.argument(service, "--distributed-executor-backend"), "mp")
                self.assertEqual(self.argument(service, "--gpu-memory-utilization"), budget)
                self.assertEqual(service.env["CUDA_VISIBLE_DEVICES"], "0,1")
                self.assertEqual(service.env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(json.loads(self.argument(table["chat"], "--lora-modules"))["name"],
                         "education-v2")
        self.assertEqual(self.argument(table["chat"], "--max-num-seqs"), "4")
        # 下游向量模型保留 CUDA Graph；仅把其 TP2 all-reduce 切到 NCCL，
        # 避免当前双 A800 在图捕获期间触发自定义 kernel 的 invalid argument。
        self.assertIn("--disable-custom-all-reduce", table["embedding"].argv)
        self.assertNotIn("--enforce-eager", table["embedding"].argv)
        self.assertNotIn("--disable-custom-all-reduce", table["chat"].argv)
        self.assertNotIn("--disable-custom-all-reduce", table["summary"].argv)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", table["web"].env)

    def test_small_model_budgets_use_yaml_and_environment_precedence(self):
        for prefix, env_name, configured, overridden in (
            ("embedding", "EMBED_GPU_MEM_UTIL", 0.03, "0.04"),
            ("summary", "SUMMARY_GPU_MEM_UTIL", 0.12, "0.14"),
        ):
            with self.subTest(service=prefix):
                serve.model_conf[f"{prefix}_gpu_memory_utilization"] = configured
                service = serve._build_services()[prefix]
                self.assertEqual(self.argument(service, "--gpu-memory-utilization"), str(configured))
                os.environ[env_name] = overridden
                service = serve._build_services()[prefix]
                self.assertEqual(self.argument(service, "--gpu-memory-utilization"), overridden)
                os.environ.pop(env_name)

    def test_invalid_small_model_budgets_are_rejected(self):
        for prefix in ("embedding", "summary"):
            key = f"{prefix}_gpu_memory_utilization"
            for value in (0, -0.1, 1.1, float("nan"), float("inf"), True, "invalid"):
                with self.subTest(service=prefix, value=value):
                    serve.model_conf[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        serve._build_services()
            serve.model_conf.pop(key)

    def test_invalid_small_model_budget_environment_is_rejected(self):
        for prefix, env_name in (("embedding", "EMBED_GPU_MEM_UTIL"),
                                 ("summary", "SUMMARY_GPU_MEM_UTIL")):
            for value in ("0", "1.1", "nan", "inf", "invalid"):
                with self.subTest(service=prefix, value=value):
                    os.environ[env_name] = value
                    with self.assertRaisesRegex(ValueError, f"{prefix}_gpu_memory_utilization"):
                        serve._build_services()
            os.environ.pop(env_name)

    def test_small_model_tp1_preserves_existing_cli(self):
        for prefix in ("embedding", "summary"):
            serve.model_conf[f"{prefix}_tensor_parallel_size"] = 1
        table = serve._build_services()
        for prefix in ("embedding", "summary"):
            with self.subTest(service=prefix):
                self.assertNotIn("--tensor-parallel-size", table[prefix].argv)
                self.assertNotIn("--distributed-executor-backend", table[prefix].argv)
                self.assertNotIn("VLLM_WORKER_MULTIPROC_METHOD", table[prefix].env)
                self.assertNotIn("--disable-custom-all-reduce", table[prefix].argv)

    def test_invalid_small_model_parallel_sizes_are_rejected(self):
        for prefix in ("embedding", "summary"):
            key = f"{prefix}_tensor_parallel_size"
            for value in (0, -1, True, False, 1.5, "2", None):
                with self.subTest(service=prefix, value=value):
                    serve.model_conf[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        serve._build_services()
            serve.model_conf.pop(key)

    def test_small_model_tp2_rejects_insufficient_visible_devices(self):
        for prefix in ("embedding", "summary"):
            tp_key = f"{prefix}_tensor_parallel_size"
            device_key = f"{prefix}_cuda_visible_devices"
            serve.model_conf[tp_key] = 2
            serve.model_conf[device_key] = "0"
            with self.subTest(service=prefix), self.assertRaisesRegex(ValueError, device_key):
                serve._build_services()
            serve.model_conf.pop(tp_key)
            serve.model_conf.pop(device_key)

    def test_tp2_nf4_uses_checked_inflight_loader_and_native_fused_sharding(self):
        serve.model_conf.update({
            "chat_quantization": "nf4", "chat_tensor_parallel_size": 2,
            "chat_cuda_visible_devices": "0,1",
        })
        chat = serve._build_services()["chat"]
        quant = json.loads(self.argument(chat, "--hf-overrides"))["quantization_config"]
        self.assertEqual(quant["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(quant["bnb_4bit_use_double_quant"])
        self.assertEqual(quant["llm_int8_skip_modules"],
                         ["visual", "mtp", "lm_head", "embed_tokens", "in_proj_qkvz"])
        self.assertEqual(json.loads(self.argument(chat, "--model-loader-extra-config")),
                         {"maomi_inflight_nf4": True})
        self.assertEqual(chat.env["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(json.loads(self.argument(chat, "--lora-modules"))["name"], "education-v2")

    def test_tp1_nf4_keeps_original_quantization_scope(self):
        serve.model_conf["chat_quantization"] = "nf4"
        chat = serve._build_services()["chat"]
        quant = json.loads(self.argument(chat, "--hf-overrides"))["quantization_config"]
        self.assertNotIn("in_proj_qkvz", quant["llm_int8_skip_modules"])
        self.assertNotIn("--model-loader-extra-config", chat.argv)

    def test_tp_defaults_preserve_single_card_and_inherited_devices(self):
        serve.model_conf.pop("chat_tensor_parallel_size")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        table = serve._build_services()
        chat = table["chat"]
        self.assertEqual(self.argument(chat, "--tensor-parallel-size"), "1")
        self.assertNotIn("--distributed-executor-backend", chat.argv)
        for key in ("chat", "embedding", "summary"):
            self.assertNotIn("CUDA_VISIBLE_DEVICES", table[key].env)
        with tempfile.TemporaryDirectory() as temp, patch.object(
            serve, "LOG_DIR", Path(temp)
        ), patch.object(serve.subprocess, "Popen", return_value=Mock(pid=123)) as popen:
            serve._start_one(chat, detach=True)
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")

    def test_tp_explicit_devices_override_parent_only_for_selected_service(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        serve.model_conf["chat_tensor_parallel_size"] = 2
        serve.model_conf["chat_cuda_visible_devices"] = " 0 , 1 "
        chat = serve._build_services()["chat"]
        with tempfile.TemporaryDirectory() as temp, patch.object(
            serve, "LOG_DIR", Path(temp)
        ), patch.object(serve.subprocess, "Popen", return_value=Mock(pid=123)) as popen:
            serve._start_one(chat, detach=True)
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_tp_invalid_parallel_sizes_are_rejected(self):
        for value in (0, -1, True, False, 1.5, "2", None):
            with self.subTest(value=value):
                serve.model_conf["chat_tensor_parallel_size"] = value
                with self.assertRaisesRegex(ValueError, "chat_tensor_parallel_size"):
                    serve._build_services()

    def test_tp_invalid_chat_device_masks_are_rejected(self):
        serve.model_conf["chat_tensor_parallel_size"] = 2
        for value in ("", " ", "0,,1", "0,0", "0", None):
            with self.subTest(value=value):
                serve.model_conf["chat_cuda_visible_devices"] = value
                with self.assertRaisesRegex(ValueError, "chat_cuda_visible_devices"):
                    serve._build_services()

    def test_tp_invalid_small_model_device_masks_are_rejected(self):
        for key in ("embedding_cuda_visible_devices", "summary_cuda_visible_devices"):
            for value in ("", "0,0"):
                with self.subTest(key=key, value=value):
                    serve.model_conf[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        serve._build_services()
            serve.model_conf.pop(key)

    def test_tp_status_shows_memory_for_both_gpus(self):
        table = serve._build_services()
        result = Mock(returncode=0, stdout="0, 36000, 81920\n1, 20000, 81920\n")
        output = io.StringIO()
        with patch.object(serve, "services", return_value=table), patch.object(
            serve, "_running_pid", return_value=None
        ), patch.object(serve.subprocess, "run", return_value=result) as run, contextlib.redirect_stdout(output):
            self.assertEqual(serve.show_status(), 0)
        self.assertIn("GPU 0：36000 / 81920 MiB", output.getvalue())
        self.assertIn("GPU 1：20000 / 81920 MiB", output.getvalue())
        self.assertIn("--query-gpu=index,memory.used,memory.total", run.call_args.args[0])

    def test_tp_status_retains_single_gpu_support(self):
        result = Mock(returncode=0, stdout="0, 28000, 81920\n")
        with patch.object(serve.subprocess, "run", return_value=result):
            self.assertEqual(serve._gpu_summary(), "GPU 0：28000 / 81920 MiB")


if __name__ == "__main__":
    unittest.main()
