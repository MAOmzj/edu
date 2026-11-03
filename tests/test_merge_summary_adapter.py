# 文件用途：验证离线 LoRA 合并的目录保护、依赖延迟加载和模型导出参数。
# 调用关系：unittest 调用 scripts/merge_summary_adapter.py，模型依赖全部使用 mock。
# 修改易踩坑：测试只操作临时目录，不实际加载模型、访问网络或读取学生数据库。
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import merge_summary_adapter as merger


class MergeSummaryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base_path = self.root / "base"
        self.adapter_path = self.root / "adapter"
        self.output_path = self.root / "merged"
        self.base_path.mkdir()
        self.adapter_path.mkdir()
        (self.base_path / "config.json").write_text("{}", encoding="utf-8")
        (self.adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.torch = Mock()
        self.torch.bfloat16 = object()
        self.torch.cuda.is_available.return_value = True
        self.model_loader = Mock()
        self.tokenizer_loader = Mock()
        self.adapter_loader = Mock()
        self.base = self.model_loader.from_pretrained.return_value
        self.tokenizer = self.tokenizer_loader.from_pretrained.return_value
        self.adapted = self.adapter_loader.from_pretrained.return_value
        self.merged = self.adapted.merge_and_unload.return_value
        dependencies_patch = patch.object(
            merger,
            "_load_dependencies",
            return_value=(self.torch, self.model_loader, self.tokenizer_loader, self.adapter_loader),
        )
        self.dependencies = dependencies_patch.start()
        self.addCleanup(dependencies_patch.stop)

    def merge(self, *, device="cpu"):
        return merger.merge_adapter(self.base_path, self.adapter_path, self.output_path, device=device)

    def test_required_cli_inputs_and_help_do_not_load_model_dependencies(self):
        for arguments, expected_code in (([], 2), (["--help"], 0)):
            with self.subTest(arguments=arguments), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    merger.main(arguments)
                self.assertEqual(exc.exception.code, expected_code)
        self.dependencies.assert_not_called()

    def test_missing_local_input_is_rejected_before_dependencies(self):
        for base, adapter in (
            (self.root / "missing-base", self.adapter_path),
            (self.base_path, self.root / "missing-adapter"),
            ("Qwen/Qwen3-4B", self.adapter_path),
        ):
            with self.subTest(base=base, adapter=adapter), self.assertRaises(FileNotFoundError):
                merger.merge_adapter(base, adapter, self.output_path)
        self.dependencies.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_input_configuration_is_required_before_loading(self):
        (self.adapter_path / "adapter_config.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "adapter_config.json"):
            self.merge()
        self.dependencies.assert_not_called()

    def test_existing_output_cannot_overwrite_base_adapter_or_other_directory(self):
        self.output_path.mkdir()
        for output in (self.base_path, self.adapter_path, self.output_path):
            with self.subTest(output=output), self.assertRaises(FileExistsError):
                merger.merge_adapter(self.base_path, self.adapter_path, output)
        self.dependencies.assert_not_called()
        self.assertEqual((self.base_path / "config.json").read_text(), "{}")

    def test_dangling_output_symlink_is_rejected(self):
        self.output_path.symlink_to(self.root / "missing-target", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.merge()
        self.dependencies.assert_not_called()

    def test_cpu_merge_loads_only_local_bf16_models_and_saves_expected_paths(self):
        result = self.merge()
        self.assertEqual(result, self.output_path)
        self.model_loader.from_pretrained.assert_called_once_with(
            str(self.base_path), torch_dtype=self.torch.bfloat16, device_map="cpu",
            local_files_only=True, trust_remote_code=False,
        )
        self.tokenizer_loader.from_pretrained.assert_called_once_with(
            str(self.base_path), local_files_only=True, trust_remote_code=False,
        )
        self.adapter_loader.from_pretrained.assert_called_once_with(
            self.base, str(self.adapter_path), is_trainable=False,
            local_files_only=True, trust_remote_code=False,
        )
        self.adapted.merge_and_unload.assert_called_once_with(safe_merge=True)
        self.merged.save_pretrained.assert_called_once_with(str(self.output_path), safe_serialization=True)
        self.tokenizer.save_pretrained.assert_called_once_with(str(self.output_path))
        self.assertIs(self.merged.generation_config, self.base.generation_config)
        self.base.generation_config.save_pretrained.assert_called_once_with(str(self.output_path))
        self.torch.cuda.is_available.assert_not_called()

    def test_base_generation_config_is_copied_without_changing_sampling(self):
        source = '{"temperature":0.6,"top_p":0.95,"top_k":20,"do_sample":true}\n'
        (self.base_path / "generation_config.json").write_text(source, encoding="utf-8")
        self.merge()
        target = self.output_path / "generation_config.json"
        self.assertEqual(target.read_text(encoding="utf-8"), source)
        self.assertTrue(json.loads(target.read_text())["do_sample"])
        self.base.generation_config.save_pretrained.assert_not_called()

    def test_cuda_request_checks_availability_and_selects_cuda(self):
        self.merge(device="cuda")
        self.torch.cuda.is_available.assert_called_once_with()
        self.assertEqual(self.model_loader.from_pretrained.call_args.kwargs["device_map"], "cuda")

    def test_unavailable_cuda_fails_before_loading_models(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.merge(device="cuda")
        self.model_loader.from_pretrained.assert_not_called()
        self.adapter_loader.from_pretrained.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_output_created_by_another_task_during_load_is_not_overwritten(self):
        def create_output(**kwargs):
            self.output_path.mkdir()
            (self.output_path / "other-task.txt").write_text("preserve", encoding="utf-8")
            return self.merged

        self.adapted.merge_and_unload.side_effect = create_output
        with self.assertRaises(FileExistsError):
            self.merge()
        self.merged.save_pretrained.assert_not_called()
        self.assertEqual((self.output_path / "other-task.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
