# 文件用途：用已安装的 Transformers API 回归验证训练参数、预热比例与开训前兼容检查。
# 调用关系：unittest 调用 train_education 的参数预检、dry-run 与正式训练入口；真实参数对象仅使用 CPU。
# 修改易踩坑：缺少训练依赖时跳过；不读取27B权重、不初始化CUDA、不创建正式训练输出目录。
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import train_education
from utils.education_training import load_config


# 模块初始化 → importlib.util.find_spec：只探测依赖是否存在，普通项目环境无需安装训练依赖。
HAS_TRAINING_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "peft", "accelerate")
)


@unittest.skipUnless(HAS_TRAINING_DEPENDENCIES, "需要独立训练环境中的 Transformers/PyTorch/PEFT/Accelerate")
class EducationTrainingApiTests(unittest.TestCase):
    # 上游：unittest 在每项测试前调用；下游：只读取 YAML，并给本次测试分配尚不存在的临时输出路径。
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = load_config(train_education.DEFAULT_CONFIG)
        self.output = Path(self.temporary.name) / "new-training-output"
        self.config["output_dir"] = str(self.output)

    # 上游：unittest；下游：预检 → 真实 TrainingArguments → get_warmup_steps；验证实际库接受参数和预热含义。
    def test_installed_arguments_accept_settings_and_preserve_warmup_ratio(self):
        import torch
        from transformers import TrainingArguments

        for ratio, expected in ((0.05, 16), (0.0, 0)):
            with self.subTest(warmup_ratio=ratio):
                self.config["warmup_ratio"] = ratio
                # 测试 → build_training_arguments：取得正式训练使用的同一组参数，但此处不构造参数对象。
                kwargs = train_education.build_training_arguments(self.config, validate_only=True)
                # 测试 → 真实 TrainingArguments：只覆盖设备和计算精度，使 API 回归完全运行于 CPU。
                kwargs.update(use_cpu=True, bf16=False)
                with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("测试不能初始化CUDA")):
                    arguments = TrainingArguments(**kwargs)
                    # 测试 → get_warmup_steps：313次更新的5%应向上取整为16次，0比例则不预热。
                    self.assertEqual(arguments.get_warmup_steps(313), expected)
        self.assertFalse(self.output.exists())

    # 上游：unittest；下游：参数预检 → inspect.signature.bind；禁止进入真实参数对象的后处理。
    def test_validation_does_not_construct_arguments_or_create_output(self):
        from transformers import TrainingArguments

        with patch.object(TrainingArguments, "__post_init__", side_effect=AssertionError("预检不能构造TrainingArguments")):
            # 测试 → build_training_arguments(validate_only=True)：只验证关键字参数签名并返回普通字典。
            kwargs = train_education.build_training_arguments(self.config, validate_only=True)
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(kwargs["output_dir"], str(self.output))
        self.assertFalse(self.output.exists())

    # 上游：unittest；下游：run_training → 参数预检；模拟库升级后拒绝参数，必须在模型加载和目录创建前报错。
    def test_incompatible_signature_fails_before_model_load_and_output_creation(self):
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText

        class IncompatibleTrainingArguments:
            # 这个替代类故意只接受一个参数；预检应从签名发现不兼容，不应真正构造它。
            def __init__(self, output_dir):
                raise AssertionError("不兼容参数不能进入构造函数")

        with patch.object(transformers, "TrainingArguments", IncompatibleTrainingArguments), \
             patch.object(train_education, "ensure_gpu_idle"), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(AutoModelForImageTextToText, "from_pretrained", side_effect=AssertionError("不能提前加载27B模型")) as load_model:
            with self.assertRaises((TypeError, ValueError, RuntimeError)):
                # 测试 → run_training：空样本即可触发参数检查；模型和真正训练循环均不得执行。
                train_education.run_training(self.config, Mock(), [], [], {})
        load_model.assert_not_called()
        self.assertFalse(self.output.exists())

    # 上游：unittest；下游：main(--dry-run) → 参数预检；只替换数据与分词，验证 CLI 真正覆盖实际训练API。
    def test_dry_run_checks_training_arguments_without_model_or_argument_construction(self):
        from transformers import AutoTokenizer, TrainingArguments

        feature = {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}
        tokenizer = Mock(pad_token_id=0)
        with patch.object(train_education, "read_settings", return_value=self.config), \
             patch.object(train_education, "read_data", return_value=([{}], [{}], {"example_data": False})), \
             patch.object(train_education, "validate_base_model", return_value={"text_config": {"max_position_embeddings": 8192}}), \
             patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer), \
             patch.object(train_education, "encode_record", return_value=feature), \
             patch.object(train_education, "build_training_arguments", wraps=train_education.build_training_arguments) as validate_arguments, \
             patch.object(TrainingArguments, "__post_init__", side_effect=AssertionError("dry-run不能构造TrainingArguments")), \
             patch.object(train_education, "run_training") as run_training, \
             contextlib.redirect_stdout(io.StringIO()):
            # 测试 → main：经过真实 CLI 分支和预检函数，避免只测一个未被入口使用的辅助函数。
            code = train_education.main(["--dry-run"])
        self.assertEqual(code, 0)
        validate_arguments.assert_called_once_with(self.config, validate_only=True)
        run_training.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    # Python 直接运行本文件 → unittest.main；仅执行 CPU 回归检查，不启动教育模型训练。
    unittest.main()
