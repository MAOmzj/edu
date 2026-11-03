# 文件用途：检查教育SFT准备状态，并用完整Qwen3.8-27B执行纯文本LoRA或QLoRA训练。
# 调用关系：run_education_finetune.py或开发者运行CLI；调用education_training与Transformers/PEFT。
# 修改易踩坑：check和dry-run不加载GPU模型；正式训练需空闲GPU，且不会自动合并或更换线上模型。
# 调用树：run_education_finetune.py 启动本文件子进程 → __main__ → main。
#   main 依次调用 parse_args、read_settings、read_data、内部 encode、run_training；这些是执行顺序。
#   例如 read_settings → load_config 是嵌套调用；parse_args 不会调用 read_settings。
#   --check 提前走 preparation_report；--validate-data-only 在读完数据后返回；--dry-run 在编码后返回。
#   run_training 先调用 build_training_arguments 校验参数，再调用 load_trainable_model 加载基座和适配器。
#   单批 tensor 测试入口也调用 load_trainable_model，复用正式训练的 NF4 与语言层 LoRA 准备方式。
#   main 的 --dry-run 也调用 build_training_arguments，但只核对参数签名，不构造设备对象。
#   参数及模型准备完成后，run_training → Trainer.train/evaluate/save_model。
#   Trainer 的数据加载器 → collate → utils.education_training.pad_batch；训练循环由 Trainer 内部执行。
"""Qwen3.8-27B教育回答训练；--check仅离线检查准备状态。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

# 模块初始化 → pathlib.Path.resolve：计算当前项目根目录；sys.path.insert 让下游 utils 导入可用。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.education_training import (
    LANGUAGE_TARGET_PATTERN, configured_path, encode_record, file_sha256, load_config,
    load_records, pad_batch, reject_examples, validate_base_model, validate_split,
)

DEFAULT_CONFIG = PROJECT_ROOT / "config/education_finetune.yml"


# 上游：main(argv) → 本函数；输入为命令行字符串列表，None 表示读取终端参数。
# 下游：argparse 建立参数规则并解析；返回 Namespace，供 read_settings 和 main 的分支判断使用。
def parse_args(argv=None):
    # parse_args → argparse.ArgumentParser：建立解析器；description 使用本文件原有 docstring。
    parser = argparse.ArgumentParser(description=__doc__)
    # parse_args → parser.add_argument：注册配置、路径和方法选项；这里只登记规则，不启动训练。
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for option in ["model-path", "train-file", "validation-file", "output-dir"]:
        # parse_args → parser.add_argument（循环四次）：分别登记基座、训练集、验证集和输出目录的覆盖选项。
        parser.add_argument("--" + option, type=Path)
    # parse_args → parser.add_argument：登记 --method，只允许 qlora/lora 两个值。
    parser.add_argument("--method", choices=["qlora", "lora"])
    # parse_args → parser.add_mutually_exclusive_group：三个检查模式不能同时指定；返回参数组。
    modes = parser.add_mutually_exclusive_group()
    # parse_args → modes.add_argument：注册 --check，只检查本地文件和准备状态。
    modes.add_argument("--check", action="store_true", help="只读检查基座、环境与正式数据是否齐备")
    # parse_args → modes.add_argument：注册 --validate-data-only，允许只检查数据而不准备基座。
    modes.add_argument("--validate-data-only", action="store_true", help="仅检查JSONL与分割，可检查演示文件")
    # parse_args → modes.add_argument：注册 --dry-run，检查真实 tokenizer 编码及训练参数接口，不加载模型。
    modes.add_argument("--dry-run", action="store_true", help="检查本地tokenizer监督边界和训练参数接口，不加载模型或GPU")
    # parse_args → parser.parse_args：把 argv 解析成 Namespace；非法参数由 argparse 报错退出。
    return parser.parse_args(argv)


# 上游：main → 本函数；输入为 parse_args 的 Namespace。
# 下游：utils/education_training.py.load_config 读取并检查 YAML；命令行显式值覆盖同名配置，返回字典。
def read_settings(args):
    # read_settings → utils.education_training.load_config(args.config)：得到已校验的训练配置字典。
    config = load_config(args.config)
    for key in ["model_path", "train_file", "validation_file", "output_dir", "method"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value)
    return config


# 上游：main 或 preparation_report → 本函数；输入为配置及是否允许演示数据的开关。
# 下游：education_training 的路径、读取、分割校验、演示数据检查和哈希函数。
# 返回：(训练记录列表，验证记录列表，包含数量/路径/哈希的报告字典)；异常向上交给调用者。
def read_data(config, *, examples_allowed=False):
    # read_data → utils.education_training.configured_path(config, "train_file")：返回训练集 Path。
    train_path = configured_path(config, "train_file")
    # read_data → utils.education_training.configured_path(config, "validation_file")：返回验证集 Path。
    validation_path = configured_path(config, "validation_file")
    # read_data → utils.education_training.load_records（分别传训练/验证路径）：返回两组 JSONL 记录。
    train, validation = load_records(train_path), load_records(validation_path)
    # read_data → utils.education_training.validate_split：传两组记录和系统指令，检查重复与分组隔离；无返回值。
    validate_split(train, validation, config["instruction"])
    if not examples_allowed:
        # read_data → utils.education_training.reject_examples：正式路径拒绝演示文件/记录；无返回值，违规会抛异常。
        reject_examples([train_path, validation_path], train + validation)
    report = {
        "train_examples": len(train), "validation_examples": len(validation),
        "train_file": str(train_path), "validation_file": str(validation_path),
        # read_data → utils.education_training.file_sha256（两次）：读取两份文件，返回可追溯内容的哈希字符串。
        "train_sha256": file_sha256(train_path), "validation_sha256": file_sha256(validation_path),
        # read_data → collections.Counter：统计审核状态出现次数，转为普通字典放入报告；不改变数据内容。
        "review_status_counts": dict(Counter(str(r.get("review_status", "unspecified")) for r in train + validation)),
        "example_data": any(r.get("dataset_kind") == "example" for r in train + validation)
            or any(".example." in p.name for p in [train_path, validation_path]),
    }
    return train, validation, report


# 上游：本文件 main 的 --check 分支，或 run_education_finetune.main 在启动训练/编码子进程前调用。
# 输入：配置字典。
# 下游：路径解析、本地文件检查、validate_base_model、read_data；返回准备报告，不加载模型或占用 GPU。
def preparation_report(config):
    issues = []
    # preparation_report → utils.education_training.configured_path：依次解析五个配置路径，返回 Path 字典。
    paths = {key: configured_path(config, key) for key in ["model_path", "training_python", "train_file", "validation_file", "output_dir"]}
    for key in ["training_python", "train_file", "validation_file"]:
        # preparation_report → Path.is_file：只读检查环境解释器和两个数据文件是否存在。
        if not paths[key].is_file():
            issues.append(f"缺少{key}：{paths[key]}")
    try:
        # preparation_report → utils.education_training.validate_base_model：检查完整基座元数据/权重；此处只取成功或异常。
        validate_base_model(paths["model_path"])
    except (OSError, ValueError, TypeError) as exc:
        issues.append(f"完整未量化基座尚未就绪：{exc}")
    data = {}
    # preparation_report → Path.is_file（两次）：只有训练/验证文件都存在，才继续调用 read_data 做内容校验。
    if paths["train_file"].is_file() and paths["validation_file"].is_file():
        try:
            # preparation_report → 本文件.read_data(config)：严格检查正式数据，只保留第三项报告。
            _, _, data = read_data(config)
        except (OSError, ValueError) as exc:
            issues.append(f"正式数据尚未就绪：{exc}")
    # preparation_report → Path.exists/is_symlink：已有输出目录会阻止正式训练，避免覆盖旧适配器。
    if paths["output_dir"].exists() or paths["output_dir"].is_symlink():
        issues.append(f"输出目录已存在，请配置新目录：{paths['output_dir']}")
    return {
        "status": "ready_for_dry_run" if not issues else "missing_prerequisites",
        "paths": {key: str(value) for key, value in paths.items()},
        "method": config["method"], "issues": issues, "data": data,
        "model_loaded": False, "gpu_used": False,
        "note": "check只验证本地文件与数据，不验证CUDA训练兼容性、不下载权重、不启动训练",
    }


# 上游：run_training 或单批 tensor 测试入口在创建 CUDA 上下文之前 → 本函数；无参数。
# 下游：subprocess.run 执行只读 nvidia-smi；无返回值，发现占用或命令失败时抛异常，不停止任何进程。
def ensure_gpu_idle():
    # ensure_gpu_idle → subprocess.run → 系统 nvidia-smi：读取计算进程 PID，返回 CompletedProcess。
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    if pids:
        raise RuntimeError(f"GPU仍被进程{pids}占用；训练前请自行用scripts/serve.py stop all停止项目模型或处理其它任务")


# 上游：load_trainable_model 在加载基座后 → 本函数；输入为 Transformers 模型对象。
# 下游：model.named_modules 枚举模块，re.fullmatch 按项目规则筛选；返回要插入 LoRA 的完整模块名列表。
def discover_targets(model):
    # discover_targets → model.named_modules / re.fullmatch：筛出语言层投影，并在下方核对 64 层和两类注意力。
    targets = [name for name, _ in model.named_modules() if re.fullmatch(LANGUAGE_TARGET_PATTERN, name)]
    layers = {int(name.split(".layers.", 1)[1].split(".", 1)[0]) for name in targets}
    if layers != set(range(64)):
        raise ValueError("LoRA目标未覆盖预期64个语言层，拒绝按错误架构训练")
    if not any(".linear_attn." in name for name in targets) or not any(".self_attn." in name for name in targets):
        raise ValueError("未识别混合注意力投影模块")
    return targets


# 上游：main 编码训练/验证数据后各调用一次；输入为 encode_record 生成的特征字典列表。
# 下游：只做 Python 聚合，无模型调用；返回样本数、最长序列和 labels 中非 -100 的监督 token 总数。
def token_stats(features):
    return {"examples": len(features), "max_sequence_tokens": max(len(f["input_ids"]) for f in features),
            "supervised_tokens": sum(sum(token != -100 for token in f["labels"]) for f in features)}


# 上游：run_training 在加载27B权重前调用；main 的 --dry-run 分支只检查接口。
# 输入：训练配置，以及是否只做接口检查的 validate_only；返回：参数字典或 TrainingArguments 对象。
# 下游：configured_path 解析输出目录 → inspect 检查已安装接口 → 正式训练时才构造 TrainingArguments。
def build_training_arguments(config, *, validate_only=False):
    from transformers import TrainingArguments
    # build_training_arguments → configured_path：取得结果目录；这里只解析路径，不创建目录。
    output = configured_path(config, "output_dir")
    options = {
        "output_dir": str(output), "num_train_epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "per_device_train_batch_size": config["batch_size"], "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config["gradient_accumulation_steps"], "bf16": True,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch", "lr_scheduler_type": "cosine",
        # Transformers 5.17 用 warmup_steps 的 [0,1) 浮点值表示比例；配置中的0.05仍表示前5%预热。
        # 这里保留项目配置名 warmup_ratio，但不再向已移除此参数的库传入同名关键字。
        "warmup_steps": config["warmup_ratio"],
        "logging_steps": 1, "eval_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "prediction_loss_only": True, "report_to": [], "seed": config["seed"],
        "remove_unused_columns": False, "label_names": ["labels"], "dataloader_num_workers": 0,
    }
    try:
        # build_training_arguments → inspect.signature → Signature.bind：只核对参数名与签名，不实例化、不初始化GPU。
        # 不支持的关键字会在这里报错，使 dry-run 能提前发现本次 warmup_ratio 一类的接口问题。
        inspect.signature(TrainingArguments).bind(**options)
    except TypeError as exc:
        raise RuntimeError(f"当前Transformers训练参数接口不兼容：{exc}") from exc
    if validate_only:
        return options
    # build_training_arguments → transformers.TrainingArguments：正式分支做完整参数校验，返回 Trainer 所需对象。
    # 构造发生在 from_pretrained 前，接口/参数错误不会等加载完27B后才暴露，也不会留下空输出目录。
    return TrainingArguments(**options)


# 上游：run_training 或单批 tensor 测试入口 → 本函数；调用方须先调用 ensure_gpu_idle，再初始化 CUDA。
# 输入：已校验的训练配置；返回 (可训练 PEFT 模型, 模型信息字典)，本函数不训练、不保存文件。
# 下游：CUDA/BF16 检查 → set_seed → 本地 NF4/基座加载 → discover_targets → PEFT 准备与语言层参数检查。
# 此处不调用 ensure_gpu_idle：调用方创建的 CUDA 上下文也会出现在 nvidia-smi 中，不能误判为其它任务。
def load_trainable_model(config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, set_seed
    # load_trainable_model → torch.cuda.is_available/is_bf16_supported：独立确认 CUDA/BF16，避免其它入口绕过检查。
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("训练需要支持BF16的CUDA GPU")
    # load_trainable_model → transformers.set_seed：固定 Python/NumPy/PyTorch 等随机状态；不保证 GPU 运算完全一致。
    set_seed(config["seed"])
    options = {"local_files_only": True, "trust_remote_code": False, "dtype": torch.bfloat16,
               "attn_implementation": "sdpa", "device_map": {"": 0}}
    if config["method"] == "qlora":
        # load_trainable_model → transformers.BitsAndBytesConfig：创建 NF4 4bit 配置，返回配置对象；尚未加载权重。
        # 完整本地基座由下一步加载时量化；视觉、MTP、输出头和词嵌入列入跳过项，计算类型为 BF16。
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["model.visual", "mtp", "lm_head", "embed_tokens"],
        )
    # load_trainable_model → utils.education_training.configured_path(config, "model_path")：给下一调用提供基座目录。
    # load_trainable_model → transformers.AutoModelForImageTextToText.from_pretrained：本地权重加载到 GPU 0，返回模型。
    # 使用图文模型加载类是为匹配基座结构；本训练输入只含文字，后续只放开语言层 LoRA。
    model = AutoModelForImageTextToText.from_pretrained(str(configured_path(config, "model_path")), **options)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if config["method"] == "qlora":
        from bitsandbytes.nn import Linear4bit
        # load_trainable_model → model.named_modules：检查视觉模块是否错误变成 bitsandbytes.Linear4bit，异常则中止。
        if any(isinstance(module, Linear4bit) for name, module in model.named_modules()
               if name.startswith("model.visual.")):
            raise ValueError("视觉模块被意外量化，请核验NF4跳过模块的精确路径")
    # load_trainable_model → 本文件.discover_targets(model)：返回经架构校验的 LoRA 目标模块名列表。
    targets = discover_targets(model)
    if config["method"] == "qlora":
        # load_trainable_model → peft.prepare_model_for_kbit_training：冻结基座并准备低比特反向传播，返回处理后的模型。
        # 梯度检查点会在反向传播时重算部分前向结果，以减少显存；use_reentrant=False 指定检查点实现。
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    # load_trainable_model → peft.LoraConfig：用目标层、r、alpha、dropout 创建适配器配置；这里未写入磁盘。
    # load_trainable_model → peft.get_peft_model(model, 上述配置)：插入 LoRA 并返回 PEFT 模型，基座参数保持冻结。
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=targets, r=config["lora_r"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], bias="none",
    ))
    # load_trainable_model → model.named_parameters：收集 requires_grad=True 参数，核验只训练语言层 LoRA。
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any(".lora_" not in name or ".language_model.layers." not in name for name, _ in trainable):
        raise ValueError("可训练参数超出语言层LoRA范围，视觉与MTP必须冻结")
    # load_trainable_model → Tensor.numel：逐个统计可训练张量元素数；返回信息交给正式训练报告或单批测试报告。
    model_info = {"target_modules": targets, "trainable_parameters": sum(p.numel() for _, p in trainable),
                  "torch_version": torch.__version__}
    return model, model_info


# 上游：main 通过全部检查且未选择提前返回模式时 → 本函数。
# 输入：配置、tokenizer、已编码的训练/验证特征、可更新的报告；返回 None，结果写入新输出目录。
# 下游：空闲/设备/参数检查 → load_trainable_model → Trainer 训练/验证/保存；不合并、不部署。
def run_training(config, tokenizer, train, validation, report):
    # run_training → 本文件.ensure_gpu_idle()：须在本进程创建 CUDA 上下文前确认空闲；不停止任何进程。
    ensure_gpu_idle()
    import torch
    from transformers import Trainer
    # run_training → torch.cuda.is_available/is_bf16_supported：加载基座前确认 CUDA 和 BF16 支持。
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("训练需要支持BF16的CUDA GPU")
    # run_training → utils.education_training.configured_path：解析结果目录；随后拒绝覆盖已有路径。
    output = configured_path(config, "output_dir")
    # run_training → Path.exists/is_symlink：复查输出路径，避免启动前后有其它任务创建同名目录。
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    # run_training → build_training_arguments(config)：先校验并构造训练参数，再加载耗时的27B基座。
    arguments = build_training_arguments(config)
    # run_training → load_trainable_model(config)：复用 NF4、随机种子和语言层 LoRA 准备，返回模型及信息字典。
    model, model_info = load_trainable_model(config)
    # run_training → dict.update：将可训练参数量等信息合入当前报告，并标为 training；尚未写入磁盘。
    report.update(**model_info, status="training")
    # run_training → Path.mkdir：此时才创建全新训练输出目录；存在则失败，防止覆盖。
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "training_report.json"
    # 上游：run_training 在训练开始、完成、失败时调用；输入来自闭包 report/report_path，无显式参数。
    # 下游：json.dumps 序列化 → Path.write_text 写训练报告；无返回值，不保存模型权重。
    def save_report():
        # save_report → json.dumps → Path.write_text：把当前 report 写入 training_report.json。
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # run_training → 本函数内的 save_report()：先持久化 status=training 的报告。
    save_report()
    # 上游：Trainer 创建的数据加载器在取每个批次时回调；run_training 只注册该函数，不直接调用它。
    # 输入：该批若干已编码样本；下游：pad_batch → torch.tensor；返回可供模型 forward 使用的张量字典。
    def collate(features):
        # collate → utils.education_training.pad_batch(features, pad_token_id)：补齐本批长度，返回含三个二维列表的字典。
        # collate → torch.tensor（每个字段一次）：将 input_ids/attention_mask/labels 转为 long 张量并返回。
        return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_batch(features, tokenizer.pad_token_id).items()}
    try:
        # run_training → transformers.Trainer：传入模型、数据及 collate 回调，返回训练管理器；此处只初始化。
        # 初始化也放进异常记录范围，失败时由下面的 except 更新 training_report，而非遗留 training 状态。
        trainer = Trainer(model=model, args=arguments, train_dataset=train, eval_dataset=validation,
                          data_collator=collate, processing_class=tokenizer)
        # run_training → Trainer.train()：开始训练，返回 TrainOutput（含训练指标 trained.metrics）。
        # 下列调用发生在 Trainer/Accelerate 内部，并非本文件手写的显式调用：
        # 数据加载器 → collate；Trainer → model(**batch) → 模型 forward 根据 labels 计算 loss；
        # Trainer/Accelerate → backward 计算梯度；累积到指定批次数后 → optimizer.step 更新 LoRA、scheduler.step 调学习率。
        # Trainer 还负责清梯度、按 epoch 验证/保存检查点，并依据 eval_loss 在结束时加载最佳检查点。
        trained = trainer.train()
        # run_training → Trainer.evaluate()：使用验证集和同一 collate 计算 loss，返回指标字典；不会更新参数。
        metrics = trainer.evaluate()
        adapter = output / "adapter"
        # run_training → Trainer.save_model(adapter)：通过当前 PEFT 模型保存适配器权重/配置，不生成合并后的 27B 权重。
        trainer.save_model(str(adapter))
        # run_training → tokenizer.save_pretrained(adapter)：保存分词配置，便于以后用同样方式编码输入。
        tokenizer.save_pretrained(str(adapter))
        # run_training → Trainer.save_state()：保存 Trainer 的训练进度状态；按 epoch 的检查点另含断点续训所需内容。
        trainer.save_state()
        # run_training → datetime.now(timezone.utc)：取得完成时间；将训练/验证指标和适配器位置写回 report。
        report.update(status="completed", adapter_path=str(adapter), train_metrics=trained.metrics,
                      validation_metrics=metrics, finished_at_utc=datetime.now(timezone.utc).isoformat())
        # run_training → 本函数内的 save_report()：把 completed 状态及指标写入报告文件。
        save_report()
        print(f"教育LoRA适配器已保存：{adapter}；未合并、未启用线上模型")
    except BaseException:
        report["status"] = "interrupted_or_failed"
        # run_training 的异常分支 → save_report()：尽量记录失败/中断，再将原异常继续抛给上游。
        save_report()
        raise


# 上游：本文件 __main__、run_education_finetune.main 的 --check 分支，或测试代码直接调用本函数。
# 输入：命令行参数列表 argv；None 表示使用终端参数。
# 下游：依次完成参数/配置/数据/编码检查；只有正式分支调用 run_training。
# 返回：成功为 0，准备不足或捕获到训练异常为 1；__main__ 用这个整数作为进程退出码。
def main(argv=None):
    # main → 本文件.parse_args(argv)：取得参数对象；argparse 自身处理命令行语法错误。
    args = parse_args(argv)
    try:
        # main → 本文件.read_settings(args)：读取 YAML 并应用命令行覆盖，返回配置字典。
        config = read_settings(args)
        if args.check:
            # --check 分支：main → 本文件.preparation_report(config)，只查本地文件/数据，完全跳过 tokenizer 和 GPU。
            report = preparation_report(config)
            # main → json.dumps → print：准备报告以 JSON 输出到终端；存在 issues 时返回退出码 1。
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if not report["issues"] else 1
        # main → 本文件.read_data：返回训练/验证原始记录和数据报告；仅 validate-data-only 允许演示数据。
        train, validation, data = read_data(config, examples_allowed=args.validate_data_only)
        # main → datetime.now(timezone.utc)：给本次运行生成 UTC 时间，后续报告复用此字典。
        report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "settings": config,
                  "enable_thinking": False, "task": "primary_school_teaching_answers", **data}
        if args.validate_data_only:
            # --validate-data-only 分支：已完成数据结构/重复/分割检查；在基座检查、分词器加载之前返回。
            report.update(status="data_validated", usable_for_training=not data["example_data"], model_loaded=False, gpu_used=False)
            # main → json.dumps → print：输出只校验数据的报告，不写训练输出目录。
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        # main → utils.education_training.configured_path：取得 model_path，作为下一调用的参数。
        # main → utils.education_training.validate_base_model：检查完整未量化基座，返回本地模型配置字典，不加载权重。
        model_config = validate_base_model(configured_path(config, "model_path"))
        if config["max_seq_length"] > model_config["text_config"]["max_position_embeddings"]:
            raise ValueError("max_seq_length超过底座上下文窗口")
        from transformers import AutoTokenizer
        # main → utils.education_training.configured_path：返回本地基座目录。
        # main → transformers.AutoTokenizer.from_pretrained：只加载本地分词器文件，返回 tokenizer，不加载模型/GPU。
        tokenizer = AutoTokenizer.from_pretrained(str(configured_path(config, "model_path")), local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        # 上游：main 分别传训练/验证记录；输入为原始记录列表。
        # 下游：每条记录调用 education_training.encode_record；返回含 input_ids/attention_mask/labels 的列表。
        def encode(records):
            # encode → utils.education_training.encode_record(记录, tokenizer, 系统指令, 长度上限)：
            # 返回单条特征；问题/历史的 labels=-100，最后回答和 EOS 接受监督，越长/空监督等异常向 main 抛出。
            return [encode_record(r, tokenizer, config["instruction"], config["max_seq_length"]) for r in records]
        # main → 本函数内 encode(train) 和 encode(validation)：分别完成两个数据集的文字到 token 转换。
        train_features, validation_features = encode(train), encode(validation)
        # main → 本文件.token_stats（训练/验证各一次）：生成长度与监督 token 数；dict.update 合入 report。
        report.update(train_tokens=token_stats(train_features), validation_tokens=token_stats(validation_features))
        if args.dry_run:
            # main → build_training_arguments(validate_only=True)：核对当前训练库是否接受所有参数；不构造GPU参数对象。
            argument_options = build_training_arguments(config, validate_only=True)
            # --dry-run 分支：编码和参数签名检查通过后返回，不继续调用 run_training，也不加载 GPU 模型。
            report.update(status="token_check_passed", training_arguments_api_checked=True,
                          warmup_steps=argument_options["warmup_steps"], model_loaded=False, gpu_used=False)
            # main → json.dumps → print：输出编码检查报告到终端。
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        # 正式分支：main → 本文件.run_training(配置, tokenizer, 编码后的两组数据, 报告)。
        # 此调用才会加载 GPU 模型、训练并保存适配器；无训练模式开关时默认进入本分支。
        run_training(config, tokenizer, train_features, validation_features, report)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        # main → print(..., file=sys.stderr)：把捕获的文件/配置/运行错误写到错误输出，随后返回失败码 1。
        print(f"教育微调未完成：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Python 直接运行本文件（包括启动脚本启动的子进程）→ main()；SystemExit 将返回值交给操作系统。
    # 被其他文件 import 时 __name__ 不等于 __main__，因此不会自动开始训练。
    raise SystemExit(main())
