# 文件用途：用本地 Qwen3 BF16 基座执行摘要 LoRA SFT，或只验证数据与 token 监督边界。
# 调用关系：开发者运行 CLI；调用 summary_training.py 和 Transformers/PEFT，不调用线上服务。
# 修改易踩坑：训练集按会话划分，只有核验后的目标摘要参与 loss；输出必须是新目录，模型仅本地加载。
"""摘要 LoRA 微调：旧摘要 + 新增对话 -> 核验后的新摘要。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.summary_training import encode_example, load_summary_records, pad_features, validate_split

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/root/MAOMI/models/Qwen3-4B"))
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="尚不存在的训练输出目录")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="校验数据并用本地 tokenizer 检查 token，不加载模型")
    parser.add_argument("--validate-data-only", action="store_true", help="只校验 JSONL 与分割，不需要训练依赖或模型")
    args = parser.parse_args(argv)
    for key in ("max_seq_length", "batch_size", "gradient_accumulation_steps", "lora_r", "lora_alpha"):
        if getattr(args, key) < 1:
            parser.error(f"{key} 必须大于 0")
    for key in ("epochs", "learning_rate"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} 必须是有限正数")
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        parser.error("lora_dropout 必须在 [0,1) 范围")
    if not (args.dry_run or args.validate_data_only) and args.output_dir is None:
        parser.error("训练需要 --output-dir；数据检查无需输出目录")
    return args


def summary_budgets():
    import yaml
    config = yaml.safe_load((PROJECT_ROOT / "config/app.yml").read_text(encoding="utf-8"))["context"]
    max_summary_chars = int(config["max_summary_chars"])
    input_chars = max(int(config["summary_trigger_chars"]) * 2, max_summary_chars)
    return max_summary_chars, input_chars


def load_tokenizer(model_path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("请先在独立环境安装 requirements-summary-training.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("本地 tokenizer 缺少 EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def feature_stats(features):
    return {
        "examples": len(features),
        "max_sequence_tokens": max(len(f["input_ids"]) for f in features),
        "supervised_tokens": sum(sum(label != -100 for label in f["labels"]) for f in features),
    }


def run_training(args, tokenizer, train_features, validation_features, report):
    """模型训练只在预检完成后进行；基座参数冻结，保存 best eval_loss 的 adapter。"""
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("请先在独立环境安装 requirements-summary-training.txt") from exc
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("正式训练需要支持 BF16 的 CUDA GPU；数据检查请使用 --dry-run")
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    config = AutoConfig.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3" or getattr(config, "quantization_config", None):
        raise ValueError("训练入口需要未量化 Qwen3 基座，请勿传入主模型 AWQ 或其它架构")
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError("max-seq-length 超过基座支持的上下文窗口")
    set_seed(args.seed)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    )
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=LORA_TARGETS, bias="none",
    ))
    model.print_trainable_parameters()
    output.mkdir(parents=True, exist_ok=False)
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=1, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, report_to=[], seed=args.seed,
        remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0,
    )
    def collate(features):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_features(features, tokenizer.pad_token_id).items()}
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_features,
        eval_dataset=validation_features, data_collator=collate, processing_class=tokenizer,
    )
    report["status"] = "training"
    report_path = output / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    trained = trainer.train()
    metrics = trainer.evaluate()
    adapter = output / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    report.update(status="completed", adapter_path=str(adapter.resolve()), train_metrics=trained.metrics, validation_metrics=metrics)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"LoRA Adapter：{adapter}")


def main(argv=None):
    args = parse_args(argv)
    try:
        max_chars, input_chars = summary_budgets()
        train = load_summary_records(args.train_file, max_chars)
        validation = load_summary_records(args.validation_file, max_chars)
        validate_split(train, validation, max_summary_chars=max_chars, summary_input_chars=input_chars)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": str(args.model_path.resolve()),
            "train_file_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
            "validation_file_sha256": hashlib.sha256(args.validation_file.read_bytes()).hexdigest(),
            "train_examples": len(train), "validation_examples": len(validation),
            "max_summary_chars": max_chars, "summary_input_chars": input_chars,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        if args.validate_data_only:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if not args.model_path.is_dir():
            raise FileNotFoundError("model-path 必须是已存在的本地模型目录")
        tokenizer = load_tokenizer(args.model_path)
        def encode(records):
            return [encode_example(r, tokenizer, args.max_seq_length, max_summary_chars=max_chars, summary_input_chars=input_chars) for r in records]
        train_features, validation_features = encode(train), encode(validation)
        report.update(train_tokens=feature_stats(train_features), validation_tokens=feature_stats(validation_features))
        if args.dry_run:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        run_training(args, tokenizer, train_features, validation_features, report)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"摘要微调失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
