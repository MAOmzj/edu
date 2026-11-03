# 文件用途：把一条教学问答变成tensor，对真实27B执行一次LoRA参数更新，验证微调链路。
# 调用关系：main → encode_record构造监督 → load_trainable_model复用正式模型加载 → update_once → 保存测试适配器。
# 修改易踩坑：input_ids必须是二维整数；本脚本只做一次更新，输出独立目录，不能当作正式教学训练成果。
"""单tensor微调测试；加 --prepare-only 仅查看CPU张量，不加载27B。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_education import ensure_gpu_idle, load_trainable_model
from utils.education_training import configured_path, encode_record, load_config, validate_base_model


# 上游：main；输入为已经带LoRA的模型、同一设备上的tensor字典及学习率；返回单步更新指标。
# 下游：model前向 → loss.backward → clip_grad_norm_ → AdamW.step；不调用Trainer、不遍历2500条数据。
def update_once(model, batch, learning_rate):
    import torch
    # update_once → model.train：启用训练模式；只收集需要梯度的LoRA参数，基座由共享加载函数冻结。
    model.train()
    named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_trainable or any(".lora_" not in name for name, _ in named_trainable):
        raise ValueError("单步测试只允许更新LoRA参数")
    # update_once → torch.optim.AdamW：创建优化器；单步直接使用指定学习率，不使用预热调度器。
    # 若一次更新也使用warmup，第一个学习率可能为0；这里需要实际观察到参数改变。
    parameters = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    # update_once → torch.autocast：与正式Trainer的bf16=True一致，前向使用BF16混合精度，输入仍是long。
    with torch.autocast(device_type=batch["input_ids"].device.type, dtype=torch.bfloat16):
        # update_once → 模型forward：input_ids是输入，attention_mask标有效位置，labels指定目标回答。
        outputs = model(**batch)
        loss = outputs.loss
    if loss is None or not torch.isfinite(loss).item():
        raise ValueError("模型没有返回有限loss")
    # update_once → Tensor.backward：由loss反向计算LoRA参数的梯度，此时还没有更新权重。
    loss.backward()
    # 找到一项有非零梯度的参数，记录更新前的副本，用于证明optimizer.step确实改变了权重。
    selected = next(((name, parameter) for name, parameter in named_trainable
                     if parameter.grad is not None and torch.count_nonzero(parameter.grad).item()), None)
    if selected is None:
        raise ValueError("没有检测到非零LoRA梯度")
    name, parameter = selected
    before = parameter.detach().clone()
    # update_once → clip_grad_norm_：与正式Trainer默认上限一致；非有限梯度会在更新前报错。
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
    # update_once → AdamW.step：本文件只有这一次参数更新调用。
    optimizer.step()
    delta = (parameter.detach() - before).abs().max().item()
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("优化器执行后未检测到有效参数变化")
    metrics = {"loss": loss.detach().item(), "gradient_norm": grad_norm.item(),
               "optimizer_steps": 1, "changed_parameter": name, "parameter_max_change": delta}
    optimizer.zero_grad(set_to_none=True)
    return metrics


# 上游：main在准备、完成或失败时调用；输入为结果目录和报告字典；下游：json.dumps → Path.write_text。
def save_report(output, report):
    (output / "tensor_test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 上游：文件末尾main()；下游：参数解析 → 配置/分词器 → 编码/张量 → 可选真实27B单步更新。
# 返回：0表示准备或测试成功；异常保留traceback，若已创建测试目录则同时保存失败报告。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/education_finetune.yml")
    parser.add_argument("--question", default="2加3等于几？")
    parser.add_argument("--answer", default="2加3等于5。")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.question.strip() or not args.answer.strip():
        raise ValueError("问题和参考回答不能为空")
    # main → load_config/validate_base_model：读取正式配置并确认完整基座；不改变配置文件。
    config = load_config(args.config)
    validate_base_model(configured_path(config, "model_path"))
    import torch
    from transformers import AutoTokenizer
    # main → AutoTokenizer.from_pretrained：只读取本地分词器，此时还不加载27B权重。
    tokenizer = AutoTokenizer.from_pretrained(str(configured_path(config, "model_path")), local_files_only=True, trust_remote_code=False)
    sample = {"id": "tensor-test-one", "group_id": "tensor-test-only", "messages": [
        {"role": "user", "content": args.question}, {"role": "assistant", "content": args.answer},
    ]}
    # main → encode_record：完全复用正式训练的监督规则，系统指令/提问的labels为-100，回答及EOS有监督。
    feature = encode_record(sample, tokenizer, config["instruction"], config["max_seq_length"])
    # main → torch.tensor：外面的一对[]表示batch中只有1条样本；long是整数token ID，不是浮点随机数。
    batch = {key: torch.tensor([values], dtype=torch.long) for key, values in feature.items()}
    print("问题：", args.question, flush=True)
    print("参考回答：", args.answer, flush=True)
    for key, tensor in batch.items():
        print(f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, 前12项={tensor[0, :12].tolist()}", flush=True)
    print("labels最后12项：", batch["labels"][0, -12:].tolist(), flush=True)
    if args.prepare_only:
        print("张量准备通过；未加载27B、未反向传播、未更新参数。", flush=True)
        return 0
    # main → ensure_gpu_idle：在任何CUDA模型/张量分配前检查占用，不自动停止已有服务或训练。
    ensure_gpu_idle()
    formal_output = configured_path(config, "output_dir").resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or formal_output.parent / f"education-tensor-test-{stamp}").expanduser().resolve()
    if output == formal_output or formal_output in output.parents:
        raise ValueError("单tensor测试不能写入正式训练输出目录")
    # main → Path.mkdir：使用独立且全新的测试目录；已有目录拒绝覆盖。
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "prepared", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "task": "single_tensor_lora_update", "model_path": config["model_path"], "output_dir": str(output),
              "question": args.question, "reference_answer": args.answer, "input_shape": list(batch["input_ids"].shape),
              "tensor_dtype": str(batch["input_ids"].dtype), "supervised_tokens": (batch["labels"] != -100).sum().item(),
              "learning_rate": config["learning_rate"], "warmup_used": False,
              "formal_training_output": str(formal_output), "formal_dataset_training_started": False}
    save_report(output, report)
    print("测试输出目录：", output, flush=True)
    try:
        # main → load_trainable_model：与正式微调共享NF4加载、语言层LoRA与基座冻结检查。
        torch.cuda.reset_peak_memory_stats()
        model, model_info = load_trainable_model(config)
        report.update(model_info)
        # main → 模型输入嵌入层：取得实际设备；Tensor.to只移动设备，保留torch.long整数类型。
        device = model.get_input_embeddings().weight.device
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        print("开始一次前向、反向和参数更新；设备：", device, flush=True)
        # main → update_once：把这一批tensor真正传给27B，得到loss、梯度和参数变化证据。
        report.update(update_once(model, batch, config["learning_rate"]))
        adapter = output / "adapter"
        # main → PEFT.save_pretrained/tokenizer.save_pretrained：只保存本次单步测试适配器及分词器。
        model.save_pretrained(str(adapter))
        tokenizer.save_pretrained(str(adapter))
        report.update(status="passed", adapter_path=str(adapter),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_report(output, report)
        print(json.dumps({key: report[key] for key in ["status", "loss", "optimizer_steps", "parameter_max_change", "adapter_path", "peak_gpu_allocated_bytes"]}, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save_report(output, report)
        raise


if __name__ == "__main__":
    # Python直接运行本文件 → main → 单tensor准备或单次更新；不进入2500条数据的正式训练入口。
    raise SystemExit(main())
