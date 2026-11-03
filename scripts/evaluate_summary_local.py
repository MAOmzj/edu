# 文件用途：在本地 GPU 对 BF16 摘要基座或 LoRA 适配器保存独立测试输出与耗时。
# 调用关系：开发者运行 CLI；复用 summary_prompt.py 的线上输入，不启动 vLLM 或读取学生数据库。
# 修改易踩坑：同一输入和生成参数比较微调前后；参考答案不进 Prompt；输出记录不等于质量评分。
"""本地摘要模型的离线推理评测，保存参考答案供逐条复核。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_summary import summary_budgets
from utils.summary_training import example_messages


def load_cases(paths):
    cases = []
    seen = set()
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number} 需要 JSON 对象")
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError(f"{path}:{number} id 缺失或重复")
            previous, reference = record.get("previous_summary", ""), record.get("reference")
            if not isinstance(previous, str) or not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{path}:{number} 需要文本 previous_summary/reference")
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{number} 需要非空 messages")
            for message in messages:
                if (
                    not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    or message["role"] not in {"user", "assistant"}
                    or not isinstance(message.get("content"), str) or not message["content"].strip()
                ):
                    raise ValueError(f"{path}:{number} 仅支持 user/assistant 和非空文本")
            seen.add(identifier)
            cases.append({"id": identifier, "previous_summary": previous, "messages": messages,
                          "reference": reference, "case_source": str(path.resolve())})
    if not cases:
        raise ValueError("测试集不能为空")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens 必须大于0")
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        parser.error("输出已存在，拒绝覆盖")
    cases = load_cases([path.expanduser() for path in args.cases])
    max_chars, input_chars = summary_budgets()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("此评测需要 CUDA BF16")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False, local_files_only=True)
    model.eval()
    results = []
    for index, case in enumerate(cases):
        prompt = example_messages(case, max_chars, input_chars)
        tokens = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        set_seed(args.seed + index)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **tokens, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        continuation = generated[0, tokens["input_ids"].shape[1]:]
        summary = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        last_token = continuation[-1].item() if continuation.numel() else None
        results.append({
            "id": case["id"], "case_source": case["case_source"], "reference": case["reference"],
            "summary": summary, "summary_chars": len(summary), "latency_seconds": round(elapsed, 4),
            "input_tokens": int(tokens["input_ids"].shape[1]), "output_tokens": int(continuation.numel()),
            "stopped_with_eos": last_token in eos_ids, "over_character_budget": len(summary) > max_chars,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        })
        print(f"{index + 1}/{len(cases)} {case['id']} {elapsed:.2f}s", flush=True)
    report = {
        "base_model": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "generation": {"enable_thinking": False, "do_sample": True, "temperature": 0.7, "top_p": 0.8,
                       "top_k": 20, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "max_summary_chars": max_chars, "summary_input_chars": input_chars, "case_count": len(results),
        "median_latency_seconds": round(statistics.median(r["latency_seconds"] for r in results), 4),
        "results": results, "quality_note": "需逐条检查事实、纠错、归属、状态和隐私；不以请求成功或文本相似度作为准确率。",
        "latency_note": "未单独预热，首例含初始化开销；本地 HF 耗时不代表 vLLM 服务吞吐。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"评测输出：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
