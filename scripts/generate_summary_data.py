# 文件用途：离线生成虚构的摘要微调候选数据，并按模板族隔离训练、验证与测试集。
# 调用关系：开发者运行 CLI；调用 summary_scenarios.py 和线上共用的摘要输入格式。
# 修改易踩坑：合成参考摘要尚未人工复核；模板验证分数不能代表真实会话质量；禁止覆盖已有输出。
"""生成旧摘要 + 对话 -> 新摘要的候选 JSONL；不读取学生数据库，不调用模型。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.summary_scenarios import SCENARIO_FAMILIES, build_scenario
from utils.summary_training import example_messages, load_summary_records, validate_split


def prompt_fingerprint(record, max_summary_chars=1200, summary_input_chars=8000):
    prompt = example_messages(record, max_summary_chars, summary_input_chars)
    encoded = json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_records(count=200, seed=42, *, max_summary_chars=1200, summary_input_chars=8000):
    """模板参考答案与会话一起生成；不把模型预测当作已审核金标。"""
    if count < len(SCENARIO_FAMILIES):
        raise ValueError(f"count 至少为 {len(SCENARIO_FAMILIES)}，以覆盖全部模板族")
    rng = random.Random(seed)
    records = []
    seen = set()
    for index in range(count):
        family = SCENARIO_FAMILIES[index % len(SCENARIO_FAMILIES)]
        variant = index // len(SCENARIO_FAMILIES)
        scenario = build_scenario(family, variant, rng)
        record = {
            "conversation_id": f"synthetic-{seed}-{family}-{variant:05d}",
            "previous_summary": scenario["previous_summary"],
            "messages": scenario["messages"],
            "target_summary": scenario["target_summary"],
            "source": "synthetic_template",
            "review_status": "pending",
            "generator_version": 1,
            "template_family": family,
            "scenario_tags": scenario["scenario_tags"],
        }
        fingerprint = prompt_fingerprint(record, max_summary_chars, summary_input_chars)
        if fingerprint in seen:
            raise ValueError(f"模板 {family} 的变体重复，不能用重复输入扩充数量；请降低 count 或新增模板")
        seen.add(fingerprint)
        records.append(record)
    return records


def split_records(records, seed=42, validation_ratio=0.1, test_ratio=0.1, *, max_summary_chars=1200, summary_input_chars=8000):
    """同一模板族的所有变体放入同一集合，比仅按会话编号划分更严格。"""
    if not all(math.isfinite(ratio) and 0 < ratio < 1 for ratio in (validation_ratio, test_ratio)):
        raise ValueError("验证、测试比例必须是 (0,1) 内的有限数")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("验证与测试比例之和必须小于 1")
    families = sorted({record["template_family"] for record in records})
    if len(families) < 3:
        raise ValueError("至少需要 3 个模板族才能隔离训练、验证与测试")
    random.Random(seed).shuffle(families)
    validation_count = max(1, round(len(families) * validation_ratio))
    test_count = max(1, round(len(families) * test_ratio))
    if validation_count + test_count >= len(families):
        raise ValueError("模板族数量不足以按指定比例分割，请降低验证或测试比例")
    validation_families = set(families[:validation_count])
    test_families = set(families[validation_count:validation_count + test_count])
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        family = record["template_family"]
        split = "validation" if family in validation_families else "test" if family in test_families else "train"
        splits[split].append(record)
    validate_splits(splits, max_summary_chars, summary_input_chars)
    return splits


def validate_splits(splits, max_summary_chars=1200, summary_input_chars=8000):
    names = ("train", "validation", "test")
    fingerprints = set()
    ids = set()
    families = set()
    for name in names:
        if not splits.get(name):
            raise ValueError(f"{name} 不能为空")
        current_families = {record["template_family"] for record in splits[name]}
        if families & current_families:
            raise ValueError("模板族同时出现在不同集合")
        families.update(current_families)
        for record in splits[name]:
            if record["conversation_id"] in ids:
                raise ValueError("conversation_id 重复")
            ids.add(record["conversation_id"])
            fingerprint = prompt_fingerprint(record, max_summary_chars, summary_input_chars)
            if fingerprint in fingerprints:
                raise ValueError("候选集中出现重复输入（按实际摘要预算裁剪后）")
            fingerprints.add(fingerprint)
            if len(record["target_summary"]) > max_summary_chars:
                raise ValueError("目标摘要超过字符预算，请修订模板")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        validate_split(splits[left], splits[right], max_summary_chars=max_summary_chars, summary_input_chars=summary_input_chars)


def write_dataset(output_dir: Path, splits: dict, seed: int, *, max_summary_chars=1200, summary_input_chars=8000):
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    validate_splits(splits, max_summary_chars, summary_input_chars)
    payloads = {
        name: "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        for name, records in splits.items()
    }
    # 先复用训练入口检查完整 JSONL，再创建输出目录；坏数据不会留下半个数据集。
    with tempfile.TemporaryDirectory(prefix="edu-summary-data-check-") as temporary:
        for name, payload in payloads.items():
            candidate = Path(temporary) / f"{name}.jsonl"
            candidate.write_text(payload, encoding="utf-8")
            load_summary_records(candidate, max_summary_chars)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "source": "synthetic_template", "review_status": "pending", "seed": seed,
        "generator_version": 1,
        "scenario_library_sha256": hashlib.sha256((PROJECT_ROOT / "utils/summary_scenarios.py").read_bytes()).hexdigest(),
        "split_method": "template_family", "max_summary_chars": max_summary_chars,
        "summary_input_chars": summary_input_chars, "splits": {},
        "limitations": [
            "虚构模板数据，参考摘要尚未逐条人工复核，不能据此宣称真实摘要质量。",
            "样本主要是短窗口，不能代替真实长对话、跨窗口累积与摘要裁剪场景。",
            "固定模板的变体相关性较强，扩大数量不能代替增加真实场景。",
        ],
    }
    for name, records in splits.items():
        path = output_dir / f"{name}.candidates.jsonl"
        path.write_text(payloads[name], encoding="utf-8")
        input_lengths = [len(example_messages(record, max_summary_chars, summary_input_chars)[1]["content"]) for record in records]
        report["splits"][name] = {
            "file": path.name, "examples": len(records),
            "template_families": dict(sorted(Counter(record["template_family"] for record in records).items())),
            "input_chars_min": min(input_lengths), "input_chars_max": max(input_lengths),
            "target_chars_max": max(len(record["target_summary"]) for record in records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    eval_path = output_dir / "test_eval.candidates.jsonl"
    evaluation = [
        {"id": record["conversation_id"], "previous_summary": record["previous_summary"],
         "messages": record["messages"], "reference": record["target_summary"],
         "source": record["source"], "review_status": record["review_status"]}
        for record in splits["test"]
    ]
    eval_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in evaluation), encoding="utf-8")
    report["test_eval_file"] = eval_path.name
    report["test_eval_sha256"] = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    (output_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的候选数据目录")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-summary-chars", type=int, default=1200)
    parser.add_argument("--summary-input-chars", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.max_summary_chars < 1 or args.summary_input_chars < 1:
        parser.error("摘要和输入预算必须大于 0")
    try:
        budgets = {"max_summary_chars": args.max_summary_chars, "summary_input_chars": args.summary_input_chars}
        records = generate_records(args.count, args.seed, **budgets)
        splits = split_records(records, args.seed, args.validation_ratio, args.test_ratio, **budgets)
        report = write_dataset(args.output_dir, splits, args.seed, max_summary_chars=args.max_summary_chars, summary_input_chars=args.summary_input_chars)
    except (OSError, ValueError) as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(args.output_dir.expanduser().resolve()), "counts": {name: info["examples"] for name, info in report["splits"].items()}, "review_status": "pending"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
