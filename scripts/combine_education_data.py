# 文件用途：将原教学数据与独立新增批次合为新版本，保留原验证/测试划分和逐条来源。
# 调用关系：读取两批已发布JSONL与校验清单，离线分词验证后供education_finetune配置读取。
# 修改易踩坑：新增只能是训练样本，必须排除重复输入与留出组；原文件不覆盖，反馈元数据不作训练消息。
"""Combine published teaching datasets without changing the held-out data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_education_data import digest, publish_splits, validate_candidate, write_json
from utils.education_training import encode_record, load_config, load_records, prompt_messages, validate_split


def verified_batch(root, config, tokenizer):
    manifest = json.loads((root / "generation_manifest.json").read_text())
    publication = json.loads((root / "publication_manifest.json").read_text())
    if manifest["instruction_sha256"] != digest(config["instruction"]):
        raise ValueError("原批次的教学指令与当前配置不同")
    seeds = [json.loads(line) for line in (root / "inputs.jsonl").read_text().splitlines() if line.strip()]
    if digest(seeds) != manifest["inputs_sha256"]:
        raise ValueError("原批次的输入校验不一致")
    seed_map = {seed["id"]: seed for seed in seeds}
    splits, sources = {}, {}
    for split, count in manifest["expected_counts"].items():
        if not count:
            splits[split] = []
            continue
        path = root / (split + ".jsonl")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if publication.get(path.name) != sha:
            raise ValueError("源文件与批次发布清单不一致")
        load_records(path)  # 只做结构检查，保留原始行，不能改写来源hash对应的messages。
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != count or any(row["split"] != split for row in rows):
            raise ValueError("源文件数量或split标签不一致")
        for row in rows:
            validate_candidate(row, seed_map[row["id"]], tokenizer, config, manifest["parameters"])
        splits[split], sources[path.name] = rows, sha
    return splits, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("/root/MAOMI/training/education-dataset-v1"))
    parser.add_argument("--additional-dir", type=Path, default=Path("/root/MAOMI/training/education-addition-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/MAOMI/training/education-dataset-v2"))
    parser.add_argument("--expected-train-count", type=int, default=2500)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/education_finetune.yml")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/root/MAOMI/models/Qwen3.8-27B-AWQ-INT4"))
    args = parser.parse_args()
    config = load_config(args.config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True, trust_remote_code=False)
    base, base_sha = verified_batch(args.base_dir, config, tokenizer)
    extra, extra_sha = verified_batch(args.additional_dir, config, tokenizer)
    if extra["validation"] or extra["test"]:
        raise ValueError("新增批次只能含训练数据，不能改变原留出集合")
    splits = {"train": base["train"] + extra["train"], "validation": base["validation"], "test": base["test"]}
    if len(splits["train"]) != args.expected_train_count:
        raise ValueError("合并后的训练数量不等于指定目标")
    rows = [row for part in splits.values() for row in part]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("两批存在重复id")
    if len({digest(prompt_messages(row, config["instruction"])) for row in rows}) != len(rows):
        raise ValueError("合并后实际输入重复")
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        validate_split(splits[first], splits[second], config["instruction"])
    for split_rows in splits.values():
        counts = Counter(row["subject"] for row in split_rows)
        if len(counts) != 4 or len(set(counts.values())) != 1:
            raise ValueError("合并后四科学科数量不均衡")
    maximum = 0
    for row in rows:
        features = encode_record(row, tokenizer, config["instruction"], config["max_seq_length"])
        maximum = max(maximum, len(features["input_ids"]))
        if features["labels"][-1] != tokenizer.eos_token_id:
            raise ValueError("目标回答缺少EOS监督")
    random.Random(42).shuffle(splits["train"])
    for split in ["validation", "test"]:
        serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits[split]).encode()
        if serialized != (args.base_dir / (split + ".jsonl")).read_bytes():
            raise ValueError("源留出文件格式不能逐字节保留，拒绝在发布后才发现变化")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    publish_splits(args.output_dir, splits)
    for split in ["validation", "test"]:
        if (args.output_dir / (split + ".jsonl")).read_bytes() != (args.base_dir / (split + ".jsonl")).read_bytes():
            raise ValueError("合并不应改变原验证或测试文件")
    report = {"status": "completed", "counts": {key:len(value) for key,value in splits.items()},
        "added_train_count": len(extra["train"]), "base_train_count":len(base["train"]),
        "subject_counts":{key:dict(Counter(row["subject"] for row in value)) for key,value in splits.items()},
        "group_counts":{key:len({row["group_id"] for row in value}) for key,value in splits.items()},
        "multi_turn_counts":{key:sum(len(row["messages"])>3 for row in value) for key,value in splits.items()},
        "teacher_thinking_counts":dict(Counter(str(row["generation"]["enable_thinking"]) for row in rows)),
        "revision_count":sum(row["generation"].get("kind") in {"model_revision_after_review", "model_regeneration_after_review"} for row in rows),
        "source_batches":{str(args.base_dir):base_sha, str(args.additional_dir):extra_sha},
        "file_sha256":{split+".jsonl":hashlib.sha256((args.output_dir/(split+".jsonl")).read_bytes()).hexdigest()
                       for split in splits},
        "max_sequence_tokens":maximum, "id_or_effective_prompt_duplicates":False, "split_overlap":False,
        "held_out_files_unchanged":True, "token_and_eos_checks_passed":True,
        "review_status":"pending", "formal_training_started":False,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "quality_note":"教师模型合成教学候选；AI抽检与修订不等于人工金标或真实课堂测试。"}
    write_json(args.output_dir / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
