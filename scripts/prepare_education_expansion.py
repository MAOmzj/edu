# 文件用途：准备四科共1500条新增教学输入，核验与原数据去重及留出组隔离。
# 调用关系：读取education_seeds_expansion三个模块；输出给generate_education_data的独立输入JSONL。
# 修改易踩坑：新增仅进入原训练组；不可将验证/测试组作为训练变体，不覆盖不同内容的已有输入。
"""Prepare 1500 additional training prompts without calling the teacher model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_education_data import digest, generation_messages, save_lines, validate_seeds
from utils.education_training import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/root/MAOMI/training/education-expansion-inputs-v1.jsonl"))
    args = parser.parse_args()
    from utils.education_seeds_expansion_math import build_seeds as math
    from utils.education_seeds_expansion_science import build_seeds as science
    from utils.education_seeds_expansion_language import build_seeds as language
    from utils.education_seeds_math_science import build_seeds as original_science
    from utils.education_seeds_language import build_seeds as original_language
    config = load_config(PROJECT_ROOT / "config/education_finetune.yml")
    original = original_science() + original_language()
    extra = math() + science() + language()
    validate_seeds(extra, config["instruction"], {"train": 1500, "validation": 0, "test": 0})
    original_groups = {row["group_id"] for row in original if row["split"] == "train"}
    held_out_groups = {row["group_id"] for row in original if row["split"] != "train"}
    if not {row["group_id"] for row in extra} <= original_groups or {row["group_id"] for row in extra} & held_out_groups:
        raise ValueError("新增任务组不在原训练集合或碰到了留出集合")
    if {row["id"] for row in original} & {row["id"] for row in extra}:
        raise ValueError("新增id重复")
    fingerprints = {digest(generation_messages(row, config["instruction"])) for row in original}
    if any(digest(generation_messages(row, config["instruction"])) in fingerprints for row in extra):
        raise ValueError("新增输入与原数据重复")
    if args.output.exists():
        saved = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
        if saved != extra:
            raise FileExistsError("新增输入文件已存在且内容不同，拒绝覆盖")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_lines(args.output, extra)
    print(json.dumps({"count":len(extra), "subjects":dict(Counter(row["subject"] for row in extra)),
        "groups":len({row["group_id"] for row in extra}), "inputs_sha256":digest(extra),
        "multi_turn_count":sum(len(row["messages"])>1 for row in extra),
        "duplicate_original_prompt":False, "held_out_group_overlap":False,
        "model_called":False, "path":str(args.output)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
