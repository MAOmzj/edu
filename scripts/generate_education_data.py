# 文件用途：调用本地27B按指定条数生成教学回答候选，支持独立增量与来源追踪。
# 调用关系：读取教育题目种子和education_finetune配置；输出供train_education.py使用的JSONL。
# 修改易踩坑：按任务族先分集合；只接受正常结束的真实模型回答，拒绝截断/控制标签；完成后发布数据。
"""Generate primary-school teaching SFT candidates using an existing local vLLM server."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.education_training import encode_record, load_config, load_records, validate_split


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_url(value):
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") != "/v1"):
        raise ValueError("只允许无凭据的本地http://127.0.0.1:端口/v1服务")
    if url.port is None:
        raise ValueError("本地服务URL必须包含端口")
    return value.rstrip("/")


def generation_messages(seed, instruction):
    messages = seed["messages"]
    return messages if messages[0]["role"] == "system" else [{"role": "system", "content": instruction}] + messages


def regeneration_messages(seed, instruction, feedback):
    original = generation_messages(seed, instruction)
    return [{"role": "system", "content": original[0]["content"]
             + "\n\n生成核验提示（不属于学生题面，不在回答中提及）：\n" + feedback}] + original[1:]


def validate_seeds(seeds, instruction, expected=None):
    expected = expected or {"train": 1000, "validation": 100, "test": 100}
    if (set(expected) != {"train", "validation", "test"} or expected["train"] <= 0
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 or n % 4 for n in expected.values())):
        raise ValueError("各集合条数须为非负的4倍数，且训练集不能空")
    if Counter(r.get("split") for r in seeds) != {key: value for key, value in expected.items() if value}:
        raise ValueError(f"种子条数必须与指定划分一致：{expected}")
    if len({r["id"] for r in seeds}) != len(seeds):
        raise ValueError("种子id重复")
    groups, fingerprints = {}, set()
    for seed in seeds:
        if seed["subject"] not in {"语文", "数学", "英语", "科学"} or not 1 <= seed["grade"] <= 6:
            raise ValueError("种子学科或年级非法")
        if seed["group_id"] in groups and groups[seed["group_id"]] != seed["split"]:
            raise ValueError("同题任务族跨集合")
        groups[seed["group_id"]] = seed["split"]
        if not seed["messages"] or seed["messages"][-1]["role"] != "user":
            raise ValueError("种子必须以user问题结尾，不包含目标答案")
        previous = None
        for index, message in enumerate(seed["messages"]):
            role = message["role"]
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("种子消息不能为空")
            if role == "system" and index == 0:
                pass
            elif role == "user" and previous in {None, "system", "assistant"}:
                pass
            elif role == "assistant" and previous == "user":
                pass
            else:
                raise ValueError("种子消息角色顺序非法")
            previous = role
        prompt = generation_messages(seed, instruction)
        fingerprint = digest(prompt)
        if fingerprint in fingerprints:
            raise ValueError("实际输入重复，不能以更换id凑数量")
        fingerprints.add(fingerprint)
    for split, count in expected.items():
        if count and Counter(r["subject"] for r in seeds if r["split"] == split) != {
            subject: count // 4 for subject in ["语文", "数学", "英语", "科学"]
        }:
            raise ValueError("各集合需要四个学科均衡")


def accepted_record(seed, response, tokenizer, instruction, max_seq_length, metadata):
    if len(response.choices) != 1:
        raise ValueError("响应choices数量异常")
    choice = response.choices[0]
    message = choice.message
    if choice.finish_reason != "stop":
        raise ValueError("回答未正常结束，拒绝把截断内容作为目标")
    if getattr(message, "tool_calls", None) or getattr(message, "function_call", None):
        raise ValueError("响应含工具调用，不能充当普通教学回答")
    content = message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型回答为空")
    usage = response.usage.model_dump() if response.usage else {}
    metadata = {**metadata, "response_model": response.model, "response_id": response.id,
                "finish_reason": choice.finish_reason,
                "reasoning_content_present": bool(getattr(message, "reasoning_content", None)
                                                  or getattr(message, "reasoning", None)),
                "usage": {key: usage.get(key) for key in ["prompt_tokens", "completion_tokens", "total_tokens"]}}
    record = {key: seed[key] for key in ["id", "group_id", "subject", "grade", "skill", "split"]}
    record.update(messages=generation_messages(seed, instruction) + [{"role": "assistant", "content": content.strip()}],
                  source="local_model", input_source=seed.get("source", "synthetic_authored"),
                  dataset_kind="synthetic_teaching", review_status="pending", generation=metadata)
    features = encode_record(record, tokenizer, instruction, max_seq_length)
    record["generation"]["training_sequence_tokens"] = len(features["input_ids"])
    record["generation"]["supervised_tokens"] = sum(t != -100 for t in features["labels"])
    record["candidate_sha256"] = digest(record)
    return record


def save_lines(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_candidate(row, seed, tokenizer, config, parameters):
    for key in ["id", "group_id", "subject", "grade", "skill", "split"]:
        if row.get(key) != seed[key]:
            raise ValueError(f"候选{seed['id']}的{key}与原输入不一致")
    expected = generation_messages(seed, config["instruction"])
    messages = row.get("messages", [])
    if len(messages) != len(expected) + 1 or messages[:-1] != expected or messages[-1].get("role") != "assistant":
        raise ValueError("已有候选的原始题面或角色被改变")
    if row.get("source") != "local_model" or row.get("review_status") != "pending":
        raise ValueError("候选来源或核验状态被改变")
    metadata = row.get("generation", {})
    prompt_sha256 = digest(expected)
    expected_thinking = parameters.get("enable_thinking", False)
    if "kind" in metadata:
        if metadata["kind"] not in {"model_revision_after_review", "model_regeneration_after_review"}:
            raise ValueError("候选生成记录包含未知kind")
        if (not isinstance(metadata.get("enable_thinking"), bool)
                or (metadata.get("enable_thinking") != expected_thinking
                    and metadata.get("original_teacher_enable_thinking") != expected_thinking)
                or ("original_teacher_enable_thinking" in metadata
                    and metadata["original_teacher_enable_thinking"] != expected_thinking)):
            raise ValueError("修订教师enable_thinking设置缺少原批次来源记录")
        expected_thinking = metadata["enable_thinking"]
        if metadata.get("original_prompt_sha256") != digest(expected):
            raise ValueError("修订候选的original_prompt_sha256与原题面不一致")
        revision_messages = metadata.get("revision_request_messages")
        feedback = metadata.get("review_feedback")
        if not isinstance(feedback, str) or not feedback.strip() or not isinstance(revision_messages, list):
            raise ValueError("修订请求必须包含非空核验反馈")
        if metadata["kind"] == "model_regeneration_after_review":
            canonical_revision = regeneration_messages(seed, config["instruction"], feedback)
        else:
            if (len(revision_messages) != len(expected)+2 or not isinstance(revision_messages[-2], dict)
                    or not isinstance(revision_messages[-2].get("content"), str)
                    or not revision_messages[-2]["content"].strip()):
                raise ValueError("修订请求必须包含原回答与非空核验反馈")
            canonical_revision = expected + [
                {"role": "assistant", "content": revision_messages[-2]["content"]},
                {"role": "user", "content": feedback},
            ]
        if revision_messages != canonical_revision:
            raise ValueError("修订请求的原题面、原回答角色或反馈被改变")
        original_sha256 = metadata.get("original_candidate_sha256")
        if (not isinstance(original_sha256, str) or len(original_sha256) != 64
                or any(character not in "0123456789abcdef" for character in original_sha256)):
            raise ValueError("修订来源original_candidate_sha256必须为64位小写hex")
        prompt_sha256 = digest(revision_messages)
    for key, value in {"input_seed_sha256": digest(seed), "prompt_sha256": prompt_sha256,
                       "teacher_model": parameters["model"], "response_model": parameters["model"],
                       "teacher_endpoint": parameters["base_url"], "finish_reason": "stop",
                       "enable_thinking": expected_thinking}.items():
        if metadata.get(key) != value:
            raise ValueError(f"候选生成记录的{key}与本批次不一致")
    original = {key: value for key, value in row.items() if key != "candidate_sha256"}
    if row.get("candidate_sha256") != digest(original):
        raise ValueError("已有候选内容的校验值不一致")
    encode_record(row, tokenizer, config["instruction"], config["max_seq_length"])


def read_candidates(path):
    """只隔离写到一半的末行，中间损坏必须停止，不能静默丢数据。"""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (UnicodeError, json.JSONDecodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError("候选记录损坏，拒绝忽略中间或完整行错误")
            with path.with_suffix(".incomplete-last-line.bin").open("xb") as handle:
                handle.write(line)
            with path.open("r+b") as handle:
                handle.truncate(sum(len(item) for item in lines[:-1]))
        else:
            if index == len(lines)-1 and not line.endswith(b"\n"):
                with path.open("ab") as handle: handle.write(b"\n")
    return rows


def publish_splits(output, splits):
    """记录本批次发布hash；中断可继续，任何外来数据文件都不覆盖。"""
    expected = {}
    for split, rows in splits.items():
        contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        expected[split + ".jsonl"] = hashlib.sha256(contents).hexdigest()
    state = output / "publication_manifest.json"
    if state.exists():
        if json.loads(state.read_text()) != expected:
            raise ValueError("发布清单与本次候选不一致")
    else:
        if any((output / name).exists() for name in expected):
            raise FileExistsError("数据文件已存在但不属于本批次发布，拒绝覆盖")
        write_json(state, expected)
    for split, rows in splits.items():
        destination = output / (split + ".jsonl")
        if destination.is_symlink():
            raise FileExistsError("正式数据路径已有链接，拒绝覆盖")
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected[destination.name]:
                raise ValueError("已发布数据与本批次hash不符，拒绝覆盖")
            continue
        temporary = output / (split + ".publish.tmp")
        if temporary.exists(): temporary.unlink()
        save_lines(temporary, rows)
        load_records(temporary)
        # link原子排他创建，不能像rename一样覆盖检查后出现的文件。
        os.link(temporary, destination)
        temporary.unlink()


async def generate(args, seeds, config, tokenizer):
    import httpx
    from openai import AsyncOpenAI
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    parameters = {"model": args.model, "base_url": local_url(args.base_url), "temperature": args.temperature,
                  "top_p": 0.9, "top_k": 20, "enable_thinking": args.teacher_thinking,
                  "max_tokens": args.max_tokens, "max_attempts": 3, "max_sequence_length": config["max_seq_length"],
                  "concurrency": args.concurrency, "seed": 42}
    expected = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    manifest = {"version": 1, "inputs_sha256": digest(seeds), "instruction_sha256": digest(config["instruction"]),
                "parameters": parameters, "expected_counts": expected}
    manifest_path = output / "generation_manifest.json"
    candidate_path = output / "model_candidates.jsonl"
    failure_path = output / "failed_attempts.jsonl"
    published = [output / (split + ".jsonl") for split, count in expected.items() if count]
    if any((output / (split + ".jsonl")).exists() for split in expected) and not (args.resume and (output / "publication_manifest.json").exists()):
        raise FileExistsError("正式数据文件已存在，拒绝覆盖；请指定新输出目录")
    completed = {}
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("已有生成记录；需要--resume继续，或指定新目录")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("种子、指令或生成参数改变，不能混入旧任务")
        for row in read_candidates(candidate_path):
            if row["id"] in completed:
                raise ValueError("候选记录id重复")
            completed[row["id"]] = row
    else:
        if args.resume or candidate_path.exists() or failure_path.exists():
            raise ValueError("生成清单缺失，拒绝复用无法确认来源的记录")
        save_lines(output / "inputs.jsonl", seeds)
        write_json(manifest_path, manifest)
    seed_map = {row["id"]: row for row in seeds}
    all_ids = set(seed_map)
    if not set(completed) <= all_ids:
        raise ValueError("已有候选包含未知id")
    for row in completed.values():
        validate_candidate(row, seed_map[row["id"]], tokenizer, config, parameters)
    progress_path = output / "generation_progress.json"
    started = time.perf_counter()
    prior = len(completed)
    failures = []
    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(base_url=parameters["base_url"], api_key="local-only", max_retries=0,
                         timeout=180, http_client=httpx.AsyncClient(trust_env=False))
    try:
        models = await client.models.list()
        if args.model not in {m.id for m in models.data}:
            raise ValueError("本地服务没有所指定的教师模型")
        def update(status="running"):
            report = {"status": status, "total": len(seeds), "succeeded": len(completed),
                      "pending": len(seeds) - len(completed), "failed_samples": len(failures),
                      "elapsed_seconds": round(time.perf_counter() - started, 2),
                      "updated_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(progress_path, report)
        update()
        with candidate_path.open("a", encoding="utf-8") as candidate_file, failure_path.open("a", encoding="utf-8") as failure_file:
            async def one(seed):
                async with semaphore:
                    for attempt in range(1, 4):
                        messages = generation_messages(seed, config["instruction"])
                        budget = parameters["max_tokens"] + (attempt - 1) * (parameters["max_tokens"] // 2)
                        tick = time.perf_counter()
                        response = None
                        try:
                            response = await client.chat.completions.create(
                                model=args.model, messages=messages, temperature=parameters["temperature"],
                                top_p=parameters["top_p"], max_tokens=budget,
                                seed=42 + attempt - 1,
                                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": parameters["enable_thinking"]}},
                            )
                            metadata = {"teacher_model": args.model, "teacher_endpoint": parameters["base_url"],
                                        "attempt": attempt, "max_tokens": budget, "seed": 42 + attempt - 1,
                                        "temperature": parameters["temperature"], "top_p": parameters["top_p"],
                                        "top_k": 20, "enable_thinking": parameters["enable_thinking"],
                                        "prompt_sha256": digest(messages), "input_seed_sha256": digest(seed),
                                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "latency_seconds": round(time.perf_counter() - tick, 4)}
                            row = accepted_record(seed, response, tokenizer, config["instruction"],
                                                  config["max_seq_length"], metadata)
                        except Exception as exc:
                            reason = str(exc) if isinstance(exc, ValueError) else "本地模型请求失败"
                            failure_file.write(json.dumps({"id": seed["id"], "attempt": attempt,
                                "error_type": type(exc).__name__, "reason": reason,
                                "finish_reason": (response.choices[0].finish_reason
                                    if response is not None and response.choices else None)}, ensure_ascii=False) + "\n")
                            failure_file.flush()
                            if attempt < 3:
                                await asyncio.sleep(attempt)
                            continue
                        completed[seed["id"]] = row
                        candidate_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        candidate_file.flush()
                        update()
                        if (len(completed) - prior) % 25 == 0:
                            print(f"已生成 {len(completed)}/{len(seeds)} 条；本轮 {round(time.perf_counter()-started, 1)} 秒", flush=True)
                        return
                    failures.append(seed["id"])
                    update()
            pending = [s for s in seeds if s["id"] not in completed]
            # 交错学科，避免中断时仅生成某一个学科。
            random.Random(42).shuffle(pending)
            tasks = [asyncio.create_task(one(seed)) for seed in pending]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done(): task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if len(completed) != len(seeds):
            update("incomplete")
            raise RuntimeError(f"还有{len(seeds)-len(completed)}条失败；修复服务后使用--resume重试")
        if args.defer_publication:
            update("candidates_ready")
            print(json.dumps({"status": "candidates_ready", "counts": expected,
                              "published": False, "output_dir": str(output)}, ensure_ascii=False), flush=True)
            return
        splits = {split: [completed[s["id"]] for s in seeds if s["split"] == split]
                  for split in ["train", "validation", "test"]}
        for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            if splits[first] and splits[second]:
                validate_split(splits[first], splits[second], config["instruction"])
        random.Random(42).shuffle(splits["train"])
        publish_splits(output, {key: value for key, value in splits.items() if value})
        all_rows = list(completed.values())
        report = {"status": "completed", "counts": {key: len(value) for key, value in splits.items()},
                  "subject_counts": {key: dict(Counter(r["subject"] for r in rows)) for key, rows in splits.items()},
                  "grade_counts": {key: dict(sorted(Counter(r["grade"] for r in rows).items())) for key, rows in splits.items()},
                  "group_counts": {key: len({r["group_id"] for r in rows}) for key, rows in splits.items()},
                  "multi_turn_counts": {key: sum(len(r["messages"])>3 for r in rows) for key,rows in splits.items()},
                  "source": "local_model", "teacher_model": args.model, "review_status": "pending",
                  "revision_count": sum(r["generation"].get("kind") in {"model_revision_after_review", "model_regeneration_after_review"} for r in all_rows),
                  "teacher_thinking_counts": dict(Counter(str(r["generation"]["enable_thinking"]) for r in all_rows)),
                  "parameters": parameters, "inputs_sha256": manifest["inputs_sha256"],
                  "file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in published},
                  "max_sequence_tokens": max(r["generation"]["training_sequence_tokens"] for r in all_rows),
                  "max_completion_tokens": max(r["generation"]["usage"].get("completion_tokens") or 0 for r in all_rows),
                  "protocol_and_token_checks_passed": True, "split_overlap": False,
                  "formal_training_started": False, "elapsed_seconds": round(time.perf_counter()-started,2),
                  "elapsed_seconds_scope": "current_invocation_only",
                  "quality_note": "合成教学候选，非人工金标；结构/长度通过不等于教学正确率；测试只代表合成分布。"}
        write_json(output / "dataset_report.json", report)
        update("completed")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        await client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/MAOMI/training/education-dataset-v1"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/root/MAOMI/models/Qwen3.8-27B-AWQ-INT4"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/education_finetune.yml")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--inputs-file", type=Path, help="读取已准备的独立输入JSONL；默认使用原1200条种子")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--teacher-thinking", action="store_true", help="仅教师生成使用思考；训练目标仍只保存最终教学回答")
    parser.add_argument("--defer-publication", action="store_true", help="候选完成后先保留，供复核修订；resume去掉此项再发布")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="只检查输入，不调用模型或创建输出")
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= 24:
        parser.error("concurrency需要在1到24之间")
    if not 128 <= args.max_tokens <= 4096 or not 0 <= args.temperature <= 2:
        parser.error("max-tokens须在128到4096之间，temperature须在0到2之间")
    from utils.education_seeds_math_science import build_seeds as math_science
    from utils.education_seeds_language import build_seeds as language
    config = load_config(args.config)
    seeds = ([json.loads(line) for line in args.inputs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
             if args.inputs_file else math_science() + language())
    validate_seeds(seeds, config["instruction"],
                   {"train": args.train_count, "validation": args.validation_count, "test": args.test_count})
    if args.prepare_only:
        print(json.dumps({"counts":dict(Counter(r["split"] for r in seeds)),"groups":len({r["group_id"] for r in seeds}),
                          "subjects":dict(Counter(r["subject"] for r in seeds)),"model_called":False},ensure_ascii=False))
        return 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True, trust_remote_code=False)
    asyncio.run(generate(args, seeds, config, tokenizer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
