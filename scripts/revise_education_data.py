# 文件用途：按事实复核反馈再次调用本地27B，修订未发布的教学数据候选并保留原始版本。
# 调用关系：复用generate_education_data的真实响应接受/来源校验；随后由生成脚本resume发布集合。
# 修改易踩坑：修订交互只放来源元数据，训练消息保持原题；备份、响应记录保存后才替换本批候选。
"""Revise generated teaching candidates using explicit factual review feedback."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.generate_education_data import (
    accepted_record, digest, generation_messages, regeneration_messages, local_url, read_candidates,
    save_lines, validate_candidate, write_json,
)
from utils.education_training import load_config


async def revise(args, tokenizer, config):
    import httpx
    from openai import AsyncOpenAI
    output = args.output_dir.expanduser()
    if any((output / name).exists() for name in ["train.jsonl", "validation.jsonl", "test.jsonl", "publication_manifest.json"]):
        raise ValueError("本脚本只修订尚未发布的候选；已发布数据应生成新版本")
    feedback = json.loads(args.feedback.read_text(encoding="utf-8"))
    if not isinstance(feedback, dict) or not feedback or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in feedback.items()
    ):
        raise ValueError("feedback必须是非空的id到事实复核反馈字符串的JSON对象")
    manifest = json.loads((output / "generation_manifest.json").read_text())
    parameters = manifest["parameters"]
    local_url(parameters["base_url"])
    seeds = [json.loads(line) for line in (output / "inputs.jsonl").read_text().splitlines()]
    if manifest["inputs_sha256"] != digest(seeds) or manifest["instruction_sha256"] != digest(config["instruction"]):
        raise ValueError("原始输入或教学指令与本批次不一致")
    seed_map = {seed["id"]: seed for seed in seeds}
    path = output / "model_candidates.jsonl"
    rows = read_candidates(path)
    candidates = {row["id"]: row for row in rows}
    if len(candidates) != len(rows) or not set(feedback) <= set(candidates):
        raise ValueError("候选id重复或待修订id不存在")
    for row in rows:
        validate_candidate(row, seed_map[row["id"]], tokenizer, config, parameters)
    original_bytes = path.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    backup = output / ("model_candidates.before_revision." + original_sha[:12] + ".jsonl")
    if backup.exists():
        if backup.read_bytes() != original_bytes:
            raise ValueError("原候选备份与本次版本不一致")
    else:
        with backup.open("xb") as handle:
            handle.write(original_bytes)
    request_sha = digest({"candidates_sha256": original_sha, "feedback": feedback})
    request_path = output / ("revision_requests." + request_sha[:12] + ".json")
    request_record = {"original_candidates_sha256": original_sha, "original_backup": backup.name,
                      "feedback": feedback, "feedback_sha256": digest(feedback)}
    if request_path.exists() and json.loads(request_path.read_text()) != request_record:
        raise ValueError("修订请求记录已存在且不一致")
    if not request_path.exists():
        write_json(request_path, request_record)
    client = AsyncOpenAI(base_url=parameters["base_url"], api_key="local-only", max_retries=0,
                        timeout=180, http_client=httpx.AsyncClient(trust_env=False))
    revised = {}
    concurrency = getattr(args, "concurrency", 4)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        models = await client.models.list()
        if parameters["model"] not in {model.id for model in models.data}:
            raise ValueError("指定本地教师模型未加载")
        with (output / "model_revision_responses.jsonl").open("a", encoding="utf-8") as ledger:
            async def one(identity, note):
                async with semaphore:
                    seed, original = seed_map[identity], candidates[identity]
                    expected = generation_messages(seed, config["instruction"])
                    regenerate = getattr(args, "regenerate", False)
                    teacher_messages = (regeneration_messages(seed, config["instruction"], note) if regenerate
                                        else expected + [original["messages"][-1], {"role": "user", "content": note}])
                    for attempt in range(1, 4):
                        started = time.perf_counter()
                        response = None
                        thinking = (parameters.get("enable_thinking", False)
                                    if getattr(args, "teacher_thinking", None) is None else args.teacher_thinking)
                        budget = 4096 if thinking else 960
                        metadata = {"kind": "model_regeneration_after_review" if regenerate else "model_revision_after_review",
                            "teacher_model": parameters["model"], "teacher_endpoint": parameters["base_url"],
                            "attempt": attempt, "max_tokens": budget, "seed": 2026 + attempt - 1,
                            "temperature": 0.25, "top_p": 0.9, "top_k": 20, "enable_thinking": thinking,
                            "revision_concurrency": concurrency,
                            "original_teacher_enable_thinking": parameters.get("enable_thinking", False),
                            "input_seed_sha256": digest(seed), "original_prompt_sha256": digest(expected),
                            "prompt_sha256": digest(teacher_messages), "revision_request_messages": teacher_messages,
                            "original_candidate_sha256": original["candidate_sha256"], "review_feedback": note,
                            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "latency_seconds": round(time.perf_counter() - started, 4)}
                        try:
                            response = await client.chat.completions.create(
                                model=parameters["model"], messages=teacher_messages,
                                temperature=0.25, top_p=0.9, max_tokens=budget,
                                seed=2026 + attempt - 1,
                                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}},
                            )
                            metadata["latency_seconds"] = round(time.perf_counter() - started, 4)
                            row = accepted_record(seed, response, tokenizer, config["instruction"],
                                                  config["max_seq_length"], metadata)
                            validate_candidate(row, seed, tokenizer, config, parameters)
                        except Exception as exc:
                            ledger.write(json.dumps({"id": identity, "accepted": False, "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "reason": str(exc) if isinstance(exc, ValueError) else "本地模型请求失败",
                                "response": ({"id": response.id,
                                    "finish_reasons": [choice.finish_reason for choice in response.choices]}
                                    if response is not None else None),
                                "metadata": metadata}, ensure_ascii=False) + "\n")
                            ledger.flush()
                            if attempt == 3:
                                raise
                            await asyncio.sleep(attempt)
                            continue
                        ledger.write(json.dumps({"id": identity, "accepted": True,
                            "original_candidates_sha256": original_sha,
                            "original_candidate_sha256": original["candidate_sha256"],
                            "revised_candidate": row}, ensure_ascii=False) + "\n")
                        ledger.flush()
                        revised[identity] = row
                        print(f"模型已修订 {identity}，{len(revised)}/{len(feedback)}", flush=True)
                        return
            tasks = [asyncio.create_task(one(identity, note)) for identity, note in feedback.items()]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.close()
    if set(revised) != set(feedback):
        raise ValueError("没有完成全部修订，保留原候选")
    updated = [revised.get(row["id"], row) for row in rows]
    for row in updated:
        validate_candidate(row, seed_map[row["id"]], tokenizer, config, parameters)
    if path.read_bytes() != original_bytes:
        raise ValueError("候选在修订期间被其他进程修改，拒绝替换")
    with tempfile.TemporaryDirectory(prefix=".education-revision-", dir=output) as directory:
        temporary = Path(directory) / "candidates.jsonl"
        save_lines(temporary, updated)
        if path.read_bytes() != original_bytes:
            raise ValueError("替换前候选版本已变更")
        temporary.replace(path)
    report = {"status": "completed", "count": len(revised), "review_status": "pending",
              "original_candidates_sha256": original_sha,
              "revised_candidates_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "original_backup": backup.name, "requests": request_path.name,
              "ids": list(revised), "teacher_model": parameters["model"], "concurrency": concurrency,
              "note": "真实模型按事实反馈修订；原候选保留；复核反馈不进入训练消息，仍是合成候选。"}
    write_json(output / ("revision_report." + request_sha[:12] + ".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/MAOMI/training/education-dataset-v1"))
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4, help="本地教师并发请求数，1到24")
    parser.add_argument("--teacher-thinking", action="store_true", default=None,
                        help="修订时启用教师思考，单独记录设置；训练只保留最终回答")
    parser.add_argument("--regenerate", action="store_true",
                        help="按原题和独立核验提示重新生成，不将上一版错误回答放入教师上下文")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/education_finetune.yml")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/root/MAOMI/models/Qwen3.8-27B-AWQ-INT4"))
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 24:
        parser.error("concurrency需要在1到24之间")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True, trust_remote_code=False)
    asyncio.run(revise(args, tokenizer, load_config(args.config)))


if __name__ == "__main__":
    main()
