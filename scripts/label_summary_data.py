# 文件用途：调用本地模型批量压缩已有对话，逐条保存待复核的摘要训练候选。
# 调用关系：开发者运行 CLI；复用线上 summary_training.example_messages 输入，不访问学生数据库。
# 修改易踩坑：旧目标不得泄漏进请求；失败不能以 fallback 冒充模型标签；只连接回环服务，输出不覆盖。
"""本地模型自动标注：旧摘要 + 待压缩对话 -> 模型生成的摘要候选。"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.summary_training import example_messages


class CandidateRejected(ValueError):
    """固定的摘要质量/协议错误；消息不包含输入或模型响应正文。"""


def normalize_input(record, location="样本"):
    if not isinstance(record, dict):
        raise ValueError(f"{location} 需要 JSON 对象")
    conversation_id = record.get("conversation_id")
    previous = record.get("previous_summary", "")
    messages = record.get("messages")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError(f"{location} 的 conversation_id 需要非空文本")
    if not isinstance(previous, str):
        raise ValueError(f"{location} 的 previous_summary 需要文本")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location} 的 messages 需要非空列表")
    normalized = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message["role"] not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"{location} 的消息仅支持 user/assistant 与非空 content")
        normalized.append({"role": message["role"], "content": message["content"].strip()})
    result = {"conversation_id": conversation_id.strip(), "previous_summary": previous.strip(), "messages": normalized}
    for key in ("template_family", "input_source"):
        if isinstance(record.get(key), str) and record[key].strip():
            result[key] = record[key].strip()
    if isinstance(record.get("source"), str) and record["source"].strip():
        result["input_source"] = record["source"].strip()
    tags = record.get("scenario_tags")
    if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
        result["scenario_tags"] = tags[:]
    # target_summary / reference 均不读取；模型必须独立根据对话生成。
    return result


def load_inputs(path: Path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} 不是有效 JSON") from exc
        records.append(normalize_input(record, f"{path}:{number}"))
    if not records:
        raise ValueError("输入对话不能为空")
    return records


def validate_local_url(base_url):
    if not isinstance(base_url, str):
        raise ValueError("base_url 需要本地 HTTP(S) 地址")
    endpoint = urlsplit(base_url.strip())
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("base_url 需要本地 HTTP(S) 地址")
    if endpoint.username is not None or endpoint.password is not None or endpoint.query or endpoint.fragment:
        raise ValueError("base_url 不得含凭据、查询参数或片段")
    if endpoint.hostname.lower() != "localhost":
        try:
            is_loopback = ipaddress.ip_address(endpoint.hostname).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("自动标注只连接本机回环服务，不发送至远端 API")
    try:
        endpoint.port
    except ValueError as exc:
        raise ValueError("本地服务端口无效") from exc
    return base_url.strip().rstrip("/")


def validate_options(options):
    normalized = dict(options)
    normalized["base_url"] = validate_local_url(options.get("base_url"))
    model_name = options.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name 需要非空文本")
    normalized["model_name"] = model_name.strip()
    for key in ("temperature", "request_timeout"):
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
    if not 0 <= options["temperature"] <= 2 or options["request_timeout"] <= 0:
        raise ValueError("temperature 需在 [0,2] 内，request_timeout 需大于0")
    tokens = options.get("max_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("max_tokens 必须是正整数")
    return {key: normalized[key] for key in ("base_url", "model_name", "temperature", "request_timeout", "max_tokens")}


def input_fingerprint(record, max_summary_chars, summary_input_chars):
    messages = example_messages(record, max_summary_chars, summary_input_chars)
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_inputs(records, max_summary_chars=1200, summary_input_chars=8000):
    if max_summary_chars < 1 or summary_input_chars < 1:
        raise ValueError("摘要和输入预算必须大于0")
    if not records:
        raise ValueError("输入对话不能为空")
    normalized = [normalize_input(record) for record in records]
    seen = set()
    for record in normalized:
        fingerprint = input_fingerprint(record, max_summary_chars, summary_input_chars)
        if fingerprint in seen:
            raise ValueError("输入包含重复的实际摘要 Prompt，请先去重")
        seen.add(fingerprint)
    return normalized


def model_summary(record, client, options, max_summary_chars=1200, summary_input_chars=8000):
    options = validate_options(options)
    record = prepare_inputs([record], max_summary_chars, summary_input_chars)[0]
    messages = example_messages(record, max_summary_chars, summary_input_chars)
    started = time.perf_counter()
    response = client.post(options["base_url"] + "/chat/completions", json={
        "model": options["model_name"], "messages": messages,
        "temperature": options["temperature"], "max_tokens": options["max_tokens"],
        "stream": False, "chat_template_kwargs": {"enable_thinking": False},
    })
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise CandidateRejected("模型响应缺少唯一摘要结果")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise CandidateRejected("模型输出未正常结束，可能达到 token 上限，不写入训练候选")
    message = choice.get("message")
    summary = message.get("content") if isinstance(message, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        raise CandidateRejected("模型摘要为空")
    summary = summary.strip()
    if len(summary) > max_summary_chars:
        raise CandidateRejected("模型摘要超过字符预算，不截断参考标签，请调整生成参数或核验后修订")
    controls = ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")
    if any(token in summary for token in controls):
        raise CandidateRejected("模型摘要含思考或模型控制标记，不写入训练候选")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    generation = {
        "model": options["model_name"], "base_url": options["base_url"],
        "temperature": options["temperature"], "enable_thinking": False,
        "max_tokens": options["max_tokens"], "max_summary_chars": max_summary_chars,
        "summary_input_chars": summary_input_chars,
        "prompt_sha256": input_fingerprint(record, max_summary_chars, summary_input_chars),
        "latency_seconds": round(time.perf_counter() - started, 4), "finish_reason": "stop",
        "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                  if isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0},
    }
    return {**record, "target_summary": summary, "source": "local_model", "review_status": "pending", "generation": generation}


def label_records(records, output_dir, client, options, *, max_summary_chars=1200, summary_input_chars=8000, progress=None):
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    options = validate_options(options)
    records = prepare_inputs(records, max_summary_chars, summary_input_chars)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = output_dir / "summaries.candidates.jsonl"
    failures = output_dir / "failed_records.jsonl"
    report_path = output_dir / "labeling_report.json"
    report = {
        "status": "running", "source": "local_model", "review_status": "pending",
        "input": len(records), "succeeded": 0, "failed": 0,
        "options": options, "max_summary_chars": max_summary_chars, "summary_input_chars": summary_input_chars,
        "candidates_file": candidates.name, "failures_file": failures.name,
        "input_prompt_sha256": hashlib.sha256("\n".join(input_fingerprint(r, max_summary_chars, summary_input_chars) for r in records).encode()).hexdigest(),
        "quality_note": "自动生成候选仍需逐条复核；未执行微调；失败与超限响应不作为标签。",
    }
    def save_report():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_report()
    with candidates.open("x", encoding="utf-8") as successful, failures.open("x", encoding="utf-8") as failed:
        for index, record in enumerate(records, 1):
            try:
                candidate = model_summary(record, client, options, max_summary_chars, summary_input_chars)
            except CandidateRejected as exc:
                failure = {"conversation_id": record["conversation_id"], "error_type": type(exc).__name__,
                           "reason": str(exc)}
                failed.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failed.flush()
                report["failed"] += 1
                status = "failed"
            except Exception as exc:
                # HTTP 客户端异常只记类型，避免将请求或响应正文复制进失败日志。
                failed.write(json.dumps({"conversation_id": record["conversation_id"], "error_type": type(exc).__name__, "reason": "模型请求失败，请检查本地服务日志"}, ensure_ascii=False) + "\n")
                failed.flush()
                report["failed"] += 1
                status = "failed"
            else:
                successful.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                successful.flush()
                report["succeeded"] += 1
                status = "succeeded"
            save_report()
            if progress is not None:
                progress({"index": index, "total": len(records), "status": status})
    report["status"] = "completed" if not report["failed"] else "completed_with_errors"
    report["candidates_sha256"] = hashlib.sha256(candidates.read_bytes()).hexdigest()
    report["failures_sha256"] = hashlib.sha256(failures.read_bytes()).hexdigest()
    save_report()
    return report


def configured_options(args):
    import yaml
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = yaml.safe_load((PROJECT_ROOT / "config/model.yml").read_text(encoding="utf-8"))
    context = yaml.safe_load((PROJECT_ROOT / "config/app.yml").read_text(encoding="utf-8"))["context"]
    if args.model == "main":
        options = {
            "base_url": os.getenv("LOCAL_VLLM_CHAT_BASE_URL") or config["chat_base_url"],
            "model_name": os.getenv("LOCAL_VLLM_CHAT_MODEL_NAME") or config["chat_model_name"],
            "temperature": 0.0, "request_timeout": config.get("request_timeout", 180), "max_tokens": 2048,
        }
    else:
        from utils.config_handler import get_summary_model_config
        summary = get_summary_model_config()
        options = {
            "base_url": os.getenv("EDUCATION_SUMMARY_BASE_URL") or summary["base_url"],
            "model_name": os.getenv("EDUCATION_SUMMARY_MODEL_NAME") or summary["model_name"],
            "temperature": 0.0, "request_timeout": summary.get("request_timeout", 60), "max_tokens": summary.get("max_tokens", 2048),
        }
    for key in options:
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    return validate_options(options), int(context["max_summary_chars"]), max(int(context["summary_trigger_chars"]) * 2, int(context["max_summary_chars"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的目录")
    parser.add_argument("--model", choices=("main", "summary"), default="main")
    parser.add_argument("--base-url")
    parser.add_argument("--model-name")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--start-index", type=int, default=0, help="从第几个有效JSONL记录开始，0为第一条")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-data-only", action="store_true")
    args = parser.parse_args(argv)
    if args.start_index < 0 or args.limit is not None and args.limit < 1:
        parser.error("start-index 必须非负，limit 必须大于0")
    try:
        options, max_chars, input_chars = configured_options(args)
        records = load_inputs(args.input_file.expanduser())
        records = records[args.start_index:]
        if args.limit is not None:
            records = records[:args.limit]
        records = prepare_inputs(records, max_chars, input_chars)
        if args.validate_data_only:
            print(json.dumps({"input": len(records), "options": options, "model_called": False}, ensure_ascii=False, indent=2))
            return 0
        if args.output_dir.expanduser().exists() or args.output_dir.expanduser().is_symlink():
            raise FileExistsError("输出目录已存在，拒绝覆盖")
        import httpx
        key = os.getenv("LOCAL_VLLM_API_KEY", "").strip() or "EMPTY"
        with httpx.Client(trust_env=False, timeout=options["request_timeout"], headers={"Authorization": "Bearer " + key}) as client:
            # 启动前确认模型已经在服务中，不创建空的假训练候选。
            health = client.get(options["base_url"] + "/models", timeout=min(5.0, options["request_timeout"]))
            health.raise_for_status()
            data = health.json().get("data", [])
            if not any(isinstance(model, dict) and model.get("id") == options["model_name"] for model in data):
                raise ValueError("本地服务未加载指定模型，请先启动对应模型服务")
            report = label_records(records, args.output_dir, client, options, max_summary_chars=max_chars, summary_input_chars=input_chars,
                                   progress=lambda event: print(f"{event['index']}/{event['total']} {event['status']}", file=sys.stderr, flush=True))
        print(json.dumps({"output_dir": str(args.output_dir.expanduser().resolve()), "succeeded": report["succeeded"], "failed": report["failed"], "review_status": "pending"}, ensure_ascii=False, indent=2))
        return 0 if not report["failed"] else 1
    except (OSError, ValueError) as exc:
        print(f"自动标注失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"本地模型请求失败（{type(exc).__name__}），请检查服务是否启动及地址/模型名称是否一致。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
