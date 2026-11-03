# 文件用途：用虚构教育会话测量当前或独立摘要模型，保存待人工核验的摘要基线。
# 调用关系：开发者运行本脚本；调用模型工厂和真实 ContextManager 摘要入口，不访问学生数据库。
# 修改易踩坑：这组小样本用于回归检查，不足以训练；请求完成不代表摘要事实正确。
"""测量摘要输出与延迟；质量需逐条对照参考摘要检查。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from agent.context_manager import EducationContextManager
from model.factory import create_chat_model, create_summary_model
from utils.config_handler import app_conf


class RecordingSummaryModel:
    """记录服务是否完成生成，供报告区分模型输出和本地 fallback。"""

    def __init__(self, delegate):
        self.delegate = delegate
        self.reset()

    def reset(self):
        self.called = False
        self.empty = False
        self.error_type = None

    def invoke(self, messages):
        self.called = True
        try:
            response = self.delegate.invoke(messages)
            self.empty = not EducationContextManager._message_content_to_text(response.content)
            return response
        except Exception as exc:
            self.error_type = type(exc).__name__
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("main", "summary"), default="summary")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "data/summary_eval.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")

    cases = []
    for number, line in enumerate(args.cases.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict) or not isinstance(case.get("messages"), list):
            parser.error(f"第 {number} 行需要 messages 列表")
        for message in case["messages"]:
            if message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
                parser.error(f"第 {number} 行消息需要 user/assistant 角色与文本 content")
        cases.append(case)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("评测集不能为空")

    model = create_summary_model() if args.model == "summary" else create_chat_model()
    if model is None:
        parser.error("独立摘要模型未开启：设置 EDUCATION_SUMMARY_ENABLED=true 后重试")
    recorder = RecordingSummaryModel(model)
    context = app_conf["context"]
    manager = EducationContextManager(
        None,
        summary_model=recorder,
        max_conversation_chars=int(context["max_conversation_chars"]),
        max_summary_chars=int(context["max_summary_chars"]),
        summary_trigger_chars=int(context["summary_trigger_chars"]),
    )
    results = []
    for case in cases:
        messages = [
            (AIMessage if message["role"] == "assistant" else HumanMessage)(content=message["content"])
            for message in case["messages"]
        ]
        recorder.reset()
        started = time.perf_counter()
        summary = manager.summarize_conversation(
            previous_summary=case.get("previous_summary", ""), older_messages=messages,
        )
        result = {
            "id": case.get("id", str(len(results) + 1)),
            "reference": case.get("reference", ""),
            "summary": summary,
            "summary_chars": len(summary),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "fallback_used": not recorder.called or recorder.empty or recorder.error_type is not None,
            "error_type": recorder.error_type,
        }
        results.append(result)
        print(f'{result["id"]}: {result["latency_seconds"]}s, {result["summary_chars"]}字符, fallback={result["fallback_used"]}')

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_name": getattr(model, "model_name", None),
        "case_count": len(results),
        "completed_without_fallback": sum(not item["fallback_used"] for item in results),
        "median_latency_seconds": round(statistics.median(item["latency_seconds"] for item in results), 3),
        "quality_review": "逐条对照参考摘要人工核验；请求成功不是准确率。",
        "results": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or PROJECT_ROOT / "storage" / f"summary_baseline_{args.model}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"基线报告：{output}")
    return 0 if report["completed_without_fallback"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
