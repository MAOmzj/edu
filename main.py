"""小学教育知识问答系统命令行入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from uuid import uuid4

from memory.runtime import EducationRuntime
from rag.vector_store import EducationVectorStore
from utils.config_handler import app_conf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="面向小学一至六年级的教育知识问答系统"
    )
    parser.add_argument("-q", "--question", help="要提问的内容")
    parser.add_argument(
        "-s",
        "--subject",
        choices=app_conf["supported_subjects"],
        default="综合",
        help="问题所属学科",
    )
    parser.add_argument(
        "-g",
        "--grade",
        type=int,
        choices=range(1, 7),
        help="学生年级（1—6）",
    )
    parser.add_argument(
        "--student-id",
        help="匿名学生 ID；重复使用可读取长期学习记忆",
    )
    parser.add_argument(
        "--thread-id",
        help="会话线程 ID；重复使用可继续短期对话",
    )

    management = parser.add_mutually_exclusive_group()
    management.add_argument(
        "--sync-index",
        action="store_true",
        help="增量同步知识库索引",
    )
    management.add_argument(
        "--rebuild-index",
        action="store_true",
        help="清空并重建知识库索引",
    )
    management.add_argument(
        "--status",
        action="store_true",
        help="查看本地索引清单状态（不调用在线模型）",
    )
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def interactive(
    agent,
    subject: str,
    grade: int | None,
    student_id: str,
    thread_id: str,
) -> None:
    print("小学知识小助手已启动。输入问题开始学习，输入 exit 退出。")
    print(f"匿名学生 ID：{student_id}")
    print(f"当前会话 ID：{thread_id}")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见，祝你学习进步！")
            return
        if question.lower() in {"exit", "quit", "退出"}:
            print("再见，祝你学习进步！")
            return
        if not question:
            continue
        try:
            print(
                "\n小助手："
                + agent.answer(
                    question,
                    subject,
                    grade,
                    student_id=student_id,
                    thread_id=thread_id,
                )
            )
        except Exception as exc:
            print(f"\n本次回答失败：{exc}")


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.status:
            _print_json(EducationVectorStore().status())
            return 0

        if args.sync_index or args.rebuild_index:
            vector_store = EducationVectorStore()
            report = (
                vector_store.rebuild()
                if args.rebuild_index
                else vector_store.sync()
            )
            _print_json(asdict(report))
            return 0

        student_id = args.student_id or f"cli-student-{uuid4().hex}"
        thread_id = args.thread_id or f"cli-thread-{uuid4().hex}"
        with EducationRuntime() as runtime:
            if args.question:
                print(
                    runtime.agent.answer(
                        args.question,
                        args.subject,
                        args.grade,
                        student_id=student_id,
                        thread_id=thread_id,
                    )
                )
                return 0
            interactive(
                runtime.agent,
                args.subject,
                args.grade,
                student_id,
                thread_id,
            )
        return 0
    except Exception as exc:
        print(f"运行失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


#python -m uvicorn app:app --host 127.0.0.1 --port 8766
