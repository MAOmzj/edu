# 文件用途：提供终端问答、同步索引、重建索引和显示状态等命令行入口。
# 调用关系：用户通过 python main.py 调用本文件；本文件调用 EducationRuntime 和 EducationQAService。
# 修改易踩坑：重建索引会改持久数据，命令参数要互斥；退出时必须关闭 SQLite 连接。
"""小学教育知识问答系统命令行入口。

这个文件和 app.py 是两个不同入口：app.py 给网页使用，main.py 给终端使用。
两者最终都会调用同一个 EducationAgent，因此回答、记忆和审核逻辑不会出现两套。

main() 的四个分支按优先级依次是：查看索引、同步/重建索引、单次提问、连续提问。
python -m uvicorn app:app --host 127.0.0.1 --port 8766
python main.py --question "1/2 + 1/3 等于多少？" --subject 数学 --grade 5
python main.py --subject 综合 --grade 5 
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from uuid import uuid4

from memory.runtime import EducationRuntime
from rag.vector_store import EducationVectorStore
from utils.config_handler import app_conf


def _configure_console_encoding() -> None:
    """把标准输出切到 UTF-8，避免 Windows GBK 终端无法打印中文与数学符号。

    Web 服务走 UTF-8 JSON 不受影响；这个函数只修复命令行入口。reconfigure
    失败或流不存在时静默跳过，不阻断索引管理、状态查询等纯逻辑命令。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            current_encoding = str(getattr(stream, "encoding", "") or "").lower()
            if current_encoding and current_encoding not in {"utf-8", "utf8"}:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_configure_console_encoding()


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器，并注册问答和知识库管理选项。"""
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
    """以保留中文且便于阅读的格式向终端输出 JSON 数据。"""
    print(json.dumps(value, ensure_ascii=False, indent=2))


def interactive(
    agent,
    subject: str,
    grade: int | None,
    student_id: str,
    thread_id: str,
) -> None:
    """启动连续提问的命令行交互循环，直至用户主动退出。"""
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
    """根据命令行参数执行单次问答、交互问答或知识库管理任务。"""
    args = build_parser().parse_args()

    try:
        # 状态查询只读取清单文件，不创建在线模型，所以没有 API Key 也能执行。
        if args.status:
            _print_json(EducationVectorStore().status())
            return 0

        # 索引管理只需要 Embedding；它不会创建 DeepSeek 聊天模型。
        if args.sync_index or args.rebuild_index:
            vector_store = EducationVectorStore()
            report = (
                vector_store.rebuild()
                if args.rebuild_index
                else vector_store.sync()
            )
            _print_json(asdict(report))
            return 0

        # student_id 标识“哪个学生”，thread_id 标识“这个学生的哪段会话”。
        # 不主动传入时生成临时编号；想继续旧会话就重复使用同一组编号。
        student_id = args.student_id or f"cli-student-{uuid4().hex}"
        thread_id = args.thread_id or f"cli-thread-{uuid4().hex}"
        # with 代码块结束时会自动关闭 SQLite，即使中途发生异常也不会遗漏清理。
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
