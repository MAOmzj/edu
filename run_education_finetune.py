# 文件用途：直接运行Python启动独立教育回答微调，或仅检查准备状态。
# 调用关系：读取education_finetune.yml，调用scripts/train_education.py；不自动暂停或恢复服务。
# 修改易踩坑：训练环境与完整底座必须事先准备；已有输出拒绝覆盖；中断要清理训练子进程组。
# 阅读顺序：文件末尾 main() → 解析参数 → 检查文件 → 启动训练子进程 → 等待或处理中断。
# --check 调用链：本文件 main → scripts.train_education.main → preparation_report；在当前进程检查后返回。
# --dry-run 和正式训练：本文件 main 先调用 preparation_report；检查通过后仍由本文件 main 启动子进程。
# --dry-run 调用链：本文件 main → subprocess.Popen → 训练环境 Python → train_education.main，编码和参数接口检查后返回。
# 正式训练调用链：本文件 main → subprocess.Popen → 训练环境 Python → train_education.main → run_training。
# 注释约定：“上游”是调用当前函数的位置；“下游”是当前函数继续调用的函数或外部程序。
"""运行：python run_education_finetune.py --check；正式训练去掉--check。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

# 模块加载 → pathlib.Path.resolve：从当前脚本位置得到项目目录，后续找配置和训练脚本都以这里为准。
PROJECT_ROOT = Path(__file__).resolve().parent
from utils.education_training import configured_path, load_config


# 上游：文件末尾的 main()，也可由测试传入 argv 列表调用。
# 输入：argv 为命令行参数列表；None 表示从终端读取。返回：检查结果或训练子进程的整数退出码。
# 常见退出码：0 成功、1 准备/启动失败、130 本入口处理中断；正常等待时透传子进程的其它退出码。
# 下游：argparse 解析参数 → 教育训练模块检查准备情况 → subprocess 启动指定 Python 和训练脚本。
def main(argv=None):
    # main → argparse.ArgumentParser：创建参数解析器；下面 add_argument 注册支持的命令行选项。
    parser = argparse.ArgumentParser(description=__doc__)
    # main → parser.add_argument：--config 接收配置文件路径；未传时使用项目内的默认 YAML。
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/education_finetune.yml")
    # main → parser.add_mutually_exclusive_group：创建互斥选项组，--check 和 --dry-run 不能同时使用。
    modes = parser.add_mutually_exclusive_group()
    # main → modes.add_argument：登记只检查文件的开关；store_true 表示写了该选项就得到 True。
    modes.add_argument("--check", action="store_true")
    # main → modes.add_argument：登记只做 token 编码和训练参数接口检查的开关，不执行 GPU 训练。
    modes.add_argument("--dry-run", action="store_true")
    # main → parser.parse_args：把终端参数转换成 args 对象，后面通过 args.check 等读取结果。
    args = parser.parse_args(argv)
    try:
        if args.check:
            # 使用普通项目解释器做文件检查，不要求训练依赖或GPU。
            # 导入别名：check 实际指向 scripts/train_education.py 的 main，不是另一个检查函数。
            from scripts.train_education import main as check
            # 本文件 main → train_education.main：传入配置路径和 --check；其返回退出码后，本入口直接结束。
            return check(["--config", str(args.config), "--check"])
        # main → utils.education_training.load_config：读取并校验 YAML，返回参数字典 config。
        config = load_config(args.config)
        from scripts.train_education import preparation_report
        # main → train_education.preparation_report：传入 config，返回路径、数据数量及缺失项；不加载模型。
        report = preparation_report(config)
        if report["issues"]:
            raise RuntimeError("；".join(report["issues"]))
        # main → utils.education_training.configured_path：把 training_python 配置转换成可用的解释器路径。
        # main → args.config.resolve：把配置路径转为绝对路径，供下游子进程读取同一份配置。
        # command 只是待执行的参数列表；构造到这里还没有启动训练。
        command = [str(configured_path(config, "training_python")),
                   str(PROJECT_ROOT / "scripts/train_education.py"), "--config", str(args.config.resolve())]
        if args.dry_run:
            # main → command.append：把 --dry-run 转交给下游 train_education.main，让它在编码及参数接口检查后返回。
            command.append("--dry-run")
        proc = None
        interrupted = False
        # main → signal.getsignal：分别保存 Ctrl+C / 终止信号的原处理方式，离开本函数前恢复。
        previous = {sig: signal.getsignal(sig) for sig in [signal.SIGINT, signal.SIGTERM]}
        # 上游：下面 signal.signal 注册本回调；收到 SIGINT/SIGTERM 时由 Python 信号机制调用。
        # 输入：信号编号和当前栈帧。下游：无业务函数调用；设置标志或抛出 KeyboardInterrupt 进入清理分支。
        def interrupt(_signum, _frame):
            nonlocal interrupted
            interrupted = True
            # Popen赋值前先记录中断，等拿到本次子进程句柄再清理。
            if proc is not None:
                raise KeyboardInterrupt
        try:
            for sig in previous:
                # main → signal.signal：注册 interrupt 回调；这里传函数本身，不会立即执行 interrupt。
                signal.signal(sig, interrupt)
            if interrupted:
                raise KeyboardInterrupt
            # main → subprocess.Popen：启动“训练环境 Python scripts/train_education.py --config ...”。
            # 下游新进程进入 train_education.py 末尾的 main()；proc 是进程句柄，不是模型对象。
            # cwd 固定工作目录；start_new_session 让本次训练及其子进程形成可一起停止的进程组。
            proc = subprocess.Popen(command, cwd=str(PROJECT_ROOT), start_new_session=True)
            if interrupted:
                raise KeyboardInterrupt
            # main → proc.wait：等待下游退出，并把它的退出码作为本入口的返回值。
            return proc.wait()
        except KeyboardInterrupt:
            # main → print：向终端显示中断状态；flush=True 表示立即输出。
            print("正在停止本次教育训练进程…", flush=True)
            for sig in previous:
                # main → signal.signal：清理期间先忽略重复中断，避免尚未释放训练进程又被打断。
                signal.signal(sig, signal.SIG_IGN)
            if proc is not None:
                try:
                    # main → os.killpg：向本次训练进程组发送 SIGTERM，请求下游结束并释放 GPU。
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    # main → proc.wait：最多等待 15 秒，让下游正常退出；超时则进入强制停止分支。
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        # main → os.killpg：下游超过等待时间仍未退出时，对同一个进程组发送 SIGKILL。
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # main → proc.wait：回收已强制停止的子进程，最多再等 10 秒。
                    proc.wait(timeout=10)
            return 130
        finally:
            for sig, handler in previous.items():
                # main → signal.signal：恢复进入 main 前的信号处理方式，正常结束或异常都执行。
                signal.signal(sig, handler)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        # main → print：将准备失败或启动异常写到标准错误，随后返回失败退出码 1。
        print(f"教育训练未启动或失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Python 直接运行本文件 → main：从这里开始执行；SystemExit 把 main 返回的退出码交给终端。
    raise SystemExit(main())
