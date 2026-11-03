# 文件用途：直接运行本文件，用当前1000条候选数据训练Qwen3-4B摘要LoRA。
# 调用关系：本文件用独立训练Python调用scripts/train_summary.py；复用serve.py识别项目服务。
# 修改易踩坑：先预检再暂停服务；原进程环境仅留内存；失败/中断时先停止训练再恢复原服务。
"""运行：python run_summary_finetune.py；仅检查：python run_summary_finetune.py --dry-run。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# 修改这些设置即可选择数据、参数和新的输出目录，不需要设置Shell变量。
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"
TRAINING_PYTHON = Path("/root/MAOMI/envs/summary-training/bin/python")
BASE_MODEL = Path("/root/MAOMI/models/Qwen3-4B")
TRAIN_FILE = PROJECT_ROOT / "storage/summary_dataset_v2/train.jsonl"
VALIDATION_FILE = PROJECT_ROOT / "storage/summary_dataset_v2/validation.jsonl"
OUTPUT_DIR = Path("/root/MAOMI/training/summary-lora-v2")  # 必须尚不存在；再训练可改为v3。
EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQUENCE_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42


class ChildCleanupError(RuntimeError):
    """子进程组未证明清理完成，此时禁止恢复GPU模型服务。"""


def training_command():
    """参数写在Python中，由子进程调用已经验证过的训练实现。"""
    return [str(TRAINING_PYTHON), str(PROJECT_ROOT / "scripts/train_summary.py"),
            "--model-path", str(BASE_MODEL), "--train-file", str(TRAIN_FILE),
            "--validation-file", str(VALIDATION_FILE), "--output-dir", str(OUTPUT_DIR),
            "--max-seq-length", str(MAX_SEQUENCE_LENGTH), "--epochs", str(EPOCHS),
            "--learning-rate", str(LEARNING_RATE), "--batch-size", str(BATCH_SIZE),
            "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
            "--lora-r", str(LORA_R), "--lora-alpha", str(LORA_ALPHA),
            "--lora-dropout", str(LORA_DROPOUT), "--seed", str(SEED)]


def process_identity(pid):
    """同时使用PID与启动时间，避免停止被系统复用的进程编号。"""
    try:
        after = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        if after[0] == "Z":
            return None
        return int(after[19])  # /proc/stat第22字段starttime。
    except (OSError, ValueError, IndexError):
        return None


@dataclass
class ServiceSnapshot:
    service: object
    pid: int
    identity: int
    argv: tuple[str, ...] = field(repr=False)
    cwd: Path
    environment: dict[str, str] = field(repr=False)


def capture_services(manager):
    """快照登记的项目服务；环境与argv不写报告，防止泄露其中的凭据。"""
    snapshots = []
    for service in manager.services().values():
        pid = manager._running_pid(service)
        if pid is None:
            if manager._port_in_use(service.port):
                raise RuntimeError(f"{service.key}端口被未登记服务占用，请先在对应终端处理")
            continue
        identity = process_identity(pid)
        if identity is None or os.getpgid(pid) != pid:
            raise RuntimeError(f"{service.key}进程身份或进程组不明确，未暂停任何服务")
        proc = Path(f"/proc/{pid}")
        argv = tuple(os.fsdecode(arg) for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg)
        if service.key in {"chat", "embedding", "summary"}:
            if "vllm.entrypoints.cli.main" not in argv or "serve" not in argv:
                raise RuntimeError(f"{service.key}不是已知的项目vLLM进程")
            model_path = service.argv[service.argv.index("serve") + 1]
            if model_path not in argv or "--port" not in argv or argv[argv.index("--port") + 1] != str(service.port):
                raise RuntimeError(f"{service.key}模型路径或端口与登记不一致")
        elif ("uvicorn" not in argv or "app:app" not in argv or "--port" not in argv
              or argv[argv.index("--port") + 1] != str(service.port)
              or Path(os.readlink(proc / "cwd")).resolve() != PROJECT_ROOT):
            raise RuntimeError("网页进程、端口或工作目录与登记不一致")
        environment = {}
        for value in (proc / "environ").read_bytes().split(b"\0"):
            if b"=" in value:
                key, content = value.split(b"=", 1)
                environment[os.fsdecode(key)] = os.fsdecode(content)
        snapshots.append(ServiceSnapshot(service, pid, identity, argv,
                                         Path(os.readlink(proc / "cwd")), environment))
        if process_identity(pid) != identity:
            raise RuntimeError("快照期间进程身份发生变化，未暂停任何服务")
    return snapshots


def verify_gpu_ownership(snapshots):
    """仅允许当前项目服务占用GPU；不自动终止其它训练或手动模型进程。"""
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    groups = {item.pid for item in snapshots if item.service.key in {"chat", "embedding", "summary"}}
    unknown = []
    for line in result.stdout.splitlines():
        if not line.strip().isdigit():
            continue
        pid = int(line.strip())
        try:
            if os.getpgid(pid) not in groups:
                unknown.append(pid)
        except ProcessLookupError:
            continue
    if unknown:
        raise RuntimeError(f"GPU有其它未登记进程 {unknown}，请先处理；本脚本未停止任何服务")


def signal_original(item, sig):
    try:
        if process_identity(item.pid) == item.identity:
            if os.getpgid(item.pid) != item.pid:
                raise RuntimeError("原服务进程组已改变，拒绝发送信号")
            if process_identity(item.pid) == item.identity:
                os.killpg(item.pid, sig)
    except ProcessLookupError:
        # 前台all监督器可能同时完成其它子服务的清理。
        pass


def pause_services(manager, snapshots):
    # 当前serve.py all前台监督器在子进程退出后会清理整组服务，因此全部登记服务一起暂停。
    for item in reversed(snapshots):
        print(f"暂停项目服务：{item.service.title}", flush=True)
        signal_original(item, signal.SIGTERM)
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        if all(not manager._process_group_alive(item.pid) for item in snapshots):
            return
        time.sleep(0.2)
    for item in snapshots:
        # 仅使用仍匹配的原进程身份，不依据可能被改写的PID文件清理。
        if manager._process_group_alive(item.pid):
            signal_original(item, signal.SIGKILL)
    if any(manager._process_group_alive(item.pid) for item in snapshots):
        raise RuntimeError("原模型子进程尚未退出，停止训练以避免显存冲突")


def stop_child(proc):
    """训练子进程独立成组，中断时等待整组结束后才恢复模型服务。"""
    if proc.poll() is not None and Path(f"/proc/{proc.pid}").exists():
        raise ChildCleanupError("训练PID已经被复用，不向该进程发送信号")
    # 训练主进程退出后也可能留有同组子进程，先清理整组。
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
    from scripts import serve
    deadline = time.monotonic() + 10
    while serve._process_group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if serve._process_group_alive(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while serve._process_group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if serve._process_group_alive(proc.pid):
            raise ChildCleanupError("训练子进程组尚未退出，不能恢复显存服务")


def run_child(command):
    proc = subprocess.Popen(command, cwd=str(PROJECT_ROOT), start_new_session=True)
    try:
        return proc.wait()
    finally:
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                stop_child(proc)
            except Exception as exc:
                raise ChildCleanupError("检查或训练子进程清理失败") from exc
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def restore_services(manager, snapshots):
    """按原实际命令和完整环境后台恢复，原来未运行的服务不启动。"""
    errors = []
    ordered = sorted(snapshots, key=lambda item: item.service.key == "web")
    for item in ordered:
        service = item.service
        proc = None
        try:
            if process_identity(item.pid) == item.identity:
                continue
            if manager._port_in_use(service.port):
                raise RuntimeError("端口已被其它实例占用，未替换或停止它")
            if manager._process_group_alive(item.pid):
                raise RuntimeError("原进程组尚未完全退出")
            log_path = manager.LOG_DIR / f"serve-{service.key}-restored-{time.time_ns()}.log"
            with log_path.open("xb") as log:
                proc = subprocess.Popen(item.argv, cwd=str(item.cwd), env=item.environment,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            service.pid_path.write_text(f"{proc.pid}\n")
            service.log_path.unlink(missing_ok=True)
            service.log_path.symlink_to(log_path.name)
            print(f"恢复原服务：{service.title}，等待就绪…", flush=True)
            ready, reason = manager._wait_ready(service, proc, service.ready_timeout)
            if not ready:
                raise RuntimeError(reason)
            print(f"已恢复：{service.title}", flush=True)
        except Exception as exc:
            if proc is not None:
                manager._stop_started(service, proc)
            errors.append(f"{service.key}: {type(exc).__name__}")
            print(f"恢复{service.key}失败：{exc}；请查看对应serve日志", file=sys.stderr, flush=True)
    return errors


def check_paths():
    for path in [TRAINING_PYTHON, TRAIN_FILE, VALIDATION_FILE, BASE_MODEL / "config.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要文件：{path}")
    if OUTPUT_DIR.exists() or OUTPUT_DIR.is_symlink():
        raise FileExistsError(f"输出目录已存在，请修改文件顶部OUTPUT_DIR：{OUTPUT_DIR}")
    config = json.loads((BASE_MODEL / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("quantization_config"):
        raise ValueError("BASE_MODEL必须是未量化的原始Qwen3基座")
    if MAX_SEQUENCE_LENGTH > config["max_position_embeddings"]:
        raise ValueError("完整训练序列长度超过模型上下文窗口")
    index = BASE_MODEL / "model.safetensors.index.json"
    if index.is_file():
        weights = set(json.loads(index.read_text())["weight_map"].values())
        if not weights or any(not (BASE_MODEL / name).is_file() for name in weights):
            raise FileNotFoundError("基座权重分片不完整")
    elif not (BASE_MODEL / "model.safetensors").is_file():
        raise FileNotFoundError("基座缺少safetensors权重")
    parent = OUTPUT_DIR.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 2 * 1024 ** 3:
        raise RuntimeError("输出磁盘空闲不足2GiB，请为适配器及检查点准备空间")


def launch(*, dry_run=False):
    from scripts import serve
    check_paths()
    command = training_command()
    reviews = Counter(json.loads(line).get("review_status", "unknown")
                      for line in TRAIN_FILE.read_text().splitlines() if line.strip())
    print(f"训练数据：{sum(reviews.values())}条，复核状态：{dict(reviews)}", flush=True)
    if reviews.get("pending"):
        print("新增模型标签仍待语义复核，本次为实验训练；数据状态不会被改成已审核。", flush=True)
    print("先校验数据和token监督范围…", flush=True)
    if run_child(command + ["--dry-run"]) != 0:
        raise RuntimeError("预检失败，未暂停服务")
    if dry_run:
        print("检查完成；未暂停服务或启动训练。", flush=True)
        return 0
    dependency_check = "import torch, peft, accelerate; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '需要支持CUDA BF16的GPU'"
    if run_child([str(TRAINING_PYTHON), "-c", dependency_check]) != 0:
        raise RuntimeError("训练环境预检失败，未暂停服务")
    check_paths()
    snapshots = capture_services(serve)
    verify_gpu_ownership(snapshots)
    report_path = serve.LOG_DIR / f"summary-finetune-launch-{time.time_ns()}.json"
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
              "train_file": str(TRAIN_FILE), "train_sha256": hashlib.sha256(TRAIN_FILE.read_bytes()).hexdigest(),
              "validation_sha256": hashlib.sha256(VALIDATION_FILE.read_bytes()).hexdigest(),
              "review_status_counts": dict(reviews), "output_dir": str(OUTPUT_DIR),
              "training_arguments": command[2:],
              "original_services": [item.service.key for item in snapshots],
              "restoration_mode": "original child commands restored as independent background services"}
    def save_report():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save_report()
    code = 1
    pause_started = False
    training_cleanup_failed = False
    try:
        pause_started = True
        pause_services(serve, snapshots)
        verify_gpu_ownership([])
        print(f"开始训练：{EPOCHS}轮；适配器将保存到{OUTPUT_DIR / 'adapter'}", flush=True)
        report["status"] = "training"
        save_report()
        code = run_child(command)
        report["status"] = "completed" if code == 0 else "training_failed"
    except KeyboardInterrupt:
        code = 130
        report["status"] = "interrupted"
        print("训练被中断，正在恢复原服务…", flush=True)
    except ChildCleanupError:
        training_cleanup_failed = True
        report["status"] = "child_cleanup_failed"
        raise
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        # 清理期间避免第二个普通中断跳过服务恢复；SIGKILL无法由Python处理。
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            # 暂停途中中断时，原启动器可能仍在清理服务，必须先完成停机再恢复。
            shutdown_errors = []
            if pause_started and any(serve._process_group_alive(item.pid) for item in snapshots):
                try:
                    pause_services(serve, snapshots)
                except Exception as exc:
                    shutdown_errors.append(f"原服务清理失败: {type(exc).__name__}")
                    print("原服务组未完成清理，无法保证恢复；请查看serve状态。", file=sys.stderr)
            restore_targets = snapshots
            gpu_unavailable = training_cleanup_failed
            if not gpu_unavailable:
                try:
                    verify_gpu_ownership([])
                except Exception:
                    gpu_unavailable = True
            if gpu_unavailable:
                # 训练组尚未清理或别的任务占用了GPU时，不启动显存服务与它竞争。
                restore_targets = [item for item in snapshots if item.service.key == "web"]
                shutdown_errors.append("训练组未退出或GPU仍有进程，未恢复模型服务")
                print("训练组尚未确认退出或GPU未释放，报告会记录未恢复的模型服务；请核查后手动恢复。", file=sys.stderr)
            report["restoration_errors"] = shutdown_errors + restore_services(serve, restore_targets)
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_report()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if report["restoration_errors"]:
        return 1
    if code == 0:
        print(f"训练完成：{OUTPUT_DIR / 'adapter'}", flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只校验数据与token，不暂停服务、不训练")
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    # 从系统Python或IDE运行也自动进入项目环境，模型训练另用TRAINING_PYTHON。
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        if not PROJECT_PYTHON.is_file():
            parser.error("项目.venv不存在，请先准备项目环境")
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *arguments])
    from scripts import serve
    serve.LOG_DIR.mkdir(parents=True, exist_ok=True)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, interrupted)
        with (serve.LOG_DIR / "summary-finetune.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("已有另一个摘要训练启动器正在运行") from exc
            return launch(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("预检被中断，未启动训练。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
