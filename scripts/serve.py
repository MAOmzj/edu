# 文件用途：用 Python 统一启动、检查和停止课堂问答助手的模型与网页常驻服务。
# 调用关系：开发者运行本脚本；本脚本用 subprocess 分别拉起 vllm-env 与项目 .venv 中的解释器。
# 修改易踩坑：vLLM 的显存配额按每张卡计算；各服务的可见 GPU 要在创建子进程前分别设置。
"""课堂问答助手的服务启动器。

替代 MAOMI/legacy_scripts 下的三个 shell 脚本，把"用哪个解释器、开哪些
环境变量、日志写到哪、怎么算启动成功"收拢到一个入口：

    python scripts/serve.py chat        # 聊天模型   :8000
    python scripts/serve.py embedding   # 向量模型   :8001
    python scripts/serve.py summary     # 独立摘要模型 :8002（可单独验证）
    python scripts/serve.py web         # 网页服务   :8766
    python scripts/serve.py all         # 依次等待模型就绪，再启动网页
    python scripts/serve.py status      # 查看谁在跑、是否就绪
    python scripts/serve.py stop all    # 优雅停止

三个服务之间只靠 HTTP 连接：网页服务自己不加载任何模型，收到提问时才去调
8000（聊天）和 8001（向量）。所以它们可以分别启动、分别重启，但一次提问要
成功，三者必须都在线——少了向量服务，RAG 检索会直接失败。
summary 是可选的独立本地摘要服务，由 config/model.yml 的 summary.enabled
或 EDUCATION_SUMMARY_ENABLED 控制是否跟随 all 启动；它只监听本机。

默认前台运行，子进程日志会同时进终端和 logs/；Ctrl-C 会先给子进程发 SIGTERM。
加 -d/--detach 转入后台；all 仍逐个等待模型就绪，最后确认网页就绪。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 端口和模型名以 config/model.yml 为唯一来源，避免脚本与配置各写一份然后慢慢漂移。
from utils.config_handler import get_summary_model_config, model_conf  # noqa: E402
from utils.quantized_base import validate_quantized_lora_base  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# vLLM 装在独立的 venv 里，和项目 .venv 不是同一个。用错解释器会直接 ImportError。
VLLM_ENV_DIR = Path(os.getenv("EDU_VLLM_ENV", "/root/MAOMI/envs/vllm"))
VLLM_PYTHON = VLLM_ENV_DIR / "bin" / "python"
MODEL_DIR = Path(os.getenv("EDU_MODEL_DIR", "/root/MAOMI/models"))

# 以下为未显式配置时的兼容默认值；聊天侧实际预算以配置或 QWEN_GPU_MEM_UTIL 为准。
# TP 部署中该比例用于每张可见 GPU，不是两张卡合计的比例。
EMBED_GPU_MEM_UTIL = os.getenv("EMBED_GPU_MEM_UTIL", "0.05")

# vLLM 监听 0.0.0.0 是为了配合 AutoDL 的端口映射；只在本机用可以设成 127.0.0.1。
VLLM_HOST = os.getenv("EDU_VLLM_HOST", "0.0.0.0")
WEB_HOST = os.getenv("EDU_HOST", "127.0.0.1")

# 所有 vLLM 实例共用的环境变量。这里每一项都不是可选的：
#   VLLM_USE_FLASHINFER_SAMPLER=0  系统 nvcc 是 CUDA 11.8，FlashInfer 采样 kernel
#      的 JIT 编译会失败；注意力本身跑在 FLASH_ATTN 上，关掉它没有损失。
#   HF_ENDPOINT                    从 hf-mirror 拉 tokenizer 等附带文件，官方源不通。
VLLM_COMMON_ENV = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "OMP_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Service:
    """一个可以被启动、探活和停止的常驻服务。"""

    key: str
    title: str
    port: int
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    health_path: str
    ready_timeout: float
    # 用于确认 pid 文件里记的进程确实是本服务，避免 pid 被系统复用后误杀无关进程。
    marker: str

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.health_path}"

    @property
    def log_path(self) -> Path:
        """最新一次运行的日志。这是一个符号链接，始终指向本次运行的日志文件，
        所以 `tail -f logs/serve-chat.log` 这类用法不受每次换文件的影响。"""
        return LOG_DIR / f"serve-{self.key}.log"

    @property
    def pid_path(self) -> Path:
        return LOG_DIR / f"serve-{self.key}.pid"

    def new_run_log(self) -> Path:
        """为本次启动分配一个带时间戳的日志文件。

        不共用一个不断追加的文件有两个原因：vLLM 启动日志很长，长期累积会撑满
        数据盘；而且日志混在一起后，失败时 tail 出来
        的末尾会夹着上一次运行的输出，反而看不清这次为什么失败。
        """
        stamp = time.strftime("%m%d_%H%M%S")
        return LOG_DIR / f"serve-{self.key}-{stamp}.log"


def _configure_console() -> None:
    """把标准输出切成 UTF-8 并对齐行缓冲。

    编码：终端不是 UTF-8 时，中文日志会变成乱码。
    缓冲：输出重定向到文件或管道时 Python 默认整块缓冲，本脚本自己的进度提示
    会一直卡在缓冲区里，`serve.py all | tee 日志` 就只能看到子进程转发的那部分
    （转发线程有显式 flush），看不到"等待就绪"这类提示，很容易以为程序卡死了。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            current = str(getattr(stream, "encoding", "") or "").lower()
            if current and current not in {"utf-8", "utf8"}:
                stream.reconfigure(encoding="utf-8", errors="replace")
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError, OSError):
            pass


def _local_port(base_url: str, service: str) -> int:
    """从配置的 base_url 中取出端口，并确认它指向本机。

    配置里写的是客户端要连的地址，本脚本把它当作服务端口的唯一来源：改端口
    只需要改 config/model.yml，脚本和网页服务会一起跟着变。
    """
    parsed = urlparse(str(base_url))
    if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        raise ValueError(
            f"{service} 配置的地址是 {base_url}，不在本机，无法由本脚本启动。\n"
            "  要改用本地模型，请把 config/model.yml 里对应地址改回 http://127.0.0.1:端口。"
        )
    port = parsed.port
    if port is None:
        raise ValueError(f"{service} 配置的地址缺少端口号：{base_url}")
    return port


def _chat_model_options() -> tuple[Path, str, tuple[str, ...]]:
    """由 _build_services 调用：读取基座/适配器，返回 vLLM 的模型名与 LoRA 参数。"""
    configured = str(model_conf.get("chat_model_path") or "").strip()
    base = Path(configured).expanduser() if configured else MODEL_DIR / "Qwen3.8-27B-AWQ-INT4"
    name = str(model_conf["chat_model_name"]).strip()
    adapter_value = str(model_conf.get("chat_lora_path") or "").strip()
    if not adapter_value:
        return base, name, ()

    # 下游 json/Path 只读取文件；启动前检查，防止适配器缺失时悄悄退回基座推理。
    adapter = Path(adapter_value).expanduser().resolve()
    for path in (base / "config.json", adapter / "adapter_config.json",
                 adapter / "adapter_model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(f"聊天微调部署缺少文件：{path}")
    info = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    recorded_base = str(info.get("base_model_name_or_path") or "")
    if not recorded_base:
        raise ValueError("chat_model_path 必须与适配器记录的训练基座一致")
    training_base = Path(recorded_base).expanduser().resolve()
    quantization = str(model_conf.get("chat_quantization") or "").strip().lower()
    if quantization == "compressed-tensors":
        # 下游校验器绑定量化来源、检查点快照及原适配器；不修改适配器训练记录。
        manifest = str(model_conf.get("chat_quantized_base_manifest") or "").strip()
        if not manifest:
            raise ValueError("INT4 + LoRA 必须配置 chat_quantized_base_manifest")
        validate_quantized_lora_base(base, training_base, adapter, Path(manifest).expanduser())
    elif training_base != base.resolve():
        raise ValueError("chat_model_path 必须与适配器记录的训练基座一致")
    rank = info.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}:
        raise ValueError(f"适配器 r={rank!r} 不在当前 vLLM 支持的 rank 中")
    if info.get("peft_type") != "LORA":
        raise ValueError("chat_lora_path 必须指向 LoRA 适配器")

    base_name = name + "-base"
    # 下游 vLLM 读取 --lora-modules：客户端用 name 选中适配器，base_name 仅用于对照。
    lora = json.dumps({"name": name, "path": str(adapter), "base_model_name": base_name})
    return base, base_name, (
        "--enable-lora", "--lora-modules", lora,
        "--max-lora-rank", str(rank), "--max-loras", "1",
        "--dtype", "bfloat16", "--enforce-eager",
        # 教学数据为纯文本，关闭图片/视频输入以避免额外的视觉 profiling 显存。
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
    )


def _chat_quantization_options(tensor_parallel_size: int = 1) -> tuple[str, ...]:
    """由 _build_services 调用：选择原生 INT4 或与 QLoRA 一致的 NF4 加载。"""
    quantization = str(model_conf.get("chat_quantization") or "").strip().lower()
    if quantization in {"", "none", "bf16"}:
        return ()
    if quantization == "compressed-tensors":
        # 下游 vLLM 从量化 config 选择 INT4 内核；不传 NF4 overrides 或 BNB loader。
        configured = str(model_conf.get("chat_model_path") or "").strip()
        base = Path(configured).expanduser() if configured else MODEL_DIR / "Qwen3.8-27B-AWQ-INT4"
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        quant = config.get("quantization_config") or {}
        groups = quant.get("config_groups") or {}
        valid = (quant.get("quant_method") == "compressed-tensors"
                 and quant.get("format") == "pack-quantized"
                 and quant.get("quantization_status") == "compressed"
                 and bool(groups))
        for group in groups.values():
            weights = group.get("weights") or {}
            valid = (valid and weights.get("type") == "int" and weights.get("num_bits") == 4
                     and not group.get("input_activations") and not group.get("output_activations"))
        if not valid:
            raise ValueError("chat_quantization=compressed-tensors 要求已导出的 W4A16 INT4 权重")
        return ("--quantization", "compressed-tensors", "--load-format", "auto")
    if quantization != "nf4":
        raise ValueError("chat_quantization 只能为空、bf16、nf4或compressed-tensors")
    # 下游 BNB 插件按此配置量化原始基座；模型文件不改写，LoRA仍独立加载。
    overrides = {"quantization_config": {
        "quant_method": "bitsandbytes", "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_skip_modules": ["visual", "mtp", "lm_head", "embed_tokens"],
    }}
    loader_args = ()
    if tensor_parallel_size > 1:
        # 下游项目插件验证磁盘基座未量化，再用真正的inflight NF4按每卡分片加载。
        # 官方BNB暂不支持Qwen融合Q/K/V/Z的分组TP加载，该组走原生BF16分片。
        overrides["quantization_config"]["llm_int8_skip_modules"].append("in_proj_qkvz")
        loader_args = ("--model-loader-extra-config", '{"maomi_inflight_nf4":true}')
    return ("--quantization", "bitsandbytes", "--load-format", "bitsandbytes",
            "--hf-overrides", json.dumps(overrides), *loader_args)


def _vllm_service_env(device_key: str, tensor_parallel_size: int = 1) -> dict[str, str]:
    """由 _build_services 调用：为单个 vLLM 服务校验并设置可见 GPU。"""
    env = dict(VLLM_COMMON_ENV)
    if device_key not in model_conf:
        # 下游 _start_one 合并父环境；未配置时保留原 CUDA_VISIBLE_DEVICES 继承行为。
        return env
    configured = model_conf[device_key]
    devices = [] if configured is None else str(configured).split(",")
    devices = [device.strip() for device in devices]
    if not devices or any(not device for device in devices):
        raise ValueError(f"{device_key} 必须是非空的 GPU 列表，例如 '0,1'")
    if len(set(devices)) != len(devices):
        raise ValueError(f"{device_key} 不能包含重复的 GPU")
    if len(devices) < tensor_parallel_size:
        raise ValueError(f"{device_key} 的 GPU 数量不能少于 tensor_parallel_size={tensor_parallel_size}")
    # 下游 subprocess 用此环境启动 API 和各 TP worker，不影响其它模型服务。
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    return env


def _tensor_parallel_options(prefix: str) -> tuple[int, tuple[str, ...], dict[str, str]]:
    """由 _build_services 调用：为每个模型分别生成 TP 参数和 worker 环境。"""
    key = f"{prefix}_tensor_parallel_size"
    size = model_conf.get(key, 1)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"{key} 必须是正整数")
    # 下游 vLLM 默认 TP=1；保留旧版小模型启动命令，只有显式多卡才增加参数。
    args = ("--tensor-parallel-size", str(size)) if prefix == "chat" or size > 1 else ()
    env = _vllm_service_env(f"{prefix}_cuda_visible_devices", size)
    if size > 1:
        args += ("--distributed-executor-backend", "mp")
        # 下游 subprocess → vLLM → multiprocessing 创建两个独立 TP worker。
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return size, args, env


def _gpu_memory_utilization(prefix: str, env_key: str, default: str) -> str:
    """由 _build_services 调用：环境变量优先于 YAML，校验每模型每卡预算。"""
    key = f"{prefix}_gpu_memory_utilization"
    configured = os.getenv(env_key, model_conf.get(key, default))
    try:
        valid = not isinstance(configured, bool) and 0 < float(configured) <= 1
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{key} / {env_key} 必须大于0且不超过1")
    # 下游 vLLM 按 GPU 总显存乘此比例计算预算，不是剩余显存的比例。
    return str(configured)


def _build_services() -> dict[str, Service]:
    """按配置组装服务，包括可显式启动的独立摘要模型。"""

    chat_port = _local_port(model_conf["chat_base_url"], "聊天模型")
    embed_port = _local_port(model_conf["embedding_base_url"], "向量模型")
    # 网页服务沿用 app.py 里 __main__ 的默认端口，保持两处一致。
    web_port = int(os.getenv("EDU_PORT", "8766"))

    # 下游 _chat_model_options 校验基座/适配器，并生成 vLLM 静态加载参数。
    chat_model, chat_served_name, chat_lora_args = _chat_model_options()
    finetuned_chat = bool(chat_lora_args)
    chat_tp, chat_parallel_args, chat_env = _tensor_parallel_options("chat")
    embedding_tp, embedding_parallel_args, embedding_env = _tensor_parallel_options("embedding")
    summary_tp, summary_parallel_args, summary_env = _tensor_parallel_options("summary")
    # 下游函数选择 NF4 或已导出的 INT4 加载；适配器仍独立由 vLLM 加载。
    chat_quantization_args = _chat_quantization_options(chat_tp)
    embed_model = MODEL_DIR / "Qwen3-Embedding-0.6B"
    summary_conf = get_summary_model_config()
    summary_model = Path(summary_conf["model_path"])
    summary_port = _local_port(summary_conf["base_url"], "摘要模型")
    summary_max_model_len = int(os.getenv("SUMMARY_MAX_MODEL_LEN", "16384"))
    if summary_max_model_len < 1:
        raise ValueError("SUMMARY_MAX_MODEL_LEN 必须是正整数")
    chat_gpu_mem_util = _gpu_memory_utilization(
        "chat", "QWEN_GPU_MEM_UTIL",
        ("0.75" if finetuned_chat else "0.70") if summary_conf["enabled"] else "0.80",
    )
    embedding_gpu_mem_util = _gpu_memory_utilization(
        "embedding", "EMBED_GPU_MEM_UTIL", "0.04" if finetuned_chat else EMBED_GPU_MEM_UTIL,
    )
    summary_gpu_mem_util = _gpu_memory_utilization(
        "summary", "SUMMARY_GPU_MEM_UTIL", "0.16" if finetuned_chat else "0.20",
    )

    # vLLM 的 cwd 故意放在 vllm-env 的上一级，而不是项目根目录。
    # `python -m` 会把 cwd 放进 sys.path[0]，而项目根下有 model/、utils/、agent/
    # 这些很常见的包名，可能遮蔽 vLLM 自己的依赖。
    vllm_cwd = VLLM_ENV_DIR.parent

    chat = Service(
        key="chat",
        title="聊天模型 Qwen3.8-27B" + (
                  " INT4 W4A16" if str(model_conf.get("chat_quantization") or "").lower() == "compressed-tensors"
                  else " 4bit NF4" if chat_quantization_args else "")
              + (" 教育微调" if finetuned_chat else "")
              + (f" TP={chat_tp}" if chat_tp > 1 else ""),
        port=chat_port,
        argv=(
            str(VLLM_PYTHON),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(chat_model),
            # 无适配器时这是客户端名称；有适配器时客户端用 --lora-modules 的 name。
            "--served-model-name",
            chat_served_name,
            *chat_lora_args,
            *chat_quantization_args,
            *chat_parallel_args,
            "--max-model-len",
            "16384",
            "--kv-cache-dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            chat_gpu_mem_util,
            "--max-num-seqs",
            "4" if finetuned_chat else "24",
            "--max-num-batched-tokens",
            "2048",
            "--reasoning-parser",
            "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            # reasoning_effort=medium 在该模型的模板里没有分支，等于不注入任何指令；
            # 真正关闭思考走的是 factory 里的 enable_thinking=false，这里保持一致。
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}' if finetuned_chat else '{"reasoning_effort":"medium"}',
            "--host",
            VLLM_HOST,
            "--port",
            str(chat_port),
        ),
        cwd=vllm_cwd,
        env=chat_env,
        health_path="/v1/models",
        # 27B 从磁盘加载可能需要几分钟，超时给宽一点。
        ready_timeout=900.0,
        marker=chat_model.name,
    )

    embedding = Service(
        key="embedding",
        title="向量模型 Qwen3-Embedding-0.6B" + (f" TP={embedding_tp}" if embedding_tp > 1 else ""),
        port=embed_port,
        argv=(
            str(VLLM_PYTHON),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(embed_model),
            # vLLM 0.29 移除了 --task：向量模型改用 --runner pooling 只输出池化向量。
            # Qwen3-Embedding 的架构是 Qwen3ForCausalLM（生成式模型改作向量），
            # 必须再用 --convert embed 适配，否则会被当成生成任务加载。
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--served-model-name",
            str(model_conf["embedding_model_name"]),
            *embedding_parallel_args,
            # 下游 CUDA Graph 捕获时，本机双卡自定义 all-reduce 的 IPC 注册失败。
            # 仅向量 TP 服务改用 NCCL 通信；保留 torch.compile 与 CUDA Graph。
            *(("--disable-custom-all-reduce",) if embedding_tp > 1 else ()),
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            embedding_gpu_mem_util,
            "--host",
            VLLM_HOST,
            "--port",
            str(embed_port),
        ),
        cwd=vllm_cwd,
        env=embedding_env,
        health_path="/v1/models",
        ready_timeout=300.0,
        marker=embed_model.name,
    )

    summary = Service(
        key="summary",
        title=f"独立摘要模型 {summary_model.name}" + (f" TP={summary_tp}" if summary_tp > 1 else ""),
        port=summary_port,
        argv=(
            str(VLLM_PYTHON),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(summary_model),
            "--served-model-name",
            str(summary_conf["model_name"]),
            *summary_parallel_args,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(summary_max_model_len),
            "--gpu-memory-utilization",
            summary_gpu_mem_util,
            "--max-num-seqs",
            "1",
            # 限制每轮预填充激活峰值；长输入分块处理，仍保留 16K 上下文。
            "--max-num-batched-tokens",
            "2048",
            "--enforce-eager",
            "--reasoning-parser",
            "qwen3",
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}',
            "--host",
            "127.0.0.1",
            "--port",
            str(summary_port),
        ),
        cwd=vllm_cwd,
        env=summary_env,
        health_path="/v1/models",
        ready_timeout=300.0,
        marker=str(summary_model),
    )

    web = Service(
        key="web",
        title="网页问答服务",
        port=web_port,
        argv=(
            str(PROJECT_PYTHON),
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            WEB_HOST,
            "--port",
            str(web_port),
        ),
        # uvicorn 要能 import app:app，所以必须从项目根启动。
        cwd=PROJECT_ROOT,
        env={},
        health_path="/api/health",
        ready_timeout=120.0,
        marker="uvicorn",
    )

    return {service.key: service for service in (chat, embedding, summary, web)}


_SERVICES: dict[str, Service] | None = None


def services() -> dict[str, Service]:
    """惰性构建服务表，让配置错误在 main() 里变成一句提示而不是一长串回溯。"""
    global _SERVICES
    if _SERVICES is None:
        _SERVICES = _build_services()
    return _SERVICES


def _is_ready(url: str, timeout: float = 2.0) -> bool:
    """探测一次 HTTP 健康地址，任何网络或状态码异常都算未就绪。"""
    try:
        # 健康地址只指向本机，避免 HTTP_PROXY 把本地探活送到远程代理。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _port_in_use(port: int) -> bool:
    """判断本机端口是否已被监听。"""
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _read_pid(service: Service) -> int | None:
    """读取 pid 文件，文件不存在或内容不是数字时返回 None。"""
    try:
        return int(service.pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _is_zombie(pid: int) -> bool:
    """判断进程是否已经退出、但还没被父进程回收（僵尸态）。

    /proc/<pid>/stat 的格式是 `pid (comm) state ...`，而 comm 里可能含空格和
    括号（进程名可以自称 "a) b"），所以取最后一个右括号之后的部分才稳。
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    _, _, after_comm = stat.rpartition(")")
    return after_comm.strip().startswith("Z")


def _pid_alive(pid: int | None) -> bool:
    """判断进程是否还在运行。信号 0 不真的发信号，只做存在性和权限检查。

    已退出但还没被回收的僵尸进程仍然占着 pid，os.kill(pid, 0) 也会成功，
    但它其实早就结束了。不排掉这种情况，_stop_one 的等待循环会为每个已是
    僵尸的子进程空转满整个超时——前台跑 `serve.py all` 时三个服务都是自己
    的直接子进程，停一次要白等一分半。
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但不属于当前用户，读不到 /proc 状态，按"还活着"保守处理。
        return True
    return not _is_zombie(pid)


def _reap(pid: int) -> None:
    """回收本进程拉起的子进程，别让它一直挂着僵尸条目。

    只有自己的直接子进程才能被 waitpid 回收；由别的进程（例如之前用 -d 启动
    的实例）拉起的会抛 ChildProcessError，忽略即可。
    """
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


def _pid_matches(pid: int, marker: str) -> bool:
    """确认 pid 对应的确是目标服务，而不是系统复用同一 pid 的无关进程。"""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False
    return marker in cmdline


def _running_pid(service: Service) -> int | None:
    """返回正在运行且身份匹配的 pid，否则返回 None。"""
    pid = _read_pid(service)
    if pid is not None and _pid_alive(pid) and _pid_matches(pid, service.marker):
        return pid
    return None


def _tail(path: Path, lines: int = 30) -> list[str]:
    """取日志末尾若干行，用于服务启动失败时给出上下文。"""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:]


def _gpu_summary() -> str | None:
    """返回各张 GPU 的显存占用摘要，nvidia-smi 不可用时返回 None。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    summaries = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            index, used, total = fields
            summaries.append(f"GPU {index}：{used} / {total} MiB")
    return "\n".join(summaries) or None


def _tee_output(proc: subprocess.Popen, log_path: Path, tag: str) -> threading.Thread:
    """把子进程输出同时写进日志文件和终端，便于前台观察模型加载进度。"""

    def pump() -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as handle, proc.stdout:
            for raw in proc.stdout:
                handle.write(raw)
                handle.flush()
                text = raw.decode("utf-8", "replace").rstrip("\n")
                sys.stdout.write(f"[{tag}] {text}\n")
                sys.stdout.flush()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return thread


def _prune_old_logs(service: Service, keep: int = 10) -> None:
    """每个服务只保留最近几次运行的日志，避免长期运行撑满 30G 的 overlay 分区。"""
    candidates = sorted(
        (path for path in LOG_DIR.glob(f"serve-{service.key}-*.log") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    for stale in candidates[:-keep]:
        stale.unlink(missing_ok=True)


def _start_one(service: Service, *, detach: bool) -> subprocess.Popen:
    """拉起一个服务进程，并记录它的 pid。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(service)

    run_log = service.new_run_log()
    # 先摘掉旧链接再重建，否则 symlink_to 会因为目标已存在而报错。
    service.log_path.unlink(missing_ok=True)
    service.log_path.symlink_to(run_log.name)

    env = {**os.environ, **service.env}

    proc = None
    try:
        if detach:
            # 后台模式：直接重定向到日志文件，父进程退出后子进程继续跑。
            with run_log.open("ab") as handle:
                proc = subprocess.Popen(
                    service.argv,
                    cwd=str(service.cwd),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    # 每次启动创建自己的进程组，失败清理只能处理本次组。
                    start_new_session=True,
                )
        else:
            proc = subprocess.Popen(
                service.argv,
                cwd=str(service.cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _tee_output(proc, run_log, service.key)

        service.pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
        return proc
    except BaseException:
        # 进程已经创建、但 PID 写入或日志设置失败时，上层还未登记这个句柄。
        if proc is not None:
            _stop_started(service, proc)
        raise


def _wait_ready(
    service: Service, proc: subprocess.Popen | None, timeout: float
) -> tuple[bool, str]:
    """轮询健康地址直到就绪、进程退出或超时。返回 (是否就绪, 失败原因)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False, f"进程提前退出（退出码 {proc.returncode}）"
        if _is_ready(service.health_url):
            return True, ""
        time.sleep(1.0)
    return False, f"等待 {timeout:.0f} 秒后仍未就绪"


def _forget(service: Service) -> None:
    """抹掉已经不存在的服务留下的 pid 文件，避免 status 读到过期的编号。"""
    service.pid_path.unlink(missing_ok=True)


def _report_failure(service: Service, reason: str) -> None:
    """服务没起来时，打印原因和日志末尾，省去再翻文件。"""
    print(f"\n{service.title} 启动失败：{reason}", file=sys.stderr)
    print(f"完整日志：{service.log_path}", file=sys.stderr)
    tail = _tail(service.log_path)
    if tail:
        print("-" * 60, file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print("-" * 60, file=sys.stderr)


def _stop_one(service: Service, timeout: float = 30.0) -> bool:
    """给服务发 SIGTERM，超时未退出再 SIGKILL。返回是否真的停掉了。"""
    pid = _running_pid(service)
    if pid is None:
        _forget(service)
        return False

    # 按进程组停：vLLM 会派生引擎子进程，只杀主进程会留下占着显存的孤儿。
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            _forget(service)
            return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            _reap(pid)
            _forget(service)
            return True
        time.sleep(0.5)

    # 兜底：释放显存比"优雅退出"更重要，否则下一次启动会 OOM。
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    _reap(pid)
    _forget(service)
    return True


def _process_group_alive(group_id: int) -> bool:
    """判断进程组是否仍存在；父进程退出后，同组引擎进程可能还在运行。"""
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_started(
    service: Service, proc: subprocess.Popen, timeout: float = 30.0
) -> bool:
    """只清理本次 Popen 创建的独立组，不依据可能被覆盖的 PID 文件杀进程。"""
    pid = proc.pid
    # poll 已回收原父进程后，若同一个 PID 又对应活进程，就是系统复用的编号。
    if proc.poll() is not None and Path(f"/proc/{pid}").exists():
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False

    deadline = time.monotonic() + timeout
    while True:
        proc.poll()
        if not _process_group_alive(pid):
            break
        if time.monotonic() >= deadline:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
            break
        time.sleep(0.1)
    proc.poll()
    # 同时启动的其它命令可能已更换管理文件，只能删除仍属于本次进程的记录。
    if _read_pid(service) == pid:
        _forget(service)
    return True


def _preflight(service: Service, *, force: bool) -> bool:
    """启动前检查：已在运行就拒绝，端口被占就报清楚。返回是否可以继续。"""
    pid = _running_pid(service)
    if pid is not None:
        if not force:
            print(
                f"{service.title} 已在运行（pid {pid}）。\n"
                f"  想重启：先 python scripts/serve.py stop {service.key}，"
                "或者加 -f/--force 让本脚本替你停掉旧的。",
                file=sys.stderr,
            )
            return False
        print(f"停掉正在运行的 {service.title}（pid {pid}）…")
        _stop_one(service)

    if _port_in_use(service.port):
        print(
            f"端口 {service.port} 已被占用，但不是本脚本启动的 "
            f"{service.title}（没有可用的 pid 文件）。\n"
            f"  先查是谁： ss -lptn 'sport = :{service.port}'  然后手动处理。",
            file=sys.stderr,
        )
        return False
    return True


def start(keys: list[str], *, detach: bool, force: bool) -> int:
    """逐个启动模型并等待就绪，模型全部就绪后再启动网页服务。"""
    table = services()
    targets = [table[key] for key in keys]

    for service in targets:
        if not _preflight(service, force=force):
            return 1

    models = [s for s in targets if s.key in {"chat", "embedding", "summary"}]
    webs = [s for s in targets if s.key == "web"]

    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"日志目录：  {LOG_DIR}\n")

    procs: dict[str, subprocess.Popen] = {}
    keep_running = False
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt(_signum, _frame) -> None:
        # 启动等待期间也必须进入 finally，不能遗留独立会话中的模型进程。
        raise KeyboardInterrupt

    for sig in previous_handlers:
        signal.signal(sig, interrupt)
    try:
        # vLLM 的显存 profiling 假设其他进程占用不变，多实例部署必须逐个加载。
        # 单模型 -d 保留异步启动；all 无论前台后台都先等待模型和网页就绪。
        wait_for_models = not detach or len(models) > 1 or bool(webs)
        for service in models:
            proc = _start_one(service, detach=detach)
            procs[service.key] = proc
            print(f"已启动 {service.title}（pid {proc.pid}，端口 {service.port}）")
            if wait_for_models:
                print(f"等待 {service.title} 就绪（最多 {service.ready_timeout:.0f} 秒）…")
                ok, reason = _wait_ready(
                    service, proc, service.ready_timeout
                )
                if not ok:
                    _report_failure(service, reason)
                    return 1
                print(f"  ✓ {service.title} 已就绪：{service.health_url}")

        for service in webs:
            proc = _start_one(service, detach=detach)
            procs[service.key] = proc
            print(f"已启动 {service.title}（pid {proc.pid}，端口 {service.port}）")
            if not detach or models:
                ok, reason = _wait_ready(service, proc, service.ready_timeout)
                if not ok:
                    _report_failure(service, reason)
                    return 1
                print(f"  ✓ {service.title} 已就绪：http://{WEB_HOST}:{service.port}")

        if detach:
            keep_running = True
            print("\n后台运行中。查看状态：python scripts/serve.py status")
            return 0

        print("\n全部就绪。按 Ctrl-C 停止。")
        return _supervise(targets, procs)
    except KeyboardInterrupt:
        print("\n正在停止本次启动的服务…")
        return 0
    finally:
        if not keep_running:
            # 保留 PID 到停止完成，超时或异常服务也不能从管理表里提前消失。
            for service in reversed(targets):
                if service.key in procs:
                    _stop_started(service, procs[service.key])
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _supervise(targets: list[Service], procs: dict[str, subprocess.Popen]) -> int:
    """前台驻留：阻塞到用户 Ctrl-C，或某个子进程意外退出。"""
    while True:
        for service in targets:
            proc = procs.get(service.key)
            if proc is not None and proc.poll() is not None:
                print(
                    f"\n{service.title} 意外退出（退出码 {proc.returncode}），"
                    "其余服务一并停止。",
                    file=sys.stderr,
                )
                return 1
        time.sleep(1.0)


def stop(keys: list[str]) -> int:
    """停止指定服务。没有在跑的会被跳过，不算失败。"""
    table = services()
    for key in keys:
        service = table[key]
        if _stop_one(service):
            print(f"已停止 {service.title}")
        else:
            print(f"{service.title} 没有在运行")
    return 0


def show_status() -> int:
    """打印所有服务的进程、端口和健康状态，包括单独启动的摘要服务。"""
    table = services()
    title_width = max(28, max(len(service.title) for service in table.values()) + 2)

    print(f"{'服务':<{title_width}}{'端口':<8}{'进程':<10}{'状态'}")
    print("-" * 62)
    for service in table.values():
        pid = _running_pid(service)
        if pid is None:
            state = "未运行"
        elif _is_ready(service.health_url):
            state = "就绪"
        else:
            state = "启动中/异常"
        print(
            f"{service.title:<{title_width}}{service.port:<8}"
            f"{(str(pid) if pid else '-'):<10}{state}"
        )

    gpu = _gpu_summary()
    if gpu:
        print(f"\nGPU 显存：\n{gpu}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器：起服务用位置参数，停服务再指定目标。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/serve.py",
        description="课堂问答助手的服务启动器（聊天 / 向量 / 可选独立摘要 / 网页）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/serve.py all            # 启动已启用的依赖和网页\n"
            "  python scripts/serve.py chat -d        # 只起聊天模型，转后台\n"
            "  python scripts/serve.py summary        # 单独验证摘要模型\n"
            "  python scripts/serve.py status         # 看谁在跑\n"
            "  python scripts/serve.py stop all       # 全部停止\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_common = argparse.ArgumentParser(add_help=False)
    start_common.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="转入后台；all 依次等待模型和网页就绪后返回",
    )
    start_common.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="已有同名服务在跑时，先把它停掉再启动",
    )

    for name, help_text in (
        ("chat", "启动聊天模型（vLLM，端口取自 config/model.yml）"),
        ("embedding", "启动向量模型（vLLM，RAG 检索必需）"),
        ("summary", "启动独立本地摘要模型（无须先开启 summary.enabled）"),
        ("web", "启动网页问答服务（FastAPI + Uvicorn）"),
        ("all", "启动聊天、向量、网页，并按配置启用独立摘要"),
    ):
        subparsers.add_parser(name, parents=[start_common], help=help_text)

    subparsers.add_parser("status", help="查看所有服务的运行状态")
    stop_parser = subparsers.add_parser("stop", help="停止服务")
    stop_parser.add_argument(
        "targets",
        nargs="+",
        choices=["chat", "embedding", "summary", "web", "all"],
        help="要停止的服务，可写多个",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行并分发到对应的处理函数。"""
    _configure_console()
    args = _build_parser().parse_args(argv)

    try:
        table = services()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"读取配置失败，无法确定服务参数：{exc}", file=sys.stderr)
        return 1

    if args.command == "status":
        return show_status()

    if args.command == "stop":
        if "all" in args.targets:
            keys = list(table)
        else:
            # 去掉重复项，`stop chat chat` 不该停两遍。
            keys = list(dict.fromkeys(args.targets))
        return stop(keys)

    keys = (
        [key for key in table if key != "summary" or get_summary_model_config()["enabled"]]
        if args.command == "all"
        else [args.command]
    )
    return start(keys, detach=args.detach, force=args.force)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # 下游管道提前关闭（例如 `serve.py status | head`）不是错误，安静退出。
        # 把 stdout 指到 /dev/null，否则解释器退出时刷新缓冲区会再报一次。
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0)
