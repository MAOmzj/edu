# 文件用途：离线合并本地摘要模型与 LoRA Adapter，导出可供 vLLM 加载的模型目录。
# 调用关系：开发者运行本 CLI；校验目录后才导入 torch、transformers、peft 并合并模型。
# 修改易踩坑：输出目录必须是新目录，加载只能读本地文件，基础生成配置必须保留。
"""将本地 LoRA Adapter 合并到 BF16 基座，并保留基础模型的采样配置。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _validate_paths(
    base_model: str | Path, adapter_path: str | Path, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """在导入模型依赖前检查本地输入和禁止覆盖的输出路径。"""
    base = Path(base_model).expanduser().resolve()
    adapter = Path(adapter_path).expanduser().resolve()
    for label, path, config_name in (
        ("基座模型", base, "config.json"),
        ("Adapter", adapter, "adapter_config.json"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label}必须是已存在的本地目录：{path}")
        if not (path / config_name).is_file():
            raise FileNotFoundError(f"{label}目录缺少 {config_name}：{path}")

    output = Path(output_dir).expanduser()
    # dangling symlink 也是已有文件系统入口，不能借此覆盖其它位置。
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    return base, adapter, output.resolve()


def _load_dependencies():
    """延迟加载大模型依赖，让路径校验和 --help 在普通 Python 环境也能运行。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("请在已安装 torch、transformers 和 peft 的环境运行合并命令。") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def merge_adapter(
    base_model: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    """合并并保存；仅读取本地模型文件，不连接模型仓库或学生数据库。"""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device 只支持 cpu 或 cuda")
    base_path, adapter_path, output_path = _validate_paths(base_model, adapter_path, output_dir)
    torch, model_loader, tokenizer_loader, adapter_loader = _load_dependencies()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请改用 --device cpu。")

    tokenizer = tokenizer_loader.from_pretrained(
        str(base_path), local_files_only=True, trust_remote_code=False
    )
    base = model_loader.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
        trust_remote_code=False,
    )
    generation_config = base.generation_config
    adapted = adapter_loader.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
        trust_remote_code=False,
    )
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.generation_config = generation_config

    # 排他创建防止校验后另一个任务占用输出目录；成功加载前不产生空目录。
    output_path.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    generation_path = base_path / "generation_config.json"
    if generation_path.is_file():
        shutil.copy2(generation_path, output_path / generation_path.name)
    elif generation_config is not None:
        generation_config.save_pretrained(str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path, help="本地 BF16 基座目录")
    parser.add_argument("--adapter-path", required=True, type=Path, help="本地 LoRA Adapter 目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="尚不存在的合并模型输出目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        output = merge_adapter(args.base_model, args.adapter_path, args.output_dir, device=args.device)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"合并失败：{exc}\n")
    print(f"合并模型已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
