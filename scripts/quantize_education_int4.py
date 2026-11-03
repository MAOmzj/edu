#!/usr/bin/env python3
"""从原始 BF16 基座生成不做 AWQ 平滑的 RTN INT4，保持原 LoRA 的坐标。

运行：/root/MAOMI/envs/vllm/bin/python scripts/quantize_education_int4.py
调用链：main -> build -> quantize_weight -> 原生 compressed_tensors API。
本脚本不加载或合并适配器，不覆盖原始模型，也不读取 AWQ 模板的权重数值。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """下游：分块读取文件；不把完整模型权重加载进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def headers(directory: Path, index: dict) -> dict:
    """下游：只读取 safetensors 头；同时记录头哈希和每个张量的布局。"""
    result = {}
    for filename in sorted(set(index["weight_map"].values())):
        path = directory / filename
        with path.open("rb") as stream:
            prefix = stream.read(8)
            length = struct.unpack("<Q", prefix)[0]
            payload = stream.read(length)
        result[filename] = {
            "header_sha256": hashlib.sha256(prefix + payload).hexdigest(),
            "file_bytes": path.stat().st_size,
            "data_start": 8 + length,
            "tensors": json.loads(payload),
        }
    return result


def raw_tensor_hash(directory: Path, index: dict, metadata: dict, key: str) -> str:
    """下游：直接散列张量的原始字节，用于验证未量化张量逐字节不变。"""
    filename = index["weight_map"][key]
    info = metadata[filename]
    start, end = info["tensors"][key]["data_offsets"]
    digest = hashlib.sha256()
    with (directory / filename).open("rb") as stream:
        stream.seek(info["data_start"] + start)
        remaining = end - start
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"权重数据被截断：{filename}: {key}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def quantize_weight(weight, scheme, device: str):
    """下游：原生 qparams -> compress -> decompress；无校准数据、无平滑变换。"""
    import torch
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    from compressed_tensors.quantization.utils import calculate_qparams

    with torch.inference_mode():
        original = weight.to(device=device)
        if original.dtype != torch.bfloat16 or original.ndim != 2:
            raise ValueError("只接受原始 BF16 二维 Linear.weight")
        group_size = scheme.weights.group_size
        if original.shape[1] % group_size:
            raise ValueError("输入维度必须整除 group_size")
        # 下游：FP32 min/max 计算稳定的逐行、逐组参数；保存前固定为 BF16 scale。
        grouped = original.float().reshape(original.shape[0], -1, group_size)
        scale, zero_point = calculate_qparams(
            grouped.amin(dim=-1), grouped.amax(dim=-1), scheme.weights
        )
        scale = scale.to(torch.bfloat16)
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError("量化 scale 非有限或非正")
        del grouped
        # 下游：官方 compressor 完成取整、截断、INT32 权重及零点打包。
        packed = PackedQuantizationCompressor.compress(
            {"weight": original, "weight_scale": scale, "weight_zero_point": zero_point},
            scheme,
        )
        restored = PackedQuantizationCompressor.decompress(packed, scheme)["weight"]
        difference = original.float() - restored.float()
        sum_sq = difference.square().sum(dtype=torch.float64).item()
        weight_sq = original.float().square().sum(dtype=torch.float64).item()
        count = original.numel()
        error = {
            "elements": count,
            "sum_squared_error": sum_sq,
            "sum_squared_weight": weight_sq,
            "rmse": math.sqrt(sum_sq / count),
            "normalized_rmse": math.sqrt(sum_sq / weight_sq) if weight_sq else 0.0,
            "max_absolute_error": difference.abs().max().item(),
        }
        if not all(math.isfinite(v) for v in error.values()):
            raise ValueError("反量化误差非有限")
        # 下游：保存函数仅接收 CPU 连续张量，避免写文件期间保留显存。
        cpu = {key: tensor.to(device="cpu").contiguous() for key, tensor in packed.items()}
        return cpu, error


def self_test(scheme) -> dict:
    """下游：CPU 小矩阵覆盖原生 packing、zero-point packing 和反量化误差。"""
    import torch
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import quantize

    generator = torch.Generator().manual_seed(20261019)
    x = (torch.randn(16, 64, generator=generator) * 0.1).to(torch.bfloat16)
    x[0].zero_()
    packed, error = quantize_weight(x, scheme, "cpu")
    restored = PackedQuantizationCompressor.decompress(packed, scheme)
    unpacked = unpack_from_int32(packed["weight_packed"], 4, x.shape)
    expected = quantize(
        x, packed["weight_scale"], restored["weight_zero_point"], scheme.weights,
        dtype=torch.int8,
    )
    assert torch.equal(unpacked, expected), "原生 packing 往返不一致"
    assert packed["weight_packed"].shape == (16, 8)
    assert packed["weight_zero_point"].shape == (2, 2)
    assert packed["weight_scale"].shape == (16, 2)
    assert packed["weight_scale"].dtype == torch.bfloat16
    assert torch.count_nonzero(restored["weight"][0]) == 0
    assert error["normalized_rmse"] < 0.15
    return {"passed": True, "packing_exact": True, **error}


def build(args) -> dict:
    """下游：检查来源 -> 自检 -> 逐张量量化/分片 -> 全量验证 -> 写溯源记录。"""
    import torch
    from compressed_tensors.quantization import QuantizationScheme
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = args.source.resolve()
    template = args.template.resolve()
    output = args.output.resolve()
    adapter = args.adapter.resolve()
    if output in (source, template, adapter):
        raise ValueError("输出目录不能覆盖原模型或适配器")
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("source 必须为未量化的原始 BF16 基座")
    template_config = json.loads((template / "config.json").read_text())
    quant_config = template_config["quantization_config"]
    # 下游：仅把现有 checkpoint 的量化配置/模块名单/张量布局作为格式模板。
    quant_config["config_groups"]["group_0"]["weights"]["observer"] = "minmax"
    scheme = QuantizationScheme.model_validate(quant_config["config_groups"]["group_0"])
    weight_args = scheme.weights
    if not (weight_args.num_bits == 4 and weight_args.group_size == 32
            and weight_args.strategy == "group" and not weight_args.symmetric
            and scheme.input_activations is None):
        raise ValueError("模板必须为 group32 非对称 W4A16 INT4")
    test_result = self_test(scheme)
    print(f"CPU packing 自检通过：{test_result}", flush=True)
    if args.self_test_only:
        return {"self_test": test_result}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，拒绝覆盖：{output}")
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    template_index = json.loads((template / "model.safetensors.index.json").read_text())
    source_headers = headers(source, source_index)
    template_headers = headers(template, template_index)
    module_names = sorted({key.rsplit(".", 1)[0] for key in template_index["weight_map"]
                           if key.endswith(".weight_packed")})
    quantized_keys = {name + ".weight" for name in module_names}
    suffixes = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")
    expected_keys = ((set(source_index["weight_map"]) - quantized_keys)
                     | {name + "." + suffix for name in module_names for suffix in suffixes})
    if expected_keys != set(template_index["weight_map"]):
        raise ValueError("原始权重与 INT4 格式模板的张量名单不一致")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(template_index["metadata"]["total_size"])
    required_free = expected_bytes + 256 * 1024**2
    free = shutil.disk_usage(output.parent).free
    if free < required_free:
        raise OSError(f"磁盘不足：需要至少 {required_free:,} bytes，可用 {free:,} bytes")
    output.mkdir(exist_ok=True)
    adapter_before = sha256_file(adapter / "adapter_model.safetensors")
    adapter_config_before = sha256_file(adapter / "adapter_config.json")
    started = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    progress_path = output / "quantization_progress.json"
    max_shard_bytes = int(args.max_shard_gib * 1024**3)
    shard_state = {}
    shard_bytes = 0
    total_bytes = 0
    shard_paths = []
    weight_map = {}
    errors = []
    float_source_hashes = {}

    def flush():
        nonlocal shard_state, shard_bytes
        if not shard_state:
            return
        name = f"temporary-shard-{len(shard_paths) + 1:05d}.safetensors"
        # 下游：safetensors 原生序列化；每个分片真实 tensor bytes 不超过设定上限。
        save_file(shard_state, str(output / name), metadata={"format": "pt"})
        shard_paths.append(name)
        for key in shard_state:
            weight_map[key] = name
        print(f"写入分片 {len(shard_paths)}：{shard_bytes / 1024**3:.3f} GiB", flush=True)
        shard_state = {}
        shard_bytes = 0

    def append(key, tensor):
        nonlocal shard_bytes, total_bytes
        size = tensor.numel() * tensor.element_size()
        if size > max_shard_bytes:
            raise ValueError(f"单个张量超过分片上限：{key}")
        if shard_bytes + size > max_shard_bytes:
            flush()
        shard_state[key] = tensor
        shard_bytes += size
        total_bytes += size

    completed = 0
    # 下游：逐原始 shard 打开 mmap，仅把当前 tensor 读入 CPU/GPU，内存有界。
    for filename in sorted(source_headers):
        keys = sorted(k for k, value in source_index["weight_map"].items() if value == filename)
        with safe_open(source / filename, framework="pt", device="cpu") as reader:
            for key in keys:
                tensor = reader.get_tensor(key)
                if key in quantized_keys:
                    name = key.rsplit(".", 1)[0]
                    packed, error = quantize_weight(tensor, scheme, args.device)
                    error["module"] = name
                    errors.append(error)
                    for suffix in suffixes:
                        out_key = name + "." + suffix
                        metadata = template_headers[template_index["weight_map"][out_key]]["tensors"][out_key]
                        value = packed[suffix]
                        actual_dtype = {torch.int32: "I32", torch.int64: "I64", torch.bfloat16: "BF16"}[value.dtype]
                        if list(value.shape) != metadata["shape"] or actual_dtype != metadata["dtype"]:
                            raise ValueError(f"与模板张量布局不一致：{out_key}")
                        append(out_key, value)
                    completed += 1
                    if completed % 20 == 0 or completed == len(module_names):
                        elapsed = time.monotonic() - started_monotonic
                        print(f"INT4 模块 {completed}/{len(module_names)}，耗时 {elapsed:.1f}s，"
                              f"最近 NRMSE={error['normalized_rmse']:.5f}：{name}", flush=True)
                        progress_path.write_text(json.dumps({"status": "quantizing", "completed": completed,
                                                             "total": len(module_names), "elapsed_seconds": elapsed}, indent=2))
                else:
                    # 下游：原始 tensor 原样保存；严禁使用 AWQ 模板的平滑后浮点权重。
                    float_source_hashes[key] = raw_tensor_hash(source, source_index, source_headers, key)
                    append(key, tensor.contiguous())
    flush()
    if completed != len(module_names) or set(weight_map) != expected_keys:
        raise ValueError("输出 tensor 不完整")
    rename_map = {}
    for number, old_name in enumerate(shard_paths, 1):
        new_name = f"model-{number:05d}-of-{len(shard_paths):05d}.safetensors"
        (output / old_name).rename(output / new_name)
        rename_map[old_name] = new_name
    output_index = {"metadata": {"total_size": total_bytes},
                    "weight_map": {key: rename_map[value] for key, value in sorted(weight_map.items())}}
    (output / "model.safetensors.index.json").write_text(json.dumps(output_index, indent=2) + "\n")
    config["quantization_config"] = quant_config
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    # 下游：复制原模型 tokenizer/processor 等小文件；不复制 .hfd、下载元数据和权重。
    for path in source.iterdir():
        if (path.is_file() and not path.name.startswith(".")
                and path.name not in {"config.json", "model.safetensors.index.json", "crc32.txt"}
                and not path.name.endswith(".safetensors")):
            shutil.copy2(path, output / path.name)
    print("开始全量验证：输出 keys/dtype/shape、未量化张量原始字节、适配器未改动。", flush=True)
    output_headers = headers(output, output_index)
    mismatch = []
    for key in sorted(expected_keys):
        actual = output_headers[output_index["weight_map"][key]]["tensors"][key]
        expected = template_headers[template_index["weight_map"][key]]["tensors"][key]
        if actual["shape"] != expected["shape"] or actual["dtype"] != expected["dtype"]:
            mismatch.append(key)
    if mismatch:
        raise ValueError(f"输出与模板布局不一致：{mismatch[:10]}")
    for key, original_hash in float_source_hashes.items():
        if raw_tensor_hash(output, output_index, output_headers, key) != original_hash:
            raise ValueError(f"未量化 tensor 的字节与基座不同：{key}")
    source_after = headers(source, source_index)
    if any(source_after[name]["header_sha256"] != info["header_sha256"]
           or source_after[name]["file_bytes"] != info["file_bytes"]
           for name, info in source_headers.items()):
        raise ValueError("原始基座文件发生变化")
    if (sha256_file(adapter / "adapter_model.safetensors") != adapter_before
            or sha256_file(adapter / "adapter_config.json") != adapter_config_before):
        raise ValueError("适配器文件发生变化")
    output_manifest = {
        filename: {"file_bytes": info["file_bytes"],
                   "header_sha256": info["header_sha256"],
                   "sha256": sha256_file(output / filename)}
        for filename, info in output_headers.items()
    }
    summed_sq = sum(error["sum_squared_error"] for error in errors)
    summed_weight_sq = sum(error["sum_squared_weight"] for error in errors)
    elements = sum(error["elements"] for error in errors)
    report = {
        "status": "quantized_and_verified", "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "source_model_canonical": str(source), "format_template_canonical": str(template),
        "output_model_canonical": str(output), "adapter_canonical": str(adapter),
        "algorithm": "RTN groupwise minmax, round-to-nearest, asymmetric W4A16",
        "num_bits": 4, "group_size": 32, "scale_dtype": "bfloat16",
        "awq_smoothing": False, "calibration_data_used": False, "lora_merged": False,
        "coordinate_transform_applied": False, "float_tensor_source": "original BF16 base only",
        "source_index_sha256": sha256_file(source / "model.safetensors.index.json"),
        "source_config_sha256": sha256_file(source / "config.json"),
        "source_shards": {name: {k: value for k, value in info.items() if k != "tensors"}
                           for name, info in source_headers.items()},
        "adapter_sha256": adapter_before, "adapter_config_sha256": adapter_config_before,
        "quantized_modules": len(module_names), "output_tensors": len(expected_keys),
        "output_weight_bytes": total_bytes, "output_shards": len(shard_paths),
        "output_shard_manifest": output_manifest,
        "quantization_script_sha256": sha256_file(Path(__file__)),
        "self_test": test_result,
        "checks": {"all_output_keys_shape_dtype_match_template": True,
                   "all_float_passthrough_bytes_equal_source": True,
                   "float_passthrough_tensors_checked": len(float_source_hashes),
                   "source_headers_unchanged": True, "adapter_unchanged": True},
        "float_passthrough_tensor_sha256": float_source_hashes,
        "error_summary": {"elements": elements, "rmse": math.sqrt(summed_sq / elements),
                          "normalized_rmse": math.sqrt(summed_sq / summed_weight_sq),
                          "max_absolute_error": max(error["max_absolute_error"] for error in errors)},
        "module_errors": errors,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "safetensors", "compressed-tensors")},
        "command": sys.argv,
    }
    (output / "quantization_provenance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    progress_path.unlink(missing_ok=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"量化完成：{output}，{total_bytes:,} bytes，NRMSE="
          f"{report['error_summary']['normalized_rmse']:.6f}，耗时 {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def main() -> int:
    """入口：解析配置，然后调用 build；不启动 vLLM，不改当前部署配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/Qwen3.8-27B")
    parser.add_argument("--template", type=Path, default=ROOT / "models/Qwen3.8-27B-AWQ-INT4")
    parser.add_argument("--output", type=Path, default=ROOT / "models/Qwen3.8-27B-RTN-INT4")
    parser.add_argument("--adapter", type=Path, default=ROOT / "training/education-lora-v2/adapter")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    parser.add_argument("--report", type=Path, default=ROOT / "organization/education-rtn-int4-quantization.json")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.max_shard_gib <= 0:
        parser.error("--max-shard-gib 必须大于 0")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
