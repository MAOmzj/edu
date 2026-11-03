"""INT4 基座与独立 LoRA 的显式绑定；只读取元数据，不导入 torch。

本文件不证明两份模型权重数值同源。部署负责人先审查 evidence，再创建
manifest；启动入口据此重新检查已审查候选、原训练基座和适配器是否改变。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "vocab.json", "merges.txt", "tokenizer.model", "special_tokens_map.json",
    "added_tokens.json",
)
_INDEX = "model.safetensors.index.json"
_PROVENANCE = (
    "Reviewed candidate provenance and file identity only; this manifest does not "
    "prove that the quantized weights were numerically derived from the source weights."
)


def _fail(message: str) -> ValueError:
    return ValueError(f"INT4 manifest: {message}")


def _directory(path: Path) -> Path:
    try:
        result = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail(f"目录不存在或不可解析: {path}") from exc
    if not result.is_dir():
        raise _fail(f"不是目录: {result}")
    return result


def _file(root: Path, name: str) -> Path:
    # 由 _model_snapshot/_adapter_snapshot 调用：索引不得逃出已绑定目录。
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name:
        raise _fail(f"禁止索引路径遍历或子目录: {name!r}")
    path = root / name
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail(f"缺少文件: {path}") from exc
    if resolved.parent != root or not resolved.is_file():
        raise _fail(f"文件越出绑定目录或不是普通文件: {path}")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(f"JSON 无法读取: {path}") from exc
    if not isinstance(data, dict):
        raise _fail(f"JSON 必须是对象: {path}")
    return data


def _sha256(path: Path) -> str:
    # 配置/索引/Tokenizer 和适配器完整哈希；大型基座分片另用有界 header 哈希。
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise _fail(f"无法计算文件 SHA256: {path}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise _fail(f"计算 SHA256 时文件发生变化: {path}")
    return digest.hexdigest()


def _header(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    # 每次启动只读取 safetensors header；不会遍历两份基座的数十 GB 数据区。
    try:
        before = path.stat()
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise _fail(f"safetensors 文件过短: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if length < 2 or length > 64 * 1024 * 1024 or length + 8 > before.st_size:
                raise _fail(f"safetensors header 长度非法: {path}")
            raw = stream.read(length)
        header = json.loads(raw)
        after = path.stat()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(f"无法读取 safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise _fail(f"safetensors header 必须是对象: {path}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise _fail(f"读取 header 时分片发生变化: {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise _fail(f"safetensors 分片没有 tensor: {path}")
    for key, spec in tensors.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("dtype"), str):
            raise _fail(f"tensor header 非法: {path.name}:{key}")
        offsets = spec.get("data_offsets")
        shape = spec.get("shape")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= before.st_size - 8 - length
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)):
            raise _fail(f"tensor shape/offset 非法: {path.name}:{key}")
    return tensors, {
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
    }


def _model_snapshot(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    # create/validate 均调用本函数，固定配置和每一个 index 引用分片。
    files = {"config.json": _sha256(_file(root, "config.json")), _INDEX: _sha256(_file(root, _INDEX))}
    for name in _TOKENIZER_FILES:
        if (root / name).exists() or (root / name).is_symlink():
            files[name] = _sha256(_file(root, name))
    weight_map = _json(_file(root, _INDEX)).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise _fail(f"模型 index 缺少非空 weight_map: {root}")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
        raise _fail(f"模型 weight_map 必须将 tensor 名映射到分片名: {root}")
    shards, tensors = {}, {}
    for name in sorted(set(weight_map.values())):
        if not name.endswith(".safetensors"):
            raise _fail(f"仅接受 safetensors 基座分片: {name}")
        entries, shards[name] = _header(_file(root, name))
        expected = {key for key, shard in weight_map.items() if shard == name}
        if set(entries) != expected:
            raise _fail(f"分片 header 与 index 不一致: {root / name}")
        tensors.update(entries)
    return {"files_sha256": files, "shards": shards}, tensors


def _adapter_snapshot(root: Path) -> dict[str, Any]:
    # 适配器远小于基座，完整计算权重 SHA，避免仅依赖 header/mtime。
    names = sorted(path.name for path in root.iterdir()
                   if path.name.startswith("adapter_model")
                   and path.name.endswith((".safetensors", ".bin")))
    if not names:
        raise _fail(f"缺少 adapter_model 权重: {root}")
    return {name: _sha256(_file(root, name)) for name in ["adapter_config.json", *names]}


def _check_compatibility(quantized: Path, source: Path, adapter: Path,
                         quant_snapshot: dict[str, Any], source_snapshot: dict[str, Any],
                         quant_tensors: dict[str, Any], source_tensors: dict[str, Any]) -> None:
    # 由 create/validate 调用：manifest 不能豁免架构、Tokenizer 或训练基座检查。
    original = _json(_file(source, "config.json"))
    compressed = _json(_file(quantized, "config.json"))
    if "quantization_config" in original:
        raise _fail("source 必须是未量化训练基座，不能含 quantization_config")
    config = compressed.pop("quantization_config", None)
    if original != compressed:
        raise _fail("INT4 与 source 的 config 去掉 quantization_config 后不一致")
    text_config = original.get("text_config", original)
    if not isinstance(text_config, dict):
        raise _fail("source text_config 必须是对象")
    dtype = text_config.get("dtype", text_config.get("torch_dtype"))
    if dtype not in ("bfloat16", "float16", "torch.bfloat16", "torch.float16"):
        raise _fail("仅接受 W4A16，原模型执行 dtype 必须是 BF16/FP16")
    if not isinstance(config, dict) or config.get("quant_method") != "compressed-tensors":
        raise _fail("候选必须是 compressed-tensors INT4 基座")
    if config.get("format") != "pack-quantized" or config.get("quantization_status") != "compressed":
        raise _fail("候选必须是已压缩的 pack-quantized INT4 权重")
    if config.get("kv_cache_scheme") or config.get("sparsity_config") or config.get("transform_config"):
        raise _fail("仅接受 W4A16，不接受 KV/稀疏/变换量化")
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise _fail("INT4 config_groups 不能为空")
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise _fail(f"INT4 group 格式非法: {name}")
        weights = group.get("weights")
        if (not isinstance(weights, dict) or type(weights.get("num_bits")) is not int
                or weights.get("num_bits") != 4 or weights.get("type") != "int"
                or weights.get("dynamic") not in (False, None)):
            raise _fail(f"INT4 所有量化组必须是静态整数 4bit 权重: {name}")
        if group.get("input_activations") is not None or group.get("output_activations") is not None:
            raise _fail(f"仅接受 W4A16，激活不能量化: {name}")
        if group.get("format", "pack-quantized") != "pack-quantized":
            raise _fail(f"INT4 group format 不兼容: {name}")
    for name in _TOKENIZER_FILES:
        if quant_snapshot["files_sha256"].get(name) != source_snapshot["files_sha256"].get(name):
            raise _fail(f"INT4 与 source 的 Tokenizer 文件不一致或缺失: {name}")
    markers = (".weight_packed", ".weight_scale", ".weight_zero_point", ".weight_shape")
    if any(key.endswith(markers) for key in source_tensors):
        raise _fail("source header 含压缩权重标记，并非未量化训练基座")
    packed = [key for key in quant_tensors if key.endswith(".weight_packed")]
    if not packed:
        raise _fail("候选 header 不含实际 INT4 weight_packed")
    for key in packed:
        prefix = key[:-len(".weight_packed")]
        scale = quant_tensors.get(prefix + ".weight_scale", {})
        shape = quant_tensors.get(prefix + ".weight_shape", {})
        if (quant_tensors[key].get("dtype") != "I32"
                or scale.get("dtype") not in ("BF16", "F16", "F32")
                or shape.get("dtype") not in ("I32", "I64")):
            raise _fail(f"INT4 packed/scale/shape header 不兼容: {prefix}")
    adapter_config = _json(_file(adapter, "adapter_config.json"))
    base = adapter_config.get("base_model_name_or_path")
    if not isinstance(base, str) or not base or not Path(base).expanduser().is_absolute():
        raise _fail("adapter 必须记录本地原始训练基座绝对路径")
    if _directory(Path(base)) != source:
        raise _fail("adapter.base_model_name_or_path 与 source 不一致")


def _evidence_snapshot(path: Path) -> dict[str, Any]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
        size = resolved.stat().st_size
    except (OSError, RuntimeError) as exc:
        raise _fail(f"evidence 文件不存在: {path}") from exc
    if not resolved.is_file() or size == 0:
        raise _fail("evidence 必须是非空普通文件，需部署负责人先审查其内容")
    return {"path": str(resolved), "size": size, "sha256": _sha256(resolved)}


def create_quantized_base_manifest(quantized: Path, source: Path, adapter: Path,
                                   evidence_path: Path, output_path: Path) -> dict:
    """审查 evidence 后显式创建绑定；serve 不应在启动时自动调用本函数。"""
    quantized, source, adapter = map(_directory, (quantized, source, adapter))
    if quantized == source:
        raise _fail("INT4 候选与未量化 source 不能是同一目录")
    evidence = _evidence_snapshot(evidence_path)
    # 调用链：创建入口 → 两份基座快照 → 兼容性检查 → 适配器全量哈希 → 落盘。
    quant_snapshot, quant_tensors = _model_snapshot(quantized)
    source_snapshot, source_tensors = _model_snapshot(source)
    _check_compatibility(quantized, source, adapter, quant_snapshot, source_snapshot,
                         quant_tensors, source_tensors)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance_limit": _PROVENANCE,
        "paths": {"quantized": str(quantized), "source": str(source), "adapter": str(adapter)},
        "evidence": evidence,
        "quantized": quant_snapshot,
        "source": source_snapshot,
        "adapter_files_sha256": _adapter_snapshot(adapter),
    }
    output_path = Path(output_path).expanduser()
    parent = _directory(output_path.parent)
    destination = parent / output_path.name
    if destination.is_symlink():
        raise _fail("manifest 输出不能是符号链接")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, destination)
    except OSError as exc:
        raise _fail(f"无法写入 manifest: {destination}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def validate_quantized_lora_base(quantized: Path, source: Path, adapter: Path,
                                manifest_path: Path) -> None:
    """serve 在 INT4 + LoRA 显式配置 manifest 时调用；任何不匹配均拒绝启动。"""
    quantized, source, adapter = map(_directory, (quantized, source, adapter))
    try:
        manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail(f"缺少显式 INT4 manifest: {manifest_path}") from exc
    manifest = _json(manifest_file)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise _fail("不支持的 schema_version")
    if manifest.get("provenance_limit") != _PROVENANCE:
        raise _fail("manifest 缺少候选来源证明范围说明")
    paths = {"quantized": str(quantized), "source": str(source), "adapter": str(adapter)}
    if manifest.get("paths") != paths or quantized == source:
        raise _fail("manifest 绑定目录与当前配置不一致")
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str):
        raise _fail("manifest 缺少 evidence 绑定")
    if _evidence_snapshot(Path(evidence["path"])) != evidence:
        raise _fail("evidence 文件指纹已改变")
    # 调用链：启动校验 → 重读 metadata/header → 重验兼容性 → 对比已审查指纹。
    quant_snapshot, quant_tensors = _model_snapshot(quantized)
    source_snapshot, source_tensors = _model_snapshot(source)
    _check_compatibility(quantized, source, adapter, quant_snapshot, source_snapshot,
                         quant_tensors, source_tensors)
    for name, current in (("quantized", quant_snapshot), ("source", source_snapshot),
                          ("adapter_files_sha256", _adapter_snapshot(adapter))):
        if manifest.get(name) != current:
            raise _fail(f"{name} 文件指纹已改变；请重新审查，不能自动重建 manifest")
