"""INT4 manifest 的离线小文件测试，不需要 torch 或 GPU。"""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from utils.quantized_base import create_quantized_base_manifest, validate_quantized_lora_base


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def write_shard(root, specs):
    header = {"__metadata__": {"tag": "a"}}
    offset = 0
    for name, dtype, shape, size in specs:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, sort_keys=True).encode()
    (root / "model-1.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    write_json(root / "model.safetensors.index.json", {
        "weight_map": {name: "model-1.safetensors" for name, *_ in specs},
    })


class QuantizedBaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original"
        self.quantized = self.root / "int4"
        self.adapter = self.root / "adapter"
        for directory in (self.source, self.quantized, self.adapter):
            directory.mkdir()
        config = {"model_type": "example", "hidden_size": 8, "torch_dtype": "bfloat16"}
        write_json(self.source / "config.json", config)
        config["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "quantization_status": "compressed", "kv_cache_scheme": None,
            "config_groups": {"group_0": {"format": "pack-quantized", "targets": ["Linear"],
                "input_activations": None, "output_activations": None,
                "weights": {"num_bits": 4, "type": "int", "dynamic": False}}},
        }
        write_json(self.quantized / "config.json", config)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
            for directory in (self.source, self.quantized):
                (directory / name).write_text("same tokenizer content", encoding="utf-8")
        write_shard(self.source, [("layer.weight", "BF16", [1, 8], 16)])
        write_shard(self.quantized, [
            ("layer.weight_packed", "I32", [1, 1], 4),
            ("layer.weight_scale", "BF16", [1, 1], 2),
            ("layer.weight_shape", "I64", [2], 16),
        ])
        write_json(self.adapter / "adapter_config.json", {"base_model_name_or_path": str(self.source), "r": 16})
        (self.adapter / "adapter_model.safetensors").write_bytes(b"tiny adapter fixture")
        self.evidence = self.root / "review.json"
        self.evidence.write_text('{"reviewed": true}', encoding="utf-8")
        self.manifest = self.root / "manifest.json"

    def create(self):
        return create_quantized_base_manifest(self.quantized, self.source, self.adapter, self.evidence, self.manifest)

    def validate(self):
        validate_quantized_lora_base(self.quantized, self.source, self.adapter, self.manifest)

    def mutate_config(self, change):
        path = self.quantized / "config.json"
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)

    def test_create_and_validate_without_loading_weights(self):
        data = self.create()
        self.validate()
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["paths"]["source"], str(self.source.resolve()))
        self.assertIn("does not prove", data["provenance_limit"])
        self.assertIn("header_sha256", data["source"]["shards"]["model-1.safetensors"])
        self.assertIn("adapter_model.safetensors", data["adapter_files_sha256"])

    def test_alias_directories_resolve_to_same_bindings(self):
        self.create()
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        validate_quantized_lora_base(self.quantized, alias, self.adapter, self.manifest)

    def test_missing_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, "INT4 manifest"):
            self.validate()

    def test_schema_and_bound_paths_cannot_be_replaced(self):
        self.create()
        data = json.loads(self.manifest.read_text())
        data["paths"]["source"] = str(self.quantized)
        write_json(self.manifest, data)
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*目录"):
            self.validate()

    def test_config_architecture_mismatch_rejected(self):
        self.mutate_config(lambda data: data.update(hidden_size=16))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*config"):
            self.create()

    def test_source_must_not_be_quantized(self):
        path = self.source / "config.json"
        data = json.loads(path.read_text())
        data["quantization_config"] = {}
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*未量化"):
            self.create()

    def test_unquantized_candidate_rejected(self):
        self.mutate_config(lambda data: data.pop("quantization_config"))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*compressed-tensors"):
            self.create()

    def test_non_four_bit_group_rejected(self):
        self.mutate_config(lambda data: data["quantization_config"]["config_groups"]["group_0"]["weights"].update(num_bits=8))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*4bit"):
            self.create()

    def test_quantized_activation_rejected(self):
        self.mutate_config(lambda data: data["quantization_config"]["config_groups"]["group_0"].update(input_activations={"num_bits": 8}))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*W4A16"):
            self.create()

    def test_float32_execution_dtype_rejected(self):
        for root in (self.source, self.quantized):
            path = root / "config.json"
            data = json.loads(path.read_text())
            data["torch_dtype"] = "float32"
            write_json(path, data)
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*W4A16"):
            self.create()

    def test_candidate_must_really_have_packed_tensors(self):
        write_shard(self.quantized, [("layer.weight", "BF16", [1, 8], 16)])
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*weight_packed"):
            self.create()

    def test_tokenizer_content_mismatch_rejected(self):
        (self.quantized / "tokenizer.json").write_text("different tokenizer")
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*Tokenizer"):
            self.create()

    def test_optional_tokenizer_must_exist_in_both_or_neither(self):
        (self.quantized / "merges.txt").unlink()
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*Tokenizer"):
            self.create()

    def test_original_adapter_training_base_must_match(self):
        write_json(self.adapter / "adapter_config.json", {"base_model_name_or_path": str(self.quantized)})
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*base_model_name_or_path"):
            self.create()

    def test_evidence_missing_or_empty_rejected(self):
        self.evidence.unlink()
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*evidence"):
            self.create()
        self.evidence.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*evidence"):
            self.create()

    def test_evidence_modified_rejected(self):
        self.create()
        self.evidence.write_text('{"reviewed": false}')
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*evidence"):
            self.validate()

    def test_shard_size_or_mtime_modified_rejected(self):
        self.create()
        shard = self.quantized / "model-1.safetensors"
        stat = shard.stat()
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*指纹"):
            self.validate()

    def test_shard_header_modified_even_with_restored_size_and_mtime(self):
        self.create()
        shard = self.source / "model-1.safetensors"
        stat = shard.stat()
        raw = shard.read_bytes()
        shard.write_bytes(raw.replace(b'"tag": "a"', b'"tag": "b"'))
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*指纹"):
            self.validate()

    def test_adapter_payload_change_detected_with_same_stat(self):
        self.create()
        weights = self.adapter / "adapter_model.safetensors"
        stat = weights.stat()
        weights.write_bytes(b"Tiny adapter fixture")
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*adapter_files"):
            self.validate()

    def test_adapter_config_change_rejected(self):
        self.create()
        path = self.adapter / "adapter_config.json"
        data = json.loads(path.read_text())
        data["r"] = 8
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*adapter_files"):
            self.validate()

    def test_index_path_traversal_rejected(self):
        write_json(self.quantized / "model.safetensors.index.json", {"weight_map": {"layer.weight_packed": "../original/model-1.safetensors"}})
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*路径遍历"):
            self.create()

    def test_external_symlink_file_rejected(self):
        tokenizer = self.quantized / "tokenizer.json"
        tokenizer.unlink()
        tokenizer.symlink_to(self.source / "tokenizer.json")
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*越出"):
            self.create()

    def test_missing_shard_rejected(self):
        (self.source / "model-1.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*缺少文件"):
            self.create()

    def test_config_whitespace_changes_are_fingerprinted(self):
        self.create()
        path = self.quantized / "config.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "INT4 manifest.*指纹"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
