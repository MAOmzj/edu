# 文件用途：校验教育回答SFT配置与数据，生成仅监督最后教学回答的训练标签。
# 调用关系：train_education.py与run_education_finetune.py调用；数据检查不加载模型或GPU。
# 修改易踩坑：示例不可正式训练；按group分割数据；历史消息和padding标签必须为-100。
# 阅读导航：本文件提供“被别人调用的工具函数”，不会自行启动训练，也不会加载模型权重。
# 配置调用链之一：run_education_finetune.main → load_config，启动入口直接读取配置。
# 配置调用链之二：train_education.main → read_settings → load_config，训练脚本再次读取同一配置。
# 数据链：train_education.read_data → configured_path/load_records/validate_split/reject_examples/file_sha256。
# 准备链：train_education.preparation_report → configured_path/validate_base_model。
# 编码链：train_education.main 的内部 encode → encode_record → prompt_messages/tokenizer。
# 批处理链：Trainer → train_education.run_training 的内部 collate → pad_batch → 返回补齐后的列表。
# 数据生成、修订、合并脚本也复用这些函数；下面“上游”列主要例子，不代表全部调用者。
"""小学多学科教学回答训练的离线数据处理。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# 模块加载 → Path.resolve：从本文件的真实位置找到项目根目录，供相对配置路径使用。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_TARGET_PATTERN = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)


# 上游：启动入口 main、train_education.read_settings，以及生成/修订/合并脚本的 main。
# 输入：YAML 配置文件路径；返回：通过必填字段、类型和数值范围检查的 config 字典。
# 下游：Path.read_text 读取文件 → yaml.safe_load 解析；这里只检查参数，不启动训练。
def load_config(path):
    import yaml
    # load_config → Path.read_text：以 UTF-8 读取 path；→ yaml.safe_load：把 YAML 文本转成字典。
    # 返回值交给下面的检查；例如 config["method"] 决定以后使用 LoRA 还是 QLoRA。
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("教育训练配置必须是YAML对象")
    required = {"model_path", "training_python", "train_file", "validation_file", "output_dir", "instruction", "method"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"教育训练配置缺少：{', '.join(sorted(missing))}")
    for key in required:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key}必须为非空字符串")
    if config["method"] not in {"qlora", "lora"}:
        raise ValueError("method只能是qlora或lora")
    for key in ["max_seq_length", "batch_size", "gradient_accumulation_steps", "lora_r", "lora_alpha", "seed"]:
        value = config.get(key)
        minimum = 0 if key == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key}必须是大于等于{minimum}的整数")
    for key in ["epochs", "learning_rate", "lora_dropout", "warmup_ratio"]:
        value = config.get(key)
        # load_config → math.isfinite：排除无穷大和 NaN，避免训练器接到无效超参数。
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key}必须是有限数值")
        if key in {"epochs", "learning_rate"} and value <= 0:
            raise ValueError(f"{key}必须大于0")
        if key in {"lora_dropout", "warmup_ratio"} and not 0 <= value < 1:
            raise ValueError(f"{key}必须在[0,1)范围")
    return config


# 上游：run_education_finetune.main、train_education.read_data/preparation_report/run_training/main。
# 输入：已读取的 config 和字段名 key；返回：该字段对应的绝对 Path 路径对象。
# 下游：Path.expanduser/is_absolute；相对路径以 PROJECT_ROOT 为起点，避免受启动目录影响。
def configured_path(config, key):
    # configured_path → Path.expanduser：把配置中可能出现的“~”展开为当前用户目录。
    path = Path(config[key]).expanduser()
    # configured_path → Path.is_absolute：绝对路径直接返回；相对路径则接到项目根目录后返回。
    return path if path.is_absolute() else PROJECT_ROOT / path


# 上游：train_education.preparation_report/main，在真正加载基座之前调用。
# 输入：本地基座目录；返回：已验证的 config.json 字典，供上游了解模型架构。
# 下游：读取 config.json/权重索引，检查分片及 tokenizer 文件是否存在；不读取整个权重。
def validate_base_model(path):
    """只读配置与权重索引；禁止当前部署的推理量化模型作为训练底座。"""
    path = Path(path)
    # validate_base_model → Path.read_text → json.loads：读取并解析模型配置，检查量化信息和架构。
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config"):
        raise ValueError("需要完整未量化Qwen3.8-27B底座；compressed-tensors/AWQ推理权重不能传入此入口")
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("此入口需要Qwen3.8-27B使用的qwen3_5 conditional-generation架构")
    text = config.get("text_config", {})
    if text.get("num_hidden_layers") != 64 or text.get("hidden_size") != 5120:
        raise ValueError("底座配置与27B的64层/5120维不符，请核验官方模型来源")
    index = path / "model.safetensors.index.json"
    # validate_base_model → Path.is_file：优先检查分片索引；没有索引时才检查单文件权重。
    if index.is_file():
        # validate_base_model → index.read_text → json.loads：weight_map 把参数名映射到分片文件名。
        # 这里只取去重后的分片名，不把几十 GB 权重读进内存，也不计算权重文件的 SHA。
        files = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        if not files:
            raise ValueError("底座权重索引为空")
        for name in files:
            candidate = Path(name)
            # validate_base_model → Path.is_absolute：防止索引把分片位置指向基座目录之外。
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("底座权重索引包含非法路径")
            # validate_base_model → Path.is_file：逐一确认索引中提到的分片真实存在。
            if not (path / candidate).is_file():
                raise FileNotFoundError(f"底座权重分片缺失：{name}")
    # validate_base_model → Path.is_file：无索引时要求存在单个 model.safetensors。
    elif not (path / "model.safetensors").is_file():
        raise FileNotFoundError("底座缺少完整safetensors权重或分片索引")
    for name in ["tokenizer_config.json", "tokenizer.json"]:
        # validate_base_model → Path.is_file：确认后续 AutoTokenizer 加载需要的本地文件存在。
        if not (path / name).is_file():
            raise FileNotFoundError(f"底座缺少{name}")
    return config


# 上游：train_education.read_data；生成脚本 publish_splits、合并脚本 verified_batch 也用于结构检查。
# 输入：JSONL 路径（一行一条样本）；返回：规范化后的样本字典列表 records。
# 下游：Path.read_text/splitlines → json.loads；逐条检查 id、题目族、角色交替和最后参考回答。
def load_records(path):
    records = []
    ids = set()
    # load_records → Path.read_text：读取 UTF-8 文本；→ splitlines：按 JSONL 约定分成独立记录。
    # number 保留原文件行号，因此报错能准确指出哪一条数据有问题。
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{number}"
        try:
            # load_records → json.loads：只解析当前一行；返回 record 字典，随后检查结构与内容。
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}不是有效JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{location}必须为对象")
        for key in ["id", "group_id"]:
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{location}的{key}必须为非空字符串")
            record[key] = record[key].strip()
        if record["id"] in ids:
            raise ValueError(f"{location}有重复id")
        ids.add(record["id"])
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{location}至少需要user问题和最后assistant回答")
        previous = None
        normalized = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"{location}仅支持system/user/assistant纯文本消息")
            if message.get("tool_calls") or message.get("function_call"):
                raise ValueError(f"{location}此入口训练普通教学回答，不接受工具调用消息")
            role = message["role"]
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{location}的content必须为非空文本")
            if role == "system":
                if i != 0:
                    raise ValueError(f"{location}的system只能出现在首条")
            elif role == "user":
                if previous not in {None, "system", "assistant"}:
                    raise ValueError(f"{location}的user/assistant必须交替")
            elif previous != "user":
                raise ValueError(f"{location}的assistant之前必须是user")
            normalized.append({"role": role, "content": content.strip()})
            previous = role
        if previous != "assistant":
            raise ValueError(f"{location}最后消息必须是assistant目标回答")
        record["messages"] = normalized
        records.append(record)
    if not records:
        raise ValueError(f"{path}不能为空")
    return records


# 上游：本文件 validate_split.inputs/encode_record；合并脚本 main 也用于实际输入去重。
# 输入：一条完整样本和默认系统指令；返回：去掉最终参考回答的提示消息列表。
# 下游：只做列表切片/拼接，无外部 API；样本已有 system 时保留，否则补默认 instruction。
def prompt_messages(record, instruction):
    messages = record["messages"][:-1]
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": instruction}] + messages
    return messages


# 上游：train_education.read_data；生成脚本 generate、合并脚本 main 也在发布前调用。
# 输入：两组样本及默认系统指令；通过检查时返回 None，发现数据泄漏则抛出 ValueError。
# 下游：内部 inputs → prompt_messages → json.dumps；比较 id、题目族和真实模型输入的交集。
def validate_split(train, validation, instruction):
    for field in ["id", "group_id"]:
        if {r[field] for r in train} & {r[field] for r in validation}:
            raise ValueError(f"训练和验证集存在相同{field}，请按完整会话/题目族分割")
    # 上游：外层 validate_split；输入一组 records，返回该组实际提示内容的字符串集合。
    # 下游：prompt_messages 去掉参考回答；json.dumps 把提示序列化，方便准确比较是否重复。
    def inputs(records):
        # inputs → prompt_messages(r, instruction)：得到学生问题及历史上下文，补上必要的系统指令。
        # inputs → json.dumps：保留中文、固定字典键顺序，使相同提示得到相同字符串用于集合比较。
        return {json.dumps(prompt_messages(r, instruction), ensure_ascii=False, sort_keys=True) for r in records}
    # validate_split → inputs(train)/inputs(validation)：分别构建两组实际输入集合，交集非空即泄漏。
    if inputs(train) & inputs(validation):
        raise ValueError("训练和验证集存在相同实际输入，即使group_id不同也会泄漏")


# 上游：train_education.read_data，在正式训练/数据检查时拒绝演示样本。
# 输入：数据路径列表和样本列表；通过时返回 None，有示例文件名/标记时抛出 ValueError。
# 下游：Path 只用于取得文件名；不重新读取文件、不改写 records，也不调用模型。
def reject_examples(paths, records):
    # reject_examples → Path(p).name：检查文件名是否含“.example.”；同时检查样本的 dataset_kind。
    if any(".example." in Path(p).name for p in paths) or any(r.get("dataset_kind") == "example" for r in records):
        raise ValueError("演示示例不能用于正式训练；请准备独立的教育train/validation数据")


# 上游：train_education.main 的内部 encode；生成脚本 accepted_record/validate_candidate、合并脚本 main。
# 输入：一条样本、已加载的 tokenizer、默认指令、长度上限；返回三个整数列表组成的字典。
# 下游：prompt_messages → tokenizer.apply_chat_template，以及 tokenizer.encode；不调用模型生成答案。
# 返回的 input_ids 给模型读取，attention_mask 标记有效位置，labels 指定哪些位置参与 loss。
def encode_record(record, tokenizer, instruction, max_seq_length):
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer缺少EOS")
    texts = [m["content"] for m in record["messages"]] + [instruction]
    # Qwen将部分思考/工具标记登记为普通added token，不能只检查all_special_tokens。
    reserved = set(tokenizer.all_special_tokens) | {
        "<think>", "</think>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
    }
    for token in reserved:
        if token and any(token in text for text in texts):
            raise ValueError(f"样本{record['id']}含模型保留控制token，请改成普通文字")
    # encode_record → tokenizer.apply_chat_template：先按模型的对话模板编码提示消息，返回 token ID 列表。
    # tokenize=True 直接返回数字；add_generation_prompt=True 在末尾添加助手开始回答的位置。
    # enable_thinking=False 使用非思考回答格式；这里仍只是编码，不是在运行模型推理。
    prompt = tokenizer.apply_chat_template(
        # encode_record → prompt_messages(record, instruction)：提供系统指令、历史对话和当前问题。
        # 最后一个 assistant 参考回答被排除，后面另行编码为训练目标，避免拼接两遍。
        prompt_messages(record, instruction), tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    # encode_record → tokenizer.encode：把最后一个 assistant 的正文转成目标 token ID 列表。
    # add_special_tokens=False 禁止自动追加控制 token；本函数在下一步显式补一个 EOS。
    target = tokenizer.encode(record["messages"][-1]["content"], add_special_tokens=False)
    if not prompt or not target:
        raise ValueError("提示或目标回答编码为空")
    # 把 EOS 纳入 target 和 labels，训练时也监督“何时结束回答”，避免只学习正文而不会停止。
    target = target + [tokenizer.eos_token_id]
    tokens = prompt + target
    if len(tokens) > max_seq_length:
        raise ValueError(f"样本{record['id']}超过{max_seq_length}token，不能截断最终回答")
    # prompt 位置的 labels=-100 仅表示“不计算这部分 loss”，它们仍在 input_ids 中供模型理解上下文。
    # target 的每个 token（包括 EOS）参与 loss；因果语言模型内部会对齐下一 token 的预测标签。
    # 此时还没有 padding，因此 attention_mask 全为 1；补齐及 padding 屏蔽交给 pad_batch。
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": [-100] * len(prompt) + target}


# 上游：Trainer 调用 train_education.run_training 内部 collate 时，collate 再调用本函数。
# 输入：若干 encode_record 返回的样本字典和补齐 token ID；返回同长度、按批次排列的列表字典。
# 下游：仅操作列表，无模型/张量 API；上游 collate 再用 torch.tensor 把本函数结果转换为张量。
def pad_batch(features, pad_token_id):
    if not features:
        raise ValueError("batch不能为空")
    # 按当前 batch 最长样本补齐，而不是把所有样本都补到 max_seq_length，减少无用计算。
    length = max(len(f["input_ids"]) for f in features)
    result = {key: [] for key in ["input_ids", "attention_mask", "labels"]}
    for feature in features:
        padding = length - len(feature["input_ids"])
        # 右侧增加 pad token，使 batch 中所有 input_ids 等长，才能堆叠为二维张量。
        result["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding)
        # padding 的 attention_mask=0：告诉模型这些补齐位置不是需要关注的真实输入。
        result["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        # padding 的 labels=-100：补齐位置不参与 loss；原有参考回答和 EOS 的监督保持不变。
        result["labels"].append(feature["labels"] + [-100] * padding)
    return result


# 上游：train_education.read_data，生成训练/验证文件的来源记录。
# 输入：文件路径；返回：64 个十六进制字符（256 比特）的 SHA-256 摘要，用于记录所用数据版本。
# 下游：Path.read_bytes → hashlib.sha256 → hexdigest；只读文件，不训练模型、不改动原数据。
def file_sha256(path):
    # file_sha256 → Path.read_bytes：取得文件原始字节；→ sha256：计算内容摘要；→ hexdigest：转为字符串。
    # 上游把结果写入数据检查/训练报告；以后可据此判断文件内容是否发生过变化。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
