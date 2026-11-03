<!-- 文件用途：说明摘要微调数据格式、离线候选生成、标注与训练前准备。 -->
<!-- 调用关系：开发者阅读；对应 generate_summary_data.py、train_summary.py 与 evaluate_summary.py。 -->
<!-- 修改易踩坑：候选摘要不是人工金标；真实长对话和独立测试不能被模板变体代替。 -->

# 摘要微调数据怎么准备

本项目训练的是**会话增量摘要**：输入“旧摘要 + 本次要压缩的对话”，输出“更新后的摘要”。教材内容可以约束对话的学习主题和正确答案，但直接训练“教材正文 → 教材摘要”与当前线上任务不同。

每行一条 UTF-8 JSONL：

```json
{
  "conversation_id": "anonymous-conversation-001",
  "previous_summary": "正在学习时间单位，分钟换算为秒的疑问尚未解决。",
  "messages": [
    {"role": "assistant", "content": "1分钟是60秒，所以4分钟是240秒。"},
    {"role": "user", "content": "明白了，我算出5分钟是300秒。以后请用表格讲。"},
    {"role": "assistant", "content": "5乘60等于300，计算正确。"}
  ],
  "target_summary": "已澄清1分钟为60秒，4分钟为240秒；学生正确给出5分钟为300秒，本轮换算疑问已解决，明确偏好表格讲解。"
}
```

`messages` 按时间排列，只有 `user`/`assistant`，不包括仍需原样保留的最近消息。首个摘要窗口的 `previous_summary` 可以为 `""`。一个会话可以形成多个窗口，但这些窗口必须留在同一个数据集合。

标注摘要时，保留学习主题、确认的正确结论、学生明确说出的困难和偏好，以及待解决的问题。用最新消息纠正旧条件和旧结论；问题已经解决时更新状态，偏好发生变化时保留最新明确偏好。助手讲过一个知识点，不能据此写成“学生已熟练掌握”。题目或故事里的人物信息也不能变成学生的个人经历。

删去姓名、学校、联系方式及内部工具、草稿、审核过程。真实对话应先脱敏再进入任何生成或训练流程。检查目标中的每个事实是否来自**按线上预算裁剪后的输入**。发现虚构会话事实错误，应修订会话或剔除样本，不能在目标摘要里偷偷补入输入之外的答案。

## 先生成候选数据

在项目根目录执行，纯离线、无需 GPU 或模型权重：

```bash
.venv/bin/python scripts/generate_summary_data.py \
  --output-dir storage/summary_candidates_v1 \
  --count 200 \
  --seed 42
```

输出目录必须尚不存在，避免覆盖已有标注。再次生成请换新目录。默认预算与当前项目一致：目标摘要最多 1200 字符，对话输入最多 8000 字符；如果修改线上配置，请同步传入 `--max-summary-chars` 和 `--summary-input-chars`，再用训练入口核验。

生成器使用 20 个虚构教育对话模板族，覆盖数学、语文、英语、科学，以及纠错、状态更新、偏好、题目人物、隐私过滤和对话内指令干扰。参考摘要由模板同时构造，数学结果用程序计算；它没有调用强模型，也没有读取学生数据库。模板变体用不同的实质练习内容扩展，重复的实际输入会被拒绝。

默认生成 200 条候选：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `train.candidates.jsonl` | 160 | 待核验训练候选 |
| `validation.candidates.jsonl` | 20 | 待核验验证候选 |
| `test.candidates.jsonl` | 20 | 待核验测试候选 |
| `test_eval.candidates.jsonl` | 20 | 同一测试集转换为评测格式，没有新增样本 |
| `dataset_report.json` | — | 来源、分割、模板分布、长度和文件 SHA-256 |

每条额外带有 `source: synthetic_template`、`review_status: pending`、`template_family` 与 `scenario_tags`。这些字段用于追踪来源，训练加载器不会把它们送入模型，也不会据此自动认定摘要已审核。**生成成功和通过格式检查不代表目标摘要已经逐条人工核验。**

模板族整体划分为 80%/10%/10%，同一模板的不同数字或措辞不会跨集合。脚本还检查会话编号、全部候选的重复输入，以及三个集合在实际预算下的输入重叠。非默认数量和比例按模板族划分，条数可能略有偏差。

## 从候选变成训练数据

### 让本地模型自动压缩并保存记录

`scripts/label_summary_data.py` 对已有对话逐条调用本地模型，保存“实际输入 + 模型摘要 + 生成来源”。输入每行只需 `conversation_id`、`previous_summary` 和 `messages`；没有旧摘要时使用空字符串。即使输入已有 `target_summary` 或 `reference`，它们也不会进入模型请求。输入应先脱敏，格式检查不会自动删除真实身份信息。

使用已启动的本地主模型生成候选：

```bash
.venv/bin/python scripts/label_summary_data.py \
  --input-file storage/summary_candidates_v1/train.candidates.jsonl \
  --output-dir storage/summary_model_candidates_v1/train \
  --model main
```

以上命令应在输出目录尚不存在时执行。当前 `main` 对应本机 `qwen3.8-27b` 服务，地址 `http://127.0.0.1:8000/v1`；工具不会启动服务。尚未启动时，可先运行 `.venv/bin/python scripts/serve.py chat -d`，等待 `/v1/models` 就绪。服务名是本项目的部署别名。

默认关闭思考、temperature 为 0，并复用线上摘要的规则和字符预算。`--base-url` 与 `--model-name` 可以指定其它已经运行的本机回环服务。`--model summary` 对应配置中的独立摘要服务，必须先准备完整权重并启动；它不依赖在线摘要的 enabled 开关，也不会修改配置。工具只连接本机，禁用环境代理且不跟随重定向。

输出目录中：

| 文件 | 内容 |
| --- | --- |
| `summaries.candidates.jsonl` | 成功返回的摘要候选，格式可被训练加载器读取 |
| `failed_records.jsonl` | 失败会话编号、错误类型和简要原因，不包含对话正文或模型残片 |
| `labeling_report.json` | 输入/成功/失败数、服务名、参数、预算、Prompt 与结果文件哈希 |

每条候选保留原始会话编号、旧摘要和对话，标记 `source: local_model`、`review_status: pending`，并记录模型、Prompt 哈希、耗时、token 用量及输入预算。原有 `template_family` 与 `scenario_tags` 会保留用于追踪，来源元数据不会进入请求。

请求失败、空输出、达到 token 上限、摘要超过字符预算，以及输出含思考/控制标记时，均不把该结果写入成功文件，也不使用本地 fallback 代替模型标签。每条处理后刷新文件和报告；中断后已保存的结果保留。当前不支持在已有目录自动续写，可使用 `--start-index` 从指定有效记录继续到新目录；不要直接拼接含重复输入的结果。

先检查输入而不调用模型：

```bash
.venv/bin/python scripts/label_summary_data.py \
  --input-file storage/summary_candidates_v1/train.candidates.jsonl \
  --output-dir storage/summary_model_check \
  --validate-data-only
```

可以用 `--limit 3` 先生成少量样本，再逐条检查。训练、验证、测试输入分别处理，保留已有分割，不把三个集合合并后随机拆开。这些标签仍是模型预测；它们需要人工核验，才能作为正式参考摘要。即使由更大的模型生成，也可能遗漏否定、推断掌握程度或保留过时结论。

逐条对照旧摘要、对话和目标，修订或剔除有问题的记录，保留来源信息和人工复核记录。将筛选后的训练、验证数据分别另存为 `data/summary_train.jsonl`、`data/summary_validation.jsonl`；不要仅批量更改审核标记代替检查。

可以先对候选做格式检查，这不会启动训练，也不会将候选认定为正式数据：

```bash
.venv/bin/python scripts/train_summary.py \
  --train-file storage/summary_candidates_v1/train.candidates.jsonl \
  --validation-file storage/summary_candidates_v1/validation.candidates.jsonl \
  --validate-data-only
```

这批模板数据适合检查输入格式、目标监督和训练流程，主要是短对话窗口。为提高实际摘要质量，应继续加入脱敏的真实教育会话：覆盖接近压缩触发长度的对话、重复和闲聊、多轮纠正、跨学科切换，以及多次摘要累积。也可用更强的模型起草这些对话的目标摘要，再按相同标准复核。生成新对话时，应先按原始会话/模板来源分集合，再扩展窗口和变体。

保留独立真实会话测试集，测试集不要参与调参、生成训练变体或训练。`data/summary_eval.jsonl` 的现有 8 条回归样本也不应进入训练。即使合成测试模板已隔离，它仍是模板分布，不能用它的结果证明真实会话质量。

生成并过滤合成输入/输出后用于监督微调，是已有研究采用的方法，例如 [Self-Instruct（ACL 2023）](https://aclanthology.org/2023.acl-long.754/)。这支持合成数据作为补充的可行性，不构成本项目摘要质量的验证。

微调前后都使用相同测试输入和生成参数，逐条比较事实保留、纠错、问题状态、隐私过滤、信息遗漏和输出长度，再比较延迟。请求成功和 loss 下降都不是摘要正确率。训练与合并入口见 [train_summary.py](../scripts/train_summary.py) 和 [merge_summary_adapter.py](../scripts/merge_summary_adapter.py)；本地完整运行说明见 `docs/summary_finetuning.md`。

2026-10-12 已实际完成一次 Qwen3-4B LoRA 实验。使用 `storage/summary_curated_v1/` 中逐条由 Codex 助手核对输入并修订的 160/20/20 条数据，标记 `assistant_reviewed`，不是人工金标。`audit_report.json` 保留教师候选、修订目标的哈希与逐条依据，没有直接把自动生成标签更改标记后投入训练。实验和评测结论见 [summary_finetune_v1.md](summary_finetune_v1.md)；它仍是短合成数据实验，当前模型未通过隐私和指令干扰边界检查，不能作为正式上线依据。

2026-10-13 已按请求将训练数据扩展到 **1000 条**，实际文件为 `storage/summary_dataset_v2/train.jsonl`：原来的 160 条核验样本完整保留，另有 840 条新增对话由本地 `qwen3.8-27b` 实际生成摘要，全部请求成功，新增标签仍标记 `source: local_model`、`review_status: pending`。因此这 1000 条是混合来源的训练候选，不能描述成全部由 27B 直接生成或全部已经核验。

`validation.jsonl`、`test.jsonl`、`test_eval.jsonl` 从原数据集原样保留，验证和测试各 20 条。新输入只来自原有 16 个训练模板族；训练数据内部及其与验证、测试、固定 8 条回归的实际 Prompt 均无重复。新增主要是数值变体，没有增加真实学生会话或长上下文覆盖。

新增输入在 `storage/summary_dataset_v2/new_inputs.jsonl`，不包含目标摘要或参考答案；模板参考单独保存在 `new_template_references.jsonl`，未送入模型请求。840 条原始模型候选及逐条生成信息保存在 `storage/summary_model_candidates_v2/batch_000/` 与 `batch_001/`，每批 420 条。`dataset_report.json` 记录计数、来源、文件哈希和分割检查，`teacher_generation_audit.json` 为全量输入/生成元数据审计，`spot_review.json` 为提前固定的 64 条分层语义抽查。抽查存在信息遗漏和归属、状态或掌握程度问题，不把格式通过或调用成功当成质量通过；本轮只扩充数据，没有启动新一轮模型训练。

## 用 Python 文件启动实验训练

项目根目录的 [run_summary_finetune.py](../run_summary_finetune.py) 已写好训练 Python、基座、数据和参数。直接执行：

```bash
python /root/MAOMI/edu/run_summary_finetune.py
```

默认使用上述 1000/20 条训练/验证文件，训练 3 轮，batch size 2、梯度累积 4、学习率 1e-4、LoRA r16/alpha32/dropout0.05，最大完整序列 2048 token。它自动使用项目 `.venv` 处理服务生命周期，训练子进程使用独立 `MAOMI/envs/summary-training`；不会把训练依赖安装到推理环境。输出为 `/root/MAOMI/training/summary-lora-v2/adapter`，只保存适配器，不自动合并或启用线上摘要。再次训练请修改文件顶部 `OUTPUT_DIR` 为新目录；脚本拒绝覆盖已有适配器或未完成训练目录。

脚本先预检，失败时不暂停服务。正式训练会暂停当时登记的本项目模型和网页服务，因为前台 `serve.py all` 中任一子服务退出都会连带清理其它服务；训练结束、失败或普通 SIGINT/SIGTERM 中断后，先等待训练进程退出，再按快照的原命令、环境和工作目录恢复原服务为独立后台进程。它只操作经 PID、启动时间、进程组与命令验证的项目实例，不自动停止未登记的 GPU 任务；原来未运行的服务保持未运行。恢复失败会明确报告，日志与启动记录位于 `logs/`。

只做数据和 token 检查：

```bash
python /root/MAOMI/edu/run_summary_finetune.py --dry-run
```

当前新增标签仍为待核验的合成候选，入口允许实验训练并显示这一状态，不会把标签改为已审核。质量验收应继续使用独立测试集；完整训练流程仍由 [train_summary.py](../scripts/train_summary.py) 执行。
