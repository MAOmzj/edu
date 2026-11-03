<!--
文件用途：面向访客的简化说明：项目是什么、怎么运行。
调用关系：开发者按本文启动 Chroma、同步索引，再启动 FastAPI 或命令行入口。

修改易踩坑：命令和路径必须与实际实现一致，示例中不能出现真实密钥。
-->
# 小学教育知识问答系统

面向小学一至六年级学生的中文教育知识问答项目，基于 LangChain/LangGraph
多 Agent、本机 vLLM（可切回 DeepSeek）、可切换的 SQLite/独立 Chroma/独立 Qdrant
向量后端和 SQLite 记忆库，提供网页聊天与命令行两种使用方式。

代码、模型、适配器和环境已统一整理到 `/root/MAOMI`，实体位于数据盘
`/root/autodl-tmp/MAOMI`。完整布局与迁移记录见 [MAOMI 工作目录说明](../README.md)。
模型在 `../models`，环境在 `../envs`，现有摘要适配器和检查点在 `../training`；
旧路径保留兼容链接，历史实验报告内容保留。

Qwen3.8-27B 的小学多学科教学微调入口为 [run_education_finetune.py](run_education_finetune.py)，
数据格式、环境及训练步骤见 [教育微调说明](data/education_finetune.md)。
当前教学训练数据为 `../training/education-dataset-v2/train.jsonl`，共2500条，四科各625条；验证、测试各100条保留原划分。
数据由本地27B生成，经过AI抽查和定向修订，仍为`pending`合成候选。生成与校验流程见[教学数据扩充](data/education_generation.md)。
完整未量化27B基座已统一整理到 `/root/MAOMI/models/Qwen3.8-27B/`，训练配置已对齐该目录。
正式教学适配器 `../training/education-lora-v2/adapter` 已完成2500条数据、3轮训练。
2026-10-19已部署 `/root/MAOMI/models/Qwen3.8-27B-RTN-INT4` 整数4bit基座与原教育v2适配器独立加载，不合并27B模型。三个模型均使用GPU 0、TP=1；预算依次为0.70、0.04、0.16。vLLM确认使用INT4 Marlin内核，权重部分保留BF16，属于W4A16混合精度INT4部署。客户端聊天模型名仍为 `qwen3.8-27b-education-v2`，原始权重和适配器文件均保留，完整质量与负载评估需另行进行。
部署配置、启动与直接推理示例见 [教育微调模型部署](docs/education_deployment.md)。

## 主要功能

- 语文、数学、英语、科学、安全与品德知识问答，按学科和年级调整回答；
- 学科回答、首次完整审核、第二次安全审核、学习记忆四个子 Agent 协作；
- Chroma/Qdrant 原生语义召回、独立 BM25 倒排召回、加权 RRF 与可选 Rerank；
- 跨请求短期对话记忆与跨会话长期学习记忆；
- 安全的算术计算、知识文件自动增量索引。
- 数学、语文、英语的问题处理 Skill，以及学习记忆提取 Skill。

## 工作流程

```text
学生问题
  → 上下文管理（会话摘要 + 长期记忆 + 教育 Skill）
  → 学科问答 Agent（本地检索 / 联网 / 计算器）
  → 完整审核 Agent（首次不通过则携带意见重写一次）
  → 重写稿交给独立安全审核 Agent（只查安全，通过则放行，不再质量退回）
  → 学习记忆 Agent + SQLite 持久化
  → 返回审核后的最终答案
```

首次完整审核通过时，直接进入记忆保存，不调用第二次安全 Agent。
安全 Agent 不使用知识检索、计算器或联网工具，不打质量分；普通知识和表达问题
不能成为第二次拒绝的理由，危险建议、不适龄内容或隐私泄露仍可阻断。
第二次安全失败或异常只返回固定提示，不返回草稿或未经验证的审核修正版。
默认最多重写一次、审核两次；新问题会重新从首次完整审核开始。

普通工具按每道题分别限额：初稿、首次完整审核与重写共享对应工具的计数；
已执行但失败的调用也计入额度，新问题重新计数。超出额度的请求由工具中间件拦截，
不执行工具；已耗尽工具不再向模型提供，其他工具仍可使用。联网回退到本地知识库
也计入检索额度，防止绕过限制。审核规则及审核联网开关保持原有配置。
限额在 `config/app.yml` 的 `agent.tool_call_limits` 调整；计数仅存于本轮运行内存，
不进入会话 Checkpoint，也不会因重写而重置或影响其他请求。

本机 vLLM 窗口只有 16384 token，限额因此比默认值（每个工具 6 次）收紧：
`search_knowledge` 4 次、`search_web` 1 次、`calculate_expression` 6 次。
`search_knowledge` / `search_web` 每次会往上下文写入约 2000 字符资料，上一轮实测
连续约 12 次检索就会超出窗口、让整轮问答失败，收紧后重资料工具合计最多 5 次。
`calculate_expression` 的结果只有几十字符，不受影响。窗口调大后可放宽这些数值。

教育流程保存在 `skills/primary-math`、`skills/primary-chinese`、`skills/primary-english`
的 `SKILL.md` 中。单科加载对应流程；“综合”加载三科，让现有问答 Agent 按问题选择，
无需额外分类调用。流程覆盖识别任务、补足条件、检索核验、分支处理和答案检查；工具仍负责
实际检索与计算，Skill 本身不执行代码，也不改变两阶段审核规则。

`skills/learning-memory/SKILL.md` 只加载到学习记忆 Agent，以本轮学生原话、学科和年级
提取明确偏好、学习主题与自述困难，不根据助手答案推断掌握程度。候选经过结构校验及存储层
过滤后保存；年级与本轮提问主题仍由存储代码记录。成功返回空候选或提取异常时仅保存这两项，
不启用旧关键词规则推断偏好或困难，提取异常不影响最终答案。

映射与字符预算在 `config/app.yml` 的 `skills` 下配置，可用 `enabled: false` 关闭附加 Skill。
单份正文上限 6000 字符，综合拼接上限 9000 字符，超限会报配置错误而不截断规则。
Skill 文件有缓存，修改后需要重启 CLI 或 Web 服务。

## 如何运行

### 环境要求

- Python 3.10 或更高版本
- 本机 vLLM：聊天、向量、独立摘要模型各起一个实例，当前配置使用单张A800 80G、GPU 0、TP=1（不需要云端 Key）
- 嵌入式 Chroma 是本机默认向量后端，不需要 Docker；独立 Chroma / Qdrant 才需要
- 切回云端供应商时才需要 DeepSeek API Key 和阿里云百炼 API Key
- Tavily API Key（可选，仅联网搜索需要；缺失时自动回退本地知识库）

### 本机 vLLM 部署（默认）

默认配置全部在本机完成，不依赖任何云端 API Key：

```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 启动聊天模型（OpenAI 兼容接口，端口 8000）
python scripts/serve.py chat

# 3. 启动向量模型（--runner pooling，端口 8001）
python scripts/serve.py embedding

# 4. 建立知识库索引（首次运行，或更换 Embedding 模型后）
python main.py --sync-index
```

四个服务可以一次拉起：`python scripts/serve.py all` 会依次启动聊天、Embedding、
合并后的 v2 摘要模型，逐个等待就绪，再启动并检查网页服务。用 `python scripts/serve.py status` 查看运行
状态，`python scripts/serve.py stop all` 停止。默认前台运行并把日志同时打到终端
和 `logs/serve-*.log`，加 `-d` 转入后台。

三个模型各自设置 `*_tensor_parallel_size: 1` 与 `*_cuda_visible_devices: '0'`。
本次单卡INT4的配置预算为聊天0.70、Embedding 0.04、摘要0.16，合计该卡90%；
预算包含权重、缓存和运行开销，三个模型已共同启动并通过同时推理检查；实测总显存72.23GiB，空闲7.02GiB，实际占用以 `nvidia-smi` 为准。
逐个启动避免 vLLM 显存分析期间其他模型同时新增占用；配额可用环境变量覆盖，`status` 展示实际可用GPU。
调整这些配置后，先用 `.venv/bin/python scripts/serve.py stop all` 停止旧进程，再用 `.venv/bin/python scripts/serve.py all -d` 统一重启。
端口和模型名都从 `config/model.yml` 的 `chat_base_url`、`embedding_base_url`、
`chat_model_name`、`embedding_model_name` 读取，脚本与配置不会各自漂移。

`chat_quantization: compressed-tensors` 读取已经导出的RTN INT4权重，原v2 LoRA仍由vLLM独立加载。
新权重使用group32、非对称W4A16：权重4位，激活、未量化层和适配器使用BF16，KV Cache也仍为BF16。
量化入口为 [quantize_education_int4.py](scripts/quantize_education_int4.py)，不做AWQ平滑，不需要激活校准数据；
它从原BF16基座读取所有数值，不复制历史AWQ的平滑权重。INT4启动不传NF4的HF overrides或BNB加载参数。
必须同时配置 `chat_quantized_base_manifest: /root/MAOMI/edu/config/chat_int4_base.json`，
来源校验绑定INT4基座、原训练基座、正式适配器和量化证据，缺失或文件指纹变化会拒绝启动。
不要改写适配器的 `base_model_name_or_path` 来绕过检查，详细步骤和NF4回退见[部署说明](docs/education_deployment.md)。

历史NF4单卡与双卡TP报告继续保留；本次只验证单卡INT4，不将旧检查当成本次结果。
北京时间11:24:26完成INT4功能验证，报告为 `../organization/education-int4-lora-deployment.json`：4个聊天请求与摘要、Embedding同时请求通过，三个项目客户端、工具调用、318文件/50,702片段的知识检索及网页问答通过，网页示例用时39.891秒。原适配器SHA256未变，基座/适配器同一首token概率不同，证明微调增量实际参与前向。聊天 `max_num_seqs` 仍为4，摘要仍为1；这些检查不代表NF4质量等效或最大用户并发，延迟与负载需要另行压测。

本机窗口为 16384 token，`config/app.yml` 的上下文预算已相应收紧到 4000 字符。
`.env` 里的云供应商密钥可以全部留空，本机部署由 `config/model.yml` 的
`chat_provider: local_vllm` 与 `embedding_provider: local_vllm` 生效。

### 独立摘要小模型

`config/model.yml` 的 `summary` 配置让会话压缩使用合并后的
`/root/MAOMI/models/Qwen3-4B-summary-lora-v2`；学科回答、审核和学习记忆提取仍走主聊天模型。
当前 `enabled: true`，`serve.py all` 会自动启动它。合并目录包含完整权重，无需 LoRA 启动参数。
摘要服务只监听本机 `127.0.0.1:8002`，使用 16384 token 窗口、单请求并发，关闭思考模式。
每轮预填充最多处理 2048 token，长输入分块处理以降低激活显存峰值，保持 16384 token 上下文窗口。

```bash
cd /root/MAOMI/edu

# 后台部署全部四个服务；等待模型和网页就绪后返回
.venv/bin/python scripts/serve.py all -d
.venv/bin/python scripts/serve.py status

# 检查摘要服务模型名，应为 education-summary
curl --noproxy '*' http://127.0.0.1:8002/v1/models
.venv/bin/python scripts/evaluate_summary.py

# 查看日志或停止全部服务
tail -f logs/serve-summary.log
.venv/bin/python scripts/serve.py stop all
```

`data/summary_eval.jsonl` 提供 8 个虚构教育对话，用于检查事实修正、未解决问题、
隐私过滤和提示注入等边界；它不是正式训练集。评测脚本经过真实摘要入口生成
`storage/summary_baseline_*.json`，记录摘要、参考文本、延迟与 fallback，不把请求成功当作准确率。
主模型在线时也可用 `--model main` 生成对照报告。

本次三个模型均配置GPU 0、TP=1，配置预算为聊天0.70、Embedding 0.04、摘要0.16，合计0.90。测试后主模型进程55.84GiB、Embedding 3.42GiB、摘要12.95GiB；主模型日志加载18.31GiB，KV/状态缓存池36.59GiB，不能把进程总占用当作权重大小。
关闭独立摘要不会自动改写已配置的主模型配额。已有运行进程不会自动调整显存，切换配额需要重启。
可用 `QWEN_GPU_MEM_UTIL`、`EMBED_GPU_MEM_UTIL`、`SUMMARY_GPU_MEM_UTIL` 覆盖，
窗口可用 `SUMMARY_MAX_MODEL_LEN` 覆盖。8GB 只是权重大小，显存还需要容纳 KV Cache 与运行开销。
单独执行 `summary -d` 仍立即返回，可用 `status` 查看加载进度。
设置 `EDUCATION_SUMMARY_ENABLED=false` 可让网页复用主聊天模型，并让 `all` 跳过摘要服务。

微调应使用与线上一致的“旧摘要 + 新增对话 → 核验后的新摘要”样本，先划分独立会话的
训练、验证和测试集，再做 SFT + LoRA。不要将未经校正的自动摘要或原始学生信息直接当作标签。
微调后把合并模型目录填到 `summary.model_path`，保持 `model_name` 与服务名一致，再重跑基线。

训练入口是 [`scripts/train_summary.py`](scripts/train_summary.py)，使用 BF16 LoRA、
仅监督目标摘要；[`scripts/merge_summary_adapter.py`](scripts/merge_summary_adapter.py)
负责导出 vLLM 可加载的合并模型。数据格式、标注标准和候选生成命令见
[`data/summary_dataset.md`](data/summary_dataset.md)，依赖见 `requirements-summary-training.txt`；
本地完整训练与部署说明在 `docs/summary_finetuning.md`（该目录按项目规则不提交 Git）。
两个 `summary_*.example.jsonl` 只演示格式，需要替换成独立会话、人工核验后的正式数据。

[`scripts/generate_summary_data.py`](scripts/generate_summary_data.py) 可离线生成 200 条虚构候选，
按模板族划分为训练 160、验证 20、测试 20 条，标记为 `pending`，不会启动微调。
候选主要是短对话模板，需逐条核验并补充真实脱敏的长对话；合成验证结果不能代表实际质量。

已有对话可用 [`scripts/label_summary_data.py`](scripts/label_summary_data.py) 调用已运行的本地模型
自动压缩并逐条保存输入、摘要候选、生成来源和失败记录。输入无需 `target_summary`；
已有目标也不会送给模型。默认使用本地主模型，标签标记为 `pending`，复核后再用于摘要小模型微调。
命令、输出格式和集合隔离说明见上述数据文档。

当前扩展版训练候选为 `storage/summary_dataset_v2/train.jsonl`，共 1000 条：
保留原 160 条核验样本，新增 840 条由本地 27B 模型生成的待复核摘要。
验证和测试各 20 条原样保留，模板族及实际输入与训练集隔离。
生成来源、批次记录和 64 条语义抽查见该目录的报告；本轮扩充没有再次训练模型。

可以直接运行 [`run_summary_finetune.py`](run_summary_finetune.py)，参数和绝对路径已写在
文件顶部，无需设置 Shell 变量：`python run_summary_finetune.py`。默认训练 3 轮，
保存到 `/root/MAOMI/training/summary-lora-v2/adapter`，输出目录必须尚不存在。
脚本先检查数据和 token，再暂停原先运行的本项目服务，训练结束、失败或普通中断后
按原命令和环境将这些服务恢复为独立后台进程；训练期间网页问答暂时不可用。
原先未运行的服务不启动，未登记的 GPU 进程不自动停止。
`python run_summary_finetune.py --dry-run` 只检查，不暂停服务或训练。
新增 840 条标签仍待核验，这个入口用于实验训练，训练完成不代表摘要质量通过。

已完成首轮真实 Qwen3-4B LoRA 实验：训练 160 条、验证 20 条，训练 3 轮后选择
第 1 轮验证最优的适配器。v1 合并模型已按之前的清理要求删除，原始基座和适配器保留。
同输入的 28 条合成测试中，问题状态更新有所改善，但仍有指令干扰失败，且出现隐私过滤回归，
因此 v1 仅作为实验候选。当前部署按项目设置接入 v2 合并模型；其已有复核仍发现
状态更新、提示注入和信息遗漏问题，部署接入不代表摘要质量通过。v1 参数、复现命令和逐例报告见
[`data/summary_finetune_v1.md`](data/summary_finetune_v1.md)。

### 切换供应商

`config/model.yml` 的 `chat_provider` 与 `embedding_provider` 决定用哪一套供应商，
也可以用环境变量临时覆盖，不必改配置文件：

```bash
# 云端聊天（需要 DEEPSEEK_API_KEY）
EDUCATION_CHAT_PROVIDER=deepseek python main.py --question "为什么月亮会有圆缺变化？" --subject 科学 --grade 4

# 云端向量（需要 DASHSCOPE_API_KEY；更换 Embedding 后必须重建索引）
EDUCATION_EMBEDDING_PROVIDER=dashscope python main.py --sync-index
```

### 云端部署（备选）

```powershell
# 1. 配置密钥
Copy-Item .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY 等密钥

# 2. 默认后端是 Qdrant，启动独立 Qdrant Server
docker compose -f deploy/qdrant/docker-compose.yml up -d --wait --wait-timeout 60

# 3. 建立知识库索引（首次运行或从嵌入式 Chroma 迁移后）
python main.py --sync-index
```

### 双路检索与关键词索引

问答时分别执行纯向量召回和独立 BM25 倒排召回，再按分片 ID 做加权 RRF：

```text
向量召回（80 条） ─┐
                  ├→ 加权 RRF → 20 条候选 → Rerank → 最终 5 条
关键词召回（80 条）┘
```

默认排名贡献为 `0.75 / (5 + 向量排名) + 0.25 / (5 + 关键词排名)`，
排名从 1 开始，某一路没有召回时该项为 0。参数在 `config/knowledge.yml`
中的 `hybrid_vector_weight`、`hybrid_bm25_weight` 和 `hybrid_search` 调整。
常数采用 5，避免提高语义权重后仅关键词命中的资料全部被挡在 20 条候选之外。
这是一组默认配置，需要评测后再调优；Rerank 失败时返回 RRF 排名前列的资料。

倒排索引用标准库 SQLite 保存，查询只读取匹配词项，不在请求中重新分词全库。
数据库跟随各向量后端的清单，例如 `storage/index_manifest_qdrant.bm25.sqlite3`；
学科过滤同时应用于两路召回。增量同步会同时维护向量、倒排与文件清单。

已有完整向量索引的用户可离线补建关键词索引，不连接向量服务器，也不调用 Embedding：

```powershell
python main.py --sync-keywords
```

补建会核对文件哈希和分片 ID。知识文件已变化时，先执行 `--sync-index`；
已有清单与切分参数不一致时，需要 `--rebuild-index`；旧 SQLite/Chroma 清单
未记录切分指纹时也需要显式重建。所有向量集合必须由本项目独占，清单缺失或
分片数量不一致时，`--sync-index` 会清空当前后端集合并重建。
缺失或未完成的关键词索引
不会在网页请求中自动重建，接口会明确提示同步；同步成功后才允许融合两路结果。
多主机只读副本需部署同一版本的 manifest 与倒排数据库；当前方案仍需唯一索引任务。

### 选择向量后端

三种后端的数据和增量清单彼此独立，切换不会删除或覆盖原后端。本机默认使用
嵌入式 Chroma（`config/knowledge.yml` 的 `vector_backend: chroma`，不需要 Docker），
云端部署默认使用 Qdrant；也可以只在当前进程环境中切换：

```powershell
# 独立 Qdrant（首次切换后会建立自己的完整索引）
docker compose -f deploy/qdrant/docker-compose.yml up -d --wait --wait-timeout 60
$env:EDUCATION_VECTOR_BACKEND = "qdrant"
python main.py --sync-index

# 如果 SQLite 已有完整且与当前 Embedding/切分参数一致的索引，可直接迁移现成向量，
# 避免重新调用 Embedding API。旧 v4 清单不记录 schema，所以必须显式确认。
python scripts/migrate_sqlite_vectors_to_qdrant.py --confirm-current-schema

# 本地 SQLite（适合开发、离线演示和兜底）
$env:EDUCATION_VECTOR_BACKEND = "sqlite"
python main.py --sync-index

# 切回独立 Chroma
$env:EDUCATION_VECTOR_BACKEND = "chroma"
python main.py --sync-index
```

Qdrant 的 `collection_name` 必须由本项目独占。Qdrant manifest 缺失、索引指纹改变
或远端点数与 manifest 不一致时，同步任务会清空该集合并从 `data/knowledge` 全量重建，
以避免旧点、空集合或不同 Embedding 的向量混在一起。

`deploy/qdrant` 是只绑定本机端口的单节点运行基线，不等同于生产 HA 集群。生产
环境必须同时启用私网隔离、HTTPS 和 API Key，并配置多节点副本、快照备份与监控；
客户端密钥只放在环境变量 `QDRANT_API_KEY` 中。若用 `QDRANT_HTTP_PORT` 修改
Compose 发布端口，还必须同步修改 `QDRANT_URL` 中的端口。

水平扩容应用时，所有在线服务副本应设置
`EDUCATION_SYNC_ON_SEARCH=false`，只读 Qdrant；另设唯一索引任务执行
`python main.py --sync-index`。本地 JSON manifest 与进程锁不是跨主机分布式锁，不能
让多个索引任务同时写同一集合。

### 启动网页界面

```powershell
python scripts/serve.py web
```

等价于直接运行 `python -m uvicorn app:app --host 127.0.0.1 --port 8766`，
但会额外做三件事：检查并记录 pid（避免重复启动）、把日志写进 `logs/serve-web.log`、
轮询 `/api/health` 确认真的起来了。用 `EDU_HOST=0.0.0.0` 可对外开放（需自行加访问控制）。

浏览器访问 <http://127.0.0.1:8766/>

### 命令行提问

```powershell
# 单次提问
python main.py --question "为什么月亮会有圆缺变化？" --subject 科学 --grade 4

# 连续对话
python main.py --subject 数学 --grade 5

# 索引管理
python main.py --status          # 查看索引状态（不调用在线模型）
python main.py --sync-keywords   # 从已有向量清单补建关键词倒排索引
python main.py --rebuild-index   # 重建索引
```

### 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 目录结构

```text
app.py                  Web 服务入口（FastAPI）
main.py                 命令行入口
agent/                  多 Agent、上下文管理与工具
memory/                 SQLite 记忆存储
rag/                    RAG 检索与向量存储
deploy/chroma/          独立 Chroma Server 的 Docker Compose 配置
deploy/qdrant/          独立 Qdrant Server 的 Docker Compose 配置
model/                  模型工厂与 Embedding
config/                 YAML 配置
data/knowledge/         知识库文件
web/                    前端页面
skills/                 教育 Skill
scripts/                服务启动器（serve.py）与索引迁移脚本
storage/                运行时数据：向量索引、BM25 倒排、记忆库
```

## 排查与运维

[问题排查记录.md](问题排查记录.md) 记录了 2026-10-10 排查「提问无回复」时发现的问题、
根因和修改，包括：工具调用限额的由来、两次审核为什么这样分工、网页报「回答暂时失败」
怎么定位、以及**不借助 `serve.py` / `ss` / `lsof` 时怎么查某个端口被哪个进程占用**。
2026-10-18新增第八节，记录符号链接路径标识不一致导致知识检索503、保留向量索引并补建关键词索引的处理与验证。
同日第九节完整记录NF4双卡TP的兼容处理，包括预量化误判、融合Q/K/V分片、插件注册与日志误判、启动告警、显存代价及单卡回退。
2026-10-19第十节记录AWQ浮点平滑为何影响原LoRA坐标、本地生成不平滑RTN INT4、来源清单校验及检查点迁移释放数据盘空间的过程。
遇到故障时先看这份文件。

## 使用边界

本项目用于辅助学习，不代替学校课程、老师指导或紧急情况下的专业帮助。
