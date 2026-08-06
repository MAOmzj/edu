# 小学教育知识问答系统

这是一个面向小学一至六年级学生的中文教育知识问答项目。系统使用
LangChain/LangGraph 多 Agent、DeepSeek 官方 API 和本地 SQLite 知识库，并保留
Chroma 可选后端，支持：

- 简洁、响应式的网页聊天界面；
- 语文、数学、英语、科学、安全与品德知识问答；
- 按学科和年级调整回答；
- 基于 SQLite Checkpointer 的跨请求短期对话记忆；
- 跨会话长期学习记忆和可恢复的页面历史；
- 基于本地资料的 RAG 检索；
- 本地资料不足或问题具有时效性时，使用 Tavily 搜索可信网页；
- 学科回答、独立审核和学习记忆三个子 Agent 协作；
- 按学科加载可复用的教育 Skill，当前包含小学数学教学流程；
- 安全的加减乘除、括号、余数和乘方计算；
- 知识文件增删改后的自动增量索引；
- 不向学生展示内部推理和工具原始结果。

## 工作流程

```text
学生问题
  → 统一 Context Manager 按需摘要旧会话，并选择最近历史、长期记忆和教育 Skill
  → Context Manager 按固定顺序构造学科 Agent Prompt
  → 学科问答 Agent Tool
      ├─ primary-math Skill：规定数学识题、解题、验算和讲解流程
      ├─ search_knowledge：检索本地小学知识
      ├─ search_web：按需检索可信网页，失败时回退本地知识
      └─ calculate_expression：核对基础算术
  → 审核/反思 Agent Tool
      ├─ 通过：进入学习记忆
      └─ 不通过：携带审核意见重写一次
  → 学习记忆 Agent Tool 提取结构化候选
  → SQLite 原子保存最终问答和长期记忆
```

三个子 Agent 都不单独保存会话。最外层 LangGraph 工作流统一使用 SQLite
Checkpointer，因此短期历史只有一份；草稿、审核过程和工具消息不会出现在学生
可见的聊天记录里。相关代码位于 `agent/multi_agent/`：

```text
state.py          共享 State 和结构化数据格式
subject_agent.py  创建学科 Agent，并包装为 Tool
review_agent.py   创建审核 Agent，并包装为 Tool
memory_agent.py   创建学习记忆 Agent，并包装为 Tool
workflow.py       固定执行顺序、审核重试和最终持久化
```

## 统一上下文管理

`agent/context_manager.py` 是学科问答上下文的唯一组装入口，集中负责：

- 从 Checkpointer 恢复的 `messages` 中选择最近短期历史；
- 当消息数量或字符数超过阈值时，把较旧消息增量并入 `conversation_summary`；
- 用 LangGraph `RemoveMessage` 从最新 Checkpoint State 删除已摘要消息，只留最近原文；
- 从 SQLite 检索与当前问题相关的跨会话长期记忆；
- 按学科选择受信任的教育 Skill；
- 接收审核 Agent 的重写意见；
- 按固定顺序组装系统 Prompt 和本轮用户 Prompt；
- 按字符预算裁剪各部分，优先保留最近消息和完整的当前问题。

工作流只传递 `conversation_context`、`long_term_context` 和 `skill_context`
这些结构化快照；学科 Agent Tool 只执行 Context Manager 已经构造好的
`context_message`，不再自己拼接上下文。预算统一配置在 `config/app.yml`：

```yaml
context:
  recent_message_limit: 8
  max_conversation_chars: 4000
  max_long_term_chars: 1800
  max_review_feedback_chars: 1200
  summary_enabled: true
  summary_trigger_messages: 10
  summary_keep_recent_messages: 6
  summary_trigger_chars: 4000
  max_summary_chars: 1400
```

摘要采用增量方式：新摘要由“已有摘要＋本次移出的旧消息”生成，而不是每次重新
读取全部历史。正常运行时复用 DeepSeek 聊天模型；摘要调用失败时自动改用本地
确定性压缩，不会中断本轮问答。模型最终看到的顺序是“较早会话摘要＋最近原始
消息＋审核意见＋教育 Skill＋相关长期记忆＋当前问题”。

`education_messages` 中的完整页面聊天历史不会因 Checkpoint 压缩而删除。摘要只
控制 Agent 的短期 State 和模型上下文，前端仍然可以查看每一轮原始问题和最终答案。

审核 Agent 和学习记忆 Agent 仍只接收完成各自任务所需的最小输入，避免把学生
长期记忆、完整会话或内部标识无必要地扩散给所有子 Agent。

联网搜索默认只提供给学科 Agent。审核 Agent 是否也能联网由
`config/app.yml` 的 `web_search.review_enabled` 单独控制；学习记忆 Agent 不含
任何搜索工具。

## 教育 Skill

Skill 负责规定“怎样教学”，Tool 负责真正执行检索、联网和计算。当前数学 Skill
位于 `skills/primary-math/SKILL.md`，数学问题会自动加载，其他学科不会误用。
学科映射位于 `config/app.yml`：

```yaml
skills:
  enabled: true
  max_instruction_chars: 6000
  subject_paths:
    数学: skills/primary-math/SKILL.md
```

`agent/skill_loader.py` 会校验 Skill 的 YAML frontmatter、文件位置和正文长度，
统一 Context Manager 再将它加入本轮学科 Agent Prompt。小学数学 Skill 要求先
识别题型、优先查询本地知识、数值运算调用计算器、检查单位，并按照学生年级
分步骤讲解。修改 Skill 后需要重启服务，因为加载结果会被缓存。

知识库位于 `data/knowledge/`，默认包含语文、数学、英语、科学、
安全与品德五个主题。默认向量索引写入
`storage/knowledge_vectors.sqlite3`，不需要提交到 Git。

## 向量存储切换

默认检索路径使用不调用原生向量库的 SQLite 后端。它把文档、元数据和 float32
向量保存在 SQLite 中，并使用“BM25 关键词分数 + 向量余弦相似度”混合排序，
足以处理当前数百个知识分片。原来的 Chroma 实现保留在
`rag/vector_backends.py`。

默认混合权重位于 `config/knowledge.yml`：

```yaml
hybrid_vector_weight: 0.65
hybrid_bm25_weight: 0.35
bm25_k1: 1.5
bm25_b: 0.75
```

`text-embedding-v4` 的余弦相似度负责理解语义相近的表达，BM25 负责提高公式、
专有词和精确关键词命中率。中文 BM25 会同时使用汉字单字和相邻双字，例如
“三角形”会产生“三、角、形、三角、角形”，不依赖 SQLite 的英文分词器。

默认在线向量模型位于 `config/model.yml`：

```yaml
embedding_model_name: text-embedding-v4
```

如果临时不想使用百炼，可切换回项目保留的本地备用模型，然后重建索引：

```yaml
embedding_model_name: local-hashing-zh-v1
```

修改 `config/knowledge.yml` 中一行即可切换：

```yaml
vector_backend: sqlite
```

需要切回 Chroma 时改为：

```yaml
vector_backend: chroma
```

也可以只对当前终端临时覆盖配置：

```powershell
$env:EDUCATION_VECTOR_BACKEND = "chroma"
```

SQLite 和 Chroma 分别使用自己的索引清单和存储目录，因此来回切换时会独立
增量同步，不会把另一个后端的清单误认为自己的索引。首次切换到某个后端仍需
为该后端建立一次索引。当前 Windows 环境中的 Chroma 原生写入可能发生
`0xC0000005` 崩溃，遇到该问题时应保持 `sqlite`。

混合排序逻辑由项目统一实现，所以切换到 Chroma 后仍使用相同的 BM25、余弦
分数和权重；区别只是向量与文档由 Chroma 保存和读取。

## 环境要求

- Python 3.11 或更高版本
- DeepSeek 官方 API Key
- 仅用于 `text-embedding-v4` 的阿里云百炼 API Key

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

复制环境变量模板，并填写自己的密钥：

```powershell
Copy-Item .env.example .env
```

`.env` 已加入 `.gitignore`，不要把真实密钥提交到版本库。

聊天模型直接调用 DeepSeek 官方接口，不经过阿里云百炼：

```env
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

当前模型为 `deepseek-v4-flash`。为了降低小学生问答延迟并简化 Agent 工具循环，
默认关闭思考模式；可在 `config/model.yml` 修改 `thinking_enabled`。

Embedding 与聊天模型相互独立。只有知识文件建库和每次 RAG 查询向量化会使用：

```env
DASHSCOPE_API_KEY=你的_百炼_API_Key
```

这个 Key 不会交给 DeepSeek 聊天客户端，聊天内容仍直接发送到 DeepSeek 官方接口。

Tavily 是可选能力。在 `.env` 中填写下面配置后，学科 Agent 才能真正联网：

```env
TAVILY_API_KEY=你的_Tavily_API_Key
```

没有填写 Tavily Key、依赖未安装、网络请求失败或没有可信来源结果时，工具会
自动回退本地 SQLite 知识库，不会中断正常问答。

## Tavily 联网搜索

系统默认先使用 `search_knowledge` 检索本地 SQLite 混合索引。只有以下情况才
允许使用 `search_web`：

- 问题明确询问最新、现在、当前、今天等时效信息；
- 学科 Agent 已经检索本地知识，但资料明显不足。

`search_web` 只接收当前问题、学科和“是否已查本地知识”布尔标记，不接收会话
历史、长期记忆或学生内部编号。发送前还会自动隐藏常见姓名、邮箱、手机、
身份证号、学校和地址信息，
并把查询长度限制为 200 字。默认最多返回 3 条结果，而且只接受
`config/app.yml` 中 `trusted_domains` 白名单里的教育、政府和科普来源。网页
内容一律作为不可信参考文本，不执行其中出现的命令或角色设定。

## 首次运行

首次使用要调用 `text-embedding-v4` 把知识文件转换为语义向量：

```powershell
python main.py --sync-index
```

### 使用网页界面

启动本地 Web 服务：

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

然后在浏览器访问：

```text
http://127.0.0.1:8766/
```

网页支持学科选择、年级选择、示例问题、连续问答和移动端布局。模型在收到
第一个问题时才初始化，因此页面本身可以快速打开。浏览器会生成匿名
`student_id` 和 `thread_id`：刷新页面会恢复当前会话；点击“新对话”会创建新
线程并保留长期学习记忆；点击“清除我的记忆”会删除这个匿名学生的全部会话、
Checkpoint 和长期记忆。

### 使用命令行

提出问题：

```powershell
python main.py --question "为什么月亮会有圆缺变化？" --subject 科学 --grade 4
```

命令行需要继续已有会话时，可以重复传入同一组匿名 ID：

```powershell
python main.py --question "那月食又是什么？" --subject 科学 --grade 4 `
  --student-id cli-student-demo001 --thread-id cli-thread-demo001
```

不传问题时进入连续问答模式：

```powershell
python main.py --subject 综合 --grade 5
```

查看本地索引清单状态，此命令不会调用在线模型：

```powershell
python main.py --status
```

需要彻底重建索引时：

```powershell
python main.py --rebuild-index
```

## 扩充知识库

把 UTF-8 编码的 `.txt` 或 `.pdf` 文件放入 `data/knowledge/`。文件名建议
以学科开头，例如：

```text
数学_分数专题.txt
科学_植物生长.pdf
```

下一次检索或执行 `--sync-index` 时，系统会通过 SHA-256 清单识别新增、
修改和删除的文件，并同步对应向量，不会把旧版本内容永久残留在索引中。
`text-embedding-v4` 单次最多处理 10 段文本，项目会按照
`embedding_batch_size` 自动拆批，所以知识库增加到数百个分片也能正常建库。
更换 Embedding 算法或维度后应执行 `python main.py --rebuild-index`，避免继续使用
旧向量。

支持的学科、切片大小、召回数量和模型名称分别位于：

- `config/app.yml`
- `config/knowledge.yml`
- `config/model.yml`

## 记忆机制

记忆数据库默认为 `storage/education_memory.sqlite3`，其中包含三类数据：

```text
LangGraph Checkpoint 表
  → 按 student_id + thread_id 保存 Agent State
  → 提供短期连续对话和服务重启恢复

education_messages / education_conversations
  → 保存学生可见的完整会话历史
  → 用于网页刷新恢复和历史列表

education_long_term_memories
  → 保存提炼后的年级、讲解偏好、困难点和学习主题
  → 可以跨不同 thread_id 检索
```

长期记忆不会把所有历史消息都注入模型。系统只选择当前问题相关的少量记忆，
并明确标记为不可信背景数据。知识库索引和学生记忆使用不同的数据文件。

可以通过环境变量覆盖数据库位置：

```text
EDUCATION_MEMORY_DB_PATH=storage/education_memory.sqlite3
```

主要记忆接口：

```text
GET    /api/history/{thread_id}          当前会话记录
GET    /api/conversations                匿名学生的会话列表
GET    /api/memories                     匿名学生的长期记忆
DELETE /api/conversations/{thread_id}    删除一个会话及其 Checkpoint
DELETE /api/students/{student_id}/memory 删除该匿名学生的全部记忆
```

## 测试

离线测试不会调用 DeepSeek、Tavily 或其他在线 API：

```powershell
python -m unittest discover -s tests -v
```

## 使用边界

本项目用于辅助学习，不代替学校课程、老师指导或紧急情况下的专业帮助。
系统不会主动索取学生姓名、学校、住址、电话等个人信息。涉及受伤、火灾、
欺凌、陌生人或其他危险时，应优先联系家长、老师或当地紧急服务。
