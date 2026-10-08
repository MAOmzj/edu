<!--
文件用途：面向访客的简化说明：项目是什么、怎么运行。
调用关系：开发者按本文启动 Chroma、同步索引，再启动 FastAPI 或命令行入口。

修改易踩坑：命令和路径必须与实际实现一致，示例中不能出现真实密钥。
-->
# 小学教育知识问答系统

面向小学一至六年级学生的中文教育知识问答项目，基于 LangChain/LangGraph
多 Agent、DeepSeek、可切换的 SQLite/独立 Chroma/独立 Qdrant 向量后端
和 SQLite 记忆库，提供网页聊天与命令行两种使用方式。

## 主要功能

- 语文、数学、英语、科学、安全与品德知识问答，按学科和年级调整回答；
- 学科回答、独立审核、学习记忆三个子 Agent 协作；
- Chroma/Qdrant 原生语义召回、独立 BM25 倒排召回、加权 RRF 与可选 Rerank；
- 跨请求短期对话记忆与跨会话长期学习记忆；
- 安全的算术计算、知识文件自动增量索引。

## 工作流程

```text
学生问题
  → 上下文管理（会话摘要 + 长期记忆 + 教育 Skill）
  → 学科问答 Agent（本地检索 / 联网 / 计算器）
  → 审核 Agent（不通过则携带意见重写一次）
  → 学习记忆 Agent + SQLite 持久化
  → 返回审核后的最终答案
```

## 如何运行

### 环境要求

- Python 3.11 或更高版本
- Docker（使用独立 Chroma 或 Qdrant 时需要）
- DeepSeek API Key（聊天模型）
- 阿里云百炼 API Key（`text-embedding-v4` 向量化）
- Tavily API Key（可选，仅联网搜索需要）

### 安装与配置

```powershell
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 配置密钥
Copy-Item .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY 等密钥

# 3. 默认后端是 Qdrant，启动独立 Qdrant Server
docker compose -f deploy/qdrant/docker-compose.yml up -d --wait --wait-timeout 60

# 4. 建立知识库索引（首次运行或从嵌入式 Chroma 迁移后）
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

三种后端的数据和增量清单彼此独立，切换不会删除或覆盖原后端。默认配置使用
Qdrant；也可以只在当前进程环境中切换：

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
python -m uvicorn app:app --host 127.0.0.1 --port 8766
```

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
```



## 使用边界

本项目用于辅助学习，不代替学校课程、老师指导或紧急情况下的专业帮助。
