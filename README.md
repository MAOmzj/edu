<!--
文件用途：面向访客的简化说明：项目是什么、怎么运行。

修改易踩坑：命令和路径必须与实际实现一致，示例中不能出现真实密钥。
-->
# 小学教育知识问答系统

面向小学一至六年级学生的中文教育知识问答项目，基于 LangChain/LangGraph
多 Agent、DeepSeek 官方 API 和本地 SQLite 知识库，提供网页聊天与命令行
两种使用方式。

## 主要功能

- 语文、数学、英语、科学、安全与品德知识问答，按学科和年级调整回答；
- 学科回答、独立审核、学习记忆三个子 Agent 协作；
- 本地知识库 RAG 检索（SQLite 混合检索），资料不足或询问时效信息时按需联网；
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

# 3. 建立知识库索引（首次运行）
python main.py --sync-index
```

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
model/                  模型工厂与 Embedding
config/                 YAML 配置
data/knowledge/         知识库文件
web/                    前端页面
skills/                 教育 Skill
```



## 使用边界

本项目用于辅助学习，不代替学校课程、老师指导或紧急情况下的专业帮助。
