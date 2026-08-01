# 小学教育知识问答系统

这是一个面向小学一至六年级学生的中文教育知识问答项目。系统使用
LangChain Agent、通义千问和本地 Chroma 知识库，支持：

- 简洁、响应式的网页聊天界面；
- 语文、数学、英语、科学、安全与品德知识问答；
- 按学科和年级调整回答；
- 基于 SQLite Checkpointer 的跨请求短期对话记忆；
- 跨会话长期学习记忆和可恢复的页面历史；
- 基于本地资料的 RAG 检索；
- 安全的加减乘除、括号、余数和乘方计算；
- 知识文件增删改后的自动增量索引；
- 不向学生展示内部推理和工具原始结果。

## 工作流程

```text
学生问题
  → 教育 Agent
  ├─ search_knowledge：检索本地小学知识
  └─ calculate_expression：核对基础算术
  → 通义模型整理为适龄答案
```

知识库位于 `data/knowledge/`，默认包含语文、数学、英语、科学、
安全与品德五个主题。向量索引生成到 `storage/`，不需要提交到 Git。

## 环境要求

- Python 3.11 或更高版本
- 阿里云百炼 DashScope API Key

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

复制环境变量模板，并填写自己的密钥：

```powershell
Copy-Item .env.example .env
```

`.env` 已加入 `.gitignore`，不要把真实密钥提交到版本库。

## 首次运行

首次使用要把知识文件转换为向量，该步骤会调用在线 Embedding 服务：

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

离线测试不会调用聊天模型或 Embedding API：

```powershell
python -m unittest discover -s tests -v
```

## 使用边界

本项目用于辅助学习，不代替学校课程、老师指导或紧急情况下的专业帮助。
系统不会主动索取学生姓名、学校、住址、电话等个人信息。涉及受伤、火灾、
欺凌、陌生人或其他危险时，应优先联系家长、老师或当地紧急服务。
