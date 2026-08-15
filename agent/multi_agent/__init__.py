# 文件用途：集中导出三个子 Agent Tool 构造器和主协调工作流。
# 调用关系：education_agent.py 从这里导入；本文件转而导入 subject/review/memory/workflow 模块。
# 修改易踩坑：新增导出时注意循环导入，基础模块不能反向依赖这个聚合入口。
"""小学教育多 Agent：统一导出子 Agent Tool 和主协调工作流。"""

from agent.multi_agent.memory_agent import build_memory_agent_tool
from agent.multi_agent.review_agent import build_review_agent_tool
from agent.multi_agent.subject_agent import build_subject_agent_tool
from agent.multi_agent.workflow import EducationMultiAgentWorkflow

__all__ = [
    "EducationMultiAgentWorkflow",
    "build_memory_agent_tool",
    "build_review_agent_tool",
    "build_subject_agent_tool",
]
