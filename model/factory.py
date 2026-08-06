"""DeepSeek 聊天模型与可切换的在线/本地 Embedding 工厂。

“工厂”就是集中创建对象的函数。本项目故意分开两个供应商：

    create_chat_model()      -> DeepSeek 官方 API，负责理解问题和生成文字
    create_embedding_model() -> 百炼 text-embedding-v4，负责把文本变成数字向量

两者使用不同环境变量，Embedding Key 不会被传给聊天模型，反之亦然。
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from model.local_embeddings import LocalHashEmbeddings
from utils.config_handler import model_conf
from utils.path_tool import get_abs_path


# 把项目根目录 .env 中的键加入环境变量。override=False 表示系统环境中已经设置的
# 值优先，避免本地文件意外覆盖部署平台注入的正式密钥。
load_dotenv(get_abs_path(".env"), override=False)


def require_deepseek_api_key() -> str:
    """读取 DeepSeek 官方 API Key，缺失时给出小白也能理解的提示。"""

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "缺少 DEEPSEEK_API_KEY。请在 .env 中填写 DeepSeek 官方 API Key。"
        )
    return api_key


def require_dashscope_api_key() -> str:
    """读取百炼 Embedding Key；它只用于向量化，不参与聊天模型调用。"""

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "text-embedding-v4 需要 DASHSCOPE_API_KEY，请在 .env 中填写百炼 Key。"
        )
    return api_key


def create_chat_model() -> BaseChatModel:
    """创建直连 api.deepseek.com、支持 Agent 工具调用的聊天模型。"""

    # 延迟导入让索引状态、离线测试等不使用聊天模型的命令仍可正常运行。
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError as exc:
        raise EnvironmentError(
            "缺少 langchain-deepseek，请先执行 python -m pip install -r requirements.txt"
        ) from exc

    api_key = require_deepseek_api_key()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    if not base_url:
        base_url = "https://api.deepseek.com"

    # 这里只负责“配置客户端”，真正的网络请求发生在后续 model.invoke() 时。
    return ChatDeepSeek(
        model=model_conf["chat_model_name"],
        api_key=api_key,
        base_url=base_url,
        temperature=float(model_conf.get("temperature", 0.2)),
        # DeepSeek V4 默认开启思考模式。多轮工具调用时思考模式要求额外回传
        # reasoning_content；教育问答默认关闭它，可减少延迟并避免工具循环 400。
        extra_body={
            "thinking": {
                "type": (
                    "enabled"
                    if bool(model_conf.get("thinking_enabled", False))
                    else "disabled"
                )
            }
        },
        timeout=float(model_conf.get("request_timeout", 60)),
        max_retries=int(model_conf.get("max_retries", 2)),
    )


def create_embedding_model() -> Embeddings:
    """按配置创建 text-embedding-v4，或创建无需账号的本地备用模型。"""

    embedding_name = str(model_conf["embedding_model_name"]).strip().lower()
    if embedding_name == "text-embedding-v4":
        # 延迟导入，确保仅查看状态或运行不涉及 Embedding 的测试时不会加载 SDK。
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as exc:
            raise EnvironmentError(
                "缺少 DashScope Embedding 依赖，请执行 python -m pip install -r requirements.txt"
            ) from exc
        # Embedding 客户端只有 embed_documents/embed_query 能力，不负责聊天。
        return DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=require_dashscope_api_key(),
            max_retries=int(model_conf.get("embedding_max_retries", 3)),
        )

    if embedding_name == "local-hashing-zh-v1":
        return LocalHashEmbeddings(
            dimension=int(model_conf.get("local_embedding_dimension", 1024))
        )

    raise ValueError(
        "embedding_model_name 只能是 text-embedding-v4 或 local-hashing-zh-v1"
    )
