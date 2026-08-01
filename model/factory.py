"""通义模型工厂；仅在实际调用时创建模型客户端。"""

import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from utils.config_handler import model_conf
from utils.path_tool import get_abs_path


load_dotenv(get_abs_path(".env"), override=False)


def require_dashscope_api_key() -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise EnvironmentError(
            "缺少 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填写密钥。"
        )


def create_chat_model() -> BaseChatModel:
    require_dashscope_api_key()
    return ChatTongyi(
        model=model_conf["chat_model_name"],
        temperature=float(model_conf.get("temperature", 0.2)),
    )


def create_embedding_model() -> Embeddings:
    require_dashscope_api_key()
    return DashScopeEmbeddings(model=model_conf["embedding_model_name"])
