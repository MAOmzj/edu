# 文件用途：集中创建聊天、独立摘要模型与 Embedding，支持本地 vLLM 与云供应商可切换。
# 调用关系：education_agent.py 和 vector_store.py 调用本文件；本文件调用 LangChain 客户端、local_embeddings.py 和 os.environ。
# 修改易踩坑：聊天与向量密钥不能混用；建库和查询必须使用同一 Embedding 模型与维度；切换 Embedding 后必须重建索引。
"""聊天、独立摘要模型与 Embedding 工厂，按供应商创建对应客户端。

“工厂”就是集中创建对象的函数。本项目支持两套供应商，由 config/model.yml 的
``chat_provider`` / ``embedding_provider`` 选择：

    create_chat_model()
        local_vllm -> 本机 vllm serve 的 OpenAI 兼容接口（默认，离线可用）
        deepseek   -> DeepSeek 官方 API

    create_summary_model()
        local_vllm -> 独立摘要服务；关闭时返回 None，由调用方继续复用主模型

    create_embedding_model()
        local_vllm           -> 本机 vllm serve --task embed 的向量接口
        dashscope            -> 阿里云百炼 text-embedding-v4
        local-hashing-zh-v1  -> 项目自带、无需网络的哈希向量降级方案

两套供应商都可以用环境变量临时覆盖，便于不修改配置文件就切换。
"""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from model.local_embeddings import LocalHashEmbeddings
from utils.config_handler import get_summary_model_config, model_conf
from utils.path_tool import get_abs_path


# 把项目根目录 .env 中的键加入环境变量。override=False 表示系统环境中已经设置的
# 值优先，避免本地文件意外覆盖部署平台注入的正式密钥。
load_dotenv(get_abs_path(".env"), override=False)


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LOCAL_CHAT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LOCAL_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"

# 本机 vLLM 默认不校验 API Key，但 OpenAI 兼容客户端要求该字段非空，
# 所以用一个固定占位值；它不会被发送到任何外部服务。
LOCAL_VLLM_PLACEHOLDER_KEY = "EMPTY"

# Qwen3-Embedding 官方建议对“检索类查询”加任务指令、对文档侧不加。
# 不加也能用，加上能明显提升中文检索的召回质量。
DEFAULT_EMBEDDING_QUERY_INSTRUCTION = (
    "Instruct: 检索能回答小学生学习问题的中文知识资料\nQuery: "
)

CHAT_PROVIDERS = ("local_vllm", "deepseek")
EMBEDDING_PROVIDERS = ("local_vllm", "dashscope", "local-hashing-zh-v1")


def chat_provider() -> str:
    """返回聊天模型供应商；未配置时回退到 deepseek，保持升级前的行为。"""

    configured = os.getenv("EDUCATION_CHAT_PROVIDER") or model_conf.get(
        "chat_provider"
    )
    value = str(configured or "deepseek").strip().lower()
    if value not in CHAT_PROVIDERS:
        choices = "、".join(CHAT_PROVIDERS)
        raise ValueError(f"chat_provider 只能是：{choices}")
    return value


# 这两个模型名本身就唯一对应一种供应商，见到它们时直接采用，不再看 provider 字段。
# 这样旧版配置（只写 embedding_model_name）和现有测试都无需改动即可继续工作。
BUILTIN_EMBEDDING_PROVIDERS = {
    "text-embedding-v4": "dashscope",
    "local-hashing-zh-v1": "local-hashing-zh-v1",
}


def embedding_provider() -> str:
    """返回 Embedding 供应商。

    判定顺序是“环境变量 -> 内置模型名 -> 配置文件 provider 字段”：

    1. 设置了 ``EDUCATION_EMBEDDING_PROVIDER`` 时以它为准；
    2. 模型名是 ``text-embedding-v4`` / ``local-hashing-zh-v1`` 这类内置名称时，
       名称本身就唯一决定供应商，保证旧配置升级后行为不变；
    3. 其余情况（例如本机 vLLM 的 ``qwen3-embedding``）读取 provider 字段。
    """

    configured = os.getenv("EDUCATION_EMBEDDING_PROVIDER", "").strip().lower()
    if configured:
        return _validate_embedding_provider(configured)

    name = str(model_conf.get("embedding_model_name", "")).strip().lower()
    if name in BUILTIN_EMBEDDING_PROVIDERS:
        return BUILTIN_EMBEDDING_PROVIDERS[name]

    value = str(model_conf.get("embedding_provider", "")).strip().lower()
    if not value:
        raise ValueError(
            f"Embedding 模型 {name!r} 不是内置模型名，"
            "必须在 config/model.yml 中显式配置 embedding_provider"
        )
    return _validate_embedding_provider(value)


def _validate_embedding_provider(value: str) -> str:
    """校验唯一的 Embedding 供应商取值，非法时指出可选范围。"""

    if value not in EMBEDDING_PROVIDERS:
        choices = "、".join(EMBEDDING_PROVIDERS)
        raise ValueError(f"embedding_provider 只能是：{choices}")
    return value


def _config_text(key: str, env_name: str, default: str) -> str:
    """读取“环境变量优先、配置文件其次、最后取默认值”的地址类配置。"""

    value = str(os.getenv(env_name) or model_conf.get(key) or default).strip()
    return value or default


def _local_api_key() -> str:
    """返回本机 vLLM 的占位 Key，允许用环境变量覆盖成真实鉴权值。"""

    return os.getenv("LOCAL_VLLM_API_KEY", "").strip() or LOCAL_VLLM_PLACEHOLDER_KEY


def require_deepseek_api_key() -> str:
    """读取 DeepSeek 官方 API Key，缺失时给出小白也能理解的提示。"""

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "缺少 DEEPSEEK_API_KEY。请在 .env 中填写 DeepSeek 官方 API Key，"
            "或把 config/model.yml 的 chat_provider 改为 local_vllm 使用本机模型。"
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
    """按配置创建聊天模型：默认本机 vLLM，也可切回 DeepSeek 官方 API。"""

    if chat_provider() == "local_vllm":
        return _create_local_vllm_chat_model()
    return _create_deepseek_chat_model()


def create_summary_model() -> BaseChatModel | None:
    """开启时创建独立摘要客户端；关闭时返回 None，由调用方沿用主模型。"""

    options = get_summary_model_config()
    if not options["enabled"]:
        return None

    # 独立初始化避免摘要占用主模型的生成参数；只创建客户端，不连接或启动服务。
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise EnvironmentError(
            "缺少 langchain-openai，请先执行 python -m pip install langchain-openai"
        ) from exc
    return ChatOpenAI(
        model=options["model_name"],
        base_url=options["base_url"].rstrip("/"),
        api_key=_local_api_key(),
        temperature=float(options["temperature"]),
        timeout=float(options["request_timeout"]),
        max_retries=options["max_retries"],
        max_tokens=options["max_tokens"],
        # 本地摘要服务直接连接；HTTP_PROXY/HTTPS_PROXY 与 OpenAI 专用代理均不适用。
        openai_proxy="",
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
        extra_body={
            "chat_template_kwargs": {"enable_thinking": options["enable_thinking"]}
        },
    )


def _create_local_vllm_chat_model() -> BaseChatModel:
    """创建指向本机 vllm serve 的 OpenAI 兼容聊天模型。"""

    # 延迟导入让索引状态、离线测试等不使用聊天模型的命令仍可正常运行。
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise EnvironmentError(
            "缺少 langchain-openai，请先执行 python -m pip install langchain-openai"
        ) from exc

    base_url = _config_text(
        "chat_base_url", "LOCAL_VLLM_CHAT_BASE_URL", DEFAULT_LOCAL_CHAT_BASE_URL
    ).rstrip("/")
    thinking = bool(model_conf.get("thinking_enabled", False))

    # 本机 vLLM 的 Qwen 对话模板用 chat_template_kwargs.enable_thinking 控制
    # 是否输出思考段，等价于 DeepSeek 的 extra_body.thinking，但参数名不同。
    # 关闭思考能显著降低延迟，并且避免多轮工具调用回传 reasoning_content。
    return ChatOpenAI(
        model=str(model_conf["chat_model_name"]).strip(),
        base_url=base_url,
        api_key=_local_api_key(),
        temperature=float(model_conf.get("temperature", 0.2)),
        timeout=float(model_conf.get("request_timeout", 60)),
        max_retries=int(model_conf.get("max_retries", 2)),
        # 下游 httpx 直连本地 vLLM，防止机器上的 HTTP_PROXY 把请求送往代理而报502。
        openai_proxy="",
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )


def _create_deepseek_chat_model() -> BaseChatModel:
    """创建直连 api.deepseek.com、支持 Agent 工具调用的聊天模型。"""

    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError as exc:
        raise EnvironmentError(
            "缺少 langchain-deepseek，请先执行 python -m pip install -r requirements.txt"
        ) from exc

    api_key = require_deepseek_api_key()
    base_url = os.getenv(
        "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL
    ).strip() or DEFAULT_DEEPSEEK_BASE_URL

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


class QueryInstructionEmbeddings(Embeddings):
    """给查询加任务指令、文档侧保持原样的 Embedding 包装器。

    Qwen3-Embedding 这类模型对“检索查询”和“被检索文档”建议使用不同的输入格式。
    本项目建库走 ``embed_documents``、提问走 ``embed_query``，因此只在查询侧加前缀，
    不会破坏“建库与查询使用同一向量空间”的前提。
    """

    def __init__(self, delegate: Embeddings, query_instruction: str) -> None:
        self._delegate = delegate
        self._query_instruction = query_instruction or ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """按原样向量化文档，不做任何前缀改写。"""

        return self._delegate.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """在查询前拼接任务指令后再向量化。"""

        prefix = self._query_instruction
        return self._delegate.embed_query(f"{prefix}{text}" if prefix else text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步版本，行为与 embed_documents 一致。"""

        return await self._delegate.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """异步版本，行为与 embed_query 一致。"""

        prefix = self._query_instruction
        return await self._delegate.aembed_query(
            f"{prefix}{text}" if prefix else text
        )


def create_embedding_model() -> Embeddings:
    """按配置创建 Embedding：本机 vLLM、百炼，或无需账号的本地备用模型。"""

    provider = embedding_provider()
    if provider == "local_vllm":
        return _create_local_vllm_embeddings()
    if provider == "dashscope":
        return _create_dashscope_embeddings()
    return LocalHashEmbeddings(
        dimension=int(model_conf.get("local_embedding_dimension", 1024))
    )


def _create_local_vllm_embeddings() -> Embeddings:
    """创建指向本机 vllm serve --task embed 的向量模型。"""

    try:
        import httpx
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise EnvironmentError(
            "缺少 langchain-openai，请先执行 python -m pip install langchain-openai"
        ) from exc

    base_url = _config_text(
        "embedding_base_url",
        "LOCAL_VLLM_EMBEDDING_BASE_URL",
        DEFAULT_LOCAL_EMBEDDING_BASE_URL,
    ).rstrip("/")

    delegate = OpenAIEmbeddings(
        model=str(model_conf["embedding_model_name"]).strip(),
        base_url=base_url,
        api_key=_local_api_key(),
        # 非 OpenAI 服务不需要客户端按 tiktoken 预估长度并预切分，
        # 关闭后直接把原始文本交给 vLLM，避免长度估算与实际分词不一致。
        check_embedding_ctx_length=False,
        max_retries=int(model_conf.get("embedding_max_retries", 3)),
        # 下游 httpx 与聊天端一致直连本机；检索不应受外部代理环境变量影响。
        openai_proxy="",
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )
    instruction = str(
        model_conf.get("embedding_query_instruction")
        or DEFAULT_EMBEDDING_QUERY_INSTRUCTION
    )
    return QueryInstructionEmbeddings(delegate, instruction)


def _create_dashscope_embeddings() -> Embeddings:
    """创建阿里云百炼 text-embedding-v4 向量模型。"""

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
