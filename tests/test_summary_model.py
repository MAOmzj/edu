# 文件用途：离线验证独立摘要模型的配置边界、环境覆盖与客户端构造。
# 调用关系：unittest 调用本文件；本文件调用 model/factory.py 和 utils/config_handler.py。
# 修改易踩坑：必须隔离环境变量并模拟 ChatOpenAI；不读 .env、不调用在线模型、不启动服务。
"""独立摘要客户端与配置契约测试，不涉及工作流的模型路由。"""

import asyncio
import os
import unittest
from unittest.mock import patch, sentinel

import httpx

# 工厂导入时会加载 .env；本测试只使用明确的测试环境和 YAML 配置。
with patch("dotenv.load_dotenv"):
    from model import factory

from utils.config_handler import validate_summary_model_config
from utils import config_handler


class SummaryModelConfigTests(unittest.TestCase):
    """配置缺省兼容旧版本，错误值在客户端创建前拒绝。"""

    def test_missing_options_default_to_disabled_without_mutating_input(self):
        supplied = {"enabled": True, "model_name": "test-summary"}
        options = validate_summary_model_config(supplied)
        self.assertEqual(supplied, {"enabled": True, "model_name": "test-summary"})
        self.assertEqual(options["model_name"], "test-summary")
        self.assertIs(options["enable_thinking"], False)
        self.assertIs(validate_summary_model_config({})["enabled"], False)

    def test_getter_keeps_legacy_configuration_disabled(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(config_handler.model_conf, {}, clear=True),
        ):
            options = config_handler.get_summary_model_config()
        self.assertIs(options["enabled"], False)
        self.assertIs(options["enable_thinking"], False)

    def test_getter_applies_environment_overrides_without_mutating_yaml(self):
        supplied = {
            "enabled": False,
            "model_name": "test-config-summary",
            "base_url": "http://127.0.0.1:18002/v1",
        }
        snapshot = dict(supplied)
        environment = {
            "EDUCATION_SUMMARY_ENABLED": "true",
            "EDUCATION_SUMMARY_MODEL_NAME": "test-env-summary",
            "EDUCATION_SUMMARY_BASE_URL": "http://127.0.0.1:28002/v1",
        }
        with (
            patch.dict(config_handler.model_conf, {"summary": supplied}, clear=True),
            patch.dict(os.environ, environment, clear=True),
        ):
            options = config_handler.get_summary_model_config()
        self.assertEqual(supplied, snapshot)
        self.assertIs(options["enabled"], True)
        self.assertEqual(options["model_name"], "test-env-summary")
        self.assertEqual(options["base_url"], "http://127.0.0.1:28002/v1")

    def test_config_requires_object_and_strict_boolean_switches(self):
        for invalid in (None, [], True, "enabled"):
            with self.subTest(config=invalid), self.assertRaises(ValueError):
                validate_summary_model_config(invalid)
        for key in ("enabled", "enable_thinking"):
            for invalid in (None, 0, 1, "true", "false"):
                with self.subTest(field=key, value=invalid), self.assertRaises(ValueError):
                    validate_summary_model_config({key: invalid})

    def test_required_text_does_not_coerce_none_or_accept_whitespace(self):
        for key in ("provider", "model_name", "model_path", "base_url"):
            for invalid in (None, False, 123, "", " \t "):
                with self.subTest(field=key, value=invalid), self.assertRaises(ValueError):
                    validate_summary_model_config({key: invalid})

    def test_numeric_limits_reject_boolean_none_nonfinite_and_invalid_ranges(self):
        invalid_by_field = {
            "temperature": (True, None, float("nan"), float("inf"), -0.1, 2.1),
            "request_timeout": (True, None, float("nan"), float("inf"), 0, -1),
            "max_retries": (True, None, -1, 0.5),
            "max_tokens": (True, None, 0, -1, 1.5),
        }
        for key, invalid_values in invalid_by_field.items():
            for invalid in invalid_values:
                with self.subTest(field=key, value=invalid), self.assertRaises(ValueError):
                    validate_summary_model_config({key: invalid})
        options = validate_summary_model_config(
            {"temperature": 0, "request_timeout": 0.5, "max_retries": 0, "max_tokens": 1}
        )
        self.assertEqual(options["max_retries"], 0)
        self.assertEqual(options["max_tokens"], 1)

    def test_endpoint_rejects_credentials_invalid_schemes_hosts_and_ports(self):
        invalid_urls = (
            "http://test-user:test-pass@127.0.0.1:8002/v1",
            "http://test-user@127.0.0.1:8002/v1",
            "ftp://127.0.0.1:8002/v1",
            "http:///v1",
            "http://127.0.0.1:0/v1",
            "http://127.0.0.1:65536/v1",
            "http://127.0.0.1:invalid/v1",
        )
        for url in invalid_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_summary_model_config({"base_url": url})
        options = validate_summary_model_config({"base_url": "https://localhost:8002/v1"})
        self.assertEqual(options["base_url"], "https://localhost:8002/v1")


class SummaryModelFactoryTests(unittest.TestCase):
    """模拟客户端构造，验证摘要不继承主模型或外部部署环境。"""

    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client_patch = patch("langchain_openai.ChatOpenAI", return_value=sentinel.summary)
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.key_patch = patch.object(factory, "_local_api_key", return_value="test-placeholder")
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.addCleanup(self._close_http_clients)

    def _close_http_clients(self):
        """模拟模型没有生命周期管理，测试负责关闭工厂创建的客户端。"""
        for call in self.client.call_args_list:
            sync_client = call.kwargs.get("http_client")
            if isinstance(sync_client, httpx.Client):
                sync_client.close()
            async_client = call.kwargs.get("http_async_client")
            if isinstance(async_client, httpx.AsyncClient):
                asyncio.run(async_client.aclose())

    def test_disabled_and_legacy_configuration_do_not_create_client(self):
        for config in ({}, {"summary": {}}, {"summary": {"enabled": False}}):
            with self.subTest(config=config), patch.dict(factory.model_conf, config, clear=True):
                self.assertIsNone(factory.create_summary_model())
        self.client.assert_not_called()

    def test_enabled_client_uses_independent_parameters_and_has_no_tools(self):
        summary = {
            "enabled": True,
            "model_name": "test-summary-small",
            "base_url": "http://127.0.0.1:18002/v1/",
            "temperature": 0.7,
            "request_timeout": 12.5,
            "max_retries": 0,
            "max_tokens": 321,
            "enable_thinking": False,
        }
        main = {
            "chat_provider": "deepseek",
            "chat_model_name": "test-main-large",
            "chat_base_url": "http://127.0.0.1:18000/v1",
            "request_timeout": 180,
            "temperature": 0.1,
            "max_retries": 2,
            "thinking_enabled": True,
            "summary": summary,
        }
        with patch.dict(factory.model_conf, main, clear=True):
            self.assertIs(factory.create_summary_model(), sentinel.summary)
        self.client.assert_called_once()
        arguments = dict(self.client.call_args.kwargs)
        self.assertIsInstance(arguments.pop("http_client"), httpx.Client)
        self.assertIsInstance(arguments.pop("http_async_client"), httpx.AsyncClient)
        self.assertEqual(
            arguments,
            {
                "model": "test-summary-small",
                "base_url": "http://127.0.0.1:18002/v1",
                "api_key": "test-placeholder",
                "temperature": 0.7,
                "timeout": 12.5,
                "max_retries": 0,
                "max_tokens": 321,
                "openai_proxy": "",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )
        self.assertNotIn("tools", self.client.call_args.kwargs)

    def test_local_sync_and_async_routes_ignore_environment_proxies(self):
        environment = {
            "HTTP_PROXY": "http://127.0.0.1:18999",
            "HTTPS_PROXY": "http://127.0.0.1:18999",
            "ALL_PROXY": "http://127.0.0.1:18999",
            "NO_PROXY": "",
            "OPENAI_PROXY": "http://127.0.0.1:18999",
        }
        with (
            patch.dict(os.environ, environment),
            patch.dict(factory.model_conf, {"summary": {"enabled": True}}, clear=True),
            httpx.Client() as default_sync,
        ):
            default_async = httpx.AsyncClient()
            try:
                factory.create_summary_model()
                arguments = self.client.call_args.kwargs
                sync_client = arguments["http_client"]
                async_client = arguments["http_async_client"]
                self.assertEqual(arguments["openai_proxy"], "")
                for scheme in ("http", "https"):
                    url = httpx.URL(f"{scheme}://127.0.0.1:8002/v1")
                    with self.subTest(scheme=scheme):
                        # 只检查路由选择，不发送请求；对照证明测试环境确实启用了代理。
                        self.assertIsNot(default_sync._transport_for_url(url), default_sync._transport)
                        self.assertIsNot(default_async._transport_for_url(url), default_async._transport)
                        self.assertIs(sync_client._transport_for_url(url), sync_client._transport)
                        self.assertIs(async_client._transport_for_url(url), async_client._transport)
            finally:
                asyncio.run(default_async.aclose())

    def test_environment_can_enable_summary_and_override_model_and_endpoint(self):
        environment = {
            "EDUCATION_SUMMARY_ENABLED": "true",
            "EDUCATION_SUMMARY_MODEL_NAME": "test-env-summary",
            "EDUCATION_SUMMARY_BASE_URL": "http://127.0.0.1:28002/v1/",
            "EDUCATION_CHAT_PROVIDER": "deepseek",
            "LOCAL_VLLM_CHAT_BASE_URL": "http://127.0.0.1:28000/v1",
        }
        with (
            patch.dict(factory.model_conf, {"summary": {"enabled": False}}, clear=True),
            patch.dict(os.environ, environment),
        ):
            self.assertIs(factory.create_summary_model(), sentinel.summary)
        kwargs = self.client.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-env-summary")
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:28002/v1")
        self.assertIs(kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)

    def test_environment_disable_prevents_client_creation(self):
        with (
            patch.dict(factory.model_conf, {"summary": {"enabled": True}}, clear=True),
            patch.dict(os.environ, {"EDUCATION_SUMMARY_ENABLED": "false"}),
        ):
            self.assertIsNone(factory.create_summary_model())
        self.client.assert_not_called()

    def test_invalid_environment_overrides_fail_before_client_creation(self):
        invalid_overrides = (
            {"EDUCATION_SUMMARY_ENABLED": "maybe"},
            {"EDUCATION_SUMMARY_MODEL_NAME": " "},
            {"EDUCATION_SUMMARY_BASE_URL": "http://test-user:test-pass@localhost/v1"},
        )
        for environment in invalid_overrides:
            with (
                self.subTest(environment=environment),
                patch.dict(factory.model_conf, {"summary": {"enabled": True}}, clear=True),
                patch.dict(os.environ, environment, clear=True),
                self.assertRaises(ValueError),
            ):
                factory.create_summary_model()
        self.client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
