"""``create_llm`` / ``create_chunking_llm`` forward the settings through ``init_chat_model``.

The real provider classes are constructed (offline: construction makes no network
call) and their fields asserted. Patching ``langchain_openai.ChatOpenAI`` is not
reliable because ``init_chat_model`` caches the resolved class per provider.
``rag_service`` imports ``embed_service`` (heavy) and builds the default LLM at
import, so it is executed under a private module name with a stub ``embed_service``.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import services as services_pkg
from services import chunking_service

RAG_SERVICE_PATH = Path(__file__).resolve().parents[2] / "services" / "rag_service.py"
HOSTED = {"model": "test-model", "temperature": 0.3, "max_tokens": 512, "top_p": 0.9,
          "api_key": "test-key", "base_url": "http://llm.local/v1"}


def _load_rag_service():
    """Execute ``rag_service`` offline with ``services.embed_service`` stubbed during the load."""
    stub = types.ModuleType("services.embed_service")
    with patch.dict(sys.modules, {"services.embed_service": stub}), patch.object(
        services_pkg, "embed_service", stub, create=True
    ):
        spec = importlib.util.spec_from_file_location("rag_service_under_test", RAG_SERVICE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _settings(prefix, provider, *, local=False):
    """Settings stub with ``hosted_<prefix>_*`` fields."""
    fields = {"provider": provider, "model": HOSTED["model"], "temperature": HOSTED["temperature"],
              "max_tokens": HOSTED["max_tokens"], "top_p": HOSTED["top_p"],
              "api_key": HOSTED["api_key"], "base_url": HOSTED["base_url"]}
    return SimpleNamespace(use_local_llm=local, ollama_llm_model="gemma4", groq_reasoning_effort="low",
                           **{f"hosted_{prefix}_{key}": value for key, value in fields.items()})


class CreateLlmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rag_service = _load_rag_service()

    def _create(self, provider, **kwargs):
        with patch.object(self.rag_service, "settings", _settings("llm", provider, **kwargs)):
            return self.rag_service.create_llm()

    def test_openai_fields(self) -> None:
        llm = self._create("openai")
        self.assertEqual(type(llm).__name__, "ChatOpenAI")
        self.assertEqual(
            (llm.model_name, llm.temperature, llm.max_tokens, llm.top_p, llm.openai_api_base,
             llm.openai_api_key.get_secret_value()),
            ("test-model", 0.3, 512, 0.9, "http://llm.local/v1", "test-key"),
        )

    def test_groq_fields_and_reasoning_effort(self) -> None:
        llm = self._create("groq")
        self.assertEqual(type(llm).__name__, "ChatGroq")
        self.assertEqual(
            (llm.model_name, llm.temperature, llm.max_tokens, llm.model_kwargs, llm.groq_api_base,
             llm.groq_api_key.get_secret_value(), llm.reasoning_effort),
            ("test-model", 0.3, 512, {"top_p": 0.9}, "http://llm.local/v1", "test-key", "low"),
        )

    def test_local_builds_ollama(self) -> None:
        llm = self._create("openai", local=True)
        self.assertEqual((type(llm).__name__, llm.model), ("ChatOllama", "gemma4"))

    def test_unsupported_provider_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            self._create("nope")


class CreateChunkingLlmTests(unittest.TestCase):
    def _create(self, provider, **kwargs):
        with patch.object(chunking_service, "settings", _settings("chunking", provider, **kwargs)):
            return chunking_service.create_chunking_llm()

    def test_groq_fields(self) -> None:
        llm = self._create("groq")
        self.assertEqual(
            (type(llm).__name__, llm.model_name, llm.max_tokens, llm.groq_api_base, llm.reasoning_effort),
            ("ChatGroq", "test-model", 512, "http://llm.local/v1", None),
        )


if __name__ == "__main__":
    unittest.main()
